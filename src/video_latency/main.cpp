#include <gtkmm.h>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <gdk/gdkx.h>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <string>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <mutex>

#include <cmath>
#include <numeric>

#include "config.hpp"

class LatencyWindow : public Gtk::Window {
public:
    LatencyWindow(const std::string& config_path, const dc::VideoConfig& v)
        : m_config_path(config_path),
          m_video_config(v),
          m_vbox(Gtk::ORIENTATION_VERTICAL, 5),
          m_measuring(false),
          m_flash_white(false),
          m_baseline(0.0),
          m_samples_collected(0),
          m_target_samples(10)
    {
        set_title("Latency Estimation Tool - " + v.name);
        set_default_size(800, 600);

        set_border_width(10);
        add(m_vbox);

        // Preview Frame
        m_preview_frame.set_label("Preview: " + v.name);
        m_preview_frame.set_shadow_type(Gtk::SHADOW_ETCHED_IN);
        m_vbox.pack_start(m_preview_frame, Gtk::PACK_EXPAND_WIDGET);

        // Overlay for Flash
        m_overlay.set_hexpand(true);
        m_overlay.set_vexpand(true);
        m_preview_frame.add(m_overlay);

        // Container for video widget
        m_video_container.set_orientation(Gtk::ORIENTATION_HORIZONTAL);
        m_overlay.add(m_video_container);

        // Flash Layer (DrawingArea on top)
        m_flash_layer.set_hexpand(true);
        m_flash_layer.set_vexpand(true);
        m_flash_layer.set_halign(Gtk::ALIGN_FILL);
        m_flash_layer.set_valign(Gtk::ALIGN_FILL);

        // Draw black or white when visible
        m_flash_layer.signal_draw().connect([this](const Cairo::RefPtr<Cairo::Context>& cr){
            if (m_flash_white) cr->set_source_rgb(1.0, 1.0, 1.0);
            else cr->set_source_rgb(0.0, 0.0, 0.0);
            cr->paint();
            return true;
        });

        m_overlay.add_overlay(m_flash_layer);
        m_flash_layer.set_no_show_all(true);
        m_flash_layer.hide();

        // Bottom Controls
        Gtk::Box* hbox = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 5));
        m_vbox.pack_start(*hbox, Gtk::PACK_SHRINK);

        m_btn_estimate.set_label("Estimate Latency");
        m_btn_estimate.signal_clicked().connect(sigc::mem_fun(*this, &LatencyWindow::on_estimate_latency));
        hbox->pack_start(m_btn_estimate, Gtk::PACK_SHRINK);

        m_lbl_result.set_text("Result: N/A");
        hbox->pack_start(m_lbl_result, Gtk::PACK_SHRINK);

        m_lbl_level.set_text("Level: 0.0");
        hbox->pack_start(m_lbl_level, Gtk::PACK_SHRINK);

        show_all_children();

        load_pipeline();
    }

    virtual ~LatencyWindow() {
        if (m_pipeline) {
            gst_element_set_state(m_pipeline, GST_STATE_NULL);
            gst_object_unref(m_pipeline);
        }
    }

protected:
    std::string m_config_path;
    dc::VideoConfig m_video_config;
    Gtk::Box m_vbox;
    Gtk::Frame m_preview_frame;
    Gtk::Overlay m_overlay;
    Gtk::Box m_video_container;
    Gtk::DrawingArea m_flash_layer;

    Gtk::Button m_btn_estimate;
    Gtk::Label m_lbl_result;
    Gtk::Label m_lbl_level;

    GstElement* m_pipeline = nullptr;
    GstElement* m_sink = nullptr;

    // Measurement State
    std::mutex m_state_mutex;
    bool m_measuring;
    bool m_flash_white;
    std::chrono::steady_clock::time_point m_start_time;
    double m_baseline;
    double m_latched_baseline;

    // Multi-sample State
    int m_samples_collected;
    int m_target_samples;
    std::vector<double> m_latencies;
    double m_last_mean = 0.0;

    // Dispatcher for UI updates from thread
    std::string m_ui_result_text;

    void on_estimate_latency() {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_measuring) return;

        if (m_baseline < 10.0 || m_baseline > 245.0) {
            m_lbl_result.set_text("Error: Baseline extreme (Camera dark or too bright?)");
            return;
        }

        m_latencies.clear();
        m_samples_collected = 0;
        m_latched_baseline = m_baseline;
        m_flash_white = false; // Start with black

        start_single_measurement();
    }

    void start_single_measurement() {
        m_measuring = true;
        m_start_time = std::chrono::steady_clock::now();
        m_flash_layer.show();
        m_flash_layer.queue_draw();

        m_lbl_result.set_text("Measuring sample " + std::to_string(m_samples_collected + 1) + "/" + std::to_string(m_target_samples) + " (" + (m_flash_white ? "WHITE":"BLACK") + ")...");

        Glib::signal_timeout().connect_once([this]() {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            if (m_measuring) {
                m_measuring = false;
                std::cerr << "Sample " << (m_samples_collected + 1) << " timed out." << std::endl;
                m_flash_layer.hide();

                Glib::signal_timeout().connect_once([this]() {
                     std::lock_guard<std::mutex> lock(m_state_mutex);
                     // Alternate even on timeout
                     m_flash_white = !m_flash_white;
                     start_single_measurement();
                }, 1500);
            }
        }, 1000);
    }

    void analyze_frame(GstBuffer* buffer) {
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            uint64_t sum = 0;
            int step = 64;
            if (map.size > 0) {
                for (size_t i = 0; i < map.size; i += step) {
                    sum += map.data[i];
                }
                double avg = (double)sum / (double)((map.size + step - 1) / step);

                std::lock_guard<std::mutex> lock(m_state_mutex);

                if (m_measuring) {
                    bool detected = false;
                    if (m_flash_white) {
                        // Look for spike (at least 20 levels higher than baseline)
                        if (avg > m_latched_baseline + 20.0 || avg > 250.0) detected = true;
                    } else {
                        // Look for drop (at least 20 levels lower than baseline)
                        if (avg < m_latched_baseline - 20.0 || avg < 5.0) detected = true;
                    }

                    if (detected) {
                        auto t1 = std::chrono::steady_clock::now();
                        double latency_ms = std::chrono::duration<double, std::milli>(t1 - m_start_time).count();

                        m_measuring = false;
                        m_latencies.push_back(latency_ms);
                        m_samples_collected++;

                        if (m_samples_collected < m_target_samples) {
                             Glib::signal_idle().connect_once([this](){
                                m_flash_layer.hide();
                                Glib::signal_timeout().connect_once([this]() {
                                    std::lock_guard<std::mutex> lock(m_state_mutex);
                                    m_flash_white = !m_flash_white; // Alternate
                                    start_single_measurement();
                                }, 1500);
                            });
                        } else {
                            double sum = std::accumulate(m_latencies.begin(), m_latencies.end(), 0.0);
                            double mean = sum / m_latencies.size();
                            double sq_sum = std::inner_product(m_latencies.begin(), m_latencies.end(), m_latencies.begin(), 0.0);
                            double stdev = std::sqrt(sq_sum / m_latencies.size() - mean * mean);
                            m_last_mean = mean;

                            std::stringstream ss;
                            ss << "Avg: " << std::fixed << std::setprecision(1) << mean
                               << "ms | StdDev: " << stdev << "ms (" << m_latencies.size() << ")";
                            m_ui_result_text = ss.str();

                            Glib::signal_idle().connect_once([this](){
                                m_flash_layer.hide();
                                m_lbl_result.set_text(m_ui_result_text);
                                prompt_save();
                            });
                        }
                    }
                } else {
                    if (m_samples_collected == 0) {
                         m_baseline = m_baseline * 0.95 + avg * 0.05;
                    }
                }

                Glib::signal_idle().connect_once([this, avg](){
                     m_lbl_level.set_text("Level: " + std::to_string((int)avg));
                });
            }
            gst_buffer_unmap(buffer, &map);
        }
    }

    void prompt_save() {
        Gtk::MessageDialog dialog(*this, "Calibration Finished", false, Gtk::MESSAGE_QUESTION, Gtk::BUTTONS_YES_NO);
        dialog.set_secondary_text("Estimated latency for " + m_video_config.name + " is " + 
                                  std::to_string((int)m_last_mean) + "ms.\nDo you want to save this to the configuration file?");
        int result = dialog.run();
        if (result == Gtk::RESPONSE_YES) {
            save_to_config();
        }
    }

    void save_to_config() {
        Json::Value root;
        if (!dc::Config::load_from_file(m_config_path, root)) return;

        if (root.isMember("videos") && root["videos"].isArray()) {
            for (auto& v : root["videos"]) {
                if (v.isMember("name") && v["name"].asString() == m_video_config.name) {
                    v["estimated_latency"] = m_last_mean / 1000.0; // In seconds
                    break;
                }
            }
            
            std::ofstream ofs(m_config_path);
            Json::StreamWriterBuilder builder;
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(root, &ofs);
            
            Gtk::MessageDialog info(*this, "Saved", false, Gtk::MESSAGE_INFO, Gtk::BUTTONS_OK);
            info.set_secondary_text("Updated estimated_latency for " + m_video_config.name + " in " + m_config_path);
            info.run();
        }
    }

    static GstPadProbeReturn probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
        (void)pad;
        LatencyWindow* self = static_cast<LatencyWindow*>(user_data);
        GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
        if (buffer) {
            self->analyze_frame(buffer);
        }
        return GST_PAD_PROBE_OK;
    }

    void load_pipeline() {
        std::string stream = m_video_config.stream;
        std::string caps = dc::Config::make_caps_string(m_video_config.encoding);

        std::string pstr = stream + " ! " + caps + " ! identity name=probe_identity ! videoconvert ! gtksink name=sink";
        std::cout << "Launching: " << pstr << std::endl;

        GError* err = nullptr;
        m_pipeline = gst_parse_launch(pstr.c_str(), &err);
        if (!m_pipeline) {
            std::cerr << "Failed to create pipeline: " << (err ? err->message : "unknown") << std::endl;
            if (err) g_error_free(err);
            return;
        }

        GstElement* identity = gst_bin_get_by_name(GST_BIN(m_pipeline), "probe_identity");
        if (identity) {
            GstPad* pad = gst_element_get_static_pad(identity, "src");
            if (pad) {
                gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, probe_cb, this, NULL);
                gst_object_unref(pad);
            }
            gst_object_unref(identity);
        }

        m_sink = gst_bin_get_by_name(GST_BIN(m_pipeline), "sink");
        if (m_sink) {
            GtkWidget* widget = nullptr;
            g_object_get(m_sink, "widget", &widget, NULL);
            if (widget) {
                Gtk::Widget* w = Glib::wrap(widget);
                m_video_container.pack_start(*w, Gtk::PACK_EXPAND_WIDGET);
                w->show();
            }
        }

        gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
    }
};

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    Gtk::Main kit(argc, argv);

    std::string config_path;
    std::string stream_name;

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "-c" && i+1 < argc) {
            config_path = argv[++i];
        } else if(arg == "-s" && i+1 < argc) {
            stream_name = argv[++i];
        }
    }

    if(config_path.empty()) {
        std::cerr << "Usage: latency_c -c <config.json> [-s <stream_name>]" << std::endl;
        return 1;
    }

    Json::Value root;
    if (!dc::Config::load_from_file(config_path, root)) return 1;

    std::vector<dc::VideoConfig> videos = dc::Config::parse_videos(root);
    if (videos.empty()) {
        std::cerr << "No videos found in config." << std::endl;
        return 1;
    }

    if (stream_name.empty()) {
        std::cout << "Available streams in " << config_path << ":" << std::endl;
        for (const auto& v : videos) {
            std::cout << "  - " << v.name << std::endl;
        }
        std::cout << "\nPlease specify a stream with -s <stream_name>" << std::endl;
        return 0;
    }

    dc::VideoConfig selected_v;
    bool found = false;
    for (const auto& v : videos) {
        if (v.name == stream_name) {
            selected_v = v;
            found = true;
            break;
        }
    }

    if (!found) {
        std::cerr << "Stream \"" << stream_name << "\" not found in config." << std::endl;
        return 1;
    }

    LatencyWindow win(config_path, selected_v);
    Gtk::Main::run(win);
    return 0;
}
