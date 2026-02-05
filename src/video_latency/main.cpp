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
    LatencyWindow(const std::string& config_path)
        : m_vbox(Gtk::ORIENTATION_VERTICAL, 5),
          m_measuring(false),
          m_baseline(0.0),
          m_samples_collected(0),
          m_target_samples(10)
    {
        set_title("Latency Estimation Tool");
        set_default_size(800, 600);

        set_border_width(10);
        add(m_vbox);

        // Preview Frame
        m_preview_frame.set_label("Preview");
        m_preview_frame.set_shadow_type(Gtk::SHADOW_ETCHED_IN);
        m_vbox.pack_start(m_preview_frame, Gtk::PACK_EXPAND_WIDGET);

        // Overlay for Flash
        m_overlay.set_hexpand(true);
        m_overlay.set_vexpand(true);
        m_preview_frame.add(m_overlay);

        // Container for video widget
        // Use a box to hold the video widget
        m_video_container.set_orientation(Gtk::ORIENTATION_HORIZONTAL);
        m_overlay.add(m_video_container);

        // Flash Layer (DrawingArea on top)
        // Ensure it fills the overlay (expand + fill align)
        m_flash_layer.set_hexpand(true);
        m_flash_layer.set_vexpand(true);
        m_flash_layer.set_halign(Gtk::ALIGN_FILL);
        m_flash_layer.set_valign(Gtk::ALIGN_FILL);

        // Draw black when visible
        m_flash_layer.signal_draw().connect([this](const Cairo::RefPtr<Cairo::Context>& cr){
            cr->set_source_rgb(0.0, 0.0, 0.0);
            cr->paint();
            return true;
        });

        m_overlay.add_overlay(m_flash_layer);
        m_flash_layer.set_no_show_all(true); // Don't show by default
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

        load_pipeline(config_path);
    }

    virtual ~LatencyWindow() {
        if (m_pipeline) {
            gst_element_set_state(m_pipeline, GST_STATE_NULL);
            gst_object_unref(m_pipeline);
        }
    }

protected:
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
    std::chrono::steady_clock::time_point m_start_time;
    double m_baseline;
    double m_latched_baseline;

    // Multi-sample State
    int m_samples_collected;
    int m_target_samples;
    std::vector<double> m_latencies;

    // Dispatcher for UI updates from thread
    std::string m_ui_result_text;

    void on_estimate_latency() {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        if (m_measuring) return; // Already running

        if (m_baseline < 10.0) {
            m_lbl_result.set_text("Error: Baseline too low (Camera dark?)");
            return;
        }

        m_latencies.clear();
        m_samples_collected = 0;
        m_latched_baseline = m_baseline; // Freeze baseline for the sequence

        start_single_measurement();
    }

    void start_single_measurement() {
        // MUST be called with m_state_mutex LOCKED or from UI thread safely?
        // This is called initially from UI thread, and then from idle callback (UI thread)
        // so we need to lock just to set shared state.

        m_measuring = true;
        m_start_time = std::chrono::steady_clock::now();
        m_flash_layer.show();
        m_flash_layer.queue_draw(); // Force redraw

        m_lbl_result.set_text("Measuring sample " + std::to_string(m_samples_collected + 1) + "/" + std::to_string(m_target_samples) + "...");

        // Timeout for failed detection (1.0 second)
        Glib::signal_timeout().connect_once([this]() {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            if (m_measuring) {
                // Timeout logic
                m_measuring = false;
                std::cerr << "Sample " << (m_samples_collected + 1) << " timed out." << std::endl;

                // Do NOT increment m_samples_collected, retrying this sample?
                // Or just ignore it? The user said "remove from statistics", which implies ignoring.
                // But we probably want to continue the sequence or abort?
                // Let's retry the sample or just skip?
                // "Remove that sample from statistics" usually means just don't count it.
                // But we still need to reach target?
                // I'll skip it but keep going to target count.
                // Wait, if I don't increment collected, we might loop forever if camera is blocked.
                // Let's increment but NOT add to m_latencies.
                // Or even better: Retry logic? No, simplistic approach: Skip sample.

                // If we skip, we have fewer samples at the end.
                // User asked "sequence of ten", so let's try to get 10 VALID samples.
                // But avoid infinite loop. Let's abort this sample, hide flash, and schedule next attempt.

                m_flash_layer.hide();

                // Schedule next attempt
                Glib::signal_timeout().connect_once([this]() {
                     std::lock_guard<std::mutex> lock(m_state_mutex);
                     // If we have failed too many times, maybe stop? For now infinite retry until 10 good samples.
                     start_single_measurement();
                }, 1500);
            }
        }, 1000); // 1.0 second timeout
    }

    // Called from GStreamer thread
    void analyze_frame(GstBuffer* buffer) {
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            uint64_t sum = 0;
            // Simple robust sampling: sum every 64th byte
            int step = 64;
            if (map.size > 0) {
                for (size_t i = 0; i < map.size; i += step) {
                    sum += map.data[i];
                }
                double avg = (double)sum / (double)((map.size + step - 1) / step);

                std::lock_guard<std::mutex> lock(m_state_mutex);

                // Detection logic
                if (m_measuring) {
                    if (avg < m_latched_baseline * 0.5) {
                        auto t1 = std::chrono::steady_clock::now();
                        double latency_ms = std::chrono::duration<double, std::milli>(t1 - m_start_time).count();

                        m_measuring = false;
                        m_latencies.push_back(latency_ms);
                        m_samples_collected++;

                        // If we need more samples, schedule the next flash
                        if (m_samples_collected < m_target_samples) {
                             Glib::signal_idle().connect_once([this](){
                                m_flash_layer.hide(); // Hide first to reset

                                // Wait a bit before next flash (e.g., 1.5s) to allow avg to stabilize and user to see the flash
                                Glib::signal_timeout().connect_once([this]() {
                                    std::lock_guard<std::mutex> lock(m_state_mutex);
                                    start_single_measurement();
                                }, 1500);
                            });
                        } else {
                            // All done
                            // Calculate Stats
                            double sum = std::accumulate(m_latencies.begin(), m_latencies.end(), 0.0);
                            double mean = sum / m_latencies.size();
                            double sq_sum = std::inner_product(m_latencies.begin(), m_latencies.end(), m_latencies.begin(), 0.0);
                            double stdev = std::sqrt(sq_sum / m_latencies.size() - mean * mean);

                            std::stringstream ss;
                            ss << "Avg: " << std::fixed << std::setprecision(1) << mean
                               << "ms | StdDev: " << stdev << "ms (" << m_latencies.size() << ")";
                            m_ui_result_text = ss.str();

                            Glib::signal_idle().connect_once([this](){
                                m_flash_layer.hide();
                                m_lbl_result.set_text(m_ui_result_text);
                            });
                        }
                    }
                } else {
                    // Update baseline (slow moving average) only if we are NOT in a sequence
                    if (m_samples_collected == 0) {
                         m_baseline = m_baseline * 0.95 + avg * 0.05;
                    }
                }

                // Throttle UI updates or just raw?
                // Updating every frame (30/60fps) is fine for a label
                Glib::signal_idle().connect_once([this, avg](){
                     m_lbl_level.set_text("Level: " + std::to_string((int)avg));
                });
            }
            gst_buffer_unmap(buffer, &map);
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

    void load_pipeline(const std::string& config_path) {
        Json::Value root;
        if (!dc::Config::load_from_file(config_path, root)) {
            return;
        }

        std::vector<dc::VideoConfig> videos = dc::Config::parse_videos(root);

        if (videos.empty()) {
            std::cerr << "No videos in config" << std::endl;
            return;
        }

        const auto& v = videos[0]; // Pick first video config
        if (v.stream.empty()) {
             std::cerr << "Video config missing 'stream' field" << std::endl;
             return;
        }

        std::string stream = v.stream;
        std::string caps = dc::Config::make_caps_string(v.encoding);

        // Insert identity name=probe_identity to attach our analysis probe
        // Order: Source -> ... -> Identity -> Videoconvert -> GTKSink
        std::string pstr = stream + " ! " + caps + " ! identity name=probe_identity ! videoconvert ! gtksink name=sink";
        std::cout << "Launching: " << pstr << std::endl;

        GError* err = nullptr;
        m_pipeline = gst_parse_launch(pstr.c_str(), &err);
        if (!m_pipeline) {
            std::cerr << "Failed to create pipeline: " << (err ? err->message : "unknown") << std::endl;
            if (err) g_error_free(err);
            return;
        }

        // Attach Probe
        GstElement* identity = gst_bin_get_by_name(GST_BIN(m_pipeline), "probe_identity");
        if (identity) {
            GstPad* pad = gst_element_get_static_pad(identity, "src");
            if (pad) {
                gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, probe_cb, this, NULL);
                gst_object_unref(pad);
            } else {
                 std::cerr << "Could not get src pad of probe_identity" << std::endl;
            }
            gst_object_unref(identity);
        } else {
             std::cerr << "Could not find probe_identity element" << std::endl;
        }

        // Attach Widget
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
    for(int i=1; i<argc; ++i) {
        if(std::string(argv[i]) == "-c" && i+1 < argc) {
            config_path = argv[++i];
        }
    }

    if(config_path.empty()) {
        std::cerr << "Usage: latency_c -c <config.json>" << std::endl;
        return 1;
    }

    LatencyWindow win(config_path);
    Gtk::Main::run(win);
    return 0;
}
