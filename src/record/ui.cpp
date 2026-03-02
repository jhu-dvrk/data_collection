#include "ui.hpp"
#include "video_stream.hpp"
#include "audio_stream.hpp"
#include "ros_node.hpp"
#include "tags.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <gtkmm.h>
#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <gdk/gdkx.h>
#include <rclcpp/rclcpp.hpp>

// Undefine X11 macros that clash with ROS/Qt/standard names
#ifdef Bool
#undef Bool
#endif
#ifdef Status
#undef Status
#endif
#ifdef True
#undef True
#endif
#ifdef False
#undef False
#endif

#include <std_msgs/msg/bool.hpp>

static MainWindow* g_main_window_instance = nullptr;

// C-compatible callback for ROS node
extern "C" gboolean toggle_recording_idle(gpointer) {
    if (g_main_window_instance) {
        g_main_window_instance->trigger_record_toggle();
    }
    return FALSE; // Remove source
}

MainWindow::MainWindow(AppData* data)
    : m_data(data),
      m_main_vbox(Gtk::ORIENTATION_VERTICAL, BOX_SPACING_PX),
      m_top_hbox(Gtk::ORIENTATION_HORIZONTAL, BOX_SPACING_PX),
      m_session_vbox(Gtk::ORIENTATION_VERTICAL, WIDGET_SPACING_PX),
      m_stages_vbox(Gtk::ORIENTATION_VERTICAL, WIDGET_SPACING_PX),
      m_tags_vbox(Gtk::ORIENTATION_VERTICAL, WIDGET_SPACING_PX),
      m_audio_vbox(Gtk::ORIENTATION_VERTICAL, WIDGET_SPACING_PX),
      m_audio_ctrl_hbox(Gtk::ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX),
      m_bag_vbox(Gtk::ORIENTATION_VERTICAL, WIDGET_SPACING_PX),
      m_bottom_hbox(Gtk::ORIENTATION_HORIZONTAL, BOX_SPACING_PX),
      m_record_button("Start Recording"),
      m_quit_button("Quit")
{
    set_title("Data Collection Recorder");
    set_default_size(1024, 768);
    set_border_width(WINDOW_BORDER_PX);

    // Sync AppData pointer for legacy C callbacks
    m_data->window = (GtkWidget*)this->gobj();

    add(m_main_vbox);

    // Top section: Left (Session/Audio/Streams) - Right (Stages)
    m_main_vbox.pack_start(m_top_hbox, Gtk::PACK_EXPAND_WIDGET);

    // Left side container
    Gtk::Box *f1a_vbox = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_VERTICAL, BOX_SPACING_PX));
    m_top_hbox.pack_start(*f1a_vbox, Gtk::PACK_EXPAND_WIDGET);

    // --- Session Frame ---
    Gtk::Label* session_lbl = Gtk::manage(new Gtk::Label());
    session_lbl->set_markup("<b>Session</b>");
    m_session_frame.set_label_widget(*session_lbl);
    m_session_frame.set_shadow_type(Gtk::SHADOW_ETCHED_IN);
    m_session_frame.add(m_session_vbox);
    m_session_vbox.set_border_width(FRAME_PADDING_PX);
    f1a_vbox->pack_start(m_session_frame, Gtk::PACK_SHRINK);

    // Session Directory
    Gtk::Box* dir_hbox = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX));
    m_session_vbox.pack_start(*dir_hbox, Gtk::PACK_SHRINK);
    dir_hbox->pack_start(*Gtk::manage(new Gtk::Label("Directory:")), Gtk::PACK_SHRINK);

    m_data_dir_entry.set_text(m_data->data_directory);
    m_data_dir_entry.set_sensitive(false);
    dir_hbox->pack_start(m_data_dir_entry, Gtk::PACK_EXPAND_WIDGET);

    // Current Session Name
    Gtk::Box* sess_hbox = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX));
    m_session_vbox.pack_start(*sess_hbox, Gtk::PACK_SHRINK);
    sess_hbox->pack_start(*Gtk::manage(new Gtk::Label("Name:")), Gtk::PACK_SHRINK);

    m_session_entry.set_text(m_data->session_dir);
    m_session_entry.set_sensitive(false);
    sess_hbox->pack_start(m_session_entry, Gtk::PACK_EXPAND_WIDGET);

    // --- Audio Frame ---
    Gtk::Label* audio_lbl = Gtk::manage(new Gtk::Label());
    audio_lbl->set_markup("<b>Audio</b>");
    m_audio_frame.set_label_widget(*audio_lbl);
    m_audio_frame.add(m_audio_vbox);
    m_audio_vbox.set_border_width(FRAME_PADDING_PX);
    f1a_vbox->pack_start(m_audio_frame, Gtk::PACK_SHRINK);

    m_audio_vbox.pack_start(m_audio_ctrl_hbox, Gtk::PACK_SHRINK);
    m_audio_enable_check.set_label("Record");
    m_audio_enable_check.set_active(true);
    // m_audio_enable_check.signal_toggled().connect(sigc::mem_fun(*this, &MainWindow::on_audio_enable_toggled));
    m_audio_ctrl_hbox.pack_start(m_audio_enable_check, Gtk::PACK_SHRINK);

    populate_audio_sources();
    // m_audio_src_combo.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_audio_source_changed));
    m_audio_ctrl_hbox.pack_start(m_audio_src_combo, Gtk::PACK_EXPAND_WIDGET);

    m_audio_level_bar.set_min_value(0.0);
    m_audio_level_bar.set_max_value(1.0);
    m_audio_level_bar.add_offset_value(GTK_LEVEL_BAR_OFFSET_LOW, 0.4);
    m_audio_level_bar.add_offset_value(GTK_LEVEL_BAR_OFFSET_HIGH, 0.8);
    m_audio_vbox.pack_start(m_audio_level_bar, Gtk::PACK_SHRINK);

    // Expose raw pointers for non-gtkmm legacy code
    m_data->audio_level_bar = (GtkWidget*)m_audio_level_bar.gobj();
    m_data->audio_enable_checkbox = (GtkWidget*)m_audio_enable_check.gobj();
    m_data->audio_src_combo = (GtkWidget*)m_audio_src_combo.gobj();


    // --- ROS Bag Frame ---
    if (!m_data->ros_topics.empty()) {
        Gtk::Label* bag_lbl = Gtk::manage(new Gtk::Label());
        bag_lbl->set_markup("<b>ROS Bag</b>");
        m_bag_frame.set_label_widget(*bag_lbl);
        m_bag_frame.add(m_bag_vbox);
        m_bag_vbox.set_border_width(FRAME_PADDING_PX);
        f1a_vbox->pack_start(m_bag_frame, Gtk::PACK_SHRINK);

        Gtk::Box* stats_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX));
        m_bag_vbox.pack_start(*stats_box, Gtk::PACK_SHRINK);

        m_bag_stats_label.set_text("Topics: 0/0\nMessages: 0");
        m_bag_stats_label.set_alignment(0.0, 0.5);
        stats_box->pack_start(m_bag_stats_label, Gtk::PACK_EXPAND_WIDGET);

        m_bag_details_button.set_label("?");
        m_bag_details_button.set_tooltip_text("Show Topic Details");
        m_bag_details_button.signal_clicked().connect(sigc::mem_fun(*this, &MainWindow::on_bag_details_clicked));
        stats_box->pack_start(m_bag_details_button, Gtk::PACK_SHRINK);

        m_data->bag_stats_label = (GtkWidget*)m_bag_stats_label.gobj();
    }

    // --- Video Streams Grid ---
    Gtk::Label* streams_lbl = Gtk::manage(new Gtk::Label());
    streams_lbl->set_markup("<b>Video Streams</b>");
    m_streams_frame.set_label_widget(*streams_lbl);
    m_streams_frame.add(m_streams_grid);
    m_streams_grid.set_column_spacing(WIDGET_SPACING_PX);
    m_streams_grid.set_row_spacing(WIDGET_SPACING_PX);
    m_streams_grid.set_border_width(FRAME_PADDING_PX);
    f1a_vbox->pack_start(m_streams_frame, Gtk::PACK_EXPAND_WIDGET);

    int col = 0, row = 0;
    for (auto s : m_data->streams) {
        StreamWidgets sw = create_stream_widget(s);
        m_streams_grid.attach(*sw.container, col, row, 1, 1);
        col++;
        if (col > 1) { col = 0; row++; }
    }


    // --- Stages Frame (Right Side) ---
    Gtk::Box* right_side_vbox = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_VERTICAL, BOX_SPACING_PX));
    m_top_hbox.pack_start(*right_side_vbox, Gtk::PACK_SHRINK);

    if (m_data->explicit_stages) {
        Gtk::Label* stages_lbl = Gtk::manage(new Gtk::Label());
        stages_lbl->set_markup("<b>Stages</b>");
        m_stages_frame.set_label_widget(*stages_lbl);
        m_stages_frame.add(m_stages_vbox);
        m_stages_vbox.set_border_width(FRAME_PADDING_PX);
        right_side_vbox->pack_start(m_stages_frame, Gtk::PACK_SHRINK);

        for (size_t i = 0; i < m_data->config_stages.size(); ++i) {
            Gtk::EventBox* eb = Gtk::manage(new Gtk::EventBox());
            Gtk::Label* lbl = Gtk::manage(new Gtk::Label(m_data->config_stages[i]));
            lbl->set_alignment(0.0, 0.5);
            eb->add(*lbl);
            m_stages_vbox.pack_start(*eb, Gtk::PACK_SHRINK);
            m_stage_labels.push_back(lbl);

            // Legacy sync
            m_data->stage_labels.push_back((GtkWidget*)lbl->gobj());

            eb->signal_button_press_event().connect([this, i](GdkEventButton*) {
                if (!m_data->global_recording) {
                    m_data->current_stage_idx = i;
                    this->update_stage_highlighting();
                }
                return true;
            });
        }
        update_stage_highlighting();
    }

    // --- Tags Frame (Right Side) ---
    if (!m_data->config_tags.empty()) {
        Gtk::Label* tags_lbl = Gtk::manage(new Gtk::Label());
        tags_lbl->set_markup("<b>Tags</b>");
        m_tags_frame.set_label_widget(*tags_lbl);
        m_tags_frame.add(m_tags_vbox);
        m_tags_vbox.set_border_width(FRAME_PADDING_PX);
        right_side_vbox->pack_start(m_tags_frame, Gtk::PACK_EXPAND_WIDGET);

        for (const auto& tag : m_data->config_tags) {
            Gtk::Button* btn = Gtk::manage(new Gtk::Button(tag));
            m_tag_ui_buttons[tag] = btn;
            btn->signal_clicked().connect([this, tag]() {
                this->on_tag_clicked(tag);
            });
            m_tags_vbox.pack_start(*btn, Gtk::PACK_SHRINK);
        }
    }

    // --- Bottom Controls ---
    m_main_vbox.pack_end(m_bottom_hbox, Gtk::PACK_SHRINK);

    m_record_button.signal_clicked().connect(sigc::mem_fun(*this, &MainWindow::on_record_toggle));
    m_bottom_hbox.pack_start(m_record_button, Gtk::PACK_EXPAND_WIDGET);

    m_quit_button.signal_clicked().connect(sigc::mem_fun(*this, &Gtk::Window::close));
    m_bottom_hbox.pack_end(m_quit_button, Gtk::PACK_SHRINK);

    m_data->record_button = (GtkWidget*)m_record_button.gobj();

    show_all_children();

    m_update_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this, &MainWindow::on_ui_update), 33);

    g_main_window_instance = this;
}

MainWindow::~MainWindow() {
    g_main_window_instance = nullptr;
    m_update_conn.disconnect();
    // All session data/pipeline finalization is handled by finalize_session(),
    // which main() calls explicitly right after Gtk::Main::run() returns.
    // Call it here only as a safety net (e.g. if someone destroys the window
    // object without calling finalize_session() first).
    finalize_session();
}

void MainWindow::finalize_session() {
    if (m_session_finalized) return;
    m_session_finalized = true;

    m_data->is_quitting = true;

    if (m_data->global_recording) {
        close_bag_writer(m_data);
    }

    // Shut down all pipelines properly
    if (m_data->audio_stream) m_data->audio_stream->shutdown();
    for (auto s : m_data->streams) s->shutdown();

    if (m_data->audio_stream && !m_data->audio_stream->frames.empty()) {
        Json::Value root; 
        root["type"] = "dc::sidecar@1.0.0";
        root["name"] = "audio";
        root["audio_file"] = "audio_" + m_data->start_timestamp + ".wav";
        root["gstreamer_pipeline"] = m_data->audio_stream->pipeline_desc;

        Json::Value configFiles(Json::arrayValue);
        for (const auto& f : m_data->config_files) configFiles.append(f);
        root["config_files"] = configFiles;

        Json::Value framesArr(Json::arrayValue);
        for (const auto& f : m_data->audio_stream->frames) {
            Json::Value frameNode;
            frameNode["cpu_ts"] = (Json::Value::Int64)f.cpu_ts;
            frameNode["gst_ts"] = (Json::Value::Int64)f.gst_ts;
            framesArr.append(frameNode);
        }
        root["frames"] = framesArr;

        std::ofstream os(m_data->audio_stream->output_json); Json::StreamWriterBuilder b;
        std::unique_ptr<Json::StreamWriter>(b.newStreamWriter())->write(root, &os);
    }

    for (auto s : m_data->streams) {
        if (!s->frames.empty()) {
            Json::Value root; 
            root["type"] = "dc::sidecar@1.0.0";
            root["name"] = s->name;
            root["video_file"] = s->output_video.substr(s->output_video.find_last_of("/\\\\") + 1);
            root["gstreamer_pipeline"] = s->pipeline_desc;

            // Save session tags to a global file if we haven't already
            // Actually, we should save it once in index.json or tags.json

            Json::Value configFiles(Json::arrayValue);
            for (const auto& f : m_data->config_files) configFiles.append(f);
            root["config_files"] = configFiles;

            if (!m_data->session_event_tags.empty() || !m_data->session_stages.empty() || m_data->recording_start_cpu_ts > 0) {
                root["session_tags_file"] = m_data->start_timestamp + "_tags.json";
            }

            Json::Value framesArr(Json::arrayValue);
            for (const auto& f : s->frames) {
                Json::Value frameNode;
                frameNode["cpu_ts"] = (Json::Value::Int64)f.cpu_ts;
                frameNode["gst_ts"] = (Json::Value::Int64)f.gst_ts;
                framesArr.append(frameNode);
            }
            root["frames"] = framesArr;
            root["estimated_latency"] = s->estimated_latency;

            std::ofstream os(s->output_json); Json::StreamWriterBuilder b;
            std::unique_ptr<Json::StreamWriter>(b.newStreamWriter())->write(root, &os);
        }
    }

    // Write index.json
    if (!m_data->streams.empty() || (m_data->audio_stream && !m_data->audio_stream->frames.empty())) {
        Json::Value indexRoot;
        indexRoot["type"] = "dc::index@1.0.0";
        Json::Value videosArr(Json::arrayValue);

        for (auto s : m_data->streams) {
            Json::Value vid;
            std::string fname = s->output_video.substr(s->output_video.find_last_of("/\\\\") + 1);
            vid["file"] = fname;

            double duration = 0.0;
            if (s->src_fps > 0.1) duration = (double)s->frames.size() / s->src_fps;
            vid["duration"] = duration;

            videosArr.append(vid);
        }

        if (m_data->audio_stream && !m_data->audio_stream->frames.empty()) {
             Json::Value aud;
             std::string fname = "audio_" + m_data->start_timestamp + ".wav";
             aud["file"] = fname;
             double duration = 0.0;
             const auto& af = m_data->audio_stream->frames;
             if (af.size() > 1) {
                  for (size_t i = 0; i < af.size() - 1; ++i) {
                      long long diff = af[i+1].cpu_ts - af[i].cpu_ts;
                      if (diff < 500 * 1000000) { // 500ms threshold for gaps
                          duration += (double)diff / 1e9;
                      }
                  }
                  // Adjust for last frame
                  if (duration > 0) duration += (duration / (af.size() - 1));
             }
             aud["duration"] = duration;
             videosArr.append(aud);
        }

        indexRoot["videos"] = videosArr;

        if (!m_data->session_bag_path.empty()) {
            std::string bag_name = m_data->session_bag_path.substr(m_data->session_bag_path.find_last_of("/\\\\") + 1);
            indexRoot["rosbags"] = bag_name;
        }

        std::ofstream os(m_data->session_dir + "/index.json");
        Json::StreamWriterBuilder b;
        std::unique_ptr<Json::StreamWriter>(b.newStreamWriter())->write(indexRoot, &os);
    }

    // Write tags.json
    if (!m_data->session_event_tags.empty() || !m_data->session_stages.empty() || m_data->recording_start_cpu_ts > 0) {
        Json::Value tagsRoot;
        tagsRoot["type"] = "dc::session_tags@1.0.0";
        Json::Value stagesArr(Json::arrayValue);
        Json::Value tagsObj(Json::objectValue);

        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        std::string current_time_str = dc::get_current_timestamp_iso8601();

        // Add already recorded stages
        for (const auto& s : m_data->session_stages) {
            Json::Value stageEntry;
            stageEntry["name"] = s.name;
            
            Json::Value startObj(Json::objectValue);
            startObj["cpu_ts"] = (Json::Value::Int64)s.start_ts;
            startObj["generated_at"] = s.start_generated_at;
            
            Json::Value endObj(Json::objectValue);
            endObj["cpu_ts"] = (Json::Value::Int64)s.end_ts;
            endObj["generated_at"] = s.end_generated_at;
            
            stageEntry["start"] = startObj;
            stageEntry["end"] = endObj;
            stagesArr.append(stageEntry);
        }

        // Add the current session stage if still active
        if (m_data->recording_start_cpu_ts > 0) {
             long long end_cpu_ts = (long long)now.tv_sec * 1000000000LL + now.tv_nsec;
             std::string end_generated_at = dc::get_current_timestamp_iso8601();

             std::string stage_name = m_data->config_stages.empty() ? "stage" : m_data->config_stages[m_data->current_stage_idx];

             Json::Value stageEntry;
             stageEntry["name"] = stage_name;
             
             Json::Value startObj(Json::objectValue);
             startObj["cpu_ts"] = (Json::Value::Int64)m_data->recording_start_cpu_ts;
             startObj["generated_at"] = m_data->recording_start_generated_at;
             
             Json::Value endObj(Json::objectValue);
             endObj["cpu_ts"] = (Json::Value::Int64)end_cpu_ts;
             endObj["generated_at"] = end_generated_at;
             
             stageEntry["start"] = startObj;
             stageEntry["end"] = endObj;
             stagesArr.append(stageEntry);
        }

        for (const auto& tag : m_data->session_event_tags) {
            Json::Value tag_obj(Json::objectValue);
            tag_obj["cpu_ts"] = (Json::Value::Int64)tag.cpu_ts;
            tag_obj["generated_at"] = tag.generated_at;
            tagsObj[tag.name].append(tag_obj);
        }

        tagsRoot["generated_at"] = current_time_str;
        tagsRoot["stages"] = stagesArr;
        tagsRoot["tags"] = tagsObj;

        std::string tags_path = m_data->session_dir + "/" + m_data->start_timestamp + "_tags.json";
        std::ofstream os(tags_path);
        Json::StreamWriterBuilder b;
        std::unique_ptr<Json::StreamWriter>(b.newStreamWriter())->write(tagsRoot, &os);
    }

    std::cout << "Session finalized." << std::endl;
}

MainWindow::StreamWidgets MainWindow::create_stream_widget(VideoStream* s) {
    StreamWidgets sw;
    sw.container = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_VERTICAL, WIDGET_SPACING_PX));
    sw.container->set_hexpand(true);
    sw.container->set_vexpand(true);

    // Header: Name (Bold) <Space> Record [x]
    Gtk::Box* header_hbox = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX));
    sw.container->pack_start(*header_hbox, Gtk::PACK_SHRINK);

    Gtk::Label* name_label = Gtk::manage(new Gtk::Label());
    name_label->set_markup("<b>" + s->name + "</b>");
    name_label->set_alignment(0.0, 0.5);
    header_hbox->pack_start(*name_label, Gtk::PACK_EXPAND_WIDGET);

    sw.record_check = Gtk::manage(new Gtk::CheckButton("Record"));
    sw.record_check->set_active(true);
    header_hbox->pack_start(*sw.record_check, Gtk::PACK_SHRINK);
    s->record_checkbox = (GtkWidget*)sw.record_check->gobj();

    // Preview
    Gtk::Frame* frame = Gtk::manage(new Gtk::Frame());
    if (s->preview_widget) {
        Gtk::Widget* gtkmm_preview = Glib::wrap(s->preview_widget);
        gtkmm_preview->set_hexpand(true);
        gtkmm_preview->set_vexpand(true);
        gtkmm_preview->set_size_request(640, 480);
        frame->add(*gtkmm_preview);
    } else {
        sw.preview = Gtk::manage(new Gtk::Image());
        frame->add(*sw.preview);
    }
    sw.container->pack_start(*frame, Gtk::PACK_EXPAND_WIDGET); // Issue 5: Expand preview

    // Stats
    sw.stats = Gtk::manage(new Gtk::Label("Ready"));
    sw.stats->set_alignment(0.0, 0.5);
    // sw.stats->override_font(Pango::FontDescription("Monospace 9")); // Optional?
    sw.container->pack_start(*sw.stats, Gtk::PACK_SHRINK);
    s->stats_label = (GtkWidget*)sw.stats->gobj();

    return sw;
}

void MainWindow::populate_audio_sources() {
    m_audio_src_combo.remove_all();
    GstDeviceMonitor* mon = gst_device_monitor_new();
    gst_device_monitor_add_filter(mon, "Audio/Source", NULL);
    GList* devices = gst_device_monitor_get_devices(mon);
    int idx = 0, default_idx = 0;

    for (GList* l = devices; l; l = l->next) {
        GstDevice* d = (GstDevice*)l->data;
        gchar* name = gst_device_get_display_name(d);
        GstStructure* p = gst_device_get_properties(d);
        const char* pulse_name = gst_structure_get_string(p, "device.name");

        if (pulse_name) {
            m_audio_src_combo.append(pulse_name, name);
            if (!g_str_has_suffix(pulse_name, ".monitor") && strstr(name, "Default")) {
                default_idx = idx;
            }
            idx++;
        }
        g_free(name); if (p) gst_structure_free(p);
    }
    g_list_free_full(devices, (GDestroyNotify)gst_object_unref);
    gst_object_unref(mon);
    m_audio_src_combo.set_active(default_idx);
}

void MainWindow::update_stage_highlighting() {
    for (size_t i = 0; i < m_stage_labels.size(); ++i) {
        if ((int)i == m_data->current_stage_idx) {
            m_stage_labels[i]->set_markup("<span foreground=\"#0055ff\"><b>" + m_data->config_stages[i] + "</b></span>");
        } else {
            m_stage_labels[i]->set_text(m_data->config_stages[i]);
        }
    }
}

void MainWindow::on_bag_details_clicked() {
    Gtk::Dialog dialog("ROS Topic Details", *this, true);
    Gtk::Box* content = dialog.get_content_area();

    std::stringstream ss;

    {
        std::lock_guard<std::mutex> lock(m_data->data_mutex);
        ss << "<b>Subscribed Topics (" << m_data->bag_topics_found << "/" << m_data->ros_topics.size() << "):</b>\n";
        for (const auto& topic : m_data->ros_topics) {
            bool found = m_data->subscribed_topics.count(topic);
            if (found) {
                ss << "<span foreground='green'>✔ " << topic << "</span>\n";
            } else {
                ss << "<span foreground='red'>✘ " << topic << "</span>\n";
            }
        }
    }

    Gtk::Label* lbl = Gtk::manage(new Gtk::Label());
    lbl->set_markup(ss.str());
    lbl->set_padding(10, 10);
    content->add(*lbl);

    dialog.add_button("Close", Gtk::RESPONSE_CLOSE);
    dialog.show_all_children();
    dialog.run();
}

void MainWindow::trigger_record_toggle() {
    on_record_toggle();
}

void MainWindow::on_tag_clicked(const std::string& tag_name) {
    // Get current CPU time
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long long cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    std::string generated_at = dc::get_current_timestamp_iso8601();

    std::map<std::string, int> counts;
    {
        std::lock_guard<std::mutex> lock(m_data->data_mutex);
        m_data->session_event_tags.push_back({tag_name, cpu_ts, generated_at});

        for (const auto& t : m_data->session_event_tags) {
            counts[t.name]++;
        }
    }

    for (auto const& [name, btn] : m_tag_ui_buttons) {
        if (counts.count(name)) {
            btn->set_label(name + " (" + std::to_string(counts[name]) + ")");
        }
    }
}

void MainWindow::on_record_toggle() {
    if (m_data->is_quitting) return;

    m_data->global_recording = !m_data->global_recording;

    int active_idx = m_data->current_stage_idx;
    std::string base_stage = m_data->config_stages.empty() ? "stage" : m_data->config_stages[active_idx];

    if (m_data->global_recording) {
        m_data->any_recording_started = true;
        char stage_buf[128];
        snprintf(stage_buf, sizeof(stage_buf), "%s_%03d", base_stage.c_str(), m_data->session_stage_cycle_count);
        std::string final_stage_name = stage_buf;

        // Add start tag
        {
             struct timespec ts;
             clock_gettime(CLOCK_REALTIME, &ts);
             long long cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
             m_data->recording_start_cpu_ts = cpu_ts;
             m_data->recording_start_generated_at = dc::get_current_timestamp_iso8601();
        }

        if (m_data->audio_stream) {
            bool audio_rec = m_audio_enable_check.get_active();
            m_data->audio_stream->set_recording(audio_rec);
        }

        // ROS Bag is now continuous for the session, no open/close here.

        for (auto s : m_data->streams) {
            bool stream_rec = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(s->record_checkbox));
            s->is_recording = stream_rec;
            if (s->valve) g_object_set(s->valve, "drop", !stream_rec, NULL);
        }
    } else {
         if (m_data->audio_stream) {
            m_data->audio_stream->set_recording(false);
        }

        // Do not close ROS bag here.

        {
             std::string stage_name = m_data->config_stages.empty() ? "stage" : m_data->config_stages[active_idx];
             char stage_buf[128];
             snprintf(stage_buf, sizeof(stage_buf), "%s_%03d", stage_name.c_str(), m_data->session_stage_cycle_count);
             std::string final_stage_name = stage_buf;

             // Capture stage end
             if (m_data->recording_start_cpu_ts > 0) {
                 struct timespec ts;
                 clock_gettime(CLOCK_REALTIME, &ts);
                 long long end_cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
                 std::string end_generated_at = dc::get_current_timestamp_iso8601();

                 std::lock_guard<std::mutex> lock(m_data->data_mutex);
                 m_data->session_stages.push_back({
                     stage_name, 
                     m_data->recording_start_cpu_ts, 
                     m_data->recording_start_generated_at,
                     end_cpu_ts,
                     end_generated_at
                 });
                 m_data->recording_start_cpu_ts = 0; // Reset for next run
                 m_data->recording_start_generated_at = "";
             }

             for (auto s : m_data->streams) {
                 s->last_run_stage_name = final_stage_name;
                 s->last_run_frames_recorded = s->frames_recorded;

                 s->is_recording = false;
                 if (s->valve) g_object_set(s->valve, "drop", TRUE, NULL);
                 // if (s->rec_text) g_object_set(s->rec_text, "text", "", NULL);
                 s->frames_recorded = 0;
                 s->frames_dropped = 0;
                 s->current_fps = 0;
                 s->last_fps_ts = 0;
                 s->fps_frame_counter = 0;
             }
        }

        int num_stages = (int)m_data->config_stages.size();
        if (num_stages > 0) {
            int next_idx = active_idx + 1;
            if (next_idx >= num_stages) {
                next_idx = 0;
                m_data->session_stage_cycle_count++;
            }
            m_data->current_stage_idx = next_idx;
        }
    }

    m_record_button.set_label(m_data->global_recording ? "Stop Recording" : "Start Recording");

    // Publish current recording state
    if (m_data->node && m_data->pub_recording) {
        auto msg = std::make_unique<std_msgs::msg::Bool>();
        msg->data = m_data->global_recording;
        m_data->pub_recording->publish(std::move(msg));
    }

    m_audio_enable_check.set_sensitive(!m_data->global_recording);
    m_audio_src_combo.set_sensitive(!m_data->global_recording);
    for (auto s : m_data->streams) {
        gtk_widget_set_sensitive(s->record_checkbox, !m_data->global_recording);
    }

    update_stage_highlighting();
}

bool MainWindow::on_ui_update() {
    if (m_data->is_quitting) return false;

    for (auto s : m_data->streams) {
        char buf[256];
        if (s->is_recording) {
            snprintf(buf, sizeof(buf), "REC | FPS: %.1f | Frames: %lld",
                     s->current_fps, s->frames_recorded);
            // if (s->rec_text) g_object_set(s->rec_text, "text", "", NULL);
        } else {
             if (s->last_run_frames_recorded > 0) {
                // Approximate duration based on FPS (or timestamp diff if we kept it)
                double duration_sec = 0.0;
                if (s->src_fps > 0.1) duration_sec = (double)s->last_run_frames_recorded / s->src_fps;

                snprintf(buf, sizeof(buf), "Last: %s (%lld f, %.1fs) | Src: %.1f",
                         s->last_run_stage_name.c_str(), s->last_run_frames_recorded, duration_sec, s->src_fps);
             } else {
                snprintf(buf, sizeof(buf), "%dx%d | Src: %.1f | Ready", s->width, s->height, s->src_fps);
             }
             // Ensure overlay is clear
             // if (s->rec_text) g_object_set(s->rec_text, "text", "", NULL);
        }
        gtk_label_set_text(GTK_LABEL(s->stats_label), buf);
    }

    char bag_buf[256];
    std::lock_guard<std::mutex> lock(m_data->data_mutex);
    snprintf(bag_buf, sizeof(bag_buf), "Topics: %d/%lu\nMessages: %lld",
             m_data->bag_topics_found, m_data->ros_topics.size(), m_data->bag_messages_recorded);
    m_bag_stats_label.set_text(bag_buf);

    return true;
}
