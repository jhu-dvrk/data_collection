#include "tag_ui.hpp"
#include <gst/video/videooverlay.h>
#include <gdk/gdkx.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>

TagWindow::TagWindow(const std::string& video, const std::string& config, const std::string& tags_file, bool load_session_tags)
    : m_main_hbox(Gtk::ORIENTATION_HORIZONTAL, 10),
      m_left_vbox(Gtk::ORIENTATION_VERTICAL, 5),
      m_right_vbox(Gtk::ORIENTATION_VERTICAL, 5),
      m_video_container(Gtk::ORIENTATION_VERTICAL),
      m_timeline_hbox(Gtk::ORIENTATION_HORIZONTAL, 5),
      m_timeline_slider(Gtk::ORIENTATION_HORIZONTAL),
      m_frame_hbox(Gtk::ORIENTATION_HORIZONTAL, 5),
      m_frame_slider(Gtk::ORIENTATION_HORIZONTAL),
      m_controls_hbox(Gtk::ORIENTATION_HORIZONTAL, 5),
      m_save_quit_hbox(Gtk::ORIENTATION_HORIZONTAL, 10)
{
    m_data.video_path = video;
    m_data.config_path = config;
    m_load_session_tags = load_session_tags;

    set_title("C++ Video Tag - " + video);
    set_default_size(1300, 800);
    set_border_width(10); // Add space around main window

    // Layout setup
    add(m_main_hbox);
    m_main_hbox.pack_start(m_left_vbox, Gtk::PACK_EXPAND_WIDGET);
    m_main_hbox.pack_start(m_right_vbox, Gtk::PACK_SHRINK);

    // Left Side: Video
    m_video_container.set_size_request(800, 450);
    m_left_vbox.pack_start(m_video_container, Gtk::PACK_EXPAND_WIDGET);

    m_info_label.set_text("Loading...");
    m_left_vbox.pack_start(m_info_label, Gtk::PACK_SHRINK);

    m_stats_label.set_text("Ready");
    m_stats_label.set_alignment(0.0, 0.5);
    m_left_vbox.pack_start(m_stats_label, Gtk::PACK_SHRINK);

    // Timeline Slider
    m_timeline_hbox.pack_start(*Gtk::make_managed<Gtk::Label>("Time:"), Gtk::PACK_SHRINK);
    m_timeline_slider.set_range(0, 100);
    m_timeline_slider.set_draw_value(false);
    m_timeline_slider.signal_value_changed().connect(sigc::mem_fun(*this, &TagWindow::on_slider_moved));
    m_timeline_hbox.pack_start(m_timeline_slider, Gtk::PACK_EXPAND_WIDGET);
    m_duration_label.set_text("00:00:00 / 00:00:00");
    m_timeline_hbox.pack_start(m_duration_label, Gtk::PACK_SHRINK);
    m_left_vbox.pack_start(m_timeline_hbox, Gtk::PACK_SHRINK);

    // Frame Slider
    m_frame_hbox.pack_start(*Gtk::make_managed<Gtk::Label>("Frame:"), Gtk::PACK_SHRINK);
    m_frame_slider.set_range(0, 100);
    m_frame_slider.set_draw_value(false);
    m_frame_slider.signal_value_changed().connect(sigc::mem_fun(*this, &TagWindow::on_frame_slider_moved));
    m_frame_hbox.pack_start(m_frame_slider, Gtk::PACK_EXPAND_WIDGET);
    m_frame_label.set_text("0 / 0");
    m_frame_label.set_width_chars(15);
    m_frame_hbox.pack_start(m_frame_label, Gtk::PACK_SHRINK);
    m_left_vbox.pack_start(m_frame_hbox, Gtk::PACK_SHRINK);

    // Controls
    m_begin_btn.set_label("Begin");
    m_begin_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_begin));
    m_controls_hbox.pack_start(m_begin_btn, Gtk::PACK_SHRINK);

    m_prev_btn.set_label("Prev Frame");
    m_prev_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_prev_frame));
    m_controls_hbox.pack_start(m_prev_btn, Gtk::PACK_SHRINK);

    m_play_btn.set_label("Play/Pause");
    m_play_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_play_pause));
    m_controls_hbox.pack_start(m_play_btn, Gtk::PACK_SHRINK);

    m_next_btn.set_label("Next Frame");
    m_next_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_next_frame));
    m_controls_hbox.pack_start(m_next_btn, Gtk::PACK_SHRINK);

    m_speed_combo.append("0.1", "0.1x");
    m_speed_combo.append("0.25", "0.25x");
    m_speed_combo.append("0.5", "0.5x");
    m_speed_combo.append("0.75", "0.75x");
    m_speed_combo.append("1.0", "1.0x");
    m_speed_combo.append("1.25", "1.25x");
    m_speed_combo.append("1.5", "1.5x");
    m_speed_combo.append("2.0", "2.0x");
    m_speed_combo.set_active_id("1.0");
    m_speed_combo.signal_changed().connect(sigc::mem_fun(*this, &TagWindow::on_speed_changed));
    m_controls_hbox.pack_start(m_speed_combo, Gtk::PACK_SHRINK);

    m_left_vbox.pack_start(m_controls_hbox, Gtk::PACK_SHRINK);

    // Right Side: Tags
    m_right_vbox.set_size_request(350, -1);
    Gtk::Label* tags_title = Gtk::make_managed<Gtk::Label>();
    tags_title->set_markup("<b>Tags</b>");
    m_right_vbox.pack_start(*tags_title, Gtk::PACK_SHRINK);
    
    m_tags_scroll.set_policy(Gtk::POLICY_NEVER, Gtk::POLICY_AUTOMATIC);
    m_tags_grid.set_column_spacing(10);
    m_tags_grid.set_row_spacing(5);
    m_tags_scroll.add(m_tags_grid);
    m_right_vbox.pack_start(m_tags_scroll, Gtk::PACK_EXPAND_WIDGET);

    m_save_btn.set_label("Save Tags");
    m_save_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_save));
    m_save_quit_hbox.pack_start(m_save_btn, Gtk::PACK_EXPAND_WIDGET);

    m_quit_btn.set_label("Quit");
    m_quit_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::close));
    m_save_quit_hbox.pack_start(m_quit_btn, Gtk::PACK_SHRINK);

    m_right_vbox.pack_start(m_save_quit_hbox, Gtk::PACK_SHRINK);

    show_all_children();

    // Configuration and Pipeline
    if (!config.empty()) load_config(config);
    if (!video.empty()) {
        setup_pipeline();
        load_tags(tags_file);
        update_tag_navigation_ui();
    }

    m_timer_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this, &TagWindow::on_ui_update_timer), 100);
    signal_key_press_event().connect(sigc::mem_fun(*this, &TagWindow::on_key_press), false);
}

TagWindow::~TagWindow() {
    if (m_data.pipeline) {
        gst_element_set_state(m_data.pipeline, GST_STATE_NULL);
        gst_object_unref(m_data.pipeline);
    }
}

void TagWindow::load_config(const std::string& path) {
    if (path.empty()) return;
    std::ifstream ifs(path);
    Json::Value root;
    ifs >> root;

    auto add_row = [this](const std::string& tag_name){
        auto* btn = Gtk::make_managed<Gtk::ToggleButton>(tag_name);
        btn->set_hexpand(true);
        m_tags_grid.attach(*btn, 0, m_grid_row_count, 1, 1);
        m_tag_buttons[tag_name] = btn;
        btn->signal_clicked().connect([this, tag_name](){ on_tag_toggle(tag_name); });

        auto* count_lbl = Gtk::make_managed<Gtk::Label>("0");
        count_lbl->set_width_chars(3);
        m_tags_grid.attach(*count_lbl, 1, m_grid_row_count, 1, 1);
        m_tag_count_labels[tag_name] = count_lbl;

        auto* combo = Gtk::make_managed<Gtk::ComboBoxText>();
        combo->set_no_show_all(true);
        combo->hide();
        m_tags_grid.attach(*combo, 2, m_grid_row_count, 1, 1);
        m_tag_combos[tag_name] = combo;
        combo->signal_changed().connect([this, tag_name](){ on_tag_jump(tag_name); });

        m_grid_row_count++;
    };

    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s.asString();
            m_data.stages.push_back(name);
            add_row(name + "_start");
            add_row(name + "_end");
        }
    }

    if (root.isMember("tags")) {
        for (auto& t : root["tags"]) {
            std::string name = t.asString();
            m_data.tags.push_back(name);
            add_row(name);
        }
    }
    m_tags_grid.show_all();
}

void TagWindow::load_sidecar_json() {
    std::string base = m_data.video_path;
    size_t dot = base.find_last_of(".");
    if (dot != std::string::npos) base = base.substr(0, dot);
    std::string json_file = base + ".json";

    std::ifstream ifs(json_file);
    if (!ifs.is_open()) {
        std::cerr << "No sidecar JSON found at: " << json_file << std::endl;
        return;
    }

    Json::Value root;
    ifs >> root;
    if (root.isMember("frames") && root["frames"].isArray()) {
        m_data.frame_cpu_timestamps.clear();
        m_data.frame_gst_timestamps.clear();
        
        long long first_gst_ts = -1;
        
        for (const auto& frame : root["frames"]) {
            if (frame.isMember("cpu_ts")) {
                m_data.frame_cpu_timestamps.push_back(frame["cpu_ts"].asInt64());
            } else {
                m_data.frame_cpu_timestamps.push_back(0);
            }
            
            long long t = 0;
            if (frame.isMember("gst_ts")) {
                t = frame["gst_ts"].asInt64();
            }
            
            if (first_gst_ts == -1) first_gst_ts = t;
            
            m_data.frame_gst_timestamps.push_back(t - first_gst_ts);
        }
    }

    // Load recording_start_cpu_ts if available as session reference
    if (root.isMember("recording_start_cpu_ts")) {
        m_data.session_start_cpu_ns = root["recording_start_cpu_ts"].asInt64();
    } else if (!m_data.frame_cpu_timestamps.empty()) {
        m_data.session_start_cpu_ns = m_data.frame_cpu_timestamps[0];
    }

    if (!m_data.frame_cpu_timestamps.empty()) {
         m_data.session_duration_ns = m_data.frame_cpu_timestamps.back() - m_data.session_start_cpu_ns;
    }

    // Auto-load session tags if requested or if flag is present in JSON
    if (m_load_session_tags) {
        this->load_session_tags();
    }

    // Auto-load config labels if not provided
    if (m_data.config_path.empty() && root.isMember("config_files") && root["config_files"].isArray()) {
        for (const auto& cfg : root["config_files"]) {
            load_config(cfg.asString());
        }
    }

    m_info_label.set_text("Loaded sidecar JSON: " + json_file);
}

void TagWindow::setup_pipeline() {
    load_sidecar_json();

    std::string pipe_str = "filesrc location=\"" + m_data.video_path + "\" ! decodebin ! videoconvert ! gtksink name=vsink sync=true";
    GError* err = nullptr;
    m_data.pipeline = gst_parse_launch(pipe_str.c_str(), &err);
    if (!m_data.pipeline) {
        std::cerr << "Failed to create pipeline: " << (err ? err->message : "unknown") << std::endl;
        return;
    }

    m_data.video_sink = gst_bin_get_by_name(GST_BIN(m_data.pipeline), "vsink");
    
    // Get gtksink widget and pack it
    GtkWidget* sink_widget = nullptr;
    g_object_get(m_data.video_sink, "widget", &sink_widget, NULL);
    if (sink_widget) {
        Gtk::Widget* wrapped_widget = Glib::wrap(sink_widget);
        m_video_container.pack_start(*wrapped_widget, Gtk::PACK_EXPAND_WIDGET);
        wrapped_widget->show();
    }

    gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    
    // Get duration and FPS
    GstState state = GST_STATE_PAUSED;
    gst_element_get_state(m_data.pipeline, &state, nullptr, GST_SECOND);
    
    gint64 duration = 0;
    if (gst_element_query_duration(m_data.pipeline, GST_FORMAT_TIME, &duration)) {
        m_data.duration_ns = duration;
        if (m_data.session_duration_ns == 0) m_data.session_duration_ns = duration;
        m_duration_label.set_text(format_time(0) + " / " + format_time(m_data.session_duration_ns));
        m_timeline_slider.set_range(0, (double)duration / 1e6);
    }

    // Attempt to get FPS from sink caps
    GstPad* pad = gst_element_get_static_pad(m_data.video_sink, "sink");
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (caps) {
        GstStructure* s = gst_caps_get_structure(caps, 0);
        gint num, den;
        if (gst_structure_get_fraction(s, "framerate", &num, &den) && den != 0) {
            m_data.fps = (double)num / (double)den;
        }
        
        gint width, height;
        if (gst_structure_get_int(s, "width", &width) && gst_structure_get_int(s, "height", &height)) {
             std::stringstream ss;
             ss << "Res: " << width << "x" << height << " | FPS: " << std::fixed << std::setprecision(2) << m_data.fps;
             m_stats_label.set_text(ss.str());
        }
        gst_caps_unref(caps);
    }
    gst_object_unref(pad);

    if (!m_data.frame_gst_timestamps.empty()) {
        m_data.total_frames = m_data.frame_gst_timestamps.size();
    } else {
        m_data.total_frames = (long long)((double)m_data.duration_ns / 1e9 * m_data.fps);
    }
    m_frame_slider.set_range(0, m_data.total_frames > 0 ? m_data.total_frames - 1 : 0);
}

void TagWindow::do_seek(gint64 ns) {
    if (!m_data.pipeline) return;
    double speed = std::stod(m_speed_combo.get_active_id());
    gst_element_seek(m_data.pipeline, speed, GST_FORMAT_TIME, 
                     (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE),
                     GST_SEEK_TYPE_SET, ns, 
                     GST_SEEK_TYPE_NONE, -1);
}

void TagWindow::on_play_pause() {
    GstState current, pending;
    gst_element_get_state(m_data.pipeline, &current, &pending, 0);
    if (current == GST_STATE_PLAYING) {
        gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    } else {
        gst_element_set_state(m_data.pipeline, GST_STATE_PLAYING);
        // Ensure speed is maintained
        on_speed_changed();
    }
}

long long TagWindow::frame_to_ns(long long frame) {
    if (!m_data.frame_gst_timestamps.empty() && frame >= 0 && frame < (long long)m_data.frame_gst_timestamps.size()) {
        return m_data.frame_gst_timestamps[frame];
    }
    return (long long)((double)frame / m_data.fps * 1e9);
}

void TagWindow::on_begin() {
    do_seek(frame_to_ns(0));
}

void TagWindow::on_prev_frame() {
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    long long next_frame = std::max(0LL, m_data.current_frame - 1);
    do_seek(frame_to_ns(next_frame));
}

void TagWindow::on_next_frame() {
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    long long max_frame = m_data.total_frames > 0 ? m_data.total_frames - 1 : 0;
    long long next_frame = std::min(max_frame, m_data.current_frame + 1);
    do_seek(frame_to_ns(next_frame));
}

void TagWindow::on_slider_moved() {
    if (m_internal_update) return;
    long long ns = (long long)(m_timeline_slider.get_value() * 1e6);
    do_seek(ns);
}

void TagWindow::on_frame_slider_moved() {
    if (m_internal_update) return;
    long long frame = (long long)m_frame_slider.get_value();
    do_seek(frame_to_ns(frame));
}

void TagWindow::on_speed_changed() {
    gint64 pos = 0;
    if (gst_element_query_position(m_data.pipeline, GST_FORMAT_TIME, &pos)) {
        do_seek(pos);
    }
}

void TagWindow::on_tag_toggle(const std::string& tag_name) {
    if (m_internal_update || m_data.total_frames == 0) return;
    
    auto& tags_at_frame = m_data.frame_tags[m_data.current_frame];
    auto it = std::find(tags_at_frame.begin(), tags_at_frame.end(), tag_name);
    
    if (m_tag_buttons[tag_name]->get_active()) {
        if (it == tags_at_frame.end()) {
            tags_at_frame.push_back(tag_name);
            m_data.unsaved_changes = true;
        }
    } else {
        if (it != tags_at_frame.end()) {
            tags_at_frame.erase(it);
            if (tags_at_frame.empty()) m_data.frame_tags.erase(m_data.current_frame);
            m_data.unsaved_changes = true;
        }
    }
    update_tag_navigation_ui();
}

void TagWindow::on_tag_jump(const std::string& tag_name) {
    if (m_internal_update) return;
    auto* combo = m_tag_combos[tag_name];
    std::string id = combo->get_active_id();
    if (id.empty()) return;

    long long frame = std::stoll(id);
    m_data.current_frame = frame; // Update state immediately to prevent sync lag
    do_seek(frame_to_ns(frame));
}

void TagWindow::update_tag_navigation_ui() {
    m_internal_update = true;
    
    // Calculate counts and frames for each tag
    std::map<std::string, std::vector<long long>> tag_occurences;
    for (auto const& [frame, tags] : m_data.frame_tags) {
        for (auto const& t : tags) {
            tag_occurences[t].push_back(frame);
        }
    }

    // Sort frame numbers for each tag to ensure dropdown order
    for (auto& [name, frames] : tag_occurences) {
        std::sort(frames.begin(), frames.end());
    }

    // Update labels and combo boxes
    for (auto const& [name, btn] : m_tag_buttons) {
        int count = tag_occurences.count(name) ? tag_occurences[name].size() : 0;
        m_tag_count_labels[name]->set_text(std::to_string(count));

        auto* combo = m_tag_combos[name];
        if (count > 0) {
            combo->show();
            combo->remove_all();
            for (long long f : tag_occurences[name]) {
                long long rel_ns = 0;
                if (f >= 0 && f < (long long)m_data.frame_cpu_timestamps.size()) {
                    rel_ns = m_data.frame_cpu_timestamps[f] - m_data.session_start_cpu_ns;
                } else {
                    rel_ns = (long long)((double)f / m_data.fps * 1e9);
                }
                combo->append(std::to_string(f), format_time_simple((double)rel_ns / 1e9));
            }
        } else {
            combo->hide();
        }
    }
    m_internal_update = false;
}

std::string TagWindow::format_time_simple(double seconds) {
    int m = (int)seconds / 60;
    double s = seconds - (m * 60);
    std::stringstream ss;
    ss << m << ":" << std::fixed << std::setprecision(1) << s << "s";
    return ss.str();
}

bool TagWindow::on_ui_update_timer() {
    if (!m_data.pipeline) return true;

    gint64 pos = 0;
    if (gst_element_query_position(m_data.pipeline, GST_FORMAT_TIME, &pos)) {
        m_internal_update = true;
        
        long long frame = 0;
        if (!m_data.frame_gst_timestamps.empty()) {
            auto it = std::lower_bound(m_data.frame_gst_timestamps.begin(), m_data.frame_gst_timestamps.end(), pos);
            if (it == m_data.frame_gst_timestamps.end()) frame = m_data.frame_gst_timestamps.size() - 1;
            else if (it == m_data.frame_gst_timestamps.begin()) frame = 0;
            else {
                long long dist1 = std::abs((long long)*it - pos);
                long long dist2 = std::abs((long long)*(it - 1) - pos);
                if (dist1 < dist2) frame = std::distance(m_data.frame_gst_timestamps.begin(), it);
                else frame = std::distance(m_data.frame_gst_timestamps.begin(), it - 1);
            }
        } else {
            frame = (long long)((double)pos / 1e9 * m_data.fps);
        }

        long long session_rel_ns = pos;
        if (frame >= 0 && frame < (long long)m_data.frame_cpu_timestamps.size()) {
            session_rel_ns = m_data.frame_cpu_timestamps[frame] - m_data.session_start_cpu_ns;
        }

        m_duration_label.set_text(format_time(session_rel_ns) + " / " + format_time(m_data.session_duration_ns));
        m_timeline_slider.set_value((double)pos / 1e6);
        
        if (frame != m_data.current_frame || true) { // Always refresh display labels to ensure consistency
            m_data.current_frame = frame;
            m_frame_slider.set_value(frame);
            m_frame_label.set_text(std::to_string(frame) + " / " + std::to_string(m_data.total_frames));
            
            // Update tag buttons state for this frame
            for (auto const& [name, btn] : m_tag_buttons) {
                auto const& tags = m_data.frame_tags[frame];
                bool active = std::find(tags.begin(), tags.end(), name) != tags.end();
                btn->set_active(active);
            }
        }
        m_internal_update = false;
    }
    return true;
}

bool TagWindow::on_key_press(GdkEventKey* event) {
    switch (event->keyval) {
        case GDK_KEY_d: on_play_pause(); return true;
        case GDK_KEY_s: on_prev_frame(); return true;
        case GDK_KEY_f: on_next_frame(); return true;
        case GDK_KEY_a: on_begin(); return true;
        case GDK_KEY_q: close(); return true;
    }
    return false;
}

std::string TagWindow::format_time(long long ns) {
    long long total_sec = ns / 1e9;
    int h = total_sec / 3600;
    int m = (total_sec % 3600) / 60;
    int s = total_sec % 60;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << h << ":" 
       << std::setw(2) << m << ":" << std::setw(2) << s;
    return ss.str();
}

void TagWindow::on_save() {
    save_tags();
    update_tag_navigation_ui();
}

void TagWindow::save_tags() {
    std::string base = m_data.video_path;
    size_t dot = base.find_last_of(".");
    if (dot != std::string::npos) base = base.substr(0, dot);
    std::string tags_file = base + "_tags.json";

    Json::Value root;
    Json::Value stages_array(Json::arrayValue);
    Json::Value tags_obj(Json::objectValue);

    auto get_ts = [&](long long frame) -> Json::Value {
        if (!m_data.frame_cpu_timestamps.empty() && frame >= 0 && frame < (long long)m_data.frame_cpu_timestamps.size()) {
            return (Json::Int64)m_data.frame_cpu_timestamps[frame];
        }
        return (Json::Int64)frame;
    };

    struct Stage {
        std::string name;
        long long start = -1;
        long long end = -1;
    };
    std::map<std::string, Stage> stage_map;

    for (auto const& [frame, tags] : m_data.frame_tags) {
        for (auto const& t : tags) {
            if (t.find("_start") != std::string::npos) {
                std::string name = t.substr(0, t.length() - 6);
                stage_map[name].name = name;
                stage_map[name].start = frame;
            } else if (t.find("_end") != std::string::npos) {
                std::string name = t.substr(0, t.length() - 4);
                stage_map[name].name = name;
                stage_map[name].end = frame;
            } else {
                tags_obj[t].append(get_ts(frame));
            }
        }
    }

    for (auto const& [name, stage] : stage_map) {
        if (stage.start != -1 && stage.end != -1) {
            Json::Value stage_entry;
            stage_entry["name"] = name;
            // Ensure chronological order for start/end
            long long s_frame = std::min(stage.start, stage.end);
            long long e_frame = std::max(stage.start, stage.end);
            
            if (stage.start > stage.end) {
                std::string error_msg = "Warning: Stage '" + name + "' has start > end! (" + 
                                       format_time_simple((double)stage.start / m_data.fps) + " > " + 
                                       format_time_simple((double)stage.end / m_data.fps) + ")";
                std::cerr << error_msg << std::endl;
                m_info_label.set_text(error_msg);
            }

            stage_entry["start"] = get_ts(s_frame);
            stage_entry["end"] = get_ts(e_frame);
            stages_array.append(stage_entry);
        } else {
            std::string status = (stage.start == -1) ? "missing start" : "missing end";
            std::string error_msg = "Error: Stage '" + name + "' is " + status + ". Not saved.";
            std::cerr << error_msg << std::endl;
            m_info_label.set_text(error_msg);
        }
    }

    root["stages"] = stages_array;
    root["tags"] = tags_obj;

    std::ofstream ofs(tags_file);
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &ofs);

    m_data.unsaved_changes = false;
    m_info_label.set_text("Saved tags to " + tags_file);
}

void TagWindow::load_tags(const std::string& explicit_path) {
    std::string tags_file = explicit_path;
    if (tags_file.empty()) {
        std::string base = m_data.video_path;
        size_t dot = base.find_last_of(".");
        if (dot != std::string::npos) base = base.substr(0, dot);
        tags_file = base + "_tags.json";
    }

    std::ifstream ifs(tags_file);
    if (!ifs.is_open()) {
        std::cout << "No existing tag file found at: " << tags_file << std::endl;
        return;
    }

    Json::Value root;
    ifs >> root;

    auto find_frame = [&](long long ts) -> long long {
        if (m_data.frame_cpu_timestamps.empty()) return ts;

        auto it = std::lower_bound(m_data.frame_cpu_timestamps.begin(), m_data.frame_cpu_timestamps.end(), ts);
        if (it == m_data.frame_cpu_timestamps.end()) return m_data.frame_cpu_timestamps.size() - 1;
        if (it == m_data.frame_cpu_timestamps.begin()) return 0;
        
        long long dist1 = std::abs((long long)*it - ts);
        long long dist2 = std::abs((long long)*(it - 1) - ts);
        if (dist1 < dist2) return std::distance(m_data.frame_cpu_timestamps.begin(), it);
        else return std::distance(m_data.frame_cpu_timestamps.begin(), it - 1);
    };

    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s["name"].asString();
            long long start = find_frame(s["start"].asInt64());
            long long end = find_frame(s["end"].asInt64());
            m_data.frame_tags[start].push_back(name + "_start");
            m_data.frame_tags[end].push_back(name + "_end");
        }
    }

    if (root.isMember("tags")) {
        for (auto const& name : root["tags"].getMemberNames()) {
            for (auto& f : root["tags"][name]) {
                m_data.frame_tags[find_frame(f.asInt64())].push_back(name);
            }
        }
    }
    m_info_label.set_markup("Loaded tags from <b>" + tags_file + "</b>");
}

void TagWindow::load_session_tags() {
    std::string base = m_data.video_path;
    size_t last_slash = base.find_last_of("/\\");
    if (last_slash == std::string::npos) return;
    std::string session_dir = base.substr(0, last_slash);
    std::string tags_json = session_dir + "/tags.json";

    std::cout << "Attempting to load session tags from: " << tags_json << std::endl;

    std::ifstream ifs(tags_json);
    if (!ifs.is_open()) {
        std::cout << "No session tags found at: " << tags_json << std::endl;
        return;
    }

    Json::Value root;
    ifs >> root;
    std::cout << "Parsing unified tags.json..." << std::endl;

    auto find_frame = [&](long long ts) -> long long {
        if (m_data.frame_cpu_timestamps.empty()) return -1;
        auto it = std::lower_bound(m_data.frame_cpu_timestamps.begin(), m_data.frame_cpu_timestamps.end(), ts);
        
        long long idx = -1;
        if (it == m_data.frame_cpu_timestamps.end()) {
            idx = m_data.frame_cpu_timestamps.size() - 1;
        } else if (it == m_data.frame_cpu_timestamps.begin()) {
            idx = 0;
        } else {
            long long dist1 = std::abs((long long)*it - ts);
            long long dist2 = std::abs((long long)*(it - 1) - ts);
            if (dist1 < dist2) idx = std::distance(m_data.frame_cpu_timestamps.begin(), it);
            else idx = std::distance(m_data.frame_cpu_timestamps.begin(), it - 1);
        }

        if (idx != -1) {
            long long diff = std::abs((long long)m_data.frame_cpu_timestamps[idx] - ts);
            if (diff > 500000000LL) { // Increase to 500ms tolerance for UI latency
                return -1;
            }
        }
        return idx;
    };

    int count = 0;
    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s["name"].asString();
            
            long long start_frame = find_frame(s["start"].asInt64());
            long long end_frame = find_frame(s["end"].asInt64());
            if (start_frame != -1) {
                m_data.frame_tags[start_frame].push_back(name + "_start");
                count++;
            }
            if (end_frame != -1) {
                m_data.frame_tags[end_frame].push_back(name + "_end");
                count++;
            }
        }
    }

    if (root.isMember("tags")) {
        for (auto const& name : root["tags"].getMemberNames()) {
            for (auto& t : root["tags"][name]) {
                long long frame = find_frame(t.asInt64());
                if (frame != -1) {
                    m_data.frame_tags[frame].push_back(name);
                    count++;
                }
            }
        }
    }
    std::cout << "Successfully loaded " << count << " entries from session tags." << std::endl;
    std::string current_info = m_info_label.get_text();
    m_info_label.set_text(current_info + " | Session tags: " + std::to_string(count));
}
