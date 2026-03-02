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
#include <ctime>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "config.hpp"
#include "tags.hpp"

TagWindow::TagWindow(const std::string& video, const std::string&config, const std::string& tags_file, bool load_session_tags)
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
    m_current_stages_label.set_margin_bottom(5);
    m_current_stages_label.set_line_wrap(true);
    m_current_stages_label.set_justify(Gtk::JUSTIFY_CENTER);
    m_left_vbox.pack_start(m_current_stages_label, Gtk::PACK_SHRINK);

    m_video_container.set_size_request(800, 450);
    m_left_vbox.pack_start(m_video_container, Gtk::PACK_EXPAND_WIDGET);

    m_info_label.set_text("Loading...");
    m_left_vbox.pack_start(m_info_label, Gtk::PACK_SHRINK);

    m_stats_label.set_text("Ready");
    m_stats_label.set_alignment(0.0, 0.5);
    m_left_vbox.pack_start(m_stats_label, Gtk::PACK_SHRINK);

    // Frame Slider
    m_frame_hbox.pack_start(*Gtk::make_managed<Gtk::Label>("Frame:"), Gtk::PACK_SHRINK);
    m_frame_slider.set_range(0, 100);
    m_frame_slider.set_draw_value(false);
    m_frame_slider.set_tooltip_text("Seek by frame number");
    m_frame_slider.signal_value_changed().connect(sigc::mem_fun(*this, &TagWindow::on_frame_slider_moved));
    m_frame_hbox.pack_start(m_frame_slider, Gtk::PACK_EXPAND_WIDGET);
    m_frame_label.set_text("0 / 0 (00:00.000)");
    m_frame_label.set_width_chars(25);
    m_frame_hbox.pack_start(m_frame_label, Gtk::PACK_SHRINK);
    m_left_vbox.pack_start(m_frame_hbox, Gtk::PACK_SHRINK);

    // Controls
    m_begin_btn.set_label("Begin");
    m_begin_btn.set_tooltip_text("Jump to beginning (a)");
    m_begin_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_begin));
    m_controls_hbox.pack_start(m_begin_btn, Gtk::PACK_SHRINK);

    m_prev_btn.set_label("Previous");
    m_prev_btn.set_tooltip_text("Previous frame (s)");
    m_prev_btn.signal_pressed().connect(sigc::mem_fun(*this, &TagWindow::on_prev_btn_pressed));
    m_prev_btn.signal_released().connect(sigc::mem_fun(*this, &TagWindow::on_prev_btn_released));
    m_controls_hbox.pack_start(m_prev_btn, Gtk::PACK_SHRINK);

    m_play_btn.set_label("Play");
    m_play_btn.set_tooltip_text("Play/Pause (d)");
    m_play_btn.set_size_request(80, -1);
    m_play_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_play_pause));
    m_controls_hbox.pack_start(m_play_btn, Gtk::PACK_SHRINK);

    m_next_btn.set_label("Next");
    m_next_btn.set_tooltip_text("Next frame (f)");
    m_next_btn.signal_pressed().connect(sigc::mem_fun(*this, &TagWindow::on_next_btn_pressed));
    m_next_btn.signal_released().connect(sigc::mem_fun(*this, &TagWindow::on_next_btn_released));
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
    m_speed_combo.set_tooltip_text("Adjust playback speed");
    m_speed_combo.signal_changed().connect(sigc::mem_fun(*this, &TagWindow::on_speed_changed));
    m_controls_hbox.pack_start(m_speed_combo, Gtk::PACK_SHRINK);

    m_left_vbox.pack_start(m_controls_hbox, Gtk::PACK_SHRINK);

    // Right Side: Tags
    m_right_vbox.set_size_request(350, -1);
    Gtk::Label* tags_title = Gtk::make_managed<Gtk::Label>();
    tags_title->set_markup("<b>Tags</b>");
    m_right_vbox.pack_start(*tags_title, Gtk::PACK_SHRINK);

    m_tags_scroll.set_policy(Gtk::POLICY_NEVER, Gtk::POLICY_AUTOMATIC);
    m_tags_grid.set_column_spacing(5);
    m_tags_grid.set_row_spacing(2);
    m_tags_scroll.add(m_tags_grid);
    m_right_vbox.pack_start(m_tags_scroll, Gtk::PACK_EXPAND_WIDGET);

    m_save_btn.set_label("Save Tags");
    m_save_btn.set_tooltip_text("Save current tags to file");
    m_save_btn.signal_clicked().connect(sigc::mem_fun(*this, &TagWindow::on_save));
    m_save_quit_hbox.pack_start(m_save_btn, Gtk::PACK_EXPAND_WIDGET);

    m_quit_btn.set_label("Quit");
    m_quit_btn.set_tooltip_text("Quit (q)");
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
    signal_key_release_event().connect(sigc::mem_fun(*this, &TagWindow::on_key_release), false);
}

TagWindow::~TagWindow() {
    if (m_data.pipeline) {
        gst_element_set_state(m_data.pipeline, GST_STATE_NULL);
        gst_object_unref(m_data.pipeline);
    }
}

void TagWindow::add_tag_row(const std::string& tag_name) {
    if (m_tag_buttons.count(tag_name)) return;

    auto* btn = Gtk::make_managed<Gtk::ToggleButton>(tag_name);
    btn->set_hexpand(true);
    btn->set_margin_top(1);
    btn->set_margin_bottom(1);
    m_tags_grid.attach(*btn, 0, m_grid_row_count, 1, 1);
    m_tag_buttons[tag_name] = btn;
    btn->signal_clicked().connect([this, tag_name](){ on_tag_toggle(tag_name); });

    auto* count_lbl = Gtk::make_managed<Gtk::Label>("0");
    count_lbl->set_width_chars(3);
    count_lbl->set_margin_top(1);
    count_lbl->set_margin_bottom(1);
    m_tags_grid.attach(*count_lbl, 1, m_grid_row_count, 1, 1);
    m_tag_count_labels[tag_name] = count_lbl;

    auto* combo = Gtk::make_managed<Gtk::ComboBoxText>();
    combo->set_no_show_all(true);
    combo->hide();
    combo->set_margin_top(1);
    combo->set_margin_bottom(1);
    m_tags_grid.attach(*combo, 2, m_grid_row_count, 1, 1);
    m_tag_combos[tag_name] = combo;
    combo->signal_changed().connect([this, tag_name](){ on_tag_jump(tag_name); });

    m_grid_row_count++;
}

void TagWindow::load_config(const std::string& path) {
    if (path.empty()) return;

    Json::Value root;
    if (!dc::Config::load_from_file(path, root)) {
        std::cerr << "Failed to load config: " << path << std::endl;
        return;
    }

    // dc::Config::load_from_file already checks for dc_config_v1
    dc::AppConfig cfg = dc::Config::parse_app_config(root);

    for (const auto& s : cfg.stages) {
        m_data.stages.push_back(s);
        add_tag_row(s + "_start");
        add_tag_row(s + "_end");
    }

    for (const auto& t : cfg.tags) {
        m_data.tags.push_back(t);
        add_tag_row(t);
    }
    m_tags_grid.show_all();
}

void TagWindow::load_sidecar_json() {
    std::string base = m_data.video_path;
    size_t dot = base.find_last_of(".");
    if (dot != std::string::npos) base = base.substr(0, dot);
    std::string json_file = base + ".json";

    Json::Value root;
    if (!dc::Config::load_from_file(json_file, root)) {
        std::cerr << "No sidecar JSON found at: " << json_file << std::endl;
        return;
    }

    if (!dc::Config::check_type(root, "dc_sidecar_v1", json_file)) {
        return;
    }

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

            // DO NOT SUBTRACT first_gst_ts. 
            // GStreamer pipeline (qtdemux/h264parse) uses absolute file PTS.
            // When we seek, we want to go to the exact PTS stored in the sidecar.
            m_data.frame_gst_timestamps.push_back(t);
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

    // Use an explicit software decode chain to avoid NVMM/NVV4L2 negotiation issues during playback.
    // Assumes MP4/H.264 recording output from the recorder: filesrc -> qtdemux -> h264parse -> avdec_h264.
    std::string pipe_str =
        "filesrc location=\"" + m_data.video_path +
        "\" ! qtdemux name=demux "
        "demux.video_0 ! queue ! h264parse ! avdec_h264 ! videoconvert ! gtksink name=vsink sync=true";
    GError* err = nullptr;
    m_data.pipeline = gst_parse_launch(pipe_str.c_str(), &err);
    if (!m_data.pipeline) {
        std::cerr << "Failed to create pipeline: " << (err ? err->message : "unknown") << std::endl;
        if (err) {
            std::cerr << "GStreamer error: " << err->message << std::endl;
            g_error_free(err);
        }
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
        
        // Use recorded frame count based "virtual duration" for the UI if sidecar exists
        if (!m_data.frame_gst_timestamps.empty()) {
            m_data.session_duration_ns = (long long)((double)m_data.frame_gst_timestamps.size() / m_data.fps * 1e9);
        } else if (m_data.session_duration_ns == 0) {
            m_data.session_duration_ns = duration;
        }
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

        // Compute absolute indices based on FPS for reference
        // This helps us know "This is frame #120 in the raw video file"
        m_data.frame_abs_indices.clear();
        for (long long ts : m_data.frame_gst_timestamps) {
            long long idx = (long long)((double)ts / 1e9 * m_data.fps + 0.5); // Round
            m_data.frame_abs_indices.push_back(idx);
        }

        m_data.current_frame = 0;
        gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
        do_seek(m_data.frame_gst_timestamps[0]);
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

void TagWindow::seek_to_frame(long long frame_idx) {
    if (frame_idx < 0) frame_idx = 0;
    if (frame_idx >= m_data.total_frames) frame_idx = m_data.total_frames - 1;
    
    // Update state first
    m_data.current_frame = frame_idx;
    
    // Convert to timestamp
    gint64 target_ns = 0;
    if (!m_data.frame_gst_timestamps.empty() && frame_idx < (long long)m_data.frame_gst_timestamps.size()) {
        target_ns = m_data.frame_gst_timestamps[frame_idx];
    } else {
        // Fallback if no sidecar
        target_ns = (gint64)((double)frame_idx / m_data.fps * 1e9);
    }
    
    do_seek(target_ns);
}

void TagWindow::on_play_pause() {
    GstState current, pending;
    gst_element_get_state(m_data.pipeline, &current, &pending, 0);
    if (current == GST_STATE_PLAYING) {
        gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
        m_play_btn.set_label("Play");
    } else {
        // If we are at the beginning or before the first frame, skip to first frame
        if (!m_data.frame_gst_timestamps.empty()) {
            gint64 pos = 0;
            if (gst_element_query_position(m_data.pipeline, GST_FORMAT_TIME, &pos)) {
                if (pos < m_data.frame_gst_timestamps[0]) {
                    seek_to_frame(0);
                }
            }
        }
        gst_element_set_state(m_data.pipeline, GST_STATE_PLAYING);
        m_play_btn.set_label("Pause");
        // Ensure speed is maintained
        on_speed_changed();
    }
}

long long TagWindow::ns_to_nearest_frame(long long ns) {
    if (m_data.frame_gst_timestamps.empty()) {
        return (long long)((double)ns / 1e9 * m_data.fps);
    }
    
    // If we're before the first frame, return frame 0
    if (ns <= m_data.frame_gst_timestamps.front()) return 0;
    // If we're after the last frame, return the last frame index
    if (ns >= m_data.frame_gst_timestamps.back()) return m_data.frame_gst_timestamps.size() - 1;

    auto it = std::lower_bound(m_data.frame_gst_timestamps.begin(), m_data.frame_gst_timestamps.end(), ns);
    if (it == m_data.frame_gst_timestamps.end()) return m_data.frame_gst_timestamps.size() - 1;
    if (it == m_data.frame_gst_timestamps.begin()) return 0;

    long long dist1 = std::abs((long long)*it - ns);
    long long dist2 = std::abs((long long)*(it - 1) - ns);
    if (dist1 < dist2) return std::distance(m_data.frame_gst_timestamps.begin(), it);
    else return std::distance(m_data.frame_gst_timestamps.begin(), it - 1);
}

void TagWindow::on_begin() {
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    seek_to_frame(0);
}

void TagWindow::on_prev_frame() {
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    long long next_frame = std::max(0LL, m_data.current_frame - 1);
    seek_to_frame(next_frame);
}

void TagWindow::on_next_frame() {
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    long long max_frame = m_data.total_frames > 0 ? m_data.total_frames - 1 : 0;
    long long next_frame = std::min(max_frame, m_data.current_frame + 1);
    seek_to_frame(next_frame);
}

void TagWindow::on_frame_slider_moved() {
    if (m_internal_update) return;
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    long long frame = (long long)m_frame_slider.get_value();
    seek_to_frame(frame);
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
    if (m_data.pipeline) gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
    seek_to_frame(frame);
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

        // Skip to the first frame if the current position is before it
        if (!m_data.frame_gst_timestamps.empty() && pos < m_data.frame_gst_timestamps[0] - 10000000) {
            seek_to_frame(0);
            return true;
        }

        // Update Play/Pause button label based on actual state
        GstState current, pending;
        gst_element_get_state(m_data.pipeline, &current, &pending, 0);
        if (current == GST_STATE_PLAYING) {
            m_play_btn.set_label("Pause");

            // If playing, check if we need to skip to the next frame
            if (!m_data.frame_gst_timestamps.empty()) {
                gint64 current_frame_ts = m_data.frame_gst_timestamps[m_data.current_frame];
                
                // If GStreamer pos is significantly past the current recorded frame's expected timestamp,
                // we are moving towards the next frame.
                // If the next frame is contiguous, we do nothing.
                // If there is a large gap, we must SEEK.
                
                // Look ahead: Where should the next frame be?
                long long next_idx = m_data.current_frame + 1;
                
                if (next_idx < m_data.total_frames) {
                    long long next_frame_ts = m_data.frame_gst_timestamps[next_idx];
                    
                    // Are we approaching or past the point where we should switch to next_idx?
                    // Or are we adrift in a gap?
                    
                    // If pos > current + 10ms, we are "done" with current frame.
                    if (pos > current_frame_ts + 10000000) {
                        
                        // Check if next frame is contiguous in ABSOLUTE index time
                        bool is_contiguous = false;
                        if (!m_data.frame_abs_indices.empty()) {
                            long long cur_abs = m_data.frame_abs_indices[m_data.current_frame];
                            long long next_abs = m_data.frame_abs_indices[next_idx];
                            if (next_abs - cur_abs <= 1) is_contiguous = true;
                        } else {
                            // Fallback heuristic: timestamp difference is ~1 frame duration
                            double frame_dur = 1e9 / m_data.fps;
                            if (std::abs(next_frame_ts - current_frame_ts - frame_dur) < (frame_dur * 0.1)) {
                                is_contiguous = true;
                            }
                        }
                        
                        if (is_contiguous) {
                            // Smooth playback: Just update the index
                            // Wait, if it's strictly contiguous, the timestamp should just be +1/FPS
                            // But if we are past next_frame_ts itself, we should update index.
                            if (pos >= next_frame_ts - 5000000) { // Within 5ms of next frame or past it
                                m_data.current_frame = next_idx;
                            }
                        } else {
                            // GAP DETECTED: Jump immediately to next_frame_ts
                            // Only jump if we are NOT already close to it (to avoid loops)
                            if (pos < next_frame_ts - 10000000) { // If we are BEFORE the target (in the gap)
                                seek_to_frame(next_idx);
                                return true;
                            } else {
                                // We are already past the gap target? Just update index
                                m_data.current_frame = next_idx;
                            }
                        }
                    }
                } else {
                     // End of recorded frames - pause if we run over
                     if (pos > current_frame_ts + 50000000) {
                        gst_element_set_state(m_data.pipeline, GST_STATE_PAUSED);
                        m_play_btn.set_label("Play");
                        seek_to_frame(m_data.total_frames - 1);
                        return true;
                     }
                }
            }
        } else {
            m_play_btn.set_label("Play");
        }

        long long frame = ns_to_nearest_frame(pos);
        long long virtual_ms = (long long)((double)frame / m_data.fps * 1000.0);
        long long total_ms = (long long)((double)(m_data.total_frames - 1) / m_data.fps * 1000.0);
        
        auto format_virtual_time = [](long long ms) {
            int mins = ms / 60000;
            double secs = (ms % 60000) / 1000.0;
            std::stringstream ss;
            ss << mins << ":" << std::fixed << std::setfill('0') << std::setw(6) << std::setprecision(3) << secs;
            return ss.str();
        };
        
        std::stringstream ss_frame;
        ss_frame << frame << " / " << m_data.total_frames - 1 << " (" 
                 << format_virtual_time(virtual_ms) << " / " << format_virtual_time(total_ms) << ")";

        if (frame != m_data.current_frame || true) { // Always refresh display labels to ensure consistency
            m_data.current_frame = frame;
            m_frame_slider.set_value(frame);
            m_frame_label.set_text(ss_frame.str());

            // Update current stages label
            std::map<std::string, int> starts;
            std::map<std::string, int> ends;
            for (auto const& [f, tags] : m_data.frame_tags) {
                if (f > frame) break;
                for (const auto& t : tags) {
                    if (t.size() > 6 && t.compare(t.size() - 6, 6, "_start") == 0) {
                        starts[t.substr(0, t.size() - 6)]++;
                    } else if (t.size() > 4 && t.compare(t.size() - 4, 4, "_end") == 0) {
                        ends[t.substr(0, t.size() - 4)]++;
                    }
                }
            }

            std::string active_stages_str = "";
            for (auto const& [name, count] : starts) {
                int end_count = ends[name];
                if (count > end_count) {
                    for (int i = end_count + 1; i <= count; ++i) {
                        if (!active_stages_str.empty()) active_stages_str += ", ";
                        active_stages_str += name + " (" + std::to_string(i) + ")";
                    }
                }
            }

            if (active_stages_str.empty()) {
                m_current_stages_label.set_markup("<i>No active stages</i>");
            } else {
                m_current_stages_label.set_markup("<b>Current Stages: " + active_stages_str + "</b>");
            }

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
    if (event->keyval == GDK_KEY_s) {
        if (!m_key_s_pressed) {
            m_key_s_pressed = true;
            m_key_press_count = 1;
            on_prev_frame();
            if (!m_step_timer_conn.connected())
                m_step_timer_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this, &TagWindow::on_step_timer), 200);
        }
        return true;
    }
    if (event->keyval == GDK_KEY_f) {
        if (!m_key_f_pressed) {
            m_key_f_pressed = true;
            m_key_press_count = 1;
            on_next_frame();
            if (!m_step_timer_conn.connected())
                m_step_timer_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this, &TagWindow::on_step_timer), 200);
        }
        return true;
    }

    switch (event->keyval) {
        case GDK_KEY_d: on_play_pause(); return true;
        case GDK_KEY_a: on_begin(); return true;
        case GDK_KEY_q: close(); return true;
    }
    return false;
}

bool TagWindow::on_key_release(GdkEventKey* event) {
    if (event->keyval == GDK_KEY_s) {
        m_key_s_pressed = false;
        return true;
    }
    if (event->keyval == GDK_KEY_f) {
        m_key_f_pressed = false;
        return true;
    }
    return false;
}

bool TagWindow::on_step_timer() {
    if (m_key_s_pressed || m_btn_prev_pressed) {
        m_key_press_count++;
        on_prev_frame();
        if (m_key_press_count > 3) on_prev_frame(); // Double speed after 3 frames
        return true;
    }
    if (m_key_f_pressed || m_btn_next_pressed) {
        m_key_press_count++;
        on_next_frame();
        if (m_key_press_count > 3) on_next_frame(); // Double speed after 3 frames
        return true;
    }
    return false; // Disconnect timer
}

void TagWindow::on_prev_btn_pressed() {
    if (!m_btn_prev_pressed) {
        m_btn_prev_pressed = true;
        m_key_press_count = 1;
        on_prev_frame();
        if (!m_step_timer_conn.connected())
            m_step_timer_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this, &TagWindow::on_step_timer), 200);
    }
}

void TagWindow::on_prev_btn_released() {
    m_btn_prev_pressed = false;
}

void TagWindow::on_next_btn_pressed() {
    if (!m_btn_next_pressed) {
        m_btn_next_pressed = true;
        m_key_press_count = 1;
        on_next_frame();
        if (!m_step_timer_conn.connected())
            m_step_timer_conn = Glib::signal_timeout().connect(sigc::mem_fun(*this, &TagWindow::on_step_timer), 200);
    }
}

void TagWindow::on_next_btn_released() {
    m_btn_next_pressed = false;
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
                tags_obj[t].append((Json::Int64)frame);
            }
        }
    }

    std::string current_time_str = dc::get_current_timestamp_iso8601();
    root["type"] = "dc::video_tags@1.0.0";
    root["created_at"] = current_time_str;

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

            Json::Value start_obj;
            start_obj["frame"] = (Json::Int64)s_frame;
            start_obj["generated_at"] = current_time_str;
            
            Json::Value end_obj;
            end_obj["frame"] = (Json::Int64)e_frame;
            end_obj["generated_at"] = current_time_str;

            stage_entry["start"] = start_obj;
            stage_entry["end"] = end_obj;
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
    root["generated_at"] = current_time_str;
    
    std::string vname = m_data.video_path;
    size_t last_slash = vname.find_last_of("/\\");
    if (last_slash != std::string::npos) vname = vname.substr(last_slash + 1);
    root["video_file"] = vname;

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

    std::vector<std::string> missing_tags;
    std::vector<std::string> stages_to_add;
    std::vector<std::string> tags_to_add;

    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s["name"].asString();
            if (m_tag_buttons.count(name + "_start") == 0 || m_tag_buttons.count(name + "_end") == 0) {
                missing_tags.push_back(name + " (stage)");
                stages_to_add.push_back(name);
            }
        }
    }
    if (root.isMember("tags")) {
        for (auto const& name : root["tags"].getMemberNames()) {
            if (m_tag_buttons.count(name) == 0) {
                missing_tags.push_back(name);
                tags_to_add.push_back(name);
            }
        }
    }

    MissingTagsResult res = MissingTagsResult::IGNORE;
    if (!missing_tags.empty()) {
        res = show_missing_tags_dialog(missing_tags, tags_file);
        if (res == MissingTagsResult::ACCEPT) {
            for (const auto& s : stages_to_add) {
                add_tag_row(s + "_start");
                add_tag_row(s + "_end");
            }
            for (const auto& t : tags_to_add) {
                add_tag_row(t);
            }
            m_tags_grid.show_all();
        }
    }

    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s["name"].asString();
            if (m_tag_buttons.count(name + "_start") == 0 || m_tag_buttons.count(name + "_end") == 0) continue;

            long long start_ts = dc::parse_stage_timestamp(s["start"]);
            long long end_ts = dc::parse_stage_timestamp(s["end"]);

            long long start = find_frame(start_ts);
            long long end = find_frame(end_ts);
            m_data.frame_tags[start].push_back(name + "_start");
            m_data.frame_tags[end].push_back(name + "_end");
        }
    }

    if (root.isMember("tags")) {
        for (auto const& name : root["tags"].getMemberNames()) {
            if (m_tag_buttons.count(name) == 0) continue;
            for (auto& f : root["tags"][name]) {
                m_data.frame_tags[find_frame(f.asInt64())].push_back(name);
            }
        }
    }

    if (!missing_tags.empty()) {
        std::string inf = (res == MissingTagsResult::ACCEPT) ? " (new tags added)" : " (some tags ignored)";
        m_info_label.set_markup("Loaded tags from <b>" + tags_file + "</b>" + inf);
    } else {
        m_info_label.set_markup("Loaded tags from <b>" + tags_file + "</b>");
    }
}

TagWindow::MissingTagsResult TagWindow::show_missing_tags_dialog(const std::vector<std::string>& missing_tags, const std::string& source) {
    if (missing_tags.empty()) return MissingTagsResult::IGNORE;

    std::string message = "Tags found in " + source + " but not in config:\n";
    for (size_t i = 0; i < missing_tags.size(); ++i) {
        message += " • " + missing_tags[i] + "\n";
        if (i > 15) {
            message += " ... and " + std::to_string(missing_tags.size() - i - 1) + " more";
            break;
        }
    }

    Gtk::MessageDialog dialog(*this, "Missing Tags", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_NONE, true);
    dialog.set_secondary_text(message);
    dialog.add_button("Accept", (int)MissingTagsResult::ACCEPT);
    dialog.add_button("Ignore", (int)MissingTagsResult::IGNORE);
    dialog.add_button("Quit", (int)MissingTagsResult::QUIT);

    int result = dialog.run();
    if (result == (int)MissingTagsResult::QUIT) {
        std::exit(0);
    }
    return (MissingTagsResult)result;
}

void TagWindow::load_session_tags() {
    std::string base = m_data.video_path;
    size_t last_slash = base.find_last_of("/\\");
    if (last_slash == std::string::npos) return;
    std::string session_dir = base.substr(0, last_slash);
    std::string tags_json = session_dir + "/tags.json";

    std::cout << "Attempting to load session tags from: " << tags_json << std::endl;

    Json::Value root;
    if (!dc::Config::load_from_file(tags_json, root)) {
        std::cout << "No session tags found at: " << tags_json << std::endl;
        return;
    }

    if (!dc::Config::check_type(root, "dc_session_tags_v1", tags_json)) {
        return;
    }

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
    std::vector<std::string> missing_tags;
    std::vector<std::string> stages_to_add;
    std::vector<std::string> tags_to_add;

    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s["name"].asString();
            if (m_tag_buttons.count(name + "_start") == 0 || m_tag_buttons.count(name + "_end") == 0) {
                missing_tags.push_back(name + " (stage)");
                stages_to_add.push_back(name);
            }
        }
    }
    if (root.isMember("tags")) {
        for (auto const& name : root["tags"].getMemberNames()) {
            if (m_tag_buttons.count(name) == 0) {
                missing_tags.push_back(name);
                tags_to_add.push_back(name);
            }
        }
    }

    MissingTagsResult res = MissingTagsResult::IGNORE;
    if (!missing_tags.empty()) {
        res = show_missing_tags_dialog(missing_tags, "session file (" + tags_json + ")");
        if (res == MissingTagsResult::ACCEPT) {
            for (const auto& s : stages_to_add) {
                add_tag_row(s + "_start");
                add_tag_row(s + "_end");
            }
            for (const auto& t : tags_to_add) {
                add_tag_row(t);
            }
            m_tags_grid.show_all();
        }
    }

    if (root.isMember("stages")) {
        for (auto& s : root["stages"]) {
            std::string name = s["name"].asString();
            if (m_tag_buttons.count(name + "_start") == 0 || m_tag_buttons.count(name + "_end") == 0) continue;

            long long start_ts = dc::parse_stage_timestamp(s["start"]);
            long long end_ts = dc::parse_stage_timestamp(s["end"]);

            long long start_frame = find_frame(start_ts);
            long long end_frame = find_frame(end_ts);
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
            if (m_tag_buttons.count(name) == 0) continue;
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
