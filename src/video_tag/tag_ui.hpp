#ifndef TAG_UI_HPP
#define TAG_UI_HPP

#include <gtkmm.h>
#include <gst/gst.h>
#include <string>
#include <vector>
#include <map>
#include <json/json.h>

struct TagData {
    std::string video_path;
    std::string config_path;
    std::vector<std::string> stages;
    std::vector<std::string> tags;

    // frame_num -> list of tags
    std::map<long long, std::vector<std::string>> frame_tags;

    // Mapping from frame index to CPU timestamp
    std::vector<long long> frame_cpu_timestamps;
    std::vector<long long> frame_gst_timestamps;
    std::vector<long long> frame_abs_indices; // Computed absolute indices in source video

    bool unsaved_changes = false;

    long long current_frame = 0;
    long long total_frames = 0;
    long long duration_ns = 0;
    long long session_start_cpu_ns = 0;
    long long session_duration_ns = 0;
    double fps = 30.0;
    bool session_tags_loaded = false;

    GstElement *pipeline = nullptr;
    GstElement *video_sink = nullptr;

    TagData() = default;
};

class TagWindow : public Gtk::Window {
public:
    TagWindow(const std::string& video, const std::string& config, const std::string& tags_file = "", bool load_session_tags = false);
    virtual ~TagWindow();

protected:
    // UI Signal Handlers
    void on_begin();
    void on_play_pause();
    void on_prev_frame();
    void on_next_frame();
    void on_slider_moved();
    void on_frame_slider_moved();
    void on_speed_changed();
    void on_save();
    void on_tag_toggle(const std::string& tag_name);
    void on_tag_jump(const std::string& tag_name);

    bool on_ui_update_timer();
    bool on_key_press(GdkEventKey* event);
    bool on_key_release(GdkEventKey* event);
    bool on_step_timer();

    // Helper methods
    void load_config(const std::string& path);
    void add_tag_row(const std::string& tag_name);
    void load_sidecar_json();
    void setup_pipeline();
    void do_seek(gint64 ns);
    void seek_to_frame(long long frame_idx);
    void update_ui_state();
    void save_tags();
    void load_tags(const std::string& explicit_path = "");
    void load_session_tags();
    
    enum class MissingTagsResult { IGNORE, QUIT, ACCEPT };
    MissingTagsResult show_missing_tags_dialog(const std::vector<std::string>& missing_tags, const std::string& source);
    
    void update_tag_navigation_ui();
    long long ns_to_nearest_frame(long long ns);
    std::string format_time(long long ns);
    std::string format_time_simple(double seconds);

    TagData m_data;
    bool m_internal_update = false;
    bool m_load_session_tags = false;

    bool m_key_s_pressed = false;
    bool m_key_f_pressed = false;
    bool m_btn_prev_pressed = false;
    bool m_btn_next_pressed = false;
    int m_key_press_count = 0;
    sigc::connection m_step_timer_conn;

    // GUI Handlers
    void on_prev_btn_pressed();
    void on_prev_btn_released();
    void on_next_btn_pressed();
    void on_next_btn_released();

    // UI Widgets
    Gtk::Box m_main_hbox;
    Gtk::Box m_left_vbox;
    Gtk::Box m_right_vbox;

    Gtk::Box m_video_container; // Container for GstGtkSink widget
    Gtk::Label m_current_stages_label;
    Gtk::Label m_info_label;
    Gtk::Label m_stats_label;

    Gtk::Box m_timeline_hbox;
    Gtk::Label m_time_label;
    Gtk::Scale m_timeline_slider;
    Gtk::Label m_duration_label;

    Gtk::Box m_frame_hbox;
    Gtk::Scale m_frame_slider;
    Gtk::Label m_frame_label;

    Gtk::Box m_controls_hbox;
    Gtk::Button m_prev_btn;
    Gtk::Button m_play_btn;
    Gtk::Button m_next_btn;
    Gtk::Button m_begin_btn;
    Gtk::ComboBoxText m_speed_combo;

    Gtk::ScrolledWindow m_tags_scroll;
    Gtk::Grid m_tags_grid;
    Gtk::Box m_save_quit_hbox;
    Gtk::Button m_save_btn;
    Gtk::Button m_quit_btn;

    // Track tag widgets: tag_name -> widget pointer (checkbox/toggle)
    std::map<std::string, Gtk::ToggleButton*> m_tag_buttons;
    std::map<std::string, Gtk::Label*> m_tag_count_labels;
    std::map<std::string, Gtk::ComboBoxText*> m_tag_combos;
    int m_grid_row_count = 0;

    sigc::connection m_timer_conn;
};

#endif // TAG_UI_HPP
