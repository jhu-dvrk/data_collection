#ifndef UI_HPP
#define UI_HPP

#include <gtkmm.h>
#include "context.hpp"

// ---- Layout Constants ----
#define WINDOW_BORDER_PX 8
#define BOX_SPACING_PX 8
#define FRAME_PADDING_PX 4
#define WIDGET_SPACING_PX 4
#define WIDGET_MARGIN_PX 4
// -------------------------

class MainWindow : public Gtk::Window {
public:
    MainWindow(AppData* data);
    virtual ~MainWindow();

    // Call this once from main() right after Gtk::Main::run() returns to
    // properly finalize GStreamer pipelines and write all session metadata.
    // Safe to call multiple times (idempotent).
    void finalize_session();
    void start_ros_sync();
    void trigger_stop_recording();

protected:
    // Signal handlers
    void on_record_toggle();
    void on_audio_enable_toggled();
    void on_audio_source_changed();
    void on_stage_changed();
    void on_bag_details_clicked();
    bool on_ui_update();
    bool on_key_press_event(GdkEventKey* event) override;

    // Helpers
    void update_stage_highlighting();
    void populate_audio_sources();
    struct StreamWidgets {
        Gtk::Box* container;
        Gtk::Widget* preview;
        Gtk::CheckButton* record_check;
        Gtk::Label* stats;
    };
    StreamWidgets create_stream_widget(VideoStream* s);

    AppData* m_data;

    void on_tag_clicked(const std::string& tag_name);

    // Main Struct
    Gtk::Box m_main_vbox;
    Gtk::Box m_top_hbox;

    // Session Frame
    Gtk::Frame m_session_frame;
    Gtk::Box m_session_vbox;
    Gtk::Entry m_data_dir_entry;
    Gtk::Entry m_session_entry;

    // Stages Frame
    Gtk::Frame m_stages_frame;
    Gtk::Box m_stages_vbox;
    Gtk::ComboBoxText m_stages_combo;
    Gtk::Grid m_stages_grid;
    std::vector<Gtk::Label*> m_stage_labels; // Managed by container

    // Tags Frame
    Gtk::Frame m_tags_frame;
    Gtk::Box m_tags_vbox;
    std::map<std::string, Gtk::Button*> m_tag_ui_buttons;

    // Audio Frame
    Gtk::Frame m_audio_frame;
    Gtk::Box m_audio_vbox;
    Gtk::Box m_audio_ctrl_hbox;
    Gtk::CheckButton m_audio_enable_check;
    Gtk::ComboBoxText m_audio_src_combo;
    Gtk::LevelBar m_audio_level_bar;

    // ROS Bag Frame
    Gtk::Frame m_bag_frame;
    Gtk::Box m_bag_vbox;
    Gtk::Label m_bag_stats_label;
    Gtk::Button m_bag_details_button;

    // Stream Grid
    Gtk::Frame m_streams_frame;
    Gtk::Grid m_streams_grid;

    // Bottom Controls
    Gtk::Box m_bottom_hbox;
    Gtk::Button m_record_button;
    Gtk::Button m_quit_button;
    Gtk::Label m_time_label;

    // Update Timer
    sigc::connection m_update_conn;
    bool m_session_finalized = false;

public:
    void trigger_record_toggle();
};

// C-compatible callback for ROS node
extern "C" gboolean toggle_recording_idle(gpointer user_data);

#endif // UI_HPP
