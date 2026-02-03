#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <string>
#include <vector>
#include <mutex>
#include <gtk/gtk.h>
#include <gst/gst.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

extern int app_max_threads;

struct VideoStage {
    std::string stage;
    std::vector<long long> timestamps;
};

struct VideoStream {
    std::string name;
    GstElement *pipeline;
    GstElement *valve;
    GstElement *rec_text;
    GtkWidget *preview_widget;
    GtkWidget *record_checkbox;
    GtkWidget *stats_label;
    std::string output_video, output_json;
    std::vector<VideoStage> stages;
    bool is_recording;
    bool record_enabled;
    long long frames_recorded;
    long long frames_dropped;
    long long last_run_frames_recorded;
    std::string last_run_stage_name;
    double current_fps;
    
    // Source Stats
    int width, height;
    double src_fps;
    long long last_src_ts;
    long long src_frame_counter;

    // FPS Calculation State
    long long last_fps_ts;
    long long fps_frame_counter;

    VideoStream() : pipeline(NULL), valve(NULL), rec_text(NULL), preview_widget(NULL), record_checkbox(NULL), stats_label(NULL), 
                    is_recording(false), record_enabled(true), frames_recorded(0), frames_dropped(0), last_run_frames_recorded(0), last_run_stage_name(""), current_fps(0.0),
                    width(0), height(0), src_fps(0.0), last_src_ts(0), src_frame_counter(0),
                    last_fps_ts(0), fps_frame_counter(0) {}
};

struct AppData {
    std::vector<VideoStream*> streams;
    GstElement *audio_pipeline, *audio_sink, *audio_valve, *audio_src;
    GtkWidget *window, *record_button, *data_dir_entry, *audio_level_bar, *audio_enable_checkbox;
    GtkWidget *audio_src_combo, *stages_combo, *session_entry, *grid;
    std::string data_directory, session_dir, start_timestamp, audio_output_json;
    std::vector<VideoStage> audio_stages;
    std::vector<std::string> config_stages;
    std::vector<GtkWidget*> stage_labels;
    int session_stage_cycle_count;
    int current_stage_idx; // Track the current stage index for UI highlighting
    bool global_recording, blink_state, session_initialized, audio_is_recording, is_quitting;
    bool enable_audio; 
    int eos_received_count;

    // ROS 2 members
    std::string trigger_topic;
    rclcpp::Node::SharedPtr node;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_record;
    std::mutex data_mutex;
    
    AppData() : audio_pipeline(NULL), audio_sink(NULL), audio_valve(NULL), audio_src(NULL),
                window(NULL), record_button(NULL), data_dir_entry(NULL), 
                audio_level_bar(NULL), audio_enable_checkbox(NULL),
                audio_src_combo(NULL), stages_combo(NULL), session_entry(NULL), grid(NULL),
                data_directory("."), session_stage_cycle_count(1), current_stage_idx(0), global_recording(false), 
                blink_state(false), session_initialized(false), audio_is_recording(false), 
                is_quitting(false), enable_audio(false), eos_received_count(0) {}
};

#endif // CONTEXT_HPP
