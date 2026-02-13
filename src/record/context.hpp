#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <string>
#include <vector>
#include <set>
#include <utility>
#include <mutex>
#include <memory>
#include <gtk/gtk.h>
#include <gst/gst.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

extern int app_max_threads;

// Forward declarations to reduce compilation time
namespace rosbag2_cpp { class Writer; }

struct FrameData {
    long long cpu_ts;
    long long gst_ts;
};

struct VideoStream {
    std::string name;
    GstElement *pipeline;
    GstElement *valve;
    GstElement *rec_overlay;
    GtkWidget *preview_widget;
    GtkWidget *record_checkbox;
    GtkWidget *stats_label;
    std::string output_video, output_json;
    std::string pipeline_desc;
    std::vector<FrameData> frames;
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

    // Stitching / Gapless Recording state
    long long total_offset_ns;
    long long last_raw_pts;
    long long last_duration;

    // FPS Calculation State
    long long last_fps_ts;
    long long fps_frame_counter;

    VideoStream() : pipeline(NULL), valve(NULL), rec_overlay(NULL), preview_widget(NULL), record_checkbox(NULL), stats_label(NULL),
                    is_recording(false), record_enabled(true), frames_recorded(0), frames_dropped(0), last_run_frames_recorded(0), last_run_stage_name(""), current_fps(0.0),
                    width(0), height(0), src_fps(0.0), last_src_ts(0), src_frame_counter(0),
                    total_offset_ns(0), last_raw_pts(-1), last_duration(0),
                    last_fps_ts(0), fps_frame_counter(0) {}
};

struct AppData {
    std::vector<VideoStream*> streams;
    GstElement *audio_pipeline, *audio_sink, *audio_valve, *audio_src;
    GtkWidget *window, *record_button, *data_dir_entry, *audio_level_bar, *audio_enable_checkbox;
    GtkWidget *audio_src_combo, *stages_combo, *session_entry, *grid;
    std::string data_directory, session_dir, start_timestamp, audio_output_json;
    std::string audio_pipeline_desc;
    std::vector<FrameData> audio_frames;

    // Audio Stitching
    long long audio_total_offset_ns;
    long long audio_last_raw_pts;
    long long audio_last_duration;

    std::vector<std::pair<std::string, long long>> session_event_tags;
    struct StageEvent {
        std::string name;
        long long start_ts;
        std::string start_generated_at;
        long long end_ts;
        std::string end_generated_at;
    };
    std::vector<StageEvent> session_stages;
    std::vector<std::string> config_stages;
    std::vector<std::string> config_tags;
    std::vector<std::string> config_files;
    std::vector<GtkWidget*> stage_labels;
    int session_stage_cycle_count;
    int current_stage_idx; // Track the current stage index for UI highlighting
    long long recording_start_cpu_ts;
    std::string recording_start_generated_at;
    bool global_recording, blink_state, session_initialized, audio_is_recording, is_quitting;
    bool enable_audio;
    int eos_received_count;

    // ROS 2 members
    std::string trigger_topic;
    std::shared_ptr<rclcpp::Node> node;
    std::shared_ptr<rclcpp::Subscription<std_msgs::msg::Bool>> sub_record;

    // ROS Bag members
    std::string session_bag_path;
    std::vector<std::string> ros_topics;
    std::unique_ptr<rosbag2_cpp::Writer> bag_writer;
    std::vector<std::shared_ptr<rclcpp::GenericSubscription>> bag_subs;
    std::set<std::string> subscribed_topics; // Track existing subscriptions
    std::vector<std::shared_ptr<rclcpp::TimerBase>> timers;

    // Bag Stats
    long long bag_messages_recorded;
    int bag_topics_found;
    GtkWidget *bag_stats_label;

    bool explicit_stages;

    std::mutex data_mutex;

    AppData(); // Defined in cpp
    ~AppData(); // Defined in cpp
};

#endif // CONTEXT_HPP
