#ifndef APP_DATA_HPP
#define APP_DATA_HPP

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
#include "config.hpp"

extern int app_max_threads;

// Forward declarations to reduce compilation time
namespace rosbag2_cpp { class Writer; }
class VideoStream;
class AudioStream;

struct AppData {
    std::vector<VideoStream*> streams;
    AudioStream* audio_stream;
    GtkWidget *window, *record_button, *data_dir_entry, *audio_level_bar, *audio_enable_checkbox;
    GtkWidget *audio_src_combo, *stages_combo, *session_entry, *grid;
    std::string data_directory, session_dir, start_timestamp;

    struct TagEvent {
        std::string name;
        long long cpu_ts;
        std::string generated_at;
    };
    std::vector<TagEvent> session_event_tags;
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
    int current_stage_idx;
    long long recording_start_cpu_ts;
    std::string recording_start_generated_at;
    bool global_recording, blink_state, session_initialized, is_quitting;
    bool record_audio;
    bool any_recording_started;
    int eos_received_count;

    // ROS 2 members
    std::string trigger_topic;
    std::shared_ptr<rclcpp::Node> node;
    std::shared_ptr<rclcpp::Subscription<std_msgs::msg::Bool>> sub_record;
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Bool>> pub_recording;

    // ROS Bag members
    std::string session_bag_path;
    std::vector<dc::RosTopicConfig> ros_topics;
    std::unique_ptr<rosbag2_cpp::Writer> bag_writer;
    std::vector<std::shared_ptr<rclcpp::GenericSubscription>> bag_subs;
    std::set<std::string> subscribed_topics;
    std::vector<std::shared_ptr<rclcpp::TimerBase>> timers;

    // Bag Stats
    long long bag_messages_recorded;
    int bag_topics_found;
    GtkWidget *bag_stats_label;

    bool explicit_stages;

    std::mutex data_mutex;
    bool dump_dot;
    GstDebugGraphDetails dot_flags;

    AppData();
    ~AppData();
};

#endif // APP_DATA_HPP
