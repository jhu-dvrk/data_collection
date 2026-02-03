#include "context.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <std_msgs/msg/bool.hpp>

int app_max_threads = 1;

AppData::AppData() : audio_pipeline(NULL), audio_sink(NULL), audio_valve(NULL), audio_src(NULL),
            window(NULL), record_button(NULL), data_dir_entry(NULL), 
            audio_level_bar(NULL), audio_enable_checkbox(NULL),
            audio_src_combo(NULL), stages_combo(NULL), session_entry(NULL), grid(NULL),
            bag_stats_label(NULL),
            data_directory("."), session_stage_cycle_count(1), current_stage_idx(0), global_recording(false), 
            blink_state(false), session_initialized(false), audio_is_recording(false), 
            is_quitting(false), enable_audio(false), eos_received_count(0),
            bag_messages_recorded(0), bag_topics_found(0), explicit_stages(false) {}

AppData::~AppData() {}
