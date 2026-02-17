#include "context.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <std_msgs/msg/bool.hpp>

int app_max_threads = 1;

AppData::AppData() : audio_pipeline(NULL), audio_sink(NULL), audio_valve(NULL), audio_src(NULL),
            window(NULL), record_button(NULL), data_dir_entry(NULL),
            audio_level_bar(NULL), audio_enable_checkbox(NULL),
            audio_src_combo(NULL), stages_combo(NULL), session_entry(NULL), grid(NULL),
            data_directory("."),
            audio_total_offset_ns(0), audio_last_raw_pts(-1), audio_last_duration(0),
            session_stage_cycle_count(1), current_stage_idx(0), global_recording(false),
            blink_state(false), session_initialized(false), audio_is_recording(false),
            is_quitting(false), record_audio(false), eos_received_count(0),
            bag_messages_recorded(0), bag_topics_found(0),
            bag_stats_label(NULL),
            explicit_stages(false) {}

AppData::~AppData() {
    for (auto s : streams) {
        if (s) {
            if (s->pipeline) {
                gst_element_set_state(s->pipeline, GST_STATE_NULL);
                gst_object_unref(s->pipeline);
            }
            delete s;
        }
    }
    streams.clear();

    if (audio_pipeline) {
        gst_element_set_state(audio_pipeline, GST_STATE_NULL);
        gst_object_unref(audio_pipeline);
    }
}
