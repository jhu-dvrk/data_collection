#include "app_data.hpp"
#include "video_stream.hpp"
#include "audio_stream.hpp"

#include <rosbag2_cpp/writer.hpp>

int app_max_threads = 1;

AppData::AppData() : audio_stream(NULL),
            window(NULL), record_button(NULL), data_dir_entry(NULL),
            audio_level_bar(NULL), audio_enable_checkbox(NULL),
            audio_src_combo(NULL), stages_combo(NULL), session_entry(NULL), grid(NULL),
            data_directory("."),
            session_stage_cycle_count(1), current_stage_idx(0),
            recording_start_cpu_ts(0),
            global_recording(false),
            blink_state(false), session_initialized(false),
            is_quitting(false), record_audio(false), any_recording_started(false), eos_received_count(0),
            bag_messages_recorded(0), bag_topics_found(0),
            bag_stats_label(NULL),
            explicit_stages(false) {}

AppData::~AppData() {
    for (auto s : streams) {
        delete s;
    }
    streams.clear();

    delete audio_stream;
    audio_stream = NULL;
}
