#ifndef AUDIO_STREAM_HPP
#define AUDIO_STREAM_HPP

#include <string>
#include <vector>
#include <gst/gst.h>
#include "common_types.hpp"

struct AppData;

class AudioStream {
public:
    AudioStream();
    ~AudioStream();

    // -- Lifecycle --
    bool create(AppData* ad);
    void set_recording(bool active);
    void stop_and_save(const std::vector<std::string>& config_files);
    void shutdown();  // Properly tear down pipeline before destruction

    // -- Public Members for UI access --
    GstElement *pipeline, *sink, *valve, *src;
    std::string pipeline_desc;
    std::string output_audio, output_json;
    std::vector<FrameData> frames;
    bool is_recording;
    long long total_offset_ns, last_raw_buffer_ts, last_duration;
    long long cpu_ts_from_unixfd_count, cpu_ts_at_reception_count;

private:
    AppData* m_ad;
};

#endif // AUDIO_STREAM_HPP
