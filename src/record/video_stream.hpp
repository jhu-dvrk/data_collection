#ifndef VIDEO_STREAM_HPP
#define VIDEO_STREAM_HPP

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <gtk/gtk.h>
#include <gst/gst.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include "common_types.hpp"
#include "../common/config.hpp"

struct AppData; // Forward declaration

class VideoStream {
public:
    VideoStream();
    ~VideoStream();

    // -- Lifecycle --
    bool create(AppData* ad, const dc::VideoConfig* config);
    void set_recording(bool active);
    void stop_and_save(const std::vector<std::string>& config_files);
    void shutdown();  // Properly tear down pipeline before destruction

    // -- Public Members for UI access --
    std::string name;
    GstElement *pipeline, *valve, *rec_overlay;
    GtkWidget *preview_widget, *record_checkbox, *stats_label;
    GtkWidget *stream_name_label, *preview_stack;
    std::string output_video, output_json;
    std::string pipeline_desc;
    std::vector<FrameData> frames;
    bool is_recording, record_enabled;
    long long frames_recorded, frames_dropped;
    long long last_run_frames_recorded;
    std::string last_run_stage_name;
    double current_fps, estimated_latency;
    long long cpu_ts_from_unixfd_count, cpu_ts_at_reception_count;

    // Source Stats
    int width, height;
    double src_fps;
    long long last_src_ts, src_frame_counter;

    // Recording Config
    int rec_width, rec_height;
    double rec_fps_requested;

    // Stitching / Gapless
    long long total_offset_ns, last_raw_buffer_ts, last_duration;

    // FPS Calculation
    long long last_fps_ts, fps_frame_counter;

    std::string side_by_side;

    bool has_error;
    std::string error_message;

private:
    AppData* m_ad;
};

#endif // VIDEO_STREAM_HPP
