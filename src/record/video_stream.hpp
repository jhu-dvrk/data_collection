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
    struct SegmentSummary {
        std::string file;
        std::string sidecar;
        double duration = 0.0;
        long long frames = 0;
    };

    VideoStream();
    ~VideoStream();

    // -- Lifecycle --
    bool create(AppData* ad, const dc::VideoConfig* config);
    bool restart();
    void set_recording(bool active);
    void stop_and_save(const std::vector<std::string>& config_files);
    void shutdown();  // Properly tear down pipeline before destruction

    // -- Public Members for UI access --
    std::string name;
    GstElement *pipeline, *valve, *rec_overlay;
    GtkWidget *preview_widget, *record_checkbox, *retry_button, *stats_label;
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
    bool auto_retry_attempted;
    bool auto_retry_in_progress;
    std::string error_message;
    std::vector<SegmentSummary> segments;

private:
    AppData* m_ad;
    dc::VideoConfig m_config;
    bool m_has_config;
    int m_error_segment_index;
    std::string m_output_stem;

    void update_output_paths();
    void reset_segment_counters();
    void update_segment_summary();
    double compute_segment_duration_seconds() const;
};

#endif // VIDEO_STREAM_HPP
