#ifndef GST_HELPERS_HPP
#define GST_HELPERS_HPP

#include <string>
#include <gst/gst.h>
#include <cairo.h>
#include <json/json.h>
#include "../common/config.hpp"

// Forward declarations
struct AppData;
class VideoStream;
class AudioStream;

// Global GStreamer Helpers
std::string get_best_encoder(const dc::VideoEncoding& enc_cfg);
double get_audio_level_max(const GValue* gv);
void dump_dot(GstElement* pipeline, const std::string& session_dir, const std::string& name);
void shutdown_pipeline(GstElement* pipeline);  // Send EOS, wait, set NULL, remove bus watch

// GStreamer Callbacks
GstFlowReturn on_new_ros_sample(GstElement *sink, gpointer user_data);
void on_rec_overlay_draw(GstElement *overlay, cairo_t *cr, guint64 timestamp, guint64 duration, gpointer user_data);
GstPadProbeReturn source_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
GstPadProbeReturn audio_timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
GstPadProbeReturn timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer user_data);

#endif // GST_HELPERS_HPP
