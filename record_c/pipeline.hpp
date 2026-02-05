#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include "context.hpp"
#include <json/json.h>
#include "config.hpp"

std::string get_best_encoder(const dc::VideoEncoding& enc_cfg);
double get_audio_level_max(const GValue* gv);
GstPadProbeReturn source_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
GstPadProbeReturn audio_timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
GstPadProbeReturn timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
void create_audio_pipeline(AppData* data);
gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer user_data);
VideoStream* create_video_stream(AppData* data, const dc::VideoConfig& v_config);

#endif // PIPELINE_HPP
