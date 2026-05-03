#include "audio_stream.hpp"
#include "context.hpp"
#include "gst_helpers.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

AudioStream::AudioStream() : pipeline(NULL), sink(NULL), valve(NULL), src(NULL),
                is_recording(false), total_offset_ns(0), last_raw_buffer_ts(-1), last_duration(0),
                cpu_ts_from_unixfd_count(0), cpu_ts_at_reception_count(0), m_ad(NULL) {}

AudioStream::~AudioStream() {
    // Pipeline should already be shut down via shutdown()
    // Just release resources
    if (pipeline) {
        gst_object_unref(pipeline);
        pipeline = NULL;
    }
    if (valve) {
        gst_object_unref(valve);
        valve = NULL;
    }
}

void AudioStream::shutdown() {
    if (!pipeline) return;

    // Open valve to drain any remaining audio
    if (valve) g_object_set(valve, "drop", FALSE, NULL);

    shutdown_pipeline(pipeline);
}

bool AudioStream::create(AppData* ad) {
    this->m_ad = ad;
    std::string apstr = "pulsesrc name=asrc ! audioconvert ! audioresample ! level name=lvl ! tee name=at "
                        "at. ! queue ! fakesink sync=false async=false "
                        "at. ! queue ! valve name=av drop=true ! wavenc ! filesink name=asink sync=false async=false";
    this->pipeline_desc = apstr;
    this->pipeline = gst_parse_launch(apstr.c_str(), NULL);
    if (!this->pipeline) return false;

    GstElement *lvl = gst_bin_get_by_name(GST_BIN(this->pipeline), "lvl");
    if (lvl) {
        g_object_set(lvl, "post-messages", TRUE, "interval", (guint64)100000000, NULL);
        gst_object_unref(lvl);
    }

    GstElement *asrc = gst_bin_get_by_name(GST_BIN(this->pipeline), "asrc");
    if (asrc) {
        const gchar* id = gtk_combo_box_get_active_id(GTK_COMBO_BOX(ad->audio_src_combo));
        if (id) g_object_set(asrc, "device", id, NULL);
        gst_object_unref(asrc);
    }

    // Cache the valve element and attach timestamping probe
    this->valve = gst_bin_get_by_name(GST_BIN(this->pipeline), "av");
    if (this->valve) {
        GstPad *apad = gst_element_get_static_pad(this->valve, "src");
        if (apad) {
            gst_pad_add_probe(apad, GST_PAD_PROBE_TYPE_BUFFER, audio_timestamp_probe_cb, this, NULL);
            gst_object_unref(apad);
        }
    }

    // Set output paths from session data
    this->output_audio = ad->session_dir + "/audio.wav";
    this->output_json  = ad->session_dir + "/audio.json";

    GstElement *asink = gst_bin_get_by_name(GST_BIN(this->pipeline), "asink");
    if (asink) {
        g_object_set(asink, "location", this->output_audio.c_str(), NULL);
        gst_object_unref(asink);
    }

    if (ad->dump_dot) {
        dc::dump_dot(this->pipeline, ad->session_dir + "/audio_pipeline.dot", ad->dot_flags);
    }

    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(this->pipeline));
    gst_bus_add_watch(bus, bus_call, ad);
    gst_object_unref(bus);

    gst_element_set_state(this->pipeline, GST_STATE_PLAYING);
    return true;
}

void AudioStream::set_recording(bool active) {
    this->is_recording = active;
    if (this->valve) g_object_set(this->valve, "drop", !active, NULL);
}

void AudioStream::stop_and_save(const std::vector<std::string>& config_files) {
    this->set_recording(false);

    if (!this->output_json.empty()) {
        Json::Value aroot;
        aroot["stream_name"] = "audio";
        aroot["start_timestamp_ns"] = this->frames.empty() ? 0 : (Json::Value::Int64)this->frames.front().cpu_ts;
        aroot["end_timestamp_ns"] = this->frames.empty() ? 0 : (Json::Value::Int64)this->frames.back().cpu_ts;

        Json::Value cpuTsObj(Json::objectValue);
        cpuTsObj["from_unixfd"] = (Json::Value::Int64)this->cpu_ts_from_unixfd_count;
        cpuTsObj["at_reception"] = (Json::Value::Int64)this->cpu_ts_at_reception_count;
        aroot["cpu_ts"] = cpuTsObj;

        Json::Value alist = Json::arrayValue;
        for (const auto& f : this->frames) {
            Json::Value fv;
            fv["cpu_ns"] = (Json::Value::Int64)f.cpu_ts;
            fv["gst_ns"] = (Json::Value::Int64)f.buffer_ts_ns;
            alist.append(fv);
        }
        aroot["frames"] = alist;

        Json::Value configs = Json::arrayValue;
        for (const auto& cf : config_files) {
            std::ifstream ifs(cf);
            if (ifs.is_open()) {
                std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                configs.append(content);
            }
        }
        aroot["configs"] = configs;

        std::ofstream aofs(this->output_json);
        aofs << aroot.toStyledString();
    }
}
