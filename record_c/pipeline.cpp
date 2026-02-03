#include "pipeline.hpp"
#include <iostream>
#include <gtk/gtk.h>

std::string get_best_encoder(const Json::Value& enc_cfg) {
    int br = enc_cfg.get("bitrate", 10000).asInt();
    int preset = enc_cfg.get("speed_preset", 5).asInt();
    int keyint = enc_cfg.get("key_int_max", 30).asInt();

    const char* candidates[] = {"nvh264enc", "nvv4l2h264enc", "vaapih264enc", "x264enc"};
    for (const char* c : candidates) {
        GstElementFactory* f = gst_element_factory_find(c);
        if (f) {
            gst_object_unref(f);
            std::string name(c);
            if (name == "nvh264enc") return "nvh264enc bitrate=" + std::to_string(br * 1000) + " zerolatency=true ! h264parse";
            if (name == "nvv4l2h264enc") return "nvv4l2h264enc bitrate=" + std::to_string(br * 1000) + " preset-level=4 control-rate=1 ! h264parse";
            if (name == "vaapih264enc") return "vaapih264enc bitrate=" + std::to_string(br) + " ! h264parse";
            if (name == "x264enc") return "x264enc bitrate=" + std::to_string(br) + " speed-preset=" + std::to_string(preset) + " tune=zerolatency key-int-max=" + std::to_string(keyint) + " threads=" + std::to_string(app_max_threads) + " ! h264parse";
        }
    }
    return "x264enc ! h264parse";
}

double get_audio_level_max(const GValue* gv) {
    double m = -100.0;
    if (!gv) return m;
    
    if (GST_VALUE_HOLDS_LIST(gv)) {
        for (guint i=0; i < gst_value_list_get_size(gv); ++i) {
            const GValue *v = gst_value_list_get_value(gv, i);
            if (v) m = std::max(m, g_value_get_double(v));
        }
    } else {
        G_GNUC_BEGIN_IGNORE_DEPRECATIONS
        if (G_TYPE_CHECK_VALUE_TYPE(gv, G_TYPE_VALUE_ARRAY)) {
            GValueArray *arr = (GValueArray *)g_value_get_boxed(gv);
            if (arr) {
                for (guint i=0; i < arr->n_values; ++i) {
                    GValue *v = g_value_array_get_nth(arr, i);
                    if (v) m = std::max(m, g_value_get_double(v));
                }
            }
        }
        G_GNUC_END_IGNORE_DEPRECATIONS
        // Modern GStreamer: also check for array type
        if (GST_VALUE_HOLDS_ARRAY(gv)) {
            for (guint i=0; i < gst_value_array_get_size(gv); ++i) {
                const GValue *v = gst_value_array_get_value(gv, i);
                if (v) m = std::max(m, g_value_get_double(v));
            }
        }
    }

    if (G_VALUE_HOLDS_DOUBLE(gv)) {
        m = g_value_get_double(gv);
    }
    return m;
}

GstPadProbeReturn source_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    VideoStream *s = (VideoStream *)user_data;
    if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
        long long now = g_get_monotonic_time();
        if (s->last_src_ts == 0) s->last_src_ts = now;
        s->src_frame_counter++;
        if (now - s->last_src_ts >= 1000000) {
            s->src_fps = (double)s->src_frame_counter * 1000000.0 / (double)(now - s->last_src_ts);
            s->src_frame_counter = 0;
            s->last_src_ts = now;
        }
        
        if (s->width == 0 || s->height == 0) {
            GstCaps *caps = gst_pad_get_current_caps(pad);
            if (caps) {
                GstStructure *st = gst_caps_get_structure(caps, 0);
                gst_structure_get_int(st, "width", &s->width);
                gst_structure_get_int(st, "height", &s->height);
                gst_caps_unref(caps);
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn audio_timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    AppData *ad = (AppData *)user_data;
    if (ad->audio_is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        ad->audio_stages.back().timestamps.push_back(g_get_real_time() * 1000);
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    VideoStream *s = (VideoStream *)user_data;
    if (s->is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        s->stages.back().timestamps.push_back(g_get_real_time() * 1000);
        s->frames_recorded++;
        
        long long now = g_get_monotonic_time();
        if (s->last_fps_ts == 0) s->last_fps_ts = now;
        s->fps_frame_counter++;
        
        if (now - s->last_fps_ts >= 1000000) {
            s->current_fps = (double)s->fps_frame_counter * 1000000.0 / (double)(now - s->last_fps_ts);
            s->fps_frame_counter = 0;
            s->last_fps_ts = now;
        }
    }
    return GST_PAD_PROBE_OK;
}

gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer user_data) {
    AppData *ad = (AppData *)user_data;
    if (ad->is_quitting && GST_MESSAGE_TYPE(msg) != GST_MESSAGE_EOS) return TRUE;

    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
        GstObject *src = GST_MESSAGE_SRC(msg);
        bool is_pipeline = (ad->audio_pipeline && src == GST_OBJECT(ad->audio_pipeline));
        if (!is_pipeline) {
            for (auto s : ad->streams) if (s->pipeline && src == GST_OBJECT(s->pipeline)) { is_pipeline = true; break; }
        }

        if (is_pipeline) {
            int target = ad->streams.size() + (ad->audio_pipeline ? 1 : 0);
            ad->eos_received_count++;
            g_print("Pipeline EOS received (%d/%d)\n", ad->eos_received_count, target);
            if (ad->eos_received_count >= target) {
                g_print("All streams finished. Exiting.\n");
                gtk_main_quit();
            }
        }
    } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ELEMENT) {
        const GstStructure *s = gst_message_get_structure(msg);
        if (gst_structure_has_name(s, "level")) {
            const GValue *rms_val = gst_structure_get_value(s, "rms");
            const GValue *peak_val = gst_structure_get_value(s, "peak");
            
            double max_rms = get_audio_level_max(rms_val);
            double max_peak = get_audio_level_max(peak_val);
            
            double display_val = std::max(max_rms, max_peak);
            
            // Map -100dB..0dB to 0.0..1.0
            double lvl = (display_val + 100.0) / 100.0;
            if (lvl < 0) lvl = 0; if (lvl > 1) lvl = 1;
            
            if (ad->audio_level_bar && GTK_IS_LEVEL_BAR(ad->audio_level_bar)) {
                gtk_level_bar_set_value(GTK_LEVEL_BAR(ad->audio_level_bar), lvl);
            }
        }
    } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_STATE_CHANGED) {
        GstObject *src = GST_MESSAGE_SRC(msg);
        if (ad->audio_pipeline && src == GST_OBJECT(ad->audio_pipeline)) {
            GstState old_st, new_st, pend;
            gst_message_parse_state_changed(msg, &old_st, &new_st, &pend);
        }
    } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        gchar *dbg; GError *err; gst_message_parse_error(msg, &err, &dbg);
        std::cerr << "Error: " << err->message << std::endl; g_free(dbg); g_error_free(err);
    }
    return TRUE;
}

void create_audio_pipeline(AppData* data) {
    if (data->audio_pipeline) {
        gst_element_set_state(data->audio_pipeline, GST_STATE_NULL);
        gst_object_unref(data->audio_pipeline);
        data->audio_pipeline = NULL;
    }

    data->audio_pipeline = gst_parse_launch("pulsesrc name=asrc ! audioconvert ! audioresample ! level name=lvl ! tee name=at "
                                            "at. ! queue ! fakesink sync=false async=false "
                                            "at. ! queue ! valve name=av drop=true ! wavenc ! filesink name=asink sync=false async=false", NULL);
    if (data->audio_pipeline) {
        GstElement *lvl = gst_bin_get_by_name(GST_BIN(data->audio_pipeline), "lvl");
        if (lvl) {
            g_object_set(lvl, "post-messages", TRUE, "interval", (guint64)100000000, NULL);
            gst_object_unref(lvl);
        }
        data->audio_sink = gst_bin_get_by_name(GST_BIN(data->audio_pipeline), "asink");
        std::string af = data->session_dir + "/audio_" + data->start_timestamp + ".wav";
        data->audio_output_json = data->session_dir + "/audio_" + data->start_timestamp + ".json";
        g_object_set(data->audio_sink, "location", af.c_str(), NULL);
        data->audio_valve = gst_bin_get_by_name(GST_BIN(data->audio_pipeline), "av");
        data->audio_src = gst_bin_get_by_name(GST_BIN(data->audio_pipeline), "asrc");
        
        const gchar* id = gtk_combo_box_get_active_id(GTK_COMBO_BOX(data->audio_src_combo));
        if (id) g_object_set(data->audio_src, "device", id, NULL);

        GstPad *apad = gst_element_get_static_pad(data->audio_valve, "src");
        gst_pad_add_probe(apad, GST_PAD_PROBE_TYPE_BUFFER, audio_timestamp_probe_cb, data, NULL);
        gst_object_unref(apad);

        GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(data->audio_pipeline));
        gst_bus_add_watch(bus, bus_call, data);
        gst_object_unref(bus);

        gst_element_set_state(data->audio_pipeline, GST_STATE_PLAYING);
    }
}

VideoStream* create_video_stream(AppData* data, const Json::Value& v) {
    VideoStream *s = new VideoStream(); s->name = v["name"].asString();
    s->record_enabled = v.get("record", true).asBool();

    std::string caps = "video/x-raw";
    if (v["encoding"].isMember("width") && v["encoding"].isMember("height")) {
        caps += ",width=" + std::to_string(v["encoding"]["width"].asInt()) + ",height=" + std::to_string(v["encoding"]["height"].asInt());
    }
    if (v["encoding"].isMember("frame_rate")) {
        caps += ",framerate=" + std::to_string(v["encoding"]["frame_rate"].asInt()) + "/1";
    }

    std::string ts_overlay = "";
    if (v.get("timestamp_overlay", false).asBool()) {
        ts_overlay = " ! timeoverlay valignment=bottom halignment=left font-desc=\"Sans, 10\" shaded-background=true "
                        " ! clockoverlay valignment=bottom halignment=right time-format=\"%Y-%m-%d %H:%M:%S\" font-desc=\"Sans, 10\" shaded-background=true ";
    }

    std::string pstr = v["stream"].asString() + " do-timestamp=true ! " + caps + ts_overlay + " ! tee name=t "
        "t. ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! videoconvert n-threads=" + std::to_string(app_max_threads) + " ! textoverlay name=rec_text text=\"\" valignment=top halignment=left font-desc=\"Sans, 24\" ! gtksink name=sink sync=false async=false "
        "t. ! queue name=qrec max-size-buffers=2 max-size-time=0 max-size-bytes=0 leaky=downstream ! valve name=v drop=true ! videoconvert n-threads=" + std::to_string(app_max_threads) + " ! " + get_best_encoder(v["encoding"]) + " ! queue max-size-buffers=1 leaky=downstream ! mp4mux ! filesink name=muxer sync=false async=false";
    s->pipeline = gst_parse_launch(pstr.c_str(), NULL);
    if (!s->pipeline) { delete s; return NULL; }

    GstElement *tee = gst_bin_get_by_name(GST_BIN(s->pipeline), "t");
    if (tee) {
        GstPad *tpad = gst_element_get_static_pad(tee, "sink");
        gst_pad_add_probe(tpad, GST_PAD_PROBE_TYPE_BUFFER, source_probe_cb, s, NULL);
        gst_object_unref(tpad);
        gst_object_unref(tee);
    }

    GstElement *qrec = gst_bin_get_by_name(GST_BIN(s->pipeline), "qrec");
    if (qrec) {
        g_signal_connect(qrec, "overrun", G_CALLBACK(+[](GstElement* q, gpointer d){
            VideoStream* vs = (VideoStream*)d; vs->frames_dropped++;
        }), s);
        gst_object_unref(qrec);
    }

    std::string sn = s->name; for (char &c : sn) if (c == ' ') c = '_';
    s->output_video = data->session_dir + "/" + sn + "_" + data->start_timestamp + ".mp4";
    s->output_json = data->session_dir + "/" + sn + "_" + data->start_timestamp + ".json";
    GstElement *mux = gst_bin_get_by_name(GST_BIN(s->pipeline), "muxer");
    if (mux) { g_object_set(mux, "location", s->output_video.c_str(), NULL); gst_object_unref(mux); }

    s->valve = gst_bin_get_by_name(GST_BIN(s->pipeline), "v");
    s->rec_text = gst_bin_get_by_name(GST_BIN(s->pipeline), "rec_text");
    GstElement *sink = gst_bin_get_by_name(GST_BIN(s->pipeline), "sink");
    g_object_get(sink, "widget", &s->preview_widget, NULL);
    gtk_widget_set_size_request(s->preview_widget, 320, 240);
    GstPad *vpad = gst_element_get_static_pad(s->valve, "src");
    gst_pad_add_probe(vpad, GST_PAD_PROBE_TYPE_BUFFER, timestamp_probe_cb, s, NULL);
    gst_object_unref(vpad);
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(s->pipeline));
    gst_bus_add_watch(bus, bus_call, data); gst_object_unref(bus);
    
    return s;
}
