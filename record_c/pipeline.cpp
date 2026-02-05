
#include "pipeline.hpp"
#include <iostream>
#include <gtk/gtk.h>
#include <cairo.h>

void on_rec_overlay_draw(GstElement *overlay, cairo_t *cr, guint64 timestamp, guint64 duration, gpointer user_data) {
    (void)overlay; (void)timestamp; (void)duration;
    VideoStream *s = (VideoStream *)user_data;

    if (s->is_recording && s->frames_recorded < 0) { // Special flag for black flash
        cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
        cairo_paint(cr);
        return;
    }

    if (s->is_recording) {
        cairo_set_source_rgb(cr, 1.0, 0.0, 0.0); // Red
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 24);
        cairo_move_to(cr, 20, 40);
        cairo_show_text(cr, "REC");
    }
}

std::string get_best_encoder(const dc::VideoEncoding& enc_cfg) {
    int br = enc_cfg.bitrate;
    int preset = enc_cfg.speed_preset;
    int keyint = enc_cfg.key_int_max;

    const char* candidates[] = {"nvh264enc", "nvv4l2h264enc", "vaapih264enc", "x264enc"};
    for (const char* c : candidates) {
        GstElementFactory* f = gst_element_factory_find(c);
        if (f) {
            gst_object_unref(f);
            std::string name(c);
            std::cout << "Selected H264 Encoder: " << name << std::endl;
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
        // G_GNUC_BEGIN_IGNORE_DEPRECATIONS
        // Removed deprecated GValueArray check to fix warnings. Relying on GST_VALUE_HOLDS_ARRAY.
        // G_GNUC_END_IGNORE_DEPRECATIONS
        
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
    (void)pad;
    AppData *ad = (AppData *)user_data;
    if (ad->audio_is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
        if (!gst_buffer_is_writable(buf)) {
             // We need to modify timestamps, so we need a writable buffer.
             // gst_buffer_make_writable() might copy data if shared implicitly.
             // Since this is audio, copying is cheap.
             buf = gst_buffer_make_writable(buf);
             GST_PAD_PROBE_INFO_DATA(info) = buf;
        }

        long long pts = (long long)GST_BUFFER_PTS(buf);
        if (!GST_CLOCK_TIME_IS_VALID(pts)) return GST_PAD_PROBE_OK;
        long long duration = (long long)GST_BUFFER_DURATION(buf);
        
        // Gapless stitching logic
        if (ad->audio_last_raw_pts != -1) {
            long long delta = pts - ad->audio_last_raw_pts;
            // Use 500ms threshold for pause detection
            if (delta > 500 * 1000000LL) {
                 long long expected_gap = ad->audio_last_duration;
                 if (!GST_CLOCK_TIME_IS_VALID(expected_gap) || expected_gap == 0) expected_gap = 20 * 1000000LL; // Default 20ms?
                 
                 long long gap = delta - expected_gap;
                 if (gap > 0) ad->audio_total_offset_ns += gap;
            }
        }
        ad->audio_last_raw_pts = pts;
        ad->audio_last_duration = duration;

        // Apply offset to buffer
        GST_BUFFER_PTS(buf) = pts - ad->audio_total_offset_ns;
        if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buf))) {
             GST_BUFFER_DTS(buf) = GST_BUFFER_DTS(buf) - ad->audio_total_offset_ns;
        }

        // long long base_time = 0; // Relative to 0 now
        // if (ad->audio_pipeline) base_time = (long long)gst_element_get_base_time(ad->audio_pipeline);
        // Note: GST_TS in sidecar should be the timestamp inside the file (relative to start 0).
        // Since we are rewriting buffer timestamps to be continuous from 0 (implicitly if first pts ~ offset), 
        // we use the modified PTS.
        
        long long gst_ts = GST_BUFFER_PTS(buf);

        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        long long cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;

        ad->audio_frames.push_back({cpu_ts, gst_ts});
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    (void)pad;
    VideoStream *s = (VideoStream *)user_data;
    if (s->is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
        
        // Ensure writable for timestamp modification
        if (!gst_buffer_is_writable(buf)) {
             // For video, deep copy is expensive. Use copy to share memory but new metadata.
             // gst_buffer_copy() with default flags performs shallow copy of memory (ref)
             GstBuffer *new_buf = gst_buffer_copy(buf);
             gst_buffer_unref(buf);
             buf = new_buf;
             GST_PAD_PROBE_INFO_DATA(info) = buf;
        }

        long long pts = (long long)GST_BUFFER_PTS(buf);
        if (!GST_CLOCK_TIME_IS_VALID(pts)) return GST_PAD_PROBE_OK;
        long long duration = (long long)GST_BUFFER_DURATION(buf);

        // Gapless stitching logic
        if (s->last_raw_pts != -1) {
            long long delta = pts - s->last_raw_pts;
            // Use 500ms threshold for pause detection
            if (delta > 500 * 1000000LL) {
                 long long expected_gap = s->last_duration;
                 if (!GST_CLOCK_TIME_IS_VALID(expected_gap) || expected_gap == 0) expected_gap = 33333333LL; // Default 30fps
                 
                 long long gap = delta - expected_gap;
                 if (gap > 0) s->total_offset_ns += gap;
            }
        }
        s->last_raw_pts = pts;
        s->last_duration = duration;

        // Apply offset
        GST_BUFFER_PTS(buf) = pts - s->total_offset_ns;
        if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buf))) {
             GST_BUFFER_DTS(buf) = GST_BUFFER_DTS(buf) - s->total_offset_ns;
        }
        
        long long gst_ts = GST_BUFFER_PTS(buf);
        
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        long long cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;

        s->frames.push_back({cpu_ts, gst_ts});
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
    (void)bus;
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
            if (lvl < 0) lvl = 0;
            if (lvl > 1) lvl = 1;
            
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

    std::string apstr = "pulsesrc name=asrc ! audioconvert ! audioresample ! level name=lvl ! tee name=at "
                        "at. ! queue ! fakesink sync=false async=false "
                        "at. ! queue ! valve name=av drop=true ! wavenc ! filesink name=asink sync=false async=false";
    data->audio_pipeline_desc = apstr;
    data->audio_pipeline = gst_parse_launch(apstr.c_str(), NULL);
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

VideoStream* create_video_stream(AppData* data, const dc::VideoConfig& v) {
    VideoStream *s = new VideoStream(); s->name = v.name;
    s->record_enabled = v.record;

    std::string caps = "video/x-raw";
    if (v.encoding.width > 0 && v.encoding.height > 0) {
        caps += ",width=" + std::to_string(v.encoding.width) + ",height=" + std::to_string(v.encoding.height);
    }
    if (v.encoding.frame_rate > 0) {
        caps += ",framerate=" + std::to_string(v.encoding.frame_rate) + "/1";
    }

    std::string ts_overlay = "";
    if (v.timestamp_overlay) {
        ts_overlay = " ! timeoverlay valignment=bottom halignment=left font-desc=\"Sans, 10\" shaded-background=true shading-value=255 xpad=0 ypad=0 "
                     " ! clockoverlay valignment=bottom halignment=right time-format=\"%Y-%m-%d %H:%M:%S\" font-desc=\"Sans, 10\" shaded-background=true shading-value=255 xpad=0 ypad=0 ";
    }

    std::string pstr = v.stream + " do-timestamp=true ! " + caps + ts_overlay + " ! tee name=t "
        "t. ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! videoconvert n-threads=" + std::to_string(app_max_threads) + " ! cairooverlay name=rec_overlay ! gtksink name=sink sync=false async=false ";
    
    if (s->record_enabled) {
        pstr += "t. ! queue name=qrec max-size-buffers=2 max-size-time=0 max-size-bytes=0 leaky=downstream ! valve name=v drop=true ! videoconvert n-threads=" + std::to_string(app_max_threads) + " ! " + get_best_encoder(v.encoding) + " ! queue max-size-buffers=1 leaky=downstream ! mp4mux ! filesink name=muxer sync=false async=false";
    }

    s->pipeline_desc = pstr;
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
            (void)q;
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
    s->rec_overlay = gst_bin_get_by_name(GST_BIN(s->pipeline), "rec_overlay");
    if (s->rec_overlay) {
        g_signal_connect(s->rec_overlay, "draw", G_CALLBACK(on_rec_overlay_draw), s);
        gst_object_unref(s->rec_overlay); 
        // Note: we kept the pointer in struct but unref here? 
        // gst_bin_get_by_name returns a NEW ref.
        // We can keep it if we want, or rely on pipeline ownership.
        // The struct member is just a pointer, we don't own the ref unless recorded.
        // Actually, previous code didn't unref rec_text? 
        // Let's check. Yes, it did NOT unref rec_text.
        // It helps to keep a ref if we access it later. But we don't access rec_overlay later except in callback via user_data?
        // Wait, on_rec_overlay_draw is called by element. 
        // We don't need to touch s->rec_overlay in UI anymore.
        // So we can unref it here. But if we want to follow RAII, s->rec_overlay should probably NOT hold a ref if the pipeline holds it,
        // BUT gst_bin_get_by_name returns a full reference. We MUST unref it eventually or when destroying struct.
        // For now, let's keep the pattern.
        // Previous: s->rec_text = ...; NO unref.
        // This causes a small leak if not cleared in destructor, but pipeline unref cleans children.
        // However, the extra ref from get_by_name means pipeline won't finalize it fully/double free?
        // No, if refcount > 1, pipeline simple drops its ref. We assume we hold one.
        // We should release it in destructor of VideoStream?
        // VideoStream has no destructor cleaning GST objects currently!
        // It relies on AppData or main? No.
        // Minor leak. I'll stick to not unref-ing to match previous style, or better, unref it if not needed.
        // We don't need s->rec_overlay in UI anymore.
        // So I will NOT store it in s->rec_overlay if not needed, OR unref it.
        // I changed struct to have rec_overlay. Let's store it and NOT unref, matching previous leak/style to be safe against double-unref fears if previous code did weird things. 
        // Actually I will unref it immediately because I don't need it outside this function setup since the callback handles it.
        // I'll keep the struct member and store it but I'll update the replacing code to match previous pattern.
        // Previous line: s->rec_text = gst_bin_get_by_name(...);
    }
    
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
