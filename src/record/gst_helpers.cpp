#include "gst_helpers.hpp"
#include "context.hpp"
#include "video_stream.hpp"
#include "audio_stream.hpp"
#include "ui.hpp"

#include <iostream>
#include <gst/video/video.h>
#include <glibmm.h>
#include "../common/gst_utils.hpp"
#include "../common/cpu_timestamp_meta.hpp"

extern int app_max_threads;

void on_rec_overlay_draw(GstElement *overlay, cairo_t *cr, guint64 timestamp, guint64 duration, gpointer user_data) {
    (void)overlay; (void)timestamp; (void)duration;
    VideoStream *s = (VideoStream *)user_data;

    // Overlay is only visible during recording
    if (!s->is_recording) return;

    if (s->frames_recorded < 0) { // Special flag for black flash
        cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
        cairo_paint(cr);
        return;
    }

    double base_size = 20.0;
    double scale = 1.0;
    if (s->width > 0 && s->height > 0) {
        scale = std::min(s->width / 640.0, s->height / 480.0);
        scale = std::max(0.5, std::min(scale, 4.0));
    }

    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, base_size * scale);

    char status_buf[256];
    int rw = s->rec_width > 0 ? s->rec_width : s->width;
    int rh = s->rec_height > 0 ? s->rec_height : s->height;
    double rf = s->rec_fps_requested > 0.1 ? s->rec_fps_requested : s->src_fps;
    
    snprintf(status_buf, sizeof(status_buf), "REC %dx%d %.1f | LIVE %.1f", 
             rw, rh, rf, s->current_fps);

    // Draw shadow
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_move_to(cr, 21.0 * scale, (double)s->height - (21.0 * scale));
    cairo_show_text(cr, status_buf);

    // Draw text (Red for Recording)
    cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
    cairo_move_to(cr, 20.0 * scale, (double)s->height - (22.0 * scale));
    cairo_show_text(cr, status_buf);
}

static bool probe_encoder(const std::string& pipeline_str) {
    GError* err = nullptr;
    GstElement* pipe = gst_parse_launch(pipeline_str.c_str(), &err);
    if (!pipe) {
        if (err) g_error_free(err);
        return false;
    }
    // PAUSED forces caps negotiation and encoder initialization
    GstStateChangeReturn ret = gst_element_set_state(pipe, GST_STATE_PAUSED);
    bool ok = (ret != GST_STATE_CHANGE_FAILURE);
    if (ok) {
        // Wait up to 2s for async state change to complete
        ret = gst_element_get_state(pipe, nullptr, nullptr, 2 * GST_SECOND);
        ok = (ret != GST_STATE_CHANGE_FAILURE);
    }
    gst_element_set_state(pipe, GST_STATE_NULL);
    gst_object_unref(pipe);
    return ok;
}

std::string get_best_encoder(const dc::VideoEncoding& enc_cfg) {
    int br = enc_cfg.bitrate_kbps;
    int preset = enc_cfg.speed_preset;
    int keyint = enc_cfg.key_int_max;

    const char* candidates[] = {"nvh264enc", "nvv4l2h264enc", "vaapih264enc", "x264enc"};
    for (const char* c : candidates) {
        GstElementFactory* f = gst_element_factory_find(c);
        if (f) {
            gst_object_unref(f);
            std::string name(c);
            if (name == "nvh264enc") {
                std::string enc_str = "nvh264enc bitrate=" + std::to_string(br);
                std::string test = "videotestsrc num-buffers=1 ! video/x-raw,format=I420,width=320,height=240,framerate=30/1 ! videoconvert ! video/x-raw,format=NV12 ! " + enc_str + " ! fakesink";
                if (probe_encoder(test)) return enc_str;
                std::cerr << "\033[1;33mWarning: nvh264enc found but failed runtime probe (preset not supported on this GPU), skipping.\033[0m" << std::endl;
                continue;
            }
            if (name == "nvv4l2h264enc") {
                GstElementFactory* conv = gst_element_factory_find("nvvidconv");
                if (conv) {
                    gst_object_unref(conv);
                    return "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate=" + std::to_string(br) + " control-rate=1";
                }
            }
            if (name == "vaapih264enc") return "vaapih264enc bitrate=" + std::to_string(br);
            if (name == "x264enc") {
                int used_preset = (preset > 0 ? preset : 1);
                return "x264enc bitrate=" + std::to_string(br) + " speed-preset=" + std::to_string(used_preset) + " tune=zerolatency key-int-max=" + std::to_string(keyint) + " threads=" + std::to_string(app_max_threads) + " bframes=0";
            }
        }
    }
    return "x264enc";
}

double get_audio_level_max(const GValue* gv) {
    double m = -100.0;
    if (!gv) return m;
    if (GST_VALUE_HOLDS_LIST(gv)) {
        for (guint i=0; i < gst_value_list_get_size(gv); ++i) {
            const GValue *v = gst_value_list_get_value(gv, i);
            if (v) m = std::max(m, g_value_get_double(v));
        }
    } else if (GST_VALUE_HOLDS_ARRAY(gv)) {
        for (guint i=0; i < gst_value_array_get_size(gv); ++i) {
            const GValue *v = gst_value_array_get_value(gv, i);
            if (v) m = std::max(m, g_value_get_double(v));
        }
    }
    if (G_VALUE_HOLDS_DOUBLE(gv)) m = g_value_get_double(gv);
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
    AudioStream *as = (AudioStream *)user_data;
    if (as->is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
        if (!gst_buffer_is_writable(buf)) {
             buf = gst_buffer_make_writable(buf);
             GST_PAD_PROBE_INFO_DATA(info) = buf;
        }
        long long buffer_ts = (long long)GST_BUFFER_PTS(buf);
        if (!GST_CLOCK_TIME_IS_VALID(buffer_ts)) return GST_PAD_PROBE_OK;
        if (as->last_raw_buffer_ts != -1) {
            long long delta = buffer_ts - as->last_raw_buffer_ts;
            if (delta > 500 * 1000000LL) {
                 long long expected_gap = as->last_duration;
                 if (!GST_CLOCK_TIME_IS_VALID(expected_gap) || expected_gap == 0) expected_gap = 20 * 1000000LL;
                 long long gap = delta - expected_gap;
                 if (gap > 0) as->total_offset_ns += gap;
            }
        }
        // Capture original buffer timestamp
        long long buffer_ts_ns = buffer_ts;

        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        long long cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
        as->frames.push_back({cpu_ts, buffer_ts_ns, 0, 0, 0, 0, 0});
        as->cpu_ts_at_reception_count++;
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    (void)pad;
    VideoStream *s = (VideoStream *)user_data;
    if (s->is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
        if (!gst_buffer_is_writable(buf)) {
             buf = gst_buffer_make_writable(buf);
             GST_PAD_PROBE_INFO_DATA(info) = buf;
        }
        long long buffer_ts = (long long)GST_BUFFER_PTS(buf);
        if (!GST_CLOCK_TIME_IS_VALID(buffer_ts)) return GST_PAD_PROBE_OK;
        // Debug: Log the jump if it is significant
        if (s->last_raw_buffer_ts != -1) {
            long long delta = buffer_ts - s->last_raw_buffer_ts;
            if (delta > 500 * 1000000LL) { // 500ms threshold for offset correction
                 long long expected_gap = s->last_duration;
                 if (!GST_CLOCK_TIME_IS_VALID(expected_gap) || expected_gap == 0) expected_gap = 33333333LL;
                 long long gap = delta - expected_gap;
                 if (gap > 0) s->total_offset_ns += gap;
            }
        }
        // Capture original buffer timestamp
        long long buffer_ts_ns = buffer_ts;
        const DcFrameTimestamps frame_timestamps =
            dc_buffer_get_frame_timestamps(buf);
        const bool has_remote_timestamps = dc_buffer_has_frame_timestamps(buf);
        long long cpu_ts;
        if (frame_timestamps.overlay_output_ts != 0) {
            cpu_ts = static_cast<long long>(frame_timestamps.overlay_output_ts);
        } else if (frame_timestamps.stereo_output_ts != 0) {
            cpu_ts = static_cast<long long>(frame_timestamps.stereo_output_ts);
        } else if (frame_timestamps.mono_source_ts != 0) {
            cpu_ts = static_cast<long long>(frame_timestamps.mono_source_ts);
        } else if (frame_timestamps.left_source_ts != 0 &&
                   frame_timestamps.right_source_ts != 0) {
            cpu_ts = static_cast<long long>(
                frame_timestamps.left_source_ts / 2 +
                frame_timestamps.right_source_ts / 2 +
                (frame_timestamps.left_source_ts % 2 +
                 frame_timestamps.right_source_ts % 2) /
                    2);
        } else if (frame_timestamps.left_source_ts != 0) {
            cpu_ts = static_cast<long long>(frame_timestamps.left_source_ts);
        } else if (frame_timestamps.right_source_ts != 0) {
            cpu_ts = static_cast<long long>(frame_timestamps.right_source_ts);
        } else {
            // Fallback: capture locally (for non-unixfd sources)
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            cpu_ts = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
        }

        if (has_remote_timestamps) {
            s->cpu_ts_from_unixfd_count++;
        } else {
            s->cpu_ts_at_reception_count++;
        }

        s->frames.push_back({cpu_ts, buffer_ts_ns,
                             static_cast<long long>(frame_timestamps.mono_source_ts),
                             static_cast<long long>(frame_timestamps.left_source_ts),
                             static_cast<long long>(frame_timestamps.right_source_ts),
                             static_cast<long long>(frame_timestamps.stereo_output_ts),
                             static_cast<long long>(frame_timestamps.overlay_output_ts)});
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
        bool is_pipeline = (ad->audio_stream && ad->audio_stream->pipeline && src == GST_OBJECT(ad->audio_stream->pipeline));
        if (!is_pipeline) {
            for (auto s : ad->streams) if (s->pipeline && src == GST_OBJECT(s->pipeline)) { is_pipeline = true; break; }
        }
        if (is_pipeline) {
            int target = ad->streams.size() + ((ad->audio_stream && ad->audio_stream->pipeline) ? 1 : 0);
            ad->eos_received_count++;
            // Ignore runtime EOS from failed/disconnected streams.
            // Only quit the GTK loop when we are explicitly shutting down.
            if (ad->is_quitting && ad->eos_received_count >= target) gtk_main_quit();
        }
    } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ELEMENT) {
        const GstStructure *s = gst_message_get_structure(msg);
        if (gst_structure_has_name(s, "level")) {
            const GValue *rms_val = gst_structure_get_value(s, "rms");
            const GValue *peak_val = gst_structure_get_value(s, "peak");
            double max_rms = get_audio_level_max(rms_val);
            double max_peak = get_audio_level_max(peak_val);
            double display_val = std::max(max_rms, max_peak);
            double lvl = (display_val + 100.0) / 100.0;
            if (lvl < 0) lvl = 0;
            if (lvl > 1) lvl = 1;
            if (ad->audio_level_bar && GTK_IS_LEVEL_BAR(ad->audio_level_bar)) {
                gtk_level_bar_set_value(GTK_LEVEL_BAR(ad->audio_level_bar), lvl);
            }
        }
    } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        gchar *dbg; GError *err; gst_message_parse_error(msg, &err, &dbg);
        const gchar *msg_src_name = GST_MESSAGE_SRC_NAME(msg);
        std::cerr << "Error from " << (msg_src_name ? msg_src_name : "unknown")
                  << ": " << err->message << std::endl;

        // Identify which stream failed
        GstObject *src = GST_MESSAGE_SRC(msg);
        VideoStream *failed_vs = nullptr;
        
        // Loop through video streams
        for (auto s : ad->streams) {
            if (s->pipeline && (src == GST_OBJECT(s->pipeline) || gst_object_has_as_ancestor(src, GST_OBJECT(s->pipeline)))) {
                failed_vs = s;
                break;
            }
        }

        if (failed_vs) {
            std::cerr << "Setting has_error for stream: " << failed_vs->name << std::endl;
            {
                std::lock_guard<std::mutex> lock(ad->data_mutex);
                failed_vs->has_error = true;
                failed_vs->error_message = err->message;
            }

            // Stop recording if active
            if (ad->global_recording && ad->window) {
                // We are in the GStreamer bus thread (often the GLib main loop thread)
                // Use a dispatcher or just call via Glib::signal_idle for safe UI interaction
                Glib::signal_idle().connect_once([ad]() {
                    if (ad->window) {
                        MainWindow *mw = static_cast<MainWindow*>(Glib::wrap(ad->window));
                        mw->trigger_stop_recording();
                    }
                });
            }
        }

        g_free(dbg); g_error_free(err);
    }
    return TRUE;
}

void shutdown_pipeline(GstElement* pipeline) {
    if (!pipeline) return;

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));

    // Send EOS to the pipeline - muxers need EOS to finalize the file
    gst_element_send_event(pipeline, gst_event_new_eos());

    // Wait for EOS message on the bus (indicates muxer finalized the file)
    if (bus) {
        GstClockTime deadline = gst_util_get_timestamp() + 10 * GST_SECOND;
        while (gst_util_get_timestamp() < deadline) {
            GstMessage* msg = gst_bus_timed_pop_filtered(bus, 50 * GST_MSECOND,
                static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
            if (msg) {
                GstMessageType type = GST_MESSAGE_TYPE(msg);
                gst_message_unref(msg);
                if (type == GST_MESSAGE_EOS || type == GST_MESSAGE_ERROR) break;
            }
            // Pump GTK main loop so gtksink can process widget events
            while (g_main_context_iteration(NULL, FALSE)) {}
        }
    }

    // Now set state to NULL
    gst_element_set_state(pipeline, GST_STATE_NULL);

    // Wait for NULL state while pumping GTK events
    GstClockTime deadline = gst_util_get_timestamp() + 5 * GST_SECOND;
    while (gst_util_get_timestamp() < deadline) {
        GstState state;
        GstStateChangeReturn ret = gst_element_get_state(pipeline, &state, NULL, 50 * GST_MSECOND);
        if (ret == GST_STATE_CHANGE_SUCCESS && state == GST_STATE_NULL) break;
        if (ret == GST_STATE_CHANGE_FAILURE) break;
        while (g_main_context_iteration(NULL, FALSE)) {}
    }

    // Remove bus watch after state transitions complete
    if (bus) {
        gst_bus_remove_watch(bus);
        gst_object_unref(bus);
    }
}
