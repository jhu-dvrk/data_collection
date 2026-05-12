#include "video_stream.hpp"
#include "context.hpp"
#include "gst_helpers.hpp"
#include "ros_node.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <gst/video/video.h>
#include <pwd.h>
#include <unistd.h>

extern int app_max_threads;

namespace {
std::string base_name(const std::string &path) {
    const std::string::size_type pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}
}

VideoStream::VideoStream() : pipeline(NULL), valve(NULL), rec_overlay(NULL), preview_widget(NULL), record_checkbox(NULL), retry_button(NULL), stats_label(NULL), stream_name_label(NULL), preview_stack(NULL),
                is_recording(false), record_enabled(true), frames_recorded(0), frames_dropped(0), last_run_frames_recorded(0), last_run_stage_name(""), current_fps(0.0), estimated_latency(0.0),
                cpu_ts_from_unixfd_count(0), cpu_ts_at_reception_count(0),
                width(0), height(0), src_fps(0.0), last_src_ts(0), src_frame_counter(0),
                rec_width(0), rec_height(0), rec_fps_requested(0.0),
                total_offset_ns(0), last_raw_buffer_ts(-1), last_duration(0),
                last_fps_ts(0), fps_frame_counter(0),
                has_error(false), auto_retry_attempted(false), auto_retry_in_progress(false), m_ad(NULL), m_has_config(false), m_error_segment_index(0) {}

VideoStream::~VideoStream() {
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

void VideoStream::shutdown() {
    if (!pipeline) return;

    // Open valve to drain any remaining frames to the encoder/mux
    if (valve) g_object_set(valve, "drop", FALSE, NULL);

    shutdown_pipeline(pipeline);

    if (pipeline) {
        gst_object_unref(pipeline);
        pipeline = NULL;
    }
    if (valve) {
        gst_object_unref(valve);
        valve = NULL;
    }
}

bool VideoStream::create(AppData* ad, const dc::VideoConfig* v) {
    this->m_ad = ad;
    const bool first_create = !this->m_has_config;
    if (v) {
        this->m_config = *v;
        this->m_has_config = true;
    }
    this->preview_widget = NULL;
    this->name = v->name;
    this->record_enabled = v->record;
    this->side_by_side = v->side_by_side;
    this->estimated_latency = v->estimated_latency;
    this->has_error = false;
    this->auto_retry_in_progress = false;
    this->error_message.clear();

    this->rec_width = v->encoding.width;
    this->rec_height = v->encoding.height;
    this->rec_fps_requested = (double)v->encoding.frame_rate;

    if (first_create) {
        std::string sn = v->name;
        for (char &c : sn) if (c == ' ') c = '_';
        this->m_output_stem = ad->session_dir + "/" + sn;
        this->m_error_segment_index = 0;
        this->segments.clear();
    }

    this->update_output_paths();

    // Recording and encoding caps
    // Use an extra videoconvert to ensure we can reach the desired format/size
    std::string rec_pipeline_segment = " ! videoconvert n-threads=" + std::to_string(app_max_threads);
    std::string rec_caps = "video/x-raw,format=NV12";
    
    bool needs_scale = v->encoding.width > 0 && v->encoding.height > 0;
    bool needs_rate = v->encoding.frame_rate > 0;

    if (needs_scale) {
        rec_pipeline_segment += " ! videoscale";
        rec_caps += ",width=" + std::to_string(v->encoding.width) + ",height=" + std::to_string(v->encoding.height);
    }
    if (needs_rate) {
        rec_pipeline_segment += " ! videorate skip-to-first=true";
        rec_caps += ",framerate=" + std::to_string(v->encoding.frame_rate) + "/1";
    }

    std::string ts_overlay = "";
    if (v->timestamp_overlay) {
        ts_overlay = " ! timeoverlay valignment=bottom halignment=left font-desc=\"Sans, 10\" shaded-background=true shading-value=255 xpad=0 ypad=0 "
                     " ! clockoverlay valignment=bottom halignment=right time-format=\"%Y-%m-%d %H:%M:%S\" font-desc=\"Sans, 10\" shaded-background=true shading-value=255 xpad=0 ypad=0 ";
    }

    std::string source_stream = v->stream;

    if (v->has_unixfd_socket_path) {
        std::string socket_path = v->unixfd_socket_path;
        if (socket_path.empty()) {
            // Generate default socket path using username and stream name
            const char *username = getenv("USER");
            if (!username) {
                struct passwd *pw = getpwuid(getuid());
                username = pw ? pw->pw_name : "unknown";
            }
            socket_path = "/tmp/" + v->name + "_" + std::string(username) + ".sock";
        }
        source_stream =
            "unixfdsrc socket-path=" + socket_path +
            " do-timestamp=true ! videoconvert ! video/x-raw,format=NV12 ! "
            "queue name=__remote_offset_q__ max-size-buffers=2 "
            "max-size-time=0 max-size-bytes=0 leaky=downstream";
    }

    if (source_stream.empty()) {
        std::cerr << "Error for stream \"" << v->name << "\": stream description is empty" << std::endl;
        return false;
    }

    std::string enc = get_best_encoder(v->encoding);
    std::cout << "Using encoder for stream \"" << v->name << "\": " << enc << std::endl;

    std::string pstr = source_stream + ts_overlay + " ! tee name=__rec_t__ "
        "__rec_t__. ! queue name=__prev_q__ max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! videoconvert n-threads=" + std::to_string(app_max_threads) + " ! cairooverlay name=__prev_overlay__ ! gtksink name=__prev_sink__ sync=false async=false ";

    if (this->record_enabled) {
        pstr += "__rec_t__. ! valve name=__rec_v__ drop=true ! queue name=__rec_q_rec__ max-size-buffers=10 max-size-time=0 max-size-bytes=0 leaky=downstream" + rec_pipeline_segment + " ! " + rec_caps + " ! " + enc + " ! queue name=__rec_q_enc__ max-size-buffers=30 max-size-time=0 max-size-bytes=0 ! h264parse ! mp4mux name=__rec_mux__ ! filesink name=__rec_filesink__ sync=false async=false";
    }

    // Warning for two sources but side_by_side setting missing
    if (!v->has_unixfd_socket_path && this->side_by_side == "undefined") {
        size_t pos = 0;
        int sources = 0;
        while ((pos = source_stream.find("src", pos)) != std::string::npos) {
            sources++;
            pos += 3;
        }
        if (sources >= 2) {
            std::cerr << "\033[1;33mWarning for stream \"" << v->name 
                      << "\": Multiple sources detected but \"side_by_side\" setting is not present in the configuration.\033[0m" << std::endl;
        }
    }

    this->pipeline_desc = pstr;
    this->pipeline = gst_parse_launch(pstr.c_str(), NULL);
    if (!this->pipeline) return false;

    std::string sn = v->name;
    for (char &c : sn) if (c == ' ') c = '_';

    if (ad->dump_dot) {
        dc::dump_dot(this->pipeline, ad->session_dir + "/" + sn + "_pipeline.dot", ad->dot_flags);
    }

    // Source probe on tee sink pad (before split to preview / record)
    GstElement *tee = gst_bin_get_by_name(GST_BIN(this->pipeline), "__rec_t__");
    if (tee) {
        GstPad *tpad = gst_element_get_static_pad(tee, "sink");
        if (tpad) {
            gst_pad_add_probe(tpad, GST_PAD_PROBE_TYPE_BUFFER, source_probe_cb, this, NULL);
            gst_object_unref(tpad);
        }
        gst_object_unref(tee);
    }



    // Record queue overrun — counts dropped frames
    GstElement *qrec = gst_bin_get_by_name(GST_BIN(this->pipeline), "__rec_q_rec__");
    if (qrec) {
        g_signal_connect(qrec, "overrun", G_CALLBACK(+[](GstElement* q, gpointer d){
            (void)q;
            VideoStream* vs = (VideoStream*)d; vs->frames_dropped++;
        }), this);
        gst_object_unref(qrec);
    }

    this->valve = gst_bin_get_by_name(GST_BIN(this->pipeline), "__rec_v__");
    if (this->valve) {
        GstPad *vpad = gst_element_get_static_pad(this->valve, "src");
        if (vpad) {
            gst_pad_add_probe(vpad, GST_PAD_PROBE_TYPE_BUFFER, timestamp_probe_cb, this, NULL);
            gst_object_unref(vpad);
        }
    }

    this->rec_overlay = gst_bin_get_by_name(GST_BIN(this->pipeline), "__prev_overlay__");
    if (this->rec_overlay) {
        g_signal_connect(this->rec_overlay, "draw", G_CALLBACK(on_rec_overlay_draw), this);
        gst_object_unref(this->rec_overlay);
    }

    GstElement *sink = gst_bin_get_by_name(GST_BIN(this->pipeline), "__prev_sink__");
    if (sink) {
        g_object_get(sink, "widget", &this->preview_widget, NULL);
        // We do not set a fixed size request here to allow the widget to shrink.
        // The container in the UI will handle size through packing.
        gst_object_unref(sink);
    }

    // Set output path for current segment
    GstElement *fsink = gst_bin_get_by_name(GST_BIN(this->pipeline), "__rec_filesink__");
    if (fsink) {
        g_object_set(fsink, "location", this->output_video.c_str(), NULL);
        gst_object_unref(fsink);
    }

    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(this->pipeline));
    gst_bus_add_watch(bus, bus_call, ad);
    gst_object_unref(bus);

    // Do NOT start the pipeline here — the gtksink widget must be embedded in
    // the application window before the pipeline goes PLAYING, otherwise gtksink
    // creates its own top-level GtkWindow for the widget which causes a
    // reparent warning later.  The caller (main.cpp) starts each pipeline
    // after MainWindow is fully constructed.
    return true;
}

bool VideoStream::restart() {
    if (!m_has_config || !m_ad) {
        return false;
    }

    const bool split_segment = this->has_error;
    if (split_segment) {
        this->stop_and_save(this->m_ad->config_files);
        this->reset_segment_counters();
        this->m_error_segment_index++;
    }

    GtkWidget *old_preview_widget = this->preview_widget;
    GtkWidget *stack_widget = this->preview_stack;

    if (stack_widget && old_preview_widget &&
        gtk_widget_get_parent(old_preview_widget) != NULL) {
        gtk_container_remove(GTK_CONTAINER(stack_widget), old_preview_widget);
    }

    this->shutdown();

    if (!this->create(m_ad, &m_config)) {
        this->has_error = true;
        this->auto_retry_in_progress = false;
        this->error_message = "Failed to recreate pipeline";
        return false;
    }

    if (stack_widget && old_preview_widget && this->preview_widget &&
        old_preview_widget != this->preview_widget) {
        if (gtk_widget_get_parent(old_preview_widget) != NULL) {
            gtk_container_remove(GTK_CONTAINER(stack_widget), old_preview_widget);
        }
        gtk_stack_add_named(GTK_STACK(stack_widget), this->preview_widget, "preview");
        gtk_widget_show_all(this->preview_widget);
        gtk_stack_set_visible_child_name(GTK_STACK(stack_widget), "preview");
    }

    if (this->m_ad->global_recording && this->record_checkbox) {
        bool stream_rec = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(this->record_checkbox));
        this->set_recording(stream_rec);
    }

    if (this->pipeline) {
        gst_element_set_state(this->pipeline, GST_STATE_PLAYING);
    }

    this->auto_retry_in_progress = false;

    return true;
}

void VideoStream::set_recording(bool active) {
    this->is_recording = active;
    if (this->valve) g_object_set(this->valve, "drop", !active, NULL);
}

void VideoStream::stop_and_save(const std::vector<std::string>& config_files) {
    this->last_run_frames_recorded = this->frames_recorded;
    if (m_ad && m_ad->current_stage_idx >= 0 && m_ad->current_stage_idx < (int)m_ad->config_stages.size()) {
        this->last_run_stage_name = m_ad->config_stages[m_ad->current_stage_idx];
    }
    this->set_recording(false);


    if (this->record_enabled && !this->output_json.empty() && !this->frames.empty()) {
        Json::Value root;
        root["type"] = "dc::sidecar@1.0.0";
        root["stream_name"] = this->name;
        root["start_timestamp_ns"] = this->frames.empty() ? 0 : (Json::Value::Int64)this->frames.front().cpu_ts;
        root["end_timestamp_ns"] = this->frames.empty() ? 0 : (Json::Value::Int64)this->frames.back().cpu_ts;

        Json::Value cpuTsObj(Json::objectValue);
        cpuTsObj["from_unixfd"] = (Json::Value::Int64)this->cpu_ts_from_unixfd_count;
        cpuTsObj["at_reception"] = (Json::Value::Int64)this->cpu_ts_at_reception_count;
        root["cpu_ts"] = cpuTsObj;
        root["frames_recorded"] = (int)this->frames_recorded;
        root["frames_dropped"] = (int)this->frames_dropped;
        root["side_by_side"] = this->side_by_side;
        root["estimated_latency_ms"] = this->estimated_latency;

        Json::Value frame_list = Json::arrayValue;
        for (size_t i = 0; i < this->frames.size(); ++i) {
            auto &f = this->frames[i];
            Json::Value fv;
            fv["cpu_ns"] = (Json::Value::Int64)f.cpu_ts;
            fv["gst_ns"] = (Json::Value::Int64)f.buffer_ts_ns;
            fv["mono_source_ns"] = (Json::Value::Int64)f.mono_source_ts;
            fv["left_source_ns"] = (Json::Value::Int64)f.left_source_ts;
            fv["right_source_ns"] = (Json::Value::Int64)f.right_source_ts;
            fv["stereo_output_ns"] = (Json::Value::Int64)f.stereo_output_ts;
            fv["overlay_output_ns"] = (Json::Value::Int64)f.overlay_output_ts;
            frame_list.append(fv);
        }
        root["frames"] = frame_list;

        Json::Value configFiles(Json::arrayValue);
        for (const auto& cf : config_files) {
            configFiles.append(cf);
        }
        root["config_files"] = configFiles;

        std::ofstream ofs(this->output_json);
        ofs << root.toStyledString();

        this->update_segment_summary();
    }
}

void VideoStream::update_output_paths() {
    std::ostringstream suffix;
    if (this->m_error_segment_index > 0) {
        suffix << "_" << std::setw(3) << std::setfill('0') << this->m_error_segment_index;
    }

    this->output_video = this->m_output_stem + suffix.str() + ".mp4";
    this->output_json = this->m_output_stem + suffix.str() + ".json";
}

void VideoStream::reset_segment_counters() {
    this->frames.clear();
    this->frames_recorded = 0;
    this->frames_dropped = 0;
    this->current_fps = 0.0;
    this->cpu_ts_from_unixfd_count = 0;
    this->cpu_ts_at_reception_count = 0;
    this->last_raw_buffer_ts = -1;
    this->last_duration = 0;
    this->total_offset_ns = 0;
    this->last_fps_ts = 0;
    this->fps_frame_counter = 0;
}

double VideoStream::compute_segment_duration_seconds() const {
    if (this->frames.size() < 2) {
        return 0.0;
    }

    long long accum_ns = 0;
    for (size_t i = 0; i + 1 < this->frames.size(); ++i) {
        const long long diff = this->frames[i + 1].cpu_ts - this->frames[i].cpu_ts;
        if (diff > 0 && diff < 500 * 1000000LL) {
            accum_ns += diff;
        }
    }

    if (accum_ns > 0) {
        return static_cast<double>(accum_ns) / 1e9;
    }

    if (this->src_fps > 0.1) {
        return static_cast<double>(this->frames.size()) / this->src_fps;
    }

    return 0.0;
}

void VideoStream::update_segment_summary() {
    const std::string video_name = base_name(this->output_video);
    const std::string sidecar_name = base_name(this->output_json);

    SegmentSummary summary;
    summary.file = video_name;
    summary.sidecar = sidecar_name;
    summary.duration = this->compute_segment_duration_seconds();
    summary.frames = static_cast<long long>(this->frames.size());

    for (auto &existing : this->segments) {
        if (existing.file == summary.file) {
            existing = summary;
            return;
        }
    }
    this->segments.push_back(summary);
}
