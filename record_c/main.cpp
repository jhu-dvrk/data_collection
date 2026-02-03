#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <thread>
#include <mutex>

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <json/json.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/joy.hpp>

// ---- Layout Constants ----
#define WINDOW_BORDER_PX 8
#define BOX_SPACING_PX 8
#define FRAME_PADDING_PX 4
#define WIDGET_SPACING_PX 4
#define WIDGET_MARGIN_PX 4
// -------------------------

static int app_max_threads = 1;

struct VideoStage {
    std::string stage;
    std::vector<long long> timestamps;
};

struct VideoStream {
    std::string name;
    GstElement *pipeline;
    GstElement *valve;
    GstElement *rec_text;
    GtkWidget *preview_widget;
    GtkWidget *record_checkbox;
    GtkWidget *stats_label;
    std::string output_video, output_json;
    std::vector<VideoStage> stages;
    bool is_recording;
    bool record_enabled;
    long long frames_recorded;
    long long frames_dropped;
    long long last_run_frames_recorded;
    std::string last_run_stage_name;
    double current_fps;
    
    // Source Stats
    int width, height;
    double src_fps;
    long long last_src_ts;
    long long src_frame_counter;

    // FPS Calculation State
    long long last_fps_ts;
    long long fps_frame_counter;

    VideoStream() : pipeline(NULL), valve(NULL), rec_text(NULL), preview_widget(NULL), record_checkbox(NULL), stats_label(NULL), 
                    is_recording(false), record_enabled(true), frames_recorded(0), frames_dropped(0), last_run_frames_recorded(0), last_run_stage_name(""), current_fps(0.0),
                    width(0), height(0), src_fps(0.0), last_src_ts(0), src_frame_counter(0),
                    last_fps_ts(0), fps_frame_counter(0) {}
};

struct AppData {
    std::vector<VideoStream*> streams;
    GstElement *audio_pipeline, *audio_sink, *audio_valve, *audio_src;
    GtkWidget *window, *record_button, *data_dir_entry, *audio_level_bar, *audio_enable_checkbox;
    GtkWidget *audio_src_combo, *stages_combo, *session_entry, *grid;
    std::string data_directory, session_dir, start_timestamp, audio_output_json;
    std::vector<VideoStage> audio_stages;
    std::vector<std::string> config_stages;
    std::vector<GtkWidget*> stage_labels;
    int session_stage_cycle_count;
    int current_stage_idx; // Track the current stage index for UI highlighting
    bool global_recording, blink_state, session_initialized, audio_is_recording, is_quitting;
    int eos_received_count;

    // ROS 2 members
    std::string trigger_topic;
    rclcpp::Node::SharedPtr node;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_record;
    std::mutex data_mutex;
    
    AppData() : audio_pipeline(NULL), audio_sink(NULL), audio_valve(NULL), audio_src(NULL),
                window(NULL), record_button(NULL), data_dir_entry(NULL), 
                audio_level_bar(NULL), audio_enable_checkbox(NULL),
                audio_src_combo(NULL), stages_combo(NULL), session_entry(NULL), grid(NULL),
                data_directory("."), session_stage_cycle_count(1), current_stage_idx(0), global_recording(false), 
                blink_state(false), session_initialized(false), audio_is_recording(false), 
                is_quitting(false), eos_received_count(0) {}
};

static std::string get_best_encoder(const Json::Value& enc_cfg) {
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

static double get_audio_level_max(const GValue* gv) {
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

static GstPadProbeReturn source_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
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

static GstPadProbeReturn audio_timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    AppData *ad = (AppData *)user_data;
    if (ad->audio_is_recording && (info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        ad->audio_stages.back().timestamps.push_back(g_get_real_time() * 1000);
    }
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn timestamp_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
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

static void update_stage_highlighting(AppData *data) {
    for (size_t i = 0; i < data->stage_labels.size(); ++i) {
        if ((int)i == data->current_stage_idx) {
            gchar* markup = g_markup_printf_escaped("<span foreground=\"#0055ff\"><b>%s</b></span>", data->config_stages[i].c_str());
            gtk_label_set_markup(GTK_LABEL(data->stage_labels[i]), markup);
            g_free(markup);
        } else {
            gtk_label_set_text(GTK_LABEL(data->stage_labels[i]), data->config_stages[i].c_str());
        }
    }
}

static void toggle_recording(AppData *data) {
    if (data->is_quitting) return; // Ignore calls during shutdown
    
    data->global_recording = !data->global_recording;

    int active_idx = data->current_stage_idx;
    std::string base_stage = data->config_stages.empty() ? "stage" : data->config_stages[active_idx];

    if (data->global_recording) {
        char stage_buf[128];
        snprintf(stage_buf, sizeof(stage_buf), "%s_%03d", base_stage.c_str(), data->session_stage_cycle_count);
        std::string final_stage_name = stage_buf;

        if (data->audio_valve) {
            bool audio_rec = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(data->audio_enable_checkbox));
            data->audio_is_recording = audio_rec;
            g_object_set(data->audio_valve, "drop", !audio_rec, NULL);
            if (audio_rec) data->audio_stages.push_back({final_stage_name, {}});
        }
        for (auto s : data->streams) {
            bool stream_rec = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(s->record_checkbox));
            s->is_recording = stream_rec;
            if (s->valve) g_object_set(s->valve, "drop", !stream_rec, NULL);
            if (stream_rec) s->stages.push_back({final_stage_name, {}});
        }
    } else {
        if (data->audio_valve) {
            data->audio_is_recording = false;
            g_object_set(data->audio_valve, "drop", TRUE, NULL);
        }
        for (auto s : data->streams) {
            if (!s->stages.empty()) {
                s->last_run_stage_name = s->stages.back().stage;
                s->last_run_frames_recorded = s->stages.back().timestamps.size();
            } else {
                s->last_run_stage_name = "";
                s->last_run_frames_recorded = 0;
            }
            s->is_recording = false;
            if (s->valve) g_object_set(s->valve, "drop", TRUE, NULL);
            if (s->rec_text) g_object_set(s->rec_text, "text", "", NULL);
            s->frames_recorded = 0;
            s->frames_dropped = 0;
            s->current_fps = 0;
            s->last_fps_ts = 0;
            s->fps_frame_counter = 0;
        }

        int num_stages = (int)data->config_stages.size();
        if (num_stages > 0) {
            int next_idx = active_idx + 1;
            if (next_idx >= num_stages) {
                next_idx = 0;
                data->session_stage_cycle_count++;
            }
            data->current_stage_idx = next_idx;
        }
    }

    gtk_button_set_label(GTK_BUTTON(data->record_button), data->global_recording ? "Stop Recording" : "Start Recording");
    gtk_widget_set_sensitive(data->audio_enable_checkbox, !data->global_recording);
    gtk_widget_set_sensitive(data->audio_src_combo, !data->global_recording);
    for (auto s : data->streams) gtk_widget_set_sensitive(s->record_checkbox, !data->global_recording);
    update_stage_highlighting(data);
    // Redraw the window to update stage highlighting
    gtk_widget_queue_draw(data->window);
}

// GUI-safe toggle triggered from ROS thread
static gboolean toggle_recording_idle(gpointer user_data) {
    AppData* ad = (AppData*)user_data;
    toggle_recording(ad);
    return G_SOURCE_REMOVE;
}

static void ros_record_callback(const std_msgs::msg::Bool::SharedPtr msg, AppData* ad) {
    bool should_record = msg->data;
    std::lock_guard<std::mutex> lock(ad->data_mutex);
    if (should_record != ad->global_recording) {
        g_idle_add(toggle_recording_idle, ad);
    }
}

static void populate_audio_sources(GtkComboBoxText* combo) {
    gtk_combo_box_text_remove_all(combo);
    GstDeviceMonitor* mon = gst_device_monitor_new();
    gst_device_monitor_add_filter(mon, "Audio/Source", NULL);
    GList* devices = gst_device_monitor_get_devices(mon);
    int idx = 0, default_idx = 0;
    for (GList* l = devices; l; l = l->next) {
        GstDevice* d = (GstDevice*)l->data;
        gchar* name = gst_device_get_display_name(d);
        GstStructure* p = gst_device_get_properties(d);
        const char* pulse_name = gst_structure_get_string(p, "device.name");
        if (pulse_name) {
            gtk_combo_box_text_append(combo, pulse_name, name);
            if (!g_str_has_suffix(pulse_name, ".monitor") && strstr(name, "Default")) {
                default_idx = idx;
            }
            idx++;
        }
        g_free(name); if (p) gst_structure_free(p);
    }
    g_list_free_full(devices, (GDestroyNotify)gst_object_unref); gst_object_unref(mon);
    gtk_combo_box_set_active(GTK_COMBO_BOX(combo), default_idx);
}

static void create_audio_pipeline(AppData* data);

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer user_data) {
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

static void on_window_destroy_cb(GtkWidget *w, gpointer d) {
    AppData *ad = (AppData *)d;
    ad->is_quitting = true;
    if (ad->global_recording) toggle_recording(ad);
    
    // Unblock valves to let EOS pass through
    if (ad->audio_valve) g_object_set(ad->audio_valve, "drop", FALSE, NULL);
    for (auto s : ad->streams) if (s->valve) g_object_set(s->valve, "drop", FALSE, NULL);

    if (!ad->audio_stages.empty()) {
        Json::Value root; root["name"] = "audio";
        root["audio_file"] = "audio_" + ad->start_timestamp + ".wav";
        Json::Value stagesArr(Json::arrayValue);
        for (const auto& seg : ad->audio_stages) {
            Json::Value segNode; segNode["stage"] = seg.stage;
            Json::Value tsArr(Json::arrayValue);
            for (long long t : seg.timestamps) tsArr.append((Json::Value::Int64)t);
            segNode["timestamps_ns"] = tsArr;
            stagesArr.append(segNode);
        }
        root["stages"] = stagesArr;
        std::ofstream os(ad->audio_output_json); Json::StreamWriterBuilder b;
        std::unique_ptr<Json::StreamWriter>(b.newStreamWriter())->write(root, &os);
    }

    for (auto s : ad->streams) {
        if (!s->stages.empty()) {
            Json::Value root; root["name"] = s->name;
            root["video_file"] = s->output_video.substr(s->output_video.find_last_of("/\\\\") + 1);
            Json::Value stagesArr(Json::arrayValue);
            for (const auto& seg : s->stages) {
                Json::Value segNode; segNode["stage"] = seg.stage;
                Json::Value tsArr(Json::arrayValue);
                for (long long t : seg.timestamps) tsArr.append((Json::Value::Int64)t);
                segNode["timestamps_ns"] = tsArr;
                stagesArr.append(segNode);
            }
            root["stages"] = stagesArr;
            std::ofstream os(s->output_json); Json::StreamWriterBuilder b;
            std::unique_ptr<Json::StreamWriter>(b.newStreamWriter())->write(root, &os);
        }
        gst_element_send_event(s->pipeline, gst_event_new_eos());
    }
    if (ad->audio_pipeline) gst_element_send_event(ad->audio_pipeline, gst_event_new_eos());
    
    gtk_widget_hide(w);

    // Safety timeout: if EOS doesn't arrive in 3 seconds, force quit
    g_timeout_add(3000, (GSourceFunc)+[](gpointer d) -> gboolean {
        g_print("Shutdown timeout reached. Force exiting.\n");
        gtk_main_quit();
        return G_SOURCE_REMOVE;
    }, NULL);

    if (ad->streams.empty() && !ad->audio_pipeline) gtk_main_quit();
    
    // Shutdown ROS 2
    rclcpp::shutdown();
}

static void create_audio_pipeline(AppData* data) {
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

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv); gst_init(&argc, &argv);
    AppData data; std::vector<std::string> configs;

    unsigned int cores = std::thread::hardware_concurrency();
    app_max_threads = (cores > 0 ? cores : 2) / 2;
    if (app_max_threads < 1) app_max_threads = 1;

    time_t now_raw; char buf[80]; time(&now_raw); strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now_raw));
    data.start_timestamp = buf;

    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" && i+1 < argc) {
            configs.push_back(argv[++i]);
        } else if (arg == "-j" && i+1 < argc) {
            app_max_threads = std::stoi(argv[++i]);
            if (app_max_threads < 1) app_max_threads = 1;
        } else if (arg == "-p" && i+1 < argc) {
            data.trigger_topic = argv[++i];
        } else if (arg[0] != '-') {
            configs.push_back(argv[i]);
        }
    }
    if (configs.empty()) return 1;

    // ROS 2 Initialization
    rclcpp::init(argc, argv);
    data.node = std::make_shared<rclcpp::Node>("record_c");
    if (!data.trigger_topic.empty()) {
        data.sub_record = data.node->create_subscription<std_msgs::msg::Bool>(
            data.trigger_topic, 10, 
            [&](const std_msgs::msg::Bool::SharedPtr msg) { ros_record_callback(msg, &data); }
        );
        std::cout << "Subscribed to trigger topic: " << data.trigger_topic << std::endl;
    }

    // Spin ROS in a separate thread
    std::thread ros_thread([&]() {
        rclcpp::spin(data.node);
    });
    ros_thread.detach();

    for (const auto& path : configs) {
        std::ifstream f(path); Json::Value root; if (!f.is_open() || !Json::parseFromStream(Json::CharReaderBuilder(), f, &root, NULL)) continue;
        data.data_directory = root.get("data_directory", data.data_directory).asString();
        if (root.isMember("stages")) for (const auto& s : root["stages"]) data.config_stages.push_back(s.asString());
    }
    if (data.config_stages.empty()) data.config_stages.push_back("stage");

    data.session_dir = data.data_directory + "/" + data.start_timestamp;
    g_mkdir_with_parents(data.session_dir.c_str(), 0777);

    for (const auto& path : configs) {
        std::ifstream f(path); Json::Value root; if (!f.is_open() || !Json::parseFromStream(Json::CharReaderBuilder(), f, &root, NULL)) continue;
        for (const auto& v : root["videos"]) {
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
            if (!s->pipeline) { delete s; continue; }

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
            s->output_video = data.session_dir + "/" + sn + "_" + data.start_timestamp + ".mp4";
            s->output_json = data.session_dir + "/" + sn + "_" + data.start_timestamp + ".json";
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
            gst_bus_add_watch(bus, bus_call, &data); gst_object_unref(bus);
            data.streams.push_back(s);
        }
    }

    data.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(data.window), 1024, 768);
    g_signal_connect(data.window, "destroy", G_CALLBACK(on_window_destroy_cb), &data);

    // Main vertical split: f1 (top), f2 (bottom)
    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, BOX_SPACING_PX);
    gtk_container_add(GTK_CONTAINER(data.window), main_vbox);
    gtk_container_set_border_width(GTK_CONTAINER(main_vbox), WINDOW_BORDER_PX);

    // f1: top horizontal split (f1a, f1b)
    GtkWidget *f1_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(main_vbox), f1_hbox, TRUE, TRUE, 0);

    // f1a: session, audio, video previews (vertical)
    GtkWidget *f1a_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(f1_hbox), f1a_vbox, TRUE, TRUE, 0);
    // gtk_container_set_border_width(GTK_CONTAINER(f1a_vbox), FRAME_BORDER_PX); // Removing inner padding to align with f1b

    // f1b: stages (vertical)
    GtkWidget *f1b_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(f1_hbox), f1b_vbox, FALSE, FALSE, 0);

    // Stages frame in f1b
    GtkWidget *stages_frame = gtk_frame_new(NULL);
    gtk_container_set_border_width(GTK_CONTAINER(stages_frame), FRAME_PADDING_PX);
    GtkWidget *stages_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, WIDGET_SPACING_PX);
    gtk_container_add(GTK_CONTAINER(stages_frame), stages_vbox);
    GtkWidget *stages_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(stages_title), "<b>Stages</b>");
    gtk_label_set_xalign(GTK_LABEL(stages_title), 0.0);
    gtk_box_pack_start(GTK_BOX(stages_vbox), stages_title, FALSE, FALSE, 0);
    // Add a vertical list of all available stage names
    for (size_t i = 0; i < data.config_stages.size(); ++i) {
        GtkWidget* stage_label = gtk_label_new(NULL);
        data.stage_labels.push_back(stage_label);
        gtk_label_set_xalign(GTK_LABEL(stage_label), 0.0);

        GtkWidget* eb = gtk_event_box_new();
        gtk_container_add(GTK_CONTAINER(eb), stage_label);
        g_object_set_data(G_OBJECT(eb), "idx", GINT_TO_POINTER(i));
        
        g_signal_connect(eb, "button-press-event", G_CALLBACK(+[](GtkWidget* w, GdkEventButton* ev, gpointer d) -> gboolean {
            AppData* ad = (AppData*)d;
            if (ad->global_recording) return FALSE;
            int idx = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(w), "idx"));
            ad->current_stage_idx = idx;
            update_stage_highlighting(ad);
            return TRUE;
        }), &data);

        gtk_box_pack_start(GTK_BOX(stages_vbox), eb, FALSE, FALSE, 0);
    }
    update_stage_highlighting(&data);
    gtk_box_pack_start(GTK_BOX(f1b_vbox), stages_frame, TRUE, TRUE, 0);

    // f2: bottom bar for recording/quit
    GtkWidget *f2_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, BOX_SPACING_PX);
    gtk_box_pack_end(GTK_BOX(main_vbox), f2_hbox, FALSE, FALSE, 0);


    // --- Session Frame ---
    GtkWidget *session_frame = gtk_frame_new(NULL);
    gtk_container_set_border_width(GTK_CONTAINER(session_frame), FRAME_PADDING_PX);
    gtk_box_pack_start(GTK_BOX(f1a_vbox), session_frame, FALSE, FALSE, 0);
    GtkWidget *session_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, WIDGET_SPACING_PX);
    gtk_container_add(GTK_CONTAINER(session_frame), session_vbox);
    GtkWidget *session_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(session_title), "<b>Session</b>");
    gtk_label_set_xalign(GTK_LABEL(session_title), 0.0);
    gtk_box_pack_start(GTK_BOX(session_vbox), session_title, FALSE, FALSE, 0);

    // Session name entry (read-only)
    GtkWidget *session_name_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(session_vbox), session_name_hbox, FALSE, FALSE, 0);
    GtkWidget *session_name_label = gtk_label_new("Name:");
    gtk_widget_set_margin_start(session_name_label, WIDGET_MARGIN_PX);
    gtk_widget_set_margin_end(session_name_label, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(session_name_hbox), session_name_label, FALSE, FALSE, 0);
    GtkWidget *session_name_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(session_name_entry), data.session_dir.c_str());
    gtk_widget_set_sensitive(session_name_entry, FALSE);
    gtk_widget_set_margin_start(session_name_entry, WIDGET_MARGIN_PX);
    gtk_widget_set_margin_end(session_name_entry, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(session_name_hbox), session_name_entry, TRUE, TRUE, 0);

    // Directory row
    GtkWidget *dir_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(session_vbox), dir_hbox, FALSE, FALSE, 0);
    GtkWidget *dir_label = gtk_label_new("Directory:");
    gtk_widget_set_margin_start(dir_label, WIDGET_MARGIN_PX);
    gtk_widget_set_margin_end(dir_label, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(dir_hbox), dir_label, FALSE, FALSE, 0);
    data.data_dir_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(data.data_dir_entry), data.data_directory.c_str());
    gtk_widget_set_sensitive(data.data_dir_entry, FALSE);
    gtk_widget_set_margin_start(data.data_dir_entry, WIDGET_MARGIN_PX);
    gtk_widget_set_margin_end(data.data_dir_entry, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(dir_hbox), data.data_dir_entry, FALSE, FALSE, 0);
    GtkWidget *bbtn = gtk_button_new_with_label("Browse");
    gtk_widget_set_margin_start(bbtn, WIDGET_MARGIN_PX);
    gtk_widget_set_margin_end(bbtn, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(dir_hbox), bbtn, FALSE, FALSE, 0);
    g_signal_connect(bbtn, "clicked", G_CALLBACK(+[](GtkButton* b, gpointer d){
        AppData* ad = (AppData*)d;
        GtkWidget* dlg = gtk_file_chooser_dialog_new("Dir", GTK_WINDOW(ad->window), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, "_Cancel", GTK_RESPONSE_CANCEL, "_Select", GTK_RESPONSE_ACCEPT, NULL);
        if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
            char* f = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
            gtk_entry_set_text(GTK_ENTRY(ad->data_dir_entry), f);
            g_free(f);
        }
        gtk_widget_destroy(dlg);
    }), &data);

    // --- End Session Frame ---


    GtkWidget *audio_frame = gtk_frame_new(NULL);
    gtk_container_set_border_width(GTK_CONTAINER(audio_frame), FRAME_PADDING_PX);
    gtk_box_pack_start(GTK_BOX(f1a_vbox), audio_frame, FALSE, FALSE, 0);
    GtkWidget *audio_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, WIDGET_SPACING_PX);
    gtk_container_add(GTK_CONTAINER(audio_frame), audio_vbox);

    GtkWidget *audio_header_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(audio_vbox), audio_header_hbox, FALSE, FALSE, 0);

    GtkWidget *audio_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(audio_label), "<b>Audio</b>");
    gtk_label_set_xalign(GTK_LABEL(audio_label), 0.0);
    gtk_widget_set_margin_start(audio_label, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(audio_header_hbox), audio_label, TRUE, TRUE, 0);

    data.audio_enable_checkbox = gtk_check_button_new_with_label("Record");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data.audio_enable_checkbox), TRUE);
    gtk_box_pack_start(GTK_BOX(audio_header_hbox), data.audio_enable_checkbox, FALSE, FALSE, 0);

    GtkWidget *audio_ctrl_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(audio_vbox), audio_ctrl_hbox, FALSE, FALSE, 0);
    gtk_container_set_border_width(GTK_CONTAINER(audio_ctrl_hbox), WIDGET_MARGIN_PX);

    data.audio_src_combo = gtk_combo_box_text_new();
    populate_audio_sources(GTK_COMBO_BOX_TEXT(data.audio_src_combo));
    gtk_box_pack_start(GTK_BOX(audio_ctrl_hbox), data.audio_src_combo, FALSE, FALSE, 0);
    g_signal_connect(data.audio_src_combo, "changed", G_CALLBACK(+[](GtkComboBox* c, gpointer d) -> void {
        AppData* ad = (AppData*)d;
        if (ad->global_recording) return; 
        create_audio_pipeline(ad);
    }), &data);

    data.audio_level_bar = gtk_level_bar_new();
    gtk_level_bar_set_min_value(GTK_LEVEL_BAR(data.audio_level_bar), 0.0);
    gtk_level_bar_set_max_value(GTK_LEVEL_BAR(data.audio_level_bar), 1.0);
    gtk_level_bar_set_mode(GTK_LEVEL_BAR(data.audio_level_bar), GTK_LEVEL_BAR_MODE_CONTINUOUS);
    // Add typical VU colors
    gtk_level_bar_add_offset_value(GTK_LEVEL_BAR(data.audio_level_bar), GTK_LEVEL_BAR_OFFSET_LOW, 0.4);
    gtk_level_bar_add_offset_value(GTK_LEVEL_BAR(data.audio_level_bar), GTK_LEVEL_BAR_OFFSET_HIGH, 0.8);
    
    gtk_widget_set_size_request(data.audio_level_bar, -1, 15);
    gtk_box_pack_start(GTK_BOX(audio_ctrl_hbox), data.audio_level_bar, TRUE, TRUE, 0);

    data.grid = gtk_grid_new(); 
    gtk_grid_set_column_homogeneous(GTK_GRID(data.grid), TRUE); 
    gtk_grid_set_row_homogeneous(GTK_GRID(data.grid), FALSE); // Allow rows to shrink to content
    gtk_grid_set_row_spacing(GTK_GRID(data.grid), BOX_SPACING_PX); 
    gtk_grid_set_column_spacing(GTK_GRID(data.grid), BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(f1a_vbox), data.grid, TRUE, TRUE, 0);

    for (size_t i=0; i<data.streams.size(); ++i) {
        VideoStream* s = data.streams[i];
        GtkWidget* stream_frame = gtk_frame_new(NULL);
        gtk_container_set_border_width(GTK_CONTAINER(stream_frame), FRAME_PADDING_PX);
        GtkWidget* stream_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, WIDGET_SPACING_PX);
        gtk_container_add(GTK_CONTAINER(stream_frame), stream_vbox);

        GtkWidget* label_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, WIDGET_SPACING_PX);
        gtk_box_pack_start(GTK_BOX(stream_vbox), label_hbox, FALSE, FALSE, 0);

        GtkWidget* name_label = gtk_label_new(NULL);
        gchar* markup = g_markup_printf_escaped("<b>%s</b>", s->name.c_str());
        gtk_label_set_markup(GTK_LABEL(name_label), markup); g_free(markup);
        gtk_label_set_xalign(GTK_LABEL(name_label), 0.0);
        gtk_widget_set_margin_start(name_label, WIDGET_MARGIN_PX);
        gtk_widget_set_margin_end(name_label, WIDGET_MARGIN_PX);
        gtk_box_pack_start(GTK_BOX(label_hbox), name_label, TRUE, TRUE, 0);

        s->record_checkbox = gtk_check_button_new_with_label("Record");
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(s->record_checkbox), s->record_enabled);
        gtk_widget_set_margin_start(s->record_checkbox, WIDGET_MARGIN_PX);
        gtk_widget_set_margin_end(s->record_checkbox, WIDGET_MARGIN_PX);
        gtk_box_pack_start(GTK_BOX(label_hbox), s->record_checkbox, FALSE, FALSE, 0);

        GtkWidget* aspect_frame = gtk_aspect_frame_new(NULL, 0.5, 0.5, 4.0/3.0, FALSE);
        gtk_frame_set_shadow_type(GTK_FRAME(aspect_frame), GTK_SHADOW_NONE);
        gtk_widget_set_hexpand(aspect_frame, TRUE);
        gtk_widget_set_vexpand(aspect_frame, TRUE);
        gtk_container_add(GTK_CONTAINER(aspect_frame), s->preview_widget);
        gtk_box_pack_start(GTK_BOX(stream_vbox), aspect_frame, TRUE, TRUE, 0);

        s->stats_label = gtk_label_new("Ready");
        gtk_label_set_xalign(GTK_LABEL(s->stats_label), 0.0);
        gtk_widget_set_margin_start(s->stats_label, WIDGET_MARGIN_PX);
        gtk_widget_set_margin_end(s->stats_label, WIDGET_MARGIN_PX);
        gtk_box_pack_start(GTK_BOX(stream_vbox), s->stats_label, FALSE, FALSE, 2);

        gtk_grid_attach(GTK_GRID(data.grid), stream_frame, (int)(i%2), (int)(i/2), 1, 1);
    }

    // (Obsolete code removed: bottom_hbox and hbox_main)
    data.record_button = gtk_button_new_with_label("Start Recording");
    g_signal_connect(data.record_button, "clicked", G_CALLBACK(+[](GtkButton* b, gpointer d){ toggle_recording((AppData*)d); }), &data);
    gtk_box_pack_start(GTK_BOX(f2_hbox), data.record_button, TRUE, TRUE, 0);

    GtkWidget *quit_btn = gtk_button_new_with_label("Quit");
    g_signal_connect(quit_btn, "clicked", G_CALLBACK(+[](GtkButton* b, gpointer d){ gtk_window_close(GTK_WINDOW(((AppData*)d)->window)); }), &data);
    gtk_box_pack_end(GTK_BOX(f2_hbox), quit_btn, FALSE, FALSE, 0);

    g_timeout_add(500, (GSourceFunc)+[](gpointer d) -> gboolean {
        AppData* ad = (AppData*)d;
        if (ad->is_quitting) return G_SOURCE_REMOVE;
        ad->blink_state = !ad->blink_state;
        for (auto s : ad->streams) {
            if (s->rec_text) {
                if (ad->global_recording && s->is_recording && ad->blink_state) g_object_set(s->rec_text, "text", "REC", NULL);
                else g_object_set(s->rec_text, "text", "", NULL);
            }
            if (s->stats_label && GTK_IS_LABEL(s->stats_label)) {
                if (ad->global_recording && s->is_recording) {
                    char buf[128];
                    snprintf(buf, sizeof(buf), "%dx%d | Src: %.1f | Rec: %.1f | Drop: %lld", 
                             s->width, s->height, s->src_fps, s->current_fps, s->frames_dropped);
                    gtk_label_set_text(GTK_LABEL(s->stats_label), buf);
                } else if (!ad->global_recording) {
                    char buf[512];
                    if (!s->last_run_stage_name.empty()) {
                        snprintf(buf, sizeof(buf), "%dx%d | Src: %.1f | Last: %s (%lld frames)", 
                                 s->width, s->height, s->src_fps, s->last_run_stage_name.c_str(), s->last_run_frames_recorded);
                    } else {
                        snprintf(buf, sizeof(buf), "%dx%d | Src: %.1f | Ready", s->width, s->height, s->src_fps);
                    }
                    gtk_label_set_text(GTK_LABEL(s->stats_label), buf);
                }
            }
        }
        return G_SOURCE_CONTINUE;
    }, &data);

    gtk_widget_show_all(data.window);
    create_audio_pipeline(&data);
    for (auto s : data.streams) gst_element_set_state(s->pipeline, GST_STATE_PLAYING);
    gtk_main();
    return 0;
}
