#include "ui.hpp"
#include "pipeline.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <json/json.h>

void update_stage_highlighting(AppData *data) {
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

void toggle_recording(AppData *data) {
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

gboolean toggle_recording_idle(gpointer user_data) {
    AppData* ad = (AppData*)user_data;
    toggle_recording(ad);
    return G_SOURCE_REMOVE;
}

void populate_audio_sources(GtkComboBoxText* combo) {
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


void on_window_destroy_cb(GtkWidget *w, gpointer d) {
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

void create_main_window(AppData* data) {
    data->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_default_size(GTK_WINDOW(data->window), 1024, 768);
    g_signal_connect(data->window, "destroy", G_CALLBACK(on_window_destroy_cb), data);

    // Main vertical split: f1 (top), f2 (bottom)
    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, BOX_SPACING_PX);
    gtk_container_add(GTK_CONTAINER(data->window), main_vbox);
    gtk_container_set_border_width(GTK_CONTAINER(main_vbox), WINDOW_BORDER_PX);

    // f1: top horizontal split (f1a, f1b)
    GtkWidget *f1_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(main_vbox), f1_hbox, TRUE, TRUE, 0);

    // f1a: session, audio, video previews (vertical)
    GtkWidget *f1a_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(f1_hbox), f1a_vbox, TRUE, TRUE, 0);

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
    for (size_t i = 0; i < data->config_stages.size(); ++i) {
        GtkWidget* stage_label = gtk_label_new(NULL);
        data->stage_labels.push_back(stage_label);
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
        }), data);

        gtk_box_pack_start(GTK_BOX(stages_vbox), eb, FALSE, FALSE, 0);
    }
    update_stage_highlighting(data);
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
    gtk_entry_set_text(GTK_ENTRY(session_name_entry), data->session_dir.c_str());
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
    data->data_dir_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(data->data_dir_entry), data->data_directory.c_str());
    gtk_widget_set_sensitive(data->data_dir_entry, FALSE);
    gtk_widget_set_margin_start(data->data_dir_entry, WIDGET_MARGIN_PX);
    gtk_widget_set_margin_end(data->data_dir_entry, WIDGET_MARGIN_PX);
    gtk_box_pack_start(GTK_BOX(dir_hbox), data->data_dir_entry, FALSE, FALSE, 0);
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
    }), data);

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

    data->audio_enable_checkbox = gtk_check_button_new_with_label("Record");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->audio_enable_checkbox), data->enable_audio);
    gtk_box_pack_start(GTK_BOX(audio_header_hbox), data->audio_enable_checkbox, FALSE, FALSE, 0);

    GtkWidget *audio_ctrl_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(audio_vbox), audio_ctrl_hbox, FALSE, FALSE, 0);
    gtk_container_set_border_width(GTK_CONTAINER(audio_ctrl_hbox), WIDGET_MARGIN_PX);

    data->audio_src_combo = gtk_combo_box_text_new();
    populate_audio_sources(GTK_COMBO_BOX_TEXT(data->audio_src_combo));
    gtk_box_pack_start(GTK_BOX(audio_ctrl_hbox), data->audio_src_combo, FALSE, FALSE, 0);
    g_signal_connect(data->audio_src_combo, "changed", G_CALLBACK(+[](GtkComboBox* c, gpointer d) -> void {
        AppData* ad = (AppData*)d;
        if (ad->global_recording) return; 
        create_audio_pipeline(ad);
    }), data);

    data->audio_level_bar = gtk_level_bar_new();
    gtk_level_bar_set_min_value(GTK_LEVEL_BAR(data->audio_level_bar), 0.0);
    gtk_level_bar_set_max_value(GTK_LEVEL_BAR(data->audio_level_bar), 1.0);
    gtk_level_bar_set_mode(GTK_LEVEL_BAR(data->audio_level_bar), GTK_LEVEL_BAR_MODE_CONTINUOUS);
    // Add typical VU colors
    gtk_level_bar_add_offset_value(GTK_LEVEL_BAR(data->audio_level_bar), GTK_LEVEL_BAR_OFFSET_LOW, 0.4);
    gtk_level_bar_add_offset_value(GTK_LEVEL_BAR(data->audio_level_bar), GTK_LEVEL_BAR_OFFSET_HIGH, 0.8);
    
    gtk_widget_set_size_request(data->audio_level_bar, -1, 15);
    gtk_box_pack_start(GTK_BOX(audio_ctrl_hbox), data->audio_level_bar, TRUE, TRUE, 0);

    data->grid = gtk_grid_new(); 
    gtk_grid_set_column_homogeneous(GTK_GRID(data->grid), TRUE); 
    gtk_grid_set_row_homogeneous(GTK_GRID(data->grid), FALSE); // Allow rows to shrink to content
    gtk_grid_set_row_spacing(GTK_GRID(data->grid), BOX_SPACING_PX); 
    gtk_grid_set_column_spacing(GTK_GRID(data->grid), BOX_SPACING_PX);
    gtk_box_pack_start(GTK_BOX(f1a_vbox), data->grid, TRUE, TRUE, 0);

    for (size_t i=0; i<data->streams.size(); ++i) {
        VideoStream* s = data->streams[i];
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

        gtk_grid_attach(GTK_GRID(data->grid), stream_frame, (int)(i%2), (int)(i/2), 1, 1);
    }

    data->record_button = gtk_button_new_with_label("Start Recording");
    g_signal_connect(data->record_button, "clicked", G_CALLBACK(+[](GtkButton* b, gpointer d){ toggle_recording((AppData*)d); }), data);
    gtk_box_pack_start(GTK_BOX(f2_hbox), data->record_button, TRUE, TRUE, 0);

    GtkWidget *quit_btn = gtk_button_new_with_label("Quit");
    g_signal_connect(quit_btn, "clicked", G_CALLBACK(+[](GtkButton* b, gpointer d){ gtk_window_close(GTK_WINDOW(((AppData*)d)->window)); }), data);
    gtk_box_pack_end(GTK_BOX(f2_hbox), quit_btn, FALSE, FALSE, 0);
}

void start_ui_update_loop(AppData* data) {
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
    }, data);
}
