#include "latency_ui.hpp"
#include <gst/video/video.h>
#include <iostream>

LatencyWindow::LatencyWindow(AppData* data) : m_data(data), m_main_vbox(Gtk::ORIENTATION_VERTICAL, 10) {
    set_title("Latency Estimation Tool");
    set_default_size(800, 600);
    add(m_main_vbox);

    m_main_vbox.pack_start(m_grid, Gtk::PACK_EXPAND_WIDGET);

    int row = 0;
    for (auto s : m_data->streams) {
        Gtk::Label* lbl = Gtk::manage(new Gtk::Label(s->name));
        m_grid.attach(*lbl, 0, row, 1, 1);

        Gtk::Label* res_lbl = Gtk::manage(new Gtk::Label("Result: N/A"));
        m_grid.attach(*res_lbl, 1, row, 1, 1);
        m_latency_labels[s] = res_lbl;

        GtkWidget* sink_widget = nullptr;
        GstElement* sink = gst_bin_get_by_name(GST_BIN(s->pipeline), "sink");
        if (sink) {
            g_object_get(sink, "widget", &sink_widget, NULL);
            if (sink_widget) {
                Gtk::Widget* wrapped = Glib::wrap(sink_widget);
                wrapped->set_size_request(320, 240);
                m_grid.attach(*wrapped, 0, row + 1, 1, 1);
            }
            
            // Add probe for brightness detection
            GstPad* pad = gst_element_get_static_pad(sink, "sink");
            gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, brightness_probe_cb, &m_latency_state, NULL);
            gst_object_unref(pad);
            gst_object_unref(sink);
        }

        Gtk::Button* btn = Gtk::manage(new Gtk::Button("Estimate Latency"));
        btn->signal_clicked().connect([this, s]() { on_estimate_clicked(s); });
        m_grid.attach(*btn, 0, row + 2, 1, 1);

        row += 3;
    }

    show_all_children();
}

LatencyWindow::~LatencyWindow() {
    for (auto s : m_data->streams) {
        gst_element_set_state(s->pipeline, GST_STATE_NULL);
    }
}

void LatencyWindow::on_estimate_clicked(VideoStream* s) {
    std::cout << "Starting latency estimation for " << s->name << "..." << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(m_latency_state.mutex);
        m_latency_state.stream = s;
        m_latency_state.waiting_for_black = false;
        m_latency_state.on_result = [this, s](double ms) {
            Glib::signal_idle().connect_once([this, s, ms]() {
                update_result(s, ms);
            });
        };
    }

    // Step 1: Turn off preview for 100ms to ensure a baseline then flash black effectively
    // Actually, let's just use cairooverlay to draw a black rectangle.
    GstElement* overlay = gst_bin_get_by_name(GST_BIN(s->pipeline), "rec_overlay");
    if (!overlay) return;

    // Use a timer to trigger the flash
    Glib::signal_timeout().connect_once([this, s]() {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        
        {
            std::lock_guard<std::mutex> lock(m_latency_state.mutex);
            m_latency_state.flash_start_cpu_ns = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
            m_latency_state.waiting_for_black = true;
            m_latency_state.detection_cpu_ns = 0;
        }

        s->is_recording = true; 
        s->frames_recorded = -1; // Trigger black flash in pipeline.cpp
        
        // Wait 100ms then stop flash
        Glib::signal_timeout().connect_once([s]() {
            s->is_recording = false;
            s->frames_recorded = 0;
        }, 100);

    }, 500); // Start in 500ms
}

void LatencyWindow::update_result(VideoStream* s, double ms) {
    auto it = m_latency_labels.find(s);
    if (it != m_latency_labels.end()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "Result: %.2f ms", ms);
        it->second->set_text(buf);
    }
}

GstPadProbeReturn LatencyWindow::brightness_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    LatencyData* ld = (LatencyData*)user_data;
    if (!(info->type & GST_PAD_PROBE_TYPE_BUFFER)) return GST_PAD_PROBE_OK;

    std::lock_guard<std::mutex> lock(ld->mutex);
    if (!ld->waiting_for_black) return GST_PAD_PROBE_OK;

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        // Assume I420 or similar for simple brightness check (first half is Y)
        long long sum = 0;
        for (size_t i = 0; i < map.size / 2; i += 100) { // Sparsely sample Y plane
            sum += map.data[i];
        }
        double avg = (double)sum / (map.size / 200.0);
        
        if (avg < 30.0) { // threshold for "mostly black"
             struct timespec ts;
             clock_gettime(CLOCK_REALTIME, &ts);
             ld->detection_cpu_ns = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
             ld->waiting_for_black = false;
             
             long long diff = ld->detection_cpu_ns - ld->flash_start_cpu_ns;
             double ms = (double)diff / 1e6;
             std::cout << "Latency detected: " << ms << " ms" << std::endl;
             
             if (ld->on_result) {
                 ld->on_result(ms);
             }
        }
        gst_buffer_unmap(buf, &map);
    }

    return GST_PAD_PROBE_OK;
}
