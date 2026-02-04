#ifndef LATENCY_UI_HPP
#define LATENCY_UI_HPP

#include <gtkmm.h>
#include <gst/gst.h>
#include <vector>
#include <mutex>
#include <map>
#include "context.hpp"

class LatencyWindow : public Gtk::Window {
public:
    LatencyWindow(AppData* data);
    virtual ~LatencyWindow();

protected:
    void on_estimate_clicked(VideoStream* s);
    static GstPadProbeReturn brightness_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

    struct LatencyData {
        long long flash_start_cpu_ns = 0;
        bool waiting_for_black = false;
        long long detection_cpu_ns = 0;
        VideoStream* stream = nullptr;
        std::mutex mutex;
        std::function<void(double)> on_result;
    };

    AppData* m_data;
    Gtk::Box m_main_vbox;
    Gtk::Grid m_grid;
    LatencyData m_latency_state;
    std::map<VideoStream*, Gtk::Label*> m_latency_labels;

    void update_result(VideoStream* s, double ms);
};

#endif
