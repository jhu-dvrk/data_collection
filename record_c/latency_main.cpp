#include <iostream>
#include <gtkmm.h>
#include <gst/gst.h>
#include <json/json.h>
#include <fstream>
#include "context.hpp"
#include "pipeline.hpp"
#include "latency_ui.hpp"

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    Gtk::Main kit(argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: latency_c -c <config.json>" << std::endl;
        return 1;
    }

    AppData data;
    std::string config_path;
    for (int i=1; i<argc; ++i) {
        if (std::string(argv[i]) == "-c" && i+1 < argc) config_path = argv[++i];
    }

    if (config_path.empty()) {
        std::cerr << "Config file required (-c)" << std::endl;
        return 1;
    }

    std::ifstream f(config_path);
    Json::Value root;
    f >> root;

    for (const auto& v : root["videos"]) {
        VideoStream* s = create_video_stream(&data, v);
        if (s) {
            data.streams.push_back(s);
            gst_element_set_state(s->pipeline, GST_STATE_PLAYING);
        }
    }

    LatencyWindow win(&data);
    Gtk::Main::run(win);

    return 0;
}
