#include <gtkmm/application.h>
#include <gst/gst.h>
#include <iostream>
#include <string>
#include <filesystem>
#include "tag_ui.hpp"

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    std::string video_file;
    std::string config_file;
    std::string tags_file;
    bool load_session_tags = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-v" || arg == "--video") && i + 1 < argc) {
            video_file = argv[++i];
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "-t" || arg == "--tags") && i + 1 < argc) {
            tags_file = argv[++i];
        } else if (arg == "-T") {
            load_session_tags = true;
        }
    }

    if (video_file.empty()) {
        std::cerr << "Usage: video_tag -v <video_file> [-c <config_file>] [-t <tags_file>] [-T]" << std::endl;
        return 1;
    }

    // Existence checks
    if (!std::filesystem::exists(video_file)) {
        std::cerr << "CRITICAL: Video file does not exist: " << video_file << std::endl;
        return 1;
    }

    if (!config_file.empty() && !std::filesystem::exists(config_file)) {
        std::cerr << "CRITICAL: Config file does not exist: " << config_file << std::endl;
        return 1;
    }

    if (!tags_file.empty() && !std::filesystem::exists(tags_file)) {
        std::cerr << "CRITICAL: Specified tags file does not exist: " << tags_file << std::endl;
        return 1;
    }

    auto app = Gtk::Application::create("org.data_collection.tag_c");
    TagWindow window(video_file, config_file, tags_file, load_session_tags);
    return app->run(window);
}
