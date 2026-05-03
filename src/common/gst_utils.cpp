#include "gst_utils.hpp"
#include <iostream>
#include <cstdlib>

namespace dc {

void dump_dot(GstElement* pipeline, const std::string& path, GstDebugGraphDetails flags) {
    gchar* dot_data = gst_debug_bin_to_dot_data(GST_BIN(pipeline), flags);
    if (dot_data) {
        std::string dot_str(dot_data);
        
        // Change orientation to Top-to-Bottom (Vertical)
        size_t pos = dot_str.find("rankdir=LR");
        if (pos != std::string::npos) {
            dot_str.replace(pos, 10, "rankdir=TB");
        }

        FILE* f = fopen(path.c_str(), "w");
        if (f) {
            fputs(dot_str.c_str(), f);
            fclose(f);
            std::cout << "GStreamer pipeline graph saved to: " << path << std::endl;
        } else {
            std::cerr << "Error: Could not open file for writing: " << path << std::endl;
        }
        g_free(dot_data);
    }
}

bool parse_dot_arguments(int& i, int argc, char* argv[], bool& dump_dot, GstDebugGraphDetails& dot_flags) {
    std::string arg = argv[i];
    if (arg == "-g" || arg == "--dot") {
        dump_dot = true;
        int level = 1; // Default to "Light" if -g is present
        if (i + 1 < argc && isdigit(argv[i + 1][0])) {
            level = std::stoi(argv[++i]);
        }
        
        if (level == 0)
            dot_flags = (GstDebugGraphDetails)0;
        else if (level == 1)
            dot_flags = GST_DEBUG_GRAPH_SHOW_STATES;
        else if (level == 2)
            dot_flags = (GstDebugGraphDetails)(GST_DEBUG_GRAPH_SHOW_MEDIA_TYPE | GST_DEBUG_GRAPH_SHOW_STATES);
        else
            dot_flags = GST_DEBUG_GRAPH_SHOW_ALL;
            
        return true;
    }
    return false;
}

void print_dot_usage() {
    std::cerr << "  -g 0: Minimal (elements only)" << std::endl;
    std::cerr << "  -g 1: Light (elements + states, recommended)" << std::endl;
    std::cerr << "  -g 2: Medium (adds media types)" << std::endl;
    std::cerr << "  -g 3: Full (all parameters)" << std::endl;
}

} // namespace dc
