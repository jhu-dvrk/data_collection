#include <gst/gst.h>
#include <iostream>
#include <vector>
#include <string>
#include <json/json.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <array>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) return "Error running command";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

struct CameraMode {
    std::string caps_name; // e.g. "video/x-raw" or "image/jpeg"
    int width;
    int height;
    int fps_num;
    int fps_den;

    bool operator<(const CameraMode& other) const {
        if (width != other.width) return width > other.width;
        if (height != other.height) return height > other.height;
        if (fps_num * other.fps_den != other.fps_num * fps_den) 
            return (double)fps_num / fps_den > (double)other.fps_num / other.fps_den;
        return caps_name < other.caps_name;
    }
};

struct CameraDevice {
    std::string name;
    std::string device_path;
    std::string factory_name;
    std::vector<CameraMode> modes;
};

void parse_caps(GstCaps *caps, std::vector<CameraMode>& modes) {
    std::set<CameraMode> unique_modes;
    guint size = gst_caps_get_size(caps);
    for (guint i = 0; i < size; i++) {
        GstStructure *structure = gst_caps_get_structure(caps, i);
        const gchar *name = gst_structure_get_name(structure);
        int width, height;
        const GValue *fps;

        if (gst_structure_get_int(structure, "width", &width) &&
            gst_structure_get_int(structure, "height", &height)) {
            
            fps = gst_structure_get_value(structure, "framerate");
            if (fps) {
                if (GST_VALUE_HOLDS_FRACTION(fps)) {
                    unique_modes.insert({name, width, height, gst_value_get_fraction_numerator(fps), gst_value_get_fraction_denominator(fps)});
                } else if (GST_VALUE_HOLDS_FRACTION_RANGE(fps)) {
                    const GValue *max_val = gst_value_get_fraction_range_max(fps);
                    unique_modes.insert({name, width, height, gst_value_get_fraction_numerator(max_val), gst_value_get_fraction_denominator(max_val)});
                } else if (GST_VALUE_HOLDS_LIST(fps)) {
                    guint list_size = gst_value_list_get_size(fps);
                    for (guint j = 0; j < list_size; j++) {
                        const GValue *val = gst_value_list_get_value(fps, j);
                        if (GST_VALUE_HOLDS_FRACTION(val)) {
                             unique_modes.insert({name, width, height, gst_value_get_fraction_numerator(val), gst_value_get_fraction_denominator(val)});
                        }
                    }
                }
            }
        }
    }
    for (const auto& m : unique_modes) modes.push_back(m);
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    GstDeviceMonitor *monitor = gst_device_monitor_new();
    gst_device_monitor_add_filter(monitor, "Video/Source", NULL);

    GList *devices = gst_device_monitor_get_devices(monitor);
    if (!devices) {
        std::cerr << "No video devices found." << std::endl;
        return 1;
    }

    std::vector<CameraDevice> camera_list;

    for (GList *l = devices; l != NULL; l = l->next) {
        GstDevice *device = GST_DEVICE(l->data);
        CameraDevice cam;
        
        gchar *display_name = gst_device_get_display_name(device);
        cam.name = display_name ? display_name : "Unknown";
        g_free(display_name);

        GstStructure *props = gst_device_get_properties(device);
        if (props) {
            const gchar *path = gst_structure_get_string(props, "device.path");
            if (!path) path = gst_structure_get_string(props, "v4l2.device");
            if (path) cam.device_path = path;
            
            // If still no path, search all string fields for /dev/video
            if (cam.device_path.empty()) {
                int n = gst_structure_n_fields(props);
                for(int i=0; i<n; ++i) {
                    const gchar *field = gst_structure_nth_field_name(props, i);
                    const GValue *val = gst_structure_get_value(props, field);
                    if (G_VALUE_HOLDS_STRING(val)) {
                        const gchar *s = g_value_get_string(val);
                        if (s && g_str_has_prefix(s, "/dev/video")) {
                            cam.device_path = s;
                            break;
                        }
                    }
                }
            }

            const gchar *api = gst_structure_get_string(props, "device.api");
            if (api) cam.factory_name = api;

            gst_structure_free(props);
        }

        GstCaps *caps = gst_device_get_caps(device);
        if (caps) {
            parse_caps(caps, cam.modes);
            gst_caps_unref(caps);
        }

        if (!cam.modes.empty()) {
            camera_list.push_back(cam);
        }
    }
    g_list_free_full(devices, gst_object_unref);
    gst_object_unref(monitor);

    std::cout << "\n--- Discovered Video Devices ---\n" << std::endl;
    for (size_t i = 0; i < camera_list.size(); ++i) {
        std::cout << "[" << i << "] " << camera_list[i].name << std::endl;
        if (!camera_list[i].device_path.empty())
            std::cout << "    Path: " << camera_list[i].device_path << std::endl;
        std::cout << "    Modes found: " << camera_list[i].modes.size() << std::endl;
    }

    if (camera_list.empty()) {
        std::cout << "\nNo suitable cameras found." << std::endl;
        return 0;
    }

    int choice = -1;
    std::cout << "\nSelect device index (or ctrl-c to quit): ";
    if (!(std::cin >> choice) || choice < 0 || choice >= (int)camera_list.size()) {
        return 1;
    }

    CameraDevice& selected = camera_list[choice];
    std::cout << "\n--- Modes for " << selected.name << " ---\n" << std::endl;

    for (size_t i = 0; i < selected.modes.size(); ++i) {
        std::cout << "[" << std::setw(2) << i << "] " << selected.modes[i].width << "x" << selected.modes[i].height << " @ " 
                  << (double)selected.modes[i].fps_num / selected.modes[i].fps_den << " fps (" << selected.modes[i].caps_name << ")" << std::endl;
    }

    int mode_choice = -1;
    std::cout << "\nSelect mode index: ";
    if (!(std::cin >> mode_choice) || mode_choice < 0 || mode_choice >= (int)selected.modes.size()) {
        return 1;
    }

    CameraMode& mode = selected.modes[mode_choice];

    // Query V4L2 Controls
    const char* device_path = selected.device_path.empty() ? "/dev/video0" : selected.device_path.c_str();
    std::cout << "\n--- Available Camera Controls for " << selected.name << " ---\n" << std::endl;
    std::string cmd = "v4l2-ctl -d " + std::string(device_path) + " --list-ctrls";
    std::cout << exec(cmd.c_str()) << std::endl;

    std::string extra_ctrls;
    std::cout << "Enter any extra v4l2src properties (e.g., brightness=60 exposure-mode=auto) or press enter: ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // clear buffer
    std::getline(std::cin, extra_ctrls);

    Json::Value video;
    video["name"] = selected.name;
    
    std::string v4l_params = "v4l2src";
    if (!selected.device_path.empty()) {
        v4l_params += " device=" + selected.device_path;
    }
    if (!extra_ctrls.empty()) {
        v4l_params += " " + extra_ctrls;
    }

    std::string stream = v4l_params;
    // Add caps to stream field
    stream += " ! " + mode.caps_name + ",width=" + std::to_string(mode.width) + 
              ",height=" + std::to_string(mode.height) + 
              ",framerate=" + std::to_string(mode.fps_num) + "/" + std::to_string(mode.fps_den);
    
    // If it's MJPEG, we need a decoder to get to raw for preview/recording
    if (mode.caps_name == "image/jpeg") {
        stream += " ! jpegdec ! videoconvert";
    }
              
    video["stream"] = stream;
    video["record"] = true;
    
    Json::Value encoding;
    encoding["width"] = mode.width;
    encoding["height"] = mode.height;
    encoding["frame_rate"] = (int)((double)mode.fps_num / mode.fps_den);
    video["encoding"] = encoding;

    Json::Value root;
    root["videos"].append(video);

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "    ";
    std::string output = Json::writeString(builder, root);

    std::cout << "\nGenerated Configuration JSON:\n" << std::endl;
    std::cout << output << std::endl;

    std::cout << "\nSave to file? (y/n): ";
    char save;
    std::cin >> save;
    if (save == 'y' || save == 'Y') {
        std::string filename;
        std::cout << "Enter filename: ";
        std::cin >> filename;
        std::ofstream f(filename);
        if (f.is_open()) {
            f << output;
            std::cout << "Saved to " << filename << std::endl;
        }
    }

    return 0;
}
