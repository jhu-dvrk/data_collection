#ifndef DC_COMMON_CONFIG_HPP
#define DC_COMMON_CONFIG_HPP

#include <string>
#include <vector>
#include <json/json.h>

namespace dc {

struct VideoEncoding {
    int width = 0;
    int height = 0;
    int frame_rate = 0;
    int bitrate_kbps = 10000;
    int speed_preset = 5;
    int key_int_max = 30;
};

struct VideoConfig {
    std::string name;
    std::string stream;
    VideoEncoding encoding;
    bool record = true;
    bool timestamp_overlay = false;
};

struct AppConfig {
    std::string data_directory = ".";
    bool enable_audio = false;
    std::vector<VideoConfig> videos;
    std::vector<std::string> stages;
    std::vector<std::string> tags;
    std::vector<std::string> ros_topics;
};

class Config {
public:
    static bool load_from_file(const std::string& path, Json::Value& root);
    static std::vector<VideoConfig> parse_videos(const Json::Value& root);
    static AppConfig parse_app_config(const Json::Value& root);
    
    // Helper to generate a GStreamer caps string from the encoding
    static std::string make_caps_string(const VideoEncoding& enc);
};

}

#endif
