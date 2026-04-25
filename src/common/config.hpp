#ifndef DC_COMMON_CONFIG_HPP
#define DC_COMMON_CONFIG_HPP

#include <string>
#include <vector>
#include <set>
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
    bool has_unixfd_socket_path = false;
    std::string unixfd_socket_path = "";
    VideoEncoding encoding;
    bool record = true;
    bool timestamp_overlay = false;
    std::string side_by_side = "undefined";
    double estimated_latency = 0.0;
};

struct RosTopicConfig {
    std::string name;
    bool continuous = false;
    bool enabled = true;
};

struct AppConfig {
    std::string data_directory = ".";
    bool record_audio = false;
    std::vector<VideoConfig> videos;
    std::vector<std::string> stages;
    std::vector<std::string> tags;
    std::vector<RosTopicConfig> ros_topics;
    std::vector<std::string> configuration_files;
};

class Config {
public:
    static bool load_from_file(const std::string& path, Json::Value& root);
    static bool check_type(const Json::Value& root, const std::string& expected_type, const std::string& path);
    static std::vector<VideoConfig> parse_videos(const Json::Value& root);
    static AppConfig parse_app_config(const Json::Value& root);

    /// Recursively load a config and all its configuration_files references.
    /// @param path       Absolute or relative path to the JSON file
    /// @param master_dir Directory of the top-level config (fallback for relative paths)
    /// @param visited    Set of resolved paths already loaded (cycle detection)
    /// @param out        Accumulated list of AppConfig objects
    static bool load_recursive(const std::string& path,
                               const std::string& master_dir,
                               std::set<std::string>& visited,
                               std::vector<AppConfig>& out);

    // Helper to generate a GStreamer caps string from the encoding
    static std::string make_caps_string(const VideoEncoding& enc);
};

}

#endif
