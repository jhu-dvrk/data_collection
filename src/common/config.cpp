#include "config.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <climits>
#include <cstdlib>

namespace dc {

bool Config::load_from_file(const std::string& path, Json::Value& root) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open config: " << path << std::endl;
        return false;
    }
    try {
        ifs >> root;
    } catch (const std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

std::vector<VideoConfig> Config::parse_videos(const Json::Value& root) {
    std::vector<VideoConfig> result;
    if (!root.isMember("videos") || !root["videos"].isArray()) {
        return result;
    }

    for (const auto& v : root["videos"]) {
        VideoConfig cfg;
        if (v.isMember("name")) cfg.name = v["name"].asString();
        if (v.isMember("stream")) cfg.stream = v["stream"].asString();
        if (v.isMember("record")) cfg.record = v["record"].asBool();
        if (v.isMember("timestamp_overlay")) cfg.timestamp_overlay = v["timestamp_overlay"].asBool();
        if (v.isMember("tee_glimage_sink")) cfg.tee_glimage_sink = v["tee_glimage_sink"].asBool();
        if (v.isMember("ros_camera_name")) cfg.ros_camera_name = v["ros_camera_name"].asString();
        if (v.isMember("estimated_latency")) cfg.estimated_latency = v["estimated_latency"].asDouble();

        if (v.isMember("encoding")) {
            const auto& enc = v["encoding"];
            if (enc.isMember("width")) cfg.encoding.width = enc["width"].asInt();
            if (enc.isMember("height")) cfg.encoding.height = enc["height"].asInt();
            if (enc.isMember("frame_rate")) cfg.encoding.frame_rate = enc["frame_rate"].asInt();
            if (enc.isMember("bitrate_kbps")) cfg.encoding.bitrate_kbps = enc["bitrate_kbps"].asInt();
            else if (enc.isMember("bitrate")) cfg.encoding.bitrate_kbps = enc["bitrate"].asInt();
            if (enc.isMember("speed_preset")) cfg.encoding.speed_preset = enc["speed_preset"].asInt();
            if (enc.isMember("key_int_max")) cfg.encoding.key_int_max = enc["key_int_max"].asInt();
        }
        result.push_back(cfg);
    }
    return result;
}

AppConfig Config::parse_app_config(const Json::Value& root) {
    AppConfig cfg;
    cfg.data_directory = root.get("data_directory", ".").asString();
    cfg.record_audio = root.get("record_audio", false).asBool();

    cfg.videos = parse_videos(root);

    if (root.isMember("stages")) {
        for (const auto& s : root["stages"]) cfg.stages.push_back(s.asString());
    }
    if (root.isMember("tags")) {
        for (const auto& t : root["tags"]) cfg.tags.push_back(t.asString());
    }
    if (root.isMember("ros_topics")) {
        for (const auto& t : root["ros_topics"]) cfg.ros_topics.push_back(t.asString());
    }
    if (root.isMember("configuration_files")) {
        for (const auto& f : root["configuration_files"]) cfg.configuration_files.push_back(f.asString());
    }
    return cfg;
}

bool Config::load_recursive(const std::string& path,
                            const std::string& master_dir,
                            std::set<std::string>& visited,
                            std::vector<AppConfig>& out) {
    // Resolve to absolute path for cycle detection
    char resolved[PATH_MAX];
    const char* abs = realpath(path.c_str(), resolved);
    std::string abs_path = abs ? std::string(abs) : path;

    if (!visited.insert(abs_path).second) {
        // Already loaded — skip silently (not an error)
        return true;
    }

    Json::Value root;
    if (!load_from_file(abs_path, root)) {
        return false;
    }

    AppConfig cfg = parse_app_config(root);

    // Determine the directory of this config file for relative path resolution
    std::string config_dir = abs_path;
    auto slash = config_dir.rfind('/');
    if (slash != std::string::npos) config_dir = config_dir.substr(0, slash);
    else config_dir = ".";

    // Recursively load referenced configuration files
    for (const auto& ref : cfg.configuration_files) {
        // Try relative to this config's directory first, then master_dir
        std::string candidate = config_dir + "/" + ref;
        std::ifstream test(candidate);
        if (!test.good()) {
            candidate = master_dir + "/" + ref;
        }
        if (!load_recursive(candidate, master_dir, visited, out)) {
            return false;
        }
    }

    out.push_back(std::move(cfg));
    return true;
}

std::string Config::make_caps_string(const VideoEncoding& enc) {
    std::string caps = "video/x-raw";
    if (enc.width > 0 && enc.height > 0) {
        caps += ",width=" + std::to_string(enc.width) +
                ",height=" + std::to_string(enc.height);
    }
    if (enc.frame_rate > 0) {
        caps += ",framerate=" + std::to_string(enc.frame_rate) + "/1";
    }
    return caps;
}

}
