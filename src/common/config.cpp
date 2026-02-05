#include "config.hpp"
#include <fstream>
#include <iostream>

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
    cfg.enable_audio = root.get("enable_audio", false).asBool();

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
    return cfg;
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
