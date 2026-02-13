#include "tags.hpp"
#include <ctime>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace dc {

std::string get_current_timestamp_iso8601() {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    char buf[100];
    struct tm* tm_info = localtime(&now.tv_sec);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", tm_info);
    std::sprintf(buf + std::strlen(buf), ".%03ld", now.tv_nsec / 1000000);
    return std::string(buf);
}

long long parse_stage_timestamp(const Json::Value& val) {
    if (val.isObject()) {
        if (val.isMember("cpu_ts")) {
            return val["cpu_ts"].asInt64();
        }
    } else if (val.isNumeric() || val.isString()) {
        return val.asInt64();
    }
    return 0; // Or better: error handling. But 0 is often treated as 'bad'.
}

} // namespace dc
