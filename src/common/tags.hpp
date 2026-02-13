#ifndef TAGS_HPP
#define TAGS_HPP

#include <string>
#include <json/json.h>

namespace dc {
    /**
     * @brief Returns current date/time in "YYYY-MM-DDTHH:MM:SS.sss" format (local time)
     */
    std::string get_current_timestamp_iso8601();

    /**
     * @brief Helper to extract timestamp from a Json Value
     * Handles both legacy format (raw int64) and new format (object with "cpu_ts")
     * @param val The JSON value for "start" or "end"
     * @return The CPU timestamp as long long
     */
    long long parse_stage_timestamp(const Json::Value& val);
}

#endif // TAGS_HPP
