#include "ros_node.hpp"
#include "ui.hpp"

void ros_record_callback(const std_msgs::msg::Bool::SharedPtr msg, AppData* ad) {
    bool should_record = msg->data;
    std::lock_guard<std::mutex> lock(ad->data_mutex);
    if (should_record != ad->global_recording) {
        g_idle_add(toggle_recording_idle, ad);
    }
}
