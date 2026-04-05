#ifndef ROS_NODE_HPP
#define ROS_NODE_HPP

#include "context.hpp"

void ros_record_callback(const std_msgs::msg::Bool::SharedPtr msg, AppData* ad);
void setup_ros_monitoring(AppData* ad);
void ros_node_spin_once(AppData* ad);
void open_bag_writer(AppData* ad, const std::string& path);
void close_bag_writer(AppData* ad);


#endif // ROS_NODE_HPP
