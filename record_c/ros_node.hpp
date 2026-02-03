#ifndef ROS_NODE_HPP
#define ROS_NODE_HPP

#include "context.hpp"

void ros_record_callback(const std_msgs::msg::Bool::SharedPtr msg, AppData* ad);

#endif // ROS_NODE_HPP
