cmake_minimum_required(VERSION 3.8)
project(data_collection)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0)
pkg_check_modules(GTK REQUIRED gtk+-3.0)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# C++ Executable: record_c
add_executable(record_c 
  record_c/main.cpp
  record_c/context.cpp
  record_c/pipeline.cpp
  record_c/ui.cpp
  record_c/ros_node.cpp
)
ament_target_dependencies(record_c
  rclcpp
  std_msgs
  sensor_msgs
  rosbag2_cpp
)
target_include_directories(record_c PRIVATE
  ${GST_INCLUDE_DIRS}
  ${GTK_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(record_c
  ${GST_LIBRARIES}
  ${GTK_LIBRARIES}
  ${JSONCPP_LIBRARIES}
)


# Install C++ executable
install(TARGETS record_c
  DESTINATION lib/${PROJECT_NAME}
)

# Python Package
ament_python_install_package(${PROJECT_NAME})

# Python Scripts (wrappers)
install(PROGRAMS
  scripts/record
  scripts/extract
  scripts/video_tag
  DESTINATION lib/${PROJECT_NAME}
)

# Data files (share directory)
install(DIRECTORY share/
  DESTINATION share/${PROJECT_NAME}
)

# Resource index
ament_package()
