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
pkg_check_modules(GTKMM REQUIRED gtkmm-3.0)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# Common Library: data_collection
add_library(data_collection_lib
  src/common/config.cpp
  src/common/tags.cpp
)
target_include_directories(data_collection_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/common>
  $<INSTALL_INTERFACE:include>
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(data_collection_lib
  ${JSONCPP_LIBRARIES}
)
# Alias for easier usage if needed, though target name is fine
set_target_properties(data_collection_lib PROPERTIES OUTPUT_NAME "data_collection")

# C++ Executable: record
add_executable(record
  src/record/main.cpp
  src/record/context.cpp
  src/record/pipeline.cpp
  src/record/ui.cpp
  src/record/ros_node.cpp
)
ament_target_dependencies(record
  rclcpp
  std_msgs
  sensor_msgs
  rosbag2_cpp
)
target_include_directories(record PRIVATE
  ${GST_INCLUDE_DIRS}
  ${GTK_INCLUDE_DIRS}
  ${GTKMM_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(record
  ${GST_LIBRARIES}
  ${GTK_LIBRARIES}
  ${GTKMM_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  data_collection_lib
)


# Install C++ executable
install(TARGETS record
  DESTINATION lib/${PROJECT_NAME}
)

# C++ Executable: video_tag
add_executable(video_tag
  src/video_tag/tag_main.cpp
  src/video_tag/tag_ui.cpp
)
target_include_directories(video_tag PRIVATE
  ${GST_INCLUDE_DIRS}
  ${GTK_INCLUDE_DIRS}
  ${GTKMM_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(video_tag
  ${GST_LIBRARIES}
  ${GTK_LIBRARIES}
  ${GTKMM_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  gstvideo-1.0
  data_collection_lib
)

# Install C++ executable video_tag
install(TARGETS video_tag
  DESTINATION lib/${PROJECT_NAME}
)

# C++ Executable: video_latency
add_executable(video_latency
  src/video_latency/main.cpp
)
target_include_directories(video_latency PRIVATE
  ${GST_INCLUDE_DIRS}
  ${GTK_INCLUDE_DIRS}
  ${GTKMM_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
)
target_link_libraries(video_latency
  ${GST_LIBRARIES}
  ${GTK_LIBRARIES}
  ${GTKMM_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  gstvideo-1.0
  data_collection_lib
)
install(TARGETS video_latency
  DESTINATION lib/${PROJECT_NAME}
)

# Python Package
ament_python_install_package(${PROJECT_NAME})

# Python Scripts (wrappers)
install(PROGRAMS
  scripts/extract
  scripts/encord_to_tags
  DESTINATION lib/${PROJECT_NAME}
)

# Data files (share directory)
install(DIRECTORY share/
  DESTINATION share/${PROJECT_NAME}
)

# Resource index
ament_package()
