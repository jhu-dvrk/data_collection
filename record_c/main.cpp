#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <ctime>

#include <gtkmm.h>
#include <gst/gst.h>
#include <json/json.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <glib-unix.h>

#include "context.hpp"
#include "pipeline.hpp"
#include "ui.hpp"
#include "ros_node.hpp"


// Signal handler for Ctrl+C
static gboolean on_sigint(gpointer user_data) {
    AppData *data = static_cast<AppData*>(user_data);
    data->is_quitting = true;
    gtk_main_quit(); 
    // Gtk::Main::quit(); // Could use this too
    return FALSE;
}

int main(int argc, char *argv[]) {
    // Initialize GStreamer first (strips GST args)
    gst_init(&argc, &argv);

    // Initialize Gtkmm (strips GTK args)
    Gtk::Main kit(argc, argv);
    
    AppData data; 
    std::vector<std::string> configs;

    unsigned int cores = std::thread::hardware_concurrency();
    app_max_threads = (cores > 0 ? cores : 2) / 2;
    if (app_max_threads < 1) app_max_threads = 1;

    time_t now_raw; char buf[80]; time(&now_raw); strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now_raw));
    data.start_timestamp = buf;

    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" && i+1 < argc) {
            configs.push_back(argv[++i]);
        } else if (arg == "-j" && i+1 < argc) {
            app_max_threads = std::stoi(argv[++i]);
            if (app_max_threads < 1) app_max_threads = 1;
        } else if (arg == "-p" && i+1 < argc) {
            data.trigger_topic = argv[++i];
        } else if (arg[0] != '-') {
            configs.push_back(argv[i]);
        }
    }
    if (configs.empty()) {
        std::cerr << "Usage: " << argv[0] << " -c <scan_config.json> [options]" << std::endl;
        return 1;
    }

    // ROS 2 Initialization
    rclcpp::init(argc, argv);
    data.node = std::make_shared<rclcpp::Node>("record_c");
    if (!data.trigger_topic.empty()) {
        data.sub_record = data.node->create_subscription<std_msgs::msg::Bool>(
            data.trigger_topic, 10, 
            [&](const std_msgs::msg::Bool::SharedPtr msg) { ros_record_callback(msg, &data); }
        );
        std::cout << "Subscribed to trigger topic: " << data.trigger_topic << std::endl;
    }

    // Spin ROS in a separate thread
    std::thread ros_thread([&]() {
        rclcpp::spin(data.node);
    });
    ros_thread.detach();

    std::vector<Json::Value> config_roots;
    for (const auto& path : configs) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Error: Could not open config file: " << path << std::endl;
            return 1;
        }
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errs;
        if (!Json::parseFromStream(builder, f, &root, &errs)) {
            std::cerr << "Error: Could not parse JSON in file: " << path << "\n" << errs << std::endl;
            return 1;
        }
        config_roots.push_back(root);
    }

    for (const auto& root : config_roots) {
        data.data_directory = root.get("data_directory", data.data_directory).asString();
        data.enable_audio = data.enable_audio || root.get("enable_audio", false).asBool();
        if (root.isMember("stages")) {
            data.explicit_stages = true;
            for (const auto& s : root["stages"]) data.config_stages.push_back(s.asString());
        }
        if (root.isMember("ros_topics")) {
            for (const auto& t : root["ros_topics"]) {
                data.ros_topics.push_back(t.asString());
            }
        }
    }
    if (data.config_stages.empty()) data.config_stages.push_back("stage");

    // Setup ROS topic monitoring if topics are requested
    setup_ros_monitoring(&data);

    data.session_dir = data.data_directory + "/" + data.start_timestamp;
    g_mkdir_with_parents(data.session_dir.c_str(), 0777);

    // Initialize ROS bag for the session
    if (!data.ros_topics.empty()) {
        data.session_bag_path = data.session_dir + "/rosbag_" + data.start_timestamp;
        open_bag_writer(&data, data.session_bag_path);
    }

    for (const auto& root : config_roots) {
        for (const auto& v : root["videos"]) {
            VideoStream* s = create_video_stream(&data, v);
            if (s) data.streams.push_back(s);
        }
    }

    // Initialize UI Window
    MainWindow window(&data);
    
    create_audio_pipeline(&data);
    for (auto s : data.streams) gst_element_set_state(s->pipeline, GST_STATE_PLAYING);
    
    // Handle Ctrl+C
    g_unix_signal_add(SIGINT, on_sigint, &data);

    // Run Main Loop
    // start_ui_update_loop(&data); // Replaced by MainWindow timer
    Gtk::Main::run(window);
    
    return 0;
}
