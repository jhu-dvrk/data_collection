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
#include "config.hpp"


// Signal handler for Ctrl+C
static gboolean on_sigint(gpointer user_data) {
    AppData *data = static_cast<AppData*>(user_data);
    data->is_quitting = true;
    Gtk::Main::quit();
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
            std::string path = argv[++i];
            char *abs_path = realpath(path.c_str(), NULL);
            if (abs_path) {
                data.config_files.push_back(abs_path);
                free(abs_path);
            } else {
                data.config_files.push_back(path);
            }
            configs.push_back(data.config_files.back());
        } else if (arg == "-j" && i+1 < argc) {
            app_max_threads = std::stoi(argv[++i]);
            if (app_max_threads < 1) app_max_threads = 1;
        } else if (arg == "-p" && i+1 < argc) {
            data.trigger_topic = argv[++i];
        } else if (arg[0] != '-') {
            char *abs_path = realpath(argv[i], NULL);
            if (abs_path) {
                data.config_files.push_back(abs_path);
                free(abs_path);
            } else {
                data.config_files.push_back(argv[i]);
            }
            configs.push_back(data.config_files.back());
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

    std::vector<Json::Value> config_roots; // Keeping this for now if needed, but preferably use AppConfig objects
    std::vector<dc::AppConfig> app_configs;

    for (const auto& path : configs) {
        Json::Value root;
        if (dc::Config::load_from_file(path, root)) {
            config_roots.push_back(root);
            app_configs.push_back(dc::Config::parse_app_config(root));
        } else {
             // Error already printed by load_from_file
            return 1;
        }
    }

    for (const auto& cfg : app_configs) {
        if (!cfg.data_directory.empty() && cfg.data_directory != ".") data.data_directory = cfg.data_directory;
        data.enable_audio = data.enable_audio || cfg.enable_audio;
        
        if (!cfg.stages.empty()) {
            data.explicit_stages = true;
            data.config_stages.insert(data.config_stages.end(), cfg.stages.begin(), cfg.stages.end());
        }
        data.config_tags.insert(data.config_tags.end(), cfg.tags.begin(), cfg.tags.end());
        data.ros_topics.insert(data.ros_topics.end(), cfg.ros_topics.begin(), cfg.ros_topics.end());
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

    for (const auto& cfg : app_configs) {
        for (const auto& v : cfg.videos) {
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
