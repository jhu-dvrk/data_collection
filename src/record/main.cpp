#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <ctime>
#include <unordered_set>
#include <climits>
#include <set>
#include <filesystem>

#include <gtkmm.h>
#include <gst/gst.h>
#include <json/json.h>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <std_msgs/msg/bool.hpp>
#include <glib-unix.h>

#include "context.hpp"
#include "video_stream.hpp"
#include "audio_stream.hpp"
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
    // Clear Snap-related environment variables that cause symbol lookup errors
    unsetenv("GTK_PATH");
    unsetenv("LOCPATH");

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
        } else {
            std::cerr << "Error: Unknown argument or missing -c for config: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " -c <config.json> [-j <threads>] [-p <trigger_topic>]" << std::endl;
            return 1;
        }
    }
    if (configs.empty()) {
        std::cerr << "Usage: " << argv[0] << " -c <scan_config.json> [options]" << std::endl;
        return 1;
    }

    // Existence checks for config files
    for (const auto& cfg : data.config_files) {
        if (!std::filesystem::exists(cfg)) {
            std::cerr << "CRITICAL: Config file does not exist: " << cfg << std::endl;
            return 1;
        }
    }

    // ROS 2 Initialization
    rclcpp::init(argc, argv);
    data.node = std::make_shared<rclcpp::Node>("record_c");
    data.it = std::make_shared<image_transport::ImageTransport>(data.node);
    data.pub_recording = data.node->create_publisher<std_msgs::msg::Bool>("record/recording", 10);

    if (data.trigger_topic.empty()) {
        data.trigger_topic = "record/record";
    }

    // Remove leading slashes if any to keep them relative
    while (!data.trigger_topic.empty() && data.trigger_topic[0] == '/') data.trigger_topic.erase(0, 1);

    data.sub_record = data.node->create_subscription<std_msgs::msg::Bool>(
        data.trigger_topic, 10,
        [&](const std_msgs::msg::Bool::SharedPtr msg) { ros_record_callback(msg, &data); }
    );

    // Spin ROS in a separate thread
    auto executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor->add_node(data.node);
    std::thread ros_thread([executor]() {
        executor->spin();
    });

    // Publish initial recording state (False)
    {
        auto msg = std::make_unique<std_msgs::msg::Bool>();
        msg->data = false;
        data.pub_recording->publish(std::move(msg));
    }

    std::vector<dc::AppConfig> app_configs;

    for (const auto& path : configs) {
        // Determine master directory for relative path fallback
        std::string master_dir = ".";
        char resolved[PATH_MAX];
        const char* abs = realpath(path.c_str(), resolved);
        if (abs) {
            master_dir = std::string(abs);
            auto slash = master_dir.rfind('/');
            if (slash != std::string::npos) master_dir = master_dir.substr(0, slash);
        }

        std::set<std::string> visited;
        if (!dc::Config::load_recursive(path, master_dir, visited, app_configs)) {
            return 1;
        }
    }

    for (const auto& cfg : app_configs) {
        if (!cfg.data_directory.empty() && cfg.data_directory != ".") data.data_directory = cfg.data_directory;
        data.record_audio = data.record_audio || cfg.record_audio;

        if (!cfg.stages.empty()) {
            data.explicit_stages = true;
            data.config_stages.insert(data.config_stages.end(), cfg.stages.begin(), cfg.stages.end());
        }
        data.config_tags.insert(data.config_tags.end(), cfg.tags.begin(), cfg.tags.end());
        data.ros_topics.insert(data.ros_topics.end(), cfg.ros_topics.begin(), cfg.ros_topics.end());
    }

    // Deduplicate ROS topics while preserving first-seen order
    if (!data.ros_topics.empty()) {
        std::unordered_set<std::string> seen;
        std::vector<std::string> deduped;
        deduped.reserve(data.ros_topics.size());
        for (const auto& topic : data.ros_topics) {
            if (seen.insert(topic).second) deduped.push_back(topic);
        }
        data.ros_topics.swap(deduped);
    }
    if (data.config_stages.empty()) data.config_stages.push_back("stage");

    // Setup ROS topic monitoring if topics are requested
    setup_ros_monitoring(&data);

    data.session_dir = data.data_directory + "/" + data.start_timestamp;
    g_mkdir_with_parents(data.session_dir.c_str(), 0777);

    // Initialize ROS bag for the session
    if (!data.ros_topics.empty()) {
        data.session_bag_path = data.session_dir + "/rosbag";
        open_bag_writer(&data, data.session_bag_path);
    }

    for (const auto& cfg : app_configs) {
        for (const auto& v : cfg.videos) {
            VideoStream* s = new VideoStream();
            if (s->create(&data, &v)) {
                data.streams.push_back(s);
            } else {
                delete s;
            }
        }
    }

    // Initialize UI Window
    MainWindow window(&data);

    if (data.record_audio) {
        AudioStream* a = new AudioStream();
        if (a->create(&data)) {
            data.audio_stream = a;
        } else {
            delete a;
        }
    }

    // Start video pipelines AFTER the MainWindow has been constructed and all
    // gtksink widgets are embedded in the layout, preventing the GTK reparent
    // warning ("widget already inside a container of type GtkWindow").
    for (auto s : data.streams) gst_element_set_state(s->pipeline, GST_STATE_PLAYING);

    // Handle Ctrl+C
    g_unix_signal_add(SIGINT, on_sigint, &data);

    // Run Main Loop
    // start_ui_update_loop(&data); // Replaced by MainWindow timer
    Gtk::Main::run(window);

    // Finalize GStreamer pipelines and write session metadata FIRST, while
    // the pipelines are still alive.  This must happen before any ROS teardown
    // because the bag writer is closed here.
    window.finalize_session();

    // Stop ROS executor first
    executor->cancel();
    if (ros_thread.joinable()) {
        ros_thread.join();
    }

    // Explicitly reset ROS objects before shutdown to avoid class_loader warnings
    data.sub_record.reset();
    data.pub_recording.reset();
    for (auto s : data.streams) {
        s->ros_publisher = image_transport::CameraPublisher();
    }
    data.it.reset();
    data.timers.clear();
    data.bag_subs.clear();
    data.bag_writer.reset();
    data.node.reset();

    // Finally shutdown ROS 2
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }

    // Clean up session directory if no recording was ever triggered
    if (!data.any_recording_started && !data.session_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(data.session_dir, ec);
        if (!ec) {
            std::cout << "No recording made - removed empty session directory: " << data.session_dir << std::endl;
        } else {
            std::cerr << "Warning: could not remove session directory: " << ec.message() << std::endl;
        }
    } else if (!data.session_dir.empty()) {
        std::cout << "Session finalized: " << data.session_dir << std::endl;
    }

    return 0;
}
