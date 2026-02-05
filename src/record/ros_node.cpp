#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <std_msgs/msg/bool.hpp>
#include "ros_node.hpp"
#include "ui.hpp"

// Since context.hpp only has forward decls, we need to include headers here
// But wait, rosbag2_storage might not be included in ui.hpp either.
// We should check what context.hpp modifications broke.
// context.hpp has pointers to GtkWidget, GstElement.
// ros_node.cpp needs to know the layout of AppData.
// Compiling ros_node.cpp handles it because we include headers here.

void ros_record_callback(const std_msgs::msg::Bool::SharedPtr msg, AppData* ad) {
    bool should_record = msg->data;
    std::lock_guard<std::mutex> lock(ad->data_mutex);
    if (should_record != ad->global_recording) {
        g_idle_add(toggle_recording_idle, ad);
    }
}

void open_bag_writer(AppData* ad, const std::string& path) {
    if (ad->ros_topics.empty()) return;

    if (ad->bag_writer) {
        ad->bag_writer.reset();
    }

    // Create new bag
    ad->bag_writer = std::make_unique<rosbag2_cpp::Writer>();

    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = path;
    storage_options.storage_id = "mcap";

    rosbag2_cpp::ConverterOptions converter_options;

    try {
        ad->bag_writer->open(storage_options, converter_options);

        // Reset message counter
        std::lock_guard<std::mutex> lock(ad->data_mutex);
        ad->bag_messages_recorded = 0;

        auto topic_names_and_types = ad->node->get_topic_names_and_types();
        for (const auto& topic : ad->subscribed_topics) {
             if (topic_names_and_types.count(topic)) {
                 std::string type = topic_names_and_types[topic][0];
                 rosbag2_storage::TopicMetadata tm;
                 tm.name = topic;
                 tm.type = type;
                 tm.serialization_format = "cdr";
                 ad->bag_writer->create_topic(tm);
             }
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to open bag: " << e.what() << std::endl;
        ad->bag_writer.reset();
    }
}

void close_bag_writer(AppData* ad) {
    // Note: Mutex handling should ideally be done by caller or here.
    // Since toggle_recording operates on UI thread, and writes happen on ROS thread,
    // we have a race condition on bag_writer.
    // However, toggle_recording sets global_recording=false first?
    // The callback checks global_recording && bag_writer under lock.
    // So if we take lock here, we are safe.

    std::lock_guard<std::mutex> lock(ad->data_mutex);
    if (ad->bag_writer) {
        ad->bag_writer.reset();
    }
}

void setup_ros_monitoring(AppData* ad) {
    if (ad->ros_topics.empty()) return;

    auto timer_cb = [ad]() {
        if (ad->ros_topics.empty()) return;

        auto topic_names_and_types = ad->node->get_topic_names_and_types();

        // Update topic count stats
        {
             std::lock_guard<std::mutex> lock(ad->data_mutex);
             // We can check how many requested topics we found
             int found = 0;
             for (const auto& t : ad->ros_topics) {
                 if (topic_names_and_types.count(t) > 0) found++;
             }
             ad->bag_topics_found = found;
        }

        std::lock_guard<std::mutex> lock(ad->data_mutex);

        for (const auto& topic : ad->ros_topics) {
            if (ad->subscribed_topics.count(topic)) continue;

            if (topic_names_and_types.count(topic)) {
                std::string type = topic_names_and_types[topic][0];
                // Removed print found topic

                auto callback = [ad, topic, type](std::shared_ptr<rclcpp::SerializedMessage> msg) {
                    std::lock_guard<std::mutex> lock(ad->data_mutex);
                    if (ad->global_recording && ad->bag_writer) {
                        try {
                             // Attempt to write. Topic creation is done at open,
                             // but if this is a NEW topic discovered during recording,
                             // we must ensure it exists.
                             // rosbag2_cpp checks existence.
                             // We blindly try to create it if we suspect it's new?
                             // Optimization: Just Try-Catch write.
                             // Actually, write() crashes/throws if topic missing? Yes.

                             // Let's rely on open_bag_writer for existing ones.
                             // For late joiners:
                             // We can't query bag_writer for existence cheaply.
                             // We'll just create_topic just in case?
                             rosbag2_storage::TopicMetadata tm;
                             tm.name = topic;
                             tm.type = type;
                             tm.serialization_format = "cdr";
                             try { ad->bag_writer->create_topic(tm); } catch(...) {} // Ignore "already exists"

                             ad->bag_writer->write(msg, topic, type, ad->node->now());
                             ad->bag_messages_recorded++;
                        } catch (const std::exception& e) {
                             // std::cerr << "Write failed: " << e.what() << std::endl;
                        }
                    }
                };

                try {
                    auto sub = ad->node->create_generic_subscription(
                        topic, type, rclcpp::QoS(10), callback
                    );
                    ad->bag_subs.push_back(sub);
                    ad->subscribed_topics.insert(topic);
                } catch (const std::exception& e) {
                    std::cerr << "Failed to subscribe to " << topic << ": " << e.what() << std::endl;
                }
            }
        }
    };

    // Initial check
    timer_cb();

    // Create timer (1s period)
    auto timer = ad->node->create_wall_timer(std::chrono::seconds(1), timer_cb);
    ad->timers.push_back(timer);
}
