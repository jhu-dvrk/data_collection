#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time

class HighFrequencyPublisher(Node):
    def __init__(self):
        super().__init__('high_freq_publisher')
        self.publisher_ = self.create_publisher(Float64, 'test_topic', 10)
        self.timer = self.create_timer(0.001, self.timer_callback) # 1KHz
        self.start_time = time.time()

    def timer_callback(self):
        msg = Float64()
        msg.data = time.time() - self.start_time
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HighFrequencyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
