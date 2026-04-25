#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
import cv2
import numpy as np
from cv_bridge import CvBridge
import time
import os

class VideoPerformanceSource(Node):
    def __init__(self):
        super().__init__('video_performance_source')
        
        # Parameters
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 60)
        self.declare_parameter('topic', '/synthetic_video')
        self.declare_parameter('frame_topic', '/synthetic_frame_number')
        self.declare_parameter('stereo', False)
        self.declare_parameter('gstreamer', False)
        self.declare_parameter('left_shm_path', '/tmp/dvrk_left.sock')
        self.declare_parameter('right_shm_path', '/tmp/dvrk_right.sock')

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.topic_name = self.get_parameter('topic').value
        self.frame_topic_name = self.get_parameter('frame_topic').value
        self.stereo = self.get_parameter('stereo').value
        self.gstreamer_enabled = self.get_parameter('gstreamer').value

        # Publishers
        if self.stereo:
            self.left_pub = self.create_publisher(Image, self.topic_name + '/left', 10)
            self.right_pub = self.create_publisher(Image, self.topic_name + '/right', 10)
        else:
            self.image_pub = self.create_publisher(Image, self.topic_name, 10)
            
        self.frame_num_pub = self.create_publisher(Int64, self.frame_topic_name, 10)
        
        self.bridge = CvBridge()
        self.frame_count = 0
        
        # GStreamer Writers
        if self.gstreamer_enabled:
            self.left_writer = self._create_gstreamer_writer(self.get_parameter('left_shm_path').value)
            if self.stereo:
                self.right_writer = self._create_gstreamer_writer(self.get_parameter('right_shm_path').value)
            self.get_logger().info('GStreamer output enabled')

        # Timer for publishing
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Starting synthetic video source at {self.fps} FPS, {self.width}x{self.height}, stereo={self.stereo}')

    def _create_gstreamer_writer(self, path):
        # Remove socket file if it already exists to avoid errors
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
            
        pipeline = (
            f"appsrc ! videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"shmsink socket-path={path} wait-for-connection=false sync=true"
        )
        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(self.fps), (self.width, self.height))
        if not writer.isOpened():
            self.get_logger().error(f'Failed to open GStreamer writer for {path}')
        return writer

    def generate_frame(self, eye=None):
        # Create a black image
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add frame number text
        eye_text = f" [{eye.upper()}]" if eye else ""
        text = f'Frame: {self.frame_count}{eye_text}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 5
        color = (255, 255, 255) # White
        
        # Get text size to center or position it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Optional: add a moving element to easily see motion
        rect_size = 100
        shift = 0
        if eye == 'left':
            shift = -20
        elif eye == 'right':
            shift = 20
            
        rect_x = (self.frame_count * 5 + shift) % (self.width - rect_size)
        rect_y = 100
        
        rect_color = (0, 255, 0) # Green
        if eye == 'left':
            rect_color = (255, 200, 100) # Different color for left
        elif eye == 'right':
            rect_color = (100, 200, 255) # Different color for right

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_size, rect_y + rect_size), rect_color, -1)
        
        return frame

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        
        if self.stereo:
            left_frame = self.generate_frame('left')
            right_frame = self.generate_frame('right')
            
            # Publish ROS messages
            left_msg = self.bridge.cv2_to_imgmsg(left_frame, encoding='bgr8')
            left_msg.header.stamp = now
            left_msg.header.frame_id = 'synthetic_left'
            self.left_pub.publish(left_msg)
            
            right_msg = self.bridge.cv2_to_imgmsg(right_frame, encoding='bgr8')
            right_msg.header.stamp = now
            right_msg.header.frame_id = 'synthetic_right'
            self.right_pub.publish(right_msg)
            
            # Write to GStreamer
            if self.gstreamer_enabled:
                if self.left_writer.isOpened():
                    self.left_writer.write(left_frame)
                if self.right_writer.isOpened():
                    self.right_writer.write(right_frame)
        else:
            frame = self.generate_frame()
            
            # Publish ROS message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = now
            msg.header.frame_id = 'synthetic_frame'
            self.image_pub.publish(msg)
            
            # Write to GStreamer
            if self.gstreamer_enabled:
                if self.left_writer.isOpened():
                    self.left_writer.write(frame)
        
        # Publish frame number separately for performance tracking
        frame_num_msg = Int64()
        frame_num_msg.data = self.frame_count
        self.frame_num_pub.publish(frame_num_msg)
        
        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = VideoPerformanceSource()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
