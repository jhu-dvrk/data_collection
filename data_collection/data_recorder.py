#!/usr/bin/env python3
import sys
import os
import json
import datetime
import argparse
import cv2
import numpy as np
import time
import subprocess
import signal
import rosbag2_py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, DurabilityPolicy

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QGridLayout, QLabel, QCheckBox,
                             QLineEdit, QFileDialog, QSizePolicy, QOpenGLWidget, QMessageBox,
                             QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    _best_encoder = None

    @classmethod
    def get_best_encoder(cls):
        if cls._best_encoder is not None:
            return cls._best_encoder
        
        # Priority list of encoders: NVENC, VAAPI, software (x264)
        # We test availability using gst-inspect-1.0
        encoders = [
            ("nvv4l2h264enc", "nvv4l2h264enc bitrate={bitrate} preset-level=4 control-rate=1 ! h264parse"),
            ("vaapih264enc", "vaapih264enc bitrate={bitrate_kb} ! h264parse"),
            ("x264enc", "x264enc bitrate={bitrate_kb} speed-preset={speed_preset} key-int-max={key_int_max}")
        ]

        for plugin, pipeline in encoders:
            try:
                result = subprocess.run(["gst-inspect-1.0", plugin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    print(f"Detected GStreamer hardware encoder: {plugin}")
                    cls._best_encoder = (plugin, pipeline)
                    return cls._best_encoder
            except Exception:
                continue
        
        # Fallback to x264enc if all else fails
        cls._best_encoder = ("x264enc", "x264enc bitrate={bitrate_kb} speed-preset={speed_preset} key-int-max={key_int_max}")
        return cls._best_encoder

    def __init__(self, pipeline_config):
        super().__init__()
        self.pipeline_config = pipeline_config
        self.name = pipeline_config['name']
        self._run_flag = True
        
        # Recording state
        self.is_recording = False
        self.rec_requested = False
        self.rec_base_dir = "."
        self.stage_name = None
        self.writer = None
        
        # Track last recording metadata
        self.last_video_name = None
        self.last_json_path = None
        self.last_video_duration = 0.0
        
        # UI state
        self.preview_enabled = True
        self.popup_enabled = False
        
        # Stream info
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # Timestamps
        self.frame_timestamps = []
        self.rec_start_epoch = None
        self.rec_start_gst = None

    def run(self):
        # Support only 'stream' key
        source_str = self.pipeline_config.get('stream')
        if not source_str:
            print(f"Error: No stream defined for pipeline '{self.name}'")
            return
        
        # Construct GStreamer pipeline for OpenCV capture
        # We need BGR frames for OpenCV
        if "appsink" not in source_str:
            read_pipeline = f"{source_str} ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        else:
            read_pipeline = source_str

        # Note: We must specify cv2.CAP_GSTREAMER to force GStreamer backend
        cap = cv2.VideoCapture(read_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print(f"Error: Could not open pipeline '{self.name}'")
            return

        # Attempt to get stream properties
        gst_fps = cap.get(cv2.CAP_PROP_FPS)
        if gst_fps > 0:
            self.fps = gst_fps
        
        while self._run_flag:
            # Only read from pipeline if somebody needs it (recording, preview, or popup)
            # We also check rec_requested so we can start recording even if no preview
            if self.rec_requested or self.is_recording or self.preview_enabled or self.popup_enabled:
                ret, cv_img = cap.read()
                if ret:
                    h, w, ch = cv_img.shape
                    self.width = w
                    self.height = h
                    
                    # Get timestamp from GStreamer source (milliseconds)
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                    # Fallback to system time if GStreamer cannot provide position (common for live streams)
                    if ts <= 0:
                        ts = time.time() * 1000.0

                    # Handle Recording State Changes (using clean frame)
                    self._handle_recording(cv_img, ts)

                    # Add blinking red circle if recording
                    if self.writer:
                        # Blink every 0.5 seconds (on/off cycle 1 sec)
                        if int(time.time() * 2) % 2 == 0:
                            # Draw red circle (BGR: 0, 0, 255) at top-left
                            cv2.circle(cv_img, (30, 30), 10, (0, 0, 255), -1)

                    if self.preview_enabled or self.popup_enabled:
                        self.change_pixmap_signal.emit(cv_img)
                else:
                    # Loop ended or error
                    # Short sleep to avoid busy loop if error persists
                    self.msleep(10)
            else:
                # No one needs the frame, avoid conversion by not calling cap.read()
                self.msleep(100)

        # Cleanup
        cap.release()
        self._stop_writer()
            
    def _handle_recording(self, frame, timestamp):
        # Check if we need to start recording
        if self.rec_requested and not self.is_recording:
             if self.pipeline_config.get('record', True):
                 self._start_writer()
             else:
                 print(f"Skipping recording for video {self.name} (record=false)")
                 # We set is_recording to True to suppress further checks until stopped
                 self.is_recording = True 
                 # But we must mark that no writer exists so we don't try to write
                 pass

        # Check if we need to stop recording
        elif not self.rec_requested and self.is_recording:
            self._stop_writer()

        # Write frame if writer exists
        if self.is_recording and self.writer:
            try:
                # Calculate absolute timestamp (nanoseconds since epoch)
                if self.rec_start_epoch is None:
                    self.rec_start_epoch = time.time()
                    self.rec_start_gst = timestamp
                    
                # Calculate nanoseconds using GStreamer stream position delta
                delta_ms = (timestamp - self.rec_start_gst)
                abs_ts_ns = int(self.rec_start_epoch * 1e9) + int(delta_ms * 1e6)
                
                self.frame_timestamps.append(abs_ts_ns)
                
                self.writer.write(frame)
            except Exception as e:
                print(f"Error writing frame for video {self.name}: {e}")

    def _start_writer(self):
        # Use YYMMDD_HHMMSS format for the file timestamp
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        
        self.rec_start_epoch = None
        self.rec_start_gst = None
        # Sanitize name for filename and pipeline use
        safe_name = self.name.replace(" ", "_")

        suffix = f"_{self.stage_name}" if self.stage_name else ""
        # Format: camera_name_YYMMDD_HHMMSS_stagename
        filename = f"{safe_name}_{timestamp}{suffix}.mp4"
        filepath = os.path.join(self.rec_base_dir, filename)
        self.last_video_name = filename
        
        # Prepare JSON timestamps file path
        json_filename = f"{safe_name}_{timestamp}{suffix}.json"
        self.json_filepath = os.path.join(self.rec_base_dir, json_filename)
        self.last_json_path = self.json_filepath
        self.frame_timestamps = []
        
        encoding = self.pipeline_config.get('encoding', {})
        bitrate = encoding.get('bitrate', 10000)
        speed_preset = encoding.get('speed_preset', 5)
        key_int_max = encoding.get('key_int_max', 30)
        
        target_width = encoding.get('width', self.width)
        target_height = encoding.get('height', self.height)
        
        # Get best encoder
        plugin, encoder_template = self.get_best_encoder()
        
        # Construct parameters
        # x264enc/vaapih264enc usually use kbit/sec. NVENC uses bit/sec.
        # We assume 'bitrate' config is in kbit/sec.
        bitrate_kb = int(bitrate)
        bitrate_bits = bitrate_kb * 1000
        
        encoder_str = encoder_template.format(
            bitrate=bitrate_bits,
            bitrate_kb=bitrate_kb,
            speed_preset=int(speed_preset),
            key_int_max=int(key_int_max)
        )

        # Construct writer pipeline
        scale_str = ""
        if target_width != self.width or target_height != self.height:
            scale_str = f"videoscale ! video/x-raw,width={int(target_width)},height={int(target_height)} ! "

        writer_pipeline = (
            f"appsrc ! videoconvert ! "
            f"{scale_str}"
            f"{encoder_str} ! "
            f"mp4mux ! filesink location=\"{filepath}\""
        )
        
        print(f"Start recording video {self.name} to {filepath}")
        
        # 0 is fourcc for custom pipeline in some versions, or 'MP4V' etc.
        # With GStreamer backend, passing 0 usually implies we don't force a codec (pipeline handles it)
        # fps must be provided.
        self.writer = cv2.VideoWriter(writer_pipeline, cv2.CAP_GSTREAMER, 0, float(self.fps), (self.width, self.height))
        
        if not self.writer.isOpened():
             print(f"Failed to open VideoWriter for video {self.name}")
             self.writer = None
        
        self.is_recording = True

    def _stop_writer(self):
        if self.writer:
            duration = 0.0
            if len(self.frame_timestamps) > 1:
                duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            print(f"Stop recording video {self.name} (duration: {duration:.2f}s)")
            self.last_video_duration = duration
            self.writer.release()
            self.writer = None
            
            # Save timestamps to JSON
            if self.json_filepath and self.frame_timestamps:
                import json
                try:
                    data = {
                        "name": self.name,
                        "fps": self.fps,
                        "video_file": os.path.basename(self.json_filepath).replace(".json", ".mp4"),
                        "timestamps_ns": self.frame_timestamps
                    }
                    with open(self.json_filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved {len(self.frame_timestamps)} timestamps to {self.json_filepath}")
                except Exception as e:
                    print(f"Error saving timestamps for {self.name}: {e}")
                    print(f"Error saving timestamps for {self.name}: {e}")
            
        self.is_recording = False
        self.json_filepath = None
        self.frame_timestamps = []

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_recording(self, recording, base_dir, stage_name=None):
        self.rec_base_dir = base_dir
        self.rec_requested = recording
        self.stage_name = stage_name


class VideoPopupWindow(QOpenGLWidget):
    closed = pyqtSignal()

    def __init__(self, name, parent=None):
        super().__init__(parent)
        # Remove window title and add name to tooltip
        self.setWindowTitle("")
        self.setToolTip(name + " (Press ESC to close, b to toggle border)")
        
        # Ensure the window can be resized
        self.setMinimumSize(160, 120)
        
        self.last_image = None
        self.resize(800, 600)

    def set_image(self, qt_image):
        # We store the image. Since it was copied in the widget, it's safe.
        self.last_image = qt_image
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_B:
            # Toggle window borders
            if self.windowFlags() & Qt.FramelessWindowHint:
                self.setWindowFlags(self.windowFlags() & ~Qt.FramelessWindowHint)
            else:
                self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
            self.show() # Need to call show() after changing window flags
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.last_image:
            # Scale rect maintaining aspect ratio
            img_size = self.last_image.size()
            img_size.scale(self.size(), Qt.KeepAspectRatio)
            
            target = QRect(0, 0, img_size.width(), img_size.height())
            target.moveCenter(self.rect().center())
            painter.drawImage(target, self.last_image)
        painter.end()

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


class VideoWidget(QOpenGLWidget):
    popup_closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        
        self.popup_window = None
        self.current_image = None
        self.preview_enabled = True

    def set_preview_enabled(self, enabled):
        self.preview_enabled = enabled
        self.update()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # Convert BGR to RGB (OpenCV conversion done once here)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        
        # Create QImage and copy data once
        # Copy is still needed because rgb_img is a temporary numpy array
        self.current_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        # Trigger repaint of this widget (OpenGL) if preview enabled
        if self.preview_enabled:
            self.update()
        
        # Update popup if it exists and is visible
        if self.popup_window and self.popup_window.isVisible():
            self.popup_window.set_image(self.current_image)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.preview_enabled and self.current_image:
            # Scale rect maintaining aspect ratio
            img_size = self.current_image.size()
            img_size.scale(self.size(), Qt.KeepAspectRatio)
            
            target = QRect(0, 0, img_size.width(), img_size.height())
            target.moveCenter(self.rect().center())
            painter.drawImage(target, self.current_image)
        painter.end()

    def set_popup_visible(self, visible, name=""):
        if visible:
            if not self.popup_window:
                self.popup_window = VideoPopupWindow(name)
                self.popup_window.closed.connect(self.on_popup_closed)
            self.popup_window.show()
        else:
            if self.popup_window:
                self.popup_window.hide()

    def on_popup_closed(self):
        self.popup_closed.emit()

class RecorderWindow(QMainWindow):
    def __init__(self, config_files, joy_topic=None):
        super().__init__()
        self.setWindowTitle("Video Recorder (OpenCV/GStreamer)")
        self.config = self.load_configs(config_files)
        self.joy_topic = joy_topic
        self.threads = []
        self.video_widgets = []
        self.is_recording = False
        self.bag_process = None
        self.bag_start_time = None
        self.last_bag_name = None
        self.last_bag_duration = 0.0
        self.last_bag_message_count = 0
        self.ignore_rosbag_warnings = False

        # ROS2 Setup
        self.ros_node = None
        self.init_ros()

        # Update Loop
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_loop)
        self.update_timer.start(50) # 20Hz
        
        self.base_dir = self.config.get("data_directory", ".")
        if not os.path.exists(self.base_dir):
            try:
                os.makedirs(self.base_dir)
            except OSError as e:
                print(f"Error creating directory {self.base_dir}: {e}")

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Directory Selection
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Data Directory:"))
        self.txt_directory = QLineEdit(self.base_dir)
        self.txt_directory.setReadOnly(True)
        dir_layout.addWidget(self.txt_directory)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.btn_browse)
        
        layout.addLayout(dir_layout)
        
        # Middle Part: Videos on left, Stages on right
        content_layout = QHBoxLayout()

        # Grid for videos
        self.grid_layout = QGridLayout()
        content_layout.addLayout(self.grid_layout, 4) # Give more weight to videos
        
        # Stages list
        self.list_stages = QListWidget()
        stages = self.config.get("stages", [])
        self.list_stages.addItems(stages)
        if stages:
            stages_container = QWidget()
            stages_layout = QVBoxLayout(stages_container)
            stages_layout.addWidget(QLabel("Stages:"))
            self.list_stages.setCurrentRow(0)
            self.list_stages.setFixedWidth(150)
            stages_layout.addWidget(self.list_stages)
            content_layout.addWidget(stages_container)
        
        layout.addLayout(content_layout, 1)
        
        # Bottom bar for Controls and Quit
        bottom_layout = QHBoxLayout()
        
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setMinimumWidth(200)
        self.btn_record.clicked.connect(self.toggle_recording)
        bottom_layout.addWidget(self.btn_record)
        
        self.lbl_bag_status = QLabel("Rosbag: Idle")
        self.lbl_bag_status.setStyleSheet("color: gray; margin-left: 10px;")
        bottom_layout.addWidget(self.lbl_bag_status)
        
        bottom_layout.addStretch()
        
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)
        bottom_layout.addWidget(self.btn_quit)
        layout.addLayout(bottom_layout)
        
        self.init_pipelines()
        self.update_record_button_state()

    def update_record_button_state(self):
        # Check if any video is selected for recording
        any_video_record = any(chk.isChecked() for _, chk in self.threads)
        
        # Check if any ROS topics are available
        any_ros_topics = len(self.config.get("ros_topics", [])) > 0
        
        can_record = any_video_record or any_ros_topics
        self.btn_record.setEnabled(can_record)
        if not can_record:
            self.btn_record.setToolTip("No video streams selected for recording and no ROS topics configured")
        else:
            self.btn_record.setToolTip("")

    def init_ros(self):
        # We need rclpy.init() to be called. We can do it here or in main.
        # Check if already initialized? 
        if not rclpy.ok():
            rclpy.init(args=sys.argv)
            
        self.ros_node = rclpy.create_node('data_recorder')
        
        # Subscriber "record" (Bool)
        self.sub_record = self.ros_node.create_subscription(
            Bool, '~/record', self.ros_record_callback, 10
        )
        
        # Joy Subscriber
        if self.joy_topic:
             print(f"Subscribing to Joy topic: {self.joy_topic}")
             self.sub_joy = self.ros_node.create_subscription(
                 Joy, self.joy_topic, self.joy_callback, 10
             )

        # QoS for recording (Transient Local)
        qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Publisher "recording" (Bool)
        self.pub_recording = self.ros_node.create_publisher(
            Bool, '~/recording', qos_profile
        )

    def update_loop(self):
        # ROS Spin
        if self.ros_node and rclpy.ok():
            rclpy.spin_once(self.ros_node, timeout_sec=0)
            
        # Check Rosbag Process
        if self.bag_process:
            ret = self.bag_process.poll()
            if ret is None:
                self.lbl_bag_status.setText("Rosbag: Recording")
                self.lbl_bag_status.setStyleSheet("color: green; font-weight: bold; margin-left: 10px;")
            else:
                self.lbl_bag_status.setText(f"Rosbag: Exit ({ret})")
                self.lbl_bag_status.setStyleSheet("color: red; font-weight: bold; margin-left: 10px;")
        else:
             ros_topics = self.config.get("ros_topics", [])
             if ros_topics:
                 self.lbl_bag_status.setText(f"Rosbag: Ready ({len(ros_topics)} topics)")
                 self.lbl_bag_status.setStyleSheet("color: gray; margin-left: 10px;")
             else:
                 self.lbl_bag_status.setText("Rosbag: Disabled")
                 self.lbl_bag_status.setStyleSheet("color: gray; margin-left: 10px;")

    def ros_record_callback(self, msg):
        should_record = msg.data
        if should_record and not self.is_recording:
            print("ROS: Starting recording...")
            self.toggle_recording()
        elif not should_record and self.is_recording:
            print("ROS: Stopping recording...")
            self.toggle_recording()

    def joy_callback(self, msg):
        try:
            # Check for button 0
            if len(msg.buttons) > 0:
                val = msg.buttons[0]
                if val == 1 and not self.is_recording:
                    print("Joy: Starting recording (button 0 pressed)")
                    self.toggle_recording()
                elif val == 0 and self.is_recording:
                    print("Joy: Stopping recording (button 0 released)")
                    self.toggle_recording()
        except Exception as e:
            print(f"Error in joy callback: {e}")

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory", self.base_dir)
        if dir_path:
            self.base_dir = dir_path
            self.txt_directory.setText(dir_path)

    def load_configs(self, paths):
        merged_config = {
            "data_directory": "data",
            "pipelines": [],
            "ros_topics": [],
            "stages": []
        }
        
        # Helper to find pipeline by name in list
        def get_pipeline_index(name, pipelines):
            for i, p in enumerate(pipelines):
                if p.get("name") == name:
                    return i
            return -1
        
        # If a single string is passed, wrap it in a list
        if isinstance(paths, str):
            paths = [paths]
            
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: Configuration file not found: {path}")
                continue
                
            try:
                with open(path, 'r') as f:
                    cfg = json.load(f)
                    # Merge data_directory (last one wins)
                    if "data_directory" in cfg:
                        merged_config["data_directory"] = cfg["data_directory"]
                    
                    # Merge ros_topics (deduplicated)
                    if "ros_topics" in cfg and isinstance(cfg["ros_topics"], list):
                        current_topics = set(merged_config["ros_topics"])
                        for t in cfg["ros_topics"]:
                            if t not in current_topics:
                                merged_config["ros_topics"].append(t)
                                current_topics.add(t)

                    # Merge stages (deduplicated)
                    if "stages" in cfg and isinstance(cfg["stages"], list):
                        current_stages = set(merged_config["stages"])
                        for s in cfg["stages"]:
                            if s not in current_stages:
                                merged_config["stages"].append(s)
                                current_stages.add(s)

                    # Append pipelines, renaming duplicates
                    if "pipelines" in cfg and isinstance(cfg["pipelines"], list):
                        for p_new in cfg["pipelines"]:
                            if "name" not in p_new:
                                merged_config["pipelines"].append(p_new)
                                continue
                                
                            original_name = p_new["name"]
                            name = original_name
                            counter = 2
                            
                            # Check if name exists in currently merged pipelines and find a unique name
                            while get_pipeline_index(name, merged_config["pipelines"]) >= 0:
                                name = f"{original_name}_{counter}"
                                counter += 1
                                
                            p_new["name"] = name
                            merged_config["pipelines"].append(p_new)
                                
            except Exception as e:
                print(f"Error loading config {path}: {e}")

        return merged_config

    def init_pipelines(self):
        cols = 2
        for i, p_config in enumerate(self.config.get('pipelines', [])):
            row = i // cols
            col = i % cols
            
            stream_name = p_config.get('name', f"Stream {i}")
            # Label and Checkbox
            lbl = QLabel(stream_name)
            lbl.setStyleSheet("font-weight: bold;")
            
            chk_preview = QCheckBox("Preview")
            chk_preview.setChecked(True)
            chk_popup = QCheckBox("Popup")            
            chk_record = QCheckBox("Record")

            # Header layout
            header_layout = QHBoxLayout()
            header_layout.addWidget(lbl)
            header_layout.addWidget(chk_record)
            header_layout.addWidget(chk_preview)
            header_layout.addWidget(chk_popup)
            header_layout.addStretch()

            # Video Widget
            vw = VideoWidget()
            vw.setMinimumSize(320, 240)
            self.video_widgets.append(vw)

            # Thread
            th = VideoThread(p_config)
            th.change_pixmap_signal.connect(vw.update_image)
            th.start()
            self.threads.append((th, chk_record))

            # Connect checkboxes
            chk_preview.toggled.connect(vw.set_preview_enabled)
            chk_preview.toggled.connect(lambda checked, t=th: setattr(t, 'preview_enabled', checked))

            def update_popup(checked, w=vw, t=th, name=stream_name):
                w.set_popup_visible(checked, name)
                t.popup_enabled = checked
            
            chk_popup.toggled.connect(update_popup)
            vw.popup_closed.connect(lambda c=chk_popup: c.setChecked(False))

            # Set initial state from config
            chk_record.setChecked(p_config.get('record', True))
            chk_popup.setChecked(p_config.get('popup', False))
            
            # Container
            v_layout = QVBoxLayout()
            v_layout.addLayout(header_layout)
            v_layout.addWidget(vw)
            
            container = QWidget()
            container.setLayout(v_layout)
            self.grid_layout.addWidget(container, row, col)
            
            # Update button state when checkbox toggled
            chk_record.toggled.connect(self.update_record_button_state)

        # Publish state to ROS
        if self.ros_node:
            msg = Bool()
            msg.data = self.is_recording
            self.pub_recording.publish(msg)

    def verify_rosbag(self, bag_path):
        try:
            info = rosbag2_py.Info()
            metadata = info.read_metadata(bag_path, "")
            
            # Get duration from metadata
            d = metadata.duration
            if hasattr(d, 'nanoseconds'):
                self.last_bag_duration = d.nanoseconds / 1e9
            elif hasattr(d, 'total_seconds'):
                self.last_bag_duration = d.total_seconds()
            else:
                try:
                    # Some versions return duration in nanoseconds as int
                    self.last_bag_duration = float(d) / 1e9
                except (ValueError, TypeError):
                    self.last_bag_duration = 0.0
            
            self.last_bag_message_count = getattr(metadata, 'message_count', 0)

            topic_counts = {}
            for topic_info in metadata.topics_with_message_count:
                topic_counts[topic_info.topic_metadata.name] = topic_info.message_count
                
            requested_topics = self.config.get("ros_topics", [])
            missing_topics = [t for t in requested_topics if t not in topic_counts]
            
            msg_text = f"Rosbag Verification ({os.path.basename(bag_path)}):\n\n"
            if topic_counts:
                for topic, count in topic_counts.items():
                    msg_text += f" - {topic}: {count} samples\n"
            else:
                msg_text += " No topics recorded.\n"
                
            if missing_topics:
                if self.ignore_rosbag_warnings:
                    return

                msg_text += f"\nWARNING: Missing topics:\n"
                for t in missing_topics:
                    msg_text += f" - {t}\n"
                
                msg_box = QMessageBox(QMessageBox.Warning, "Rosbag Verification", msg_text, QMessageBox.Ok, self)
                chk_ignore = QCheckBox("Ignore future warnings")
                msg_box.setCheckBox(chk_ignore)
                msg_box.exec_()
                
                if chk_ignore.isChecked():
                    self.ignore_rosbag_warnings = True
            else:
                QMessageBox.information(self, "Rosbag Verification", msg_text)
        except Exception as e:
            QMessageBox.critical(self, "Rosbag Error", f"Failed to verify rosbag at {bag_path}: {e}")

    def save_index(self):
        if not hasattr(self, 'current_stage_dir') or not self.current_stage_dir:
            return
            
        try:
            videos = []
            for th, _ in self.threads:
                if th.last_video_name:
                    v_duration = 0.0
                    # Compute duration from the sidecar JSON file
                    if th.last_json_path and os.path.exists(th.last_json_path):
                        try:
                            with open(th.last_json_path, 'r') as f:
                                v_data = json.load(f)
                                ts_ns = v_data.get("timestamps_ns", [])
                                if len(ts_ns) > 1:
                                    v_duration = (ts_ns[-1] - ts_ns[0]) / 1e9
                                else:
                                    v_duration = 0.0
                        except Exception as e:
                            print(f"Error computing duration for {th.last_video_name}: {e}")
                            v_duration = th.last_video_duration # Fallback
                    
                    videos.append({
                        "file": th.last_video_name,
                        "duration": v_duration
                    })
            
            rosbag_data = None
            if self.last_bag_name:
                rosbag_data = {
                    "name": self.last_bag_name,
                    "duration": self.last_bag_duration,
                    "message_count": self.last_bag_message_count
                }
            
            index_data = {
                "videos": videos,
                "rosbag": rosbag_data
            }
            
            index_path = os.path.join(self.current_stage_dir, "index.json")
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            print(f"Saved session index to {index_path}")
        except Exception as e:
            print(f"Error saving index.json: {e}")

    def toggle_recording(self):
        if not self.is_recording:
            # Start
            self.btn_browse.setEnabled(False)
            self.list_stages.setEnabled(False)
            
            # Create a subdirectory based on the date and time
            stage_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Append stage name if available
            dir_name = stage_timestamp
            stage_name = None
            current_item = self.list_stages.currentItem()
            if current_item:
                stage_name = current_item.text().replace(" ", "_")
                dir_name = f"{stage_timestamp}_{stage_name}"
                
            stage_dir = os.path.join(self.base_dir, dir_name)
            self.current_stage_dir = stage_dir
            try:
                os.makedirs(stage_dir, exist_ok=True)
                print(f"Created directory: {stage_dir}")
            except OSError as e:
                print(f"Error creating directory {stage_dir}: {e}")
                # Fallback to base_dir if stage_dir creation fails
                stage_dir = self.base_dir

            # Start ROS Bag first (can be slow to initialize)
            ros_topics = self.config.get("ros_topics", [])
            self.last_bag_path = None
            self.last_bag_name = None
            self.last_bag_duration = 0.0
            self.last_bag_message_count = 0
            if ros_topics:
                try:
                    bag_name = f"rosbag_{dir_name}"
                    bag_path = os.path.join(stage_dir, bag_name)
                    self.last_bag_path = bag_path
                    self.last_bag_name = bag_name
                    self.bag_start_time = time.time()
                    
                    # Use --topics to specify the topic list explicitely
                    cmd = ["ros2", "bag", "record", "-o", bag_path, "--topics"] + ros_topics
                    print(f"Starting rosbag: {' '.join(cmd)}")
                    # Prevent ros2 bag from capturing stdin (avoids pausing on SPACE)
                    self.bag_process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL) 
                except Exception as e:
                    print(f"Failed to start rosbag: {e}")

            # Start video recordings
            for th, chk in self.threads:
                # Disable checkbox
                chk.setEnabled(False)
                # Update record config based on checkbox
                th.pipeline_config['record'] = chk.isChecked()
                th.set_recording(True, stage_dir, stage_name)

            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: red; color: white;")
            self.is_recording = True
        else:
            # Stop
            for th, chk in self.threads:
                th.set_recording(False, self.base_dir)
                chk.setEnabled(True)
            
            # Stop ROS Bag
            if self.bag_process:
                print("Stopping rosbag...")
                self.bag_process.send_signal(signal.SIGINT)
                try:
                    self.bag_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.bag_process.kill()
                self.bag_process = None
                
                # Verify the bag if path exists
                if hasattr(self, 'last_bag_path') and self.last_bag_path:
                    self.verify_rosbag(self.last_bag_path)

            # Wait for all video threads to finish writing files
            wait_start = time.time()
            while time.time() - wait_start < 3.0:
                any_recording = any(th.is_recording for th, _ in self.threads)
                if not any_recording:
                    break
                time.sleep(0.1)

            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("")
            self.is_recording = False
            self.btn_browse.setEnabled(True)
            self.list_stages.setEnabled(True)
            
            # Advance to next stage (restart if at end)
            current_row = self.list_stages.currentRow()
            if current_row >= 0:
                next_row = (current_row + 1) % self.list_stages.count()
                self.list_stages.setCurrentRow(next_row)

            # Save index.json for all files in the stage directory
            self.save_index()

        # Publish updated state to ROS
        if self.ros_node:
            msg = Bool()
            msg.data = self.is_recording
            self.pub_recording.publish(msg)

    def closeEvent(self, event):
        # 1. Signal all threads to stop (without waiting yet)
        for th, _ in self.threads:
            th._run_flag = False
            
        # 2. Stop ROS Bag immediately if recording
        if self.bag_process:
            print("Stopping rosbag during exit...")
            self.bag_process.send_signal(signal.SIGINT)
            try:
                self.bag_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.bag_process.kill()
            self.bag_process = None

        # 3. Close all popup windows
        for vw in self.video_widgets:
            if vw.popup_window:
                vw.popup_window.close()
        
        # 4. Wait for all threads to finish
        for th, _ in self.threads:
            th.wait()

        # 5. Final ROS cleanup
        if self.ros_node:
            self.ros_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Video Recorder")
    parser.add_argument("-c", "--config", help="Path to JSON configuration file", action="append", required=True)
    parser.add_argument("-p", "--joy-topic", help="ROS Joy topic for recording control (button 0)", default=None)
    args = parser.parse_args()

    config_files = args.config

    app = QApplication(sys.argv)
    
    # Allow Ctrl+C to close the application correctly
    # We need a timer to periodically yield control to the Python interpreter 
    # so it can process signals.
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(100)
    
    def handle_sigint(sig, frame):
        print("\nInterrupt received, shutting down...")
        window.close() # This will trigger closeEvent and its cleanup
        app.quit()
        
    signal.signal(signal.SIGINT, handle_sigint)
    
    window = RecorderWindow(config_files, joy_topic=args.joy_topic)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
