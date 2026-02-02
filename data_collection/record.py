#!/usr/bin/env python3
import rclpy
# Initialize ROS immediately
if not rclpy.ok():
    rclpy.init(args=[])

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
import shutil
import rosbag2_py
import queue

from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from rclpy.qos import QoSProfile, DurabilityPolicy

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QGridLayout, QLabel, QCheckBox,
                             QLineEdit, QFileDialog, QSizePolicy, QOpenGLWidget, QMessageBox,
                             QListWidget, QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QRect, QUrl, QIODevice
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtMultimedia import QAudioRecorder, QAudioEncoderSettings, QMultimedia, QAudioInput, QAudioFormat, QAudioDeviceInfo

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
            ("nvh264enc", "nvh264enc bitrate={bitrate_kb} ! h264parse"),
            ("nvv4l2h264enc", "nvv4l2h264enc bitrate={bitrate_bits} preset-level=4 control-rate=1 ! h264parse"),
            ("vaapih264enc", "vaapih264enc bitrate={bitrate_kb} ! h264parse"),
            ("x264enc", "x264enc bitrate={bitrate_kb} speed-preset={speed_preset} key-int-max={key_int_max}")
        ]

        for plugin, video in encoders:
            try:
                result = subprocess.run(["gst-inspect-1.0", plugin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    print(f"Detected GStreamer hardware encoder: {plugin}", flush=True)
                    cls._best_encoder = (plugin, video)
                    return cls._best_encoder
            except Exception:
                continue
        
        # Fallback to x264enc if all else fails
        cls._best_encoder = ("x264enc", "x264enc bitrate={bitrate_kb} speed-preset={speed_preset} key-int-max={key_int_max}")
        return cls._best_encoder

    def __init__(self, video_config):
        super().__init__()
        self.video_config = video_config
        self.name = video_config['name']
        self._run_flag = True
        
        # Capture and Recording
        self.cap = None
        self.record_proc = None
        self.session_video_path = None
        self.json_filepath = None
        
        # Recording state
        self.is_recording = False
        self.rec_requested = False
        self.rec_base_dir = "."
        self.stage_name = ""
        self.preview_enabled = True
        
        # Options
        self.time_watermark = video_config.get('time_overlay', False)
        
        # Internal state
        self._restart_cap = False
        self.frame_timestamps = []
        self.rec_start_epoch = None
        
        # Performance tracking
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = None
        self.fps_actual = 0.0
        self.fps_samples = []
        self.width = 0
        self.height = 0
        
        # Metadata for session index
        self.last_video_name = None
        self.last_json_path = None
        self.last_video_duration = 0.0

    def _get_capture_string(self):
        """Construct a GStreamer capture string for OpenCV."""
        source_str = self.video_config.get('stream')
        if not source_str:
            return None
            
        # Optional overlay in preview
        overlay_str = ""
        if self.time_watermark:
            now = datetime.datetime.now().astimezone()
            now_iso = now.isoformat(timespec='milliseconds')
            overlay_str = (
                " ! videobox bottom=-30 "
                f" ! timeoverlay time-mode=buffer-time show-times-as-dates=true datetime-epoch={now_iso} datetime-format=\"%Y-%m-%d %H:%M:%S.%f\" valignment=bottom halignment=left font-desc=\"Sans, 10\" ypad=6 "
            )
            
        # Return a GStreamer pipeline ending in appsink for OpenCV
        return f"{source_str} {overlay_str} ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"

    def run(self):
        """Main thread loop using cv2.VideoCapture."""
        while self._run_flag:
            pipeline_str = self._get_capture_string()
            if not pipeline_str:
                self.msleep(1000)
                continue
                
            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.name}")
                self.msleep(2000)
                continue
                
            self._restart_cap = False
            while self._run_flag and not self._restart_cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Track performance
                self.frame_count += 1
                if self.frame_count == 1:
                    self.height, self.width = frame.shape[:2]
                current_time = time.time()
                if self.last_frame_time is not None:
                    delta = current_time - self.last_frame_time
                    if delta > 0:
                        self.fps_samples.append(1.0 / delta)
                        if len(self.fps_samples) > 30: self.fps_samples.pop(0)
                        self.fps_actual = np.mean(self.fps_samples)
                self.last_frame_time = current_time
                
                # Check recording state
                if self.rec_requested and not self.is_recording:
                    self._start_external_recording()
                elif not self.rec_requested and self.is_recording:
                    self._stop_external_recording()
                    
                if self.is_recording:
                    self.frame_timestamps.append({
                        'ts': current_time,
                        'frame': self.frame_count
                    })
                    # Blink red circle
                    if int(time.time() * 2) % 2 == 0:
                        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                
                if self.preview_enabled:
                    self.change_pixmap_signal.emit(frame)
                    
            self.cap.release()
            self._stop_external_recording()
            if not self._run_flag:
                break
                
    def _start_external_recording(self):
        """Launch a separate gst-launch-1.0 process for recording."""
        source_str = self.video_config.get('stream')
        encoding = self.video_config.get('encoding', {})
        bitrate_kb = int(encoding.get('bitrate', 10000))
        bitrate_bits = bitrate_kb * 1000
        
        # File paths
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c if c.isalnum() else "_" for c in self.name])
        suffix = f"_{self.stage_name}" if self.stage_name else ""
        
        video_filename = f"video_{safe_name}_{timestamp}{suffix}.mp4"
        self.session_video_path = os.path.abspath(os.path.join(self.rec_base_dir, video_filename))
        
        json_filename = f"{safe_name}_{timestamp}{suffix}.json"
        self.json_filepath = os.path.join(self.rec_base_dir, json_filename)
        
        # Encoder
        plugin, template = self.get_best_encoder()
        encoder_str = template.format(
            bitrate_bits=bitrate_bits,
            bitrate_kb=bitrate_kb,
            speed_preset=int(encoding.get('speed_preset', 5)),
            key_int_max=int(encoding.get('key_int_max', 30))
        )
        
        # Watermark
        now_iso = ""
        if self.time_watermark:
            now_iso = datetime.datetime.now().astimezone().isoformat(timespec='milliseconds')

        # Build command manually for subprocess
        gst_cmd = [
            "gst-launch-1.0", "-e",
            source_str,
            "!", "queue"
        ]
        
        if self.time_watermark:
            gst_cmd.extend(["!", "videobox", "bottom=-30"])
            gst_cmd.extend(["!", "timeoverlay", "time-mode=buffer-time", "show-times-as-dates=true", 
                           f"datetime-epoch={now_iso}", "datetime-format=%Y-%m-%d %H:%M:%S.%f", 
                           "valignment=bottom", "halignment=left", "font-desc=Sans, 10", "ypad=6"])

        gst_cmd.extend(["!", "videoconvert"])
        
        # Add Encoder and Muxer
        encoder_parts = encoder_str.split("!")
        for part in encoder_parts:
            gst_cmd.append("!")
            gst_cmd.extend(part.strip().split())
            
        gst_cmd.extend(["!", "mp4mux", "!", "filesink", f"location={self.session_video_path}"])
        
        print(f"Starting recording process for {self.name}...")
        try:
            self.record_proc = subprocess.Popen(gst_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.is_recording = True
            self.frame_timestamps = []
            self.rec_start_epoch = time.time()
        except Exception as e:
            print(f"Failed to start recording process: {e}")

    def _stop_external_recording(self):
        """Terminate the recording process."""
        if self.record_proc:
            print(f"Stopping recording process for {self.name}...")
            # Send SIGINT (-e in gst-launch handles this for clean MP4 closing)
            try:
                self.record_proc.send_signal(signal.SIGINT)
                self.record_proc.wait(timeout=5)
            except Exception as e:
                print(f"Error stopping recording process: {e}")
                self.record_proc.kill()
            
            # Store metadata for session index
            if self.session_video_path:
                self.last_video_name = os.path.basename(self.session_video_path)
            self.last_json_path = self.json_filepath
            if self.frame_timestamps:
                self.last_video_duration = self.frame_timestamps[-1]['ts'] - self.frame_timestamps[0]['ts']
            else:
                self.last_video_duration = 0.0

            self.record_proc = None
            self.is_recording = False
            self._save_index_json()

    def _save_index_json(self):
        if not self.json_filepath or not self.frame_timestamps:
            return
        try:
            data = {
                'name': self.name,
                'stream': self.video_config.get('stream'),
                'width': self.width,
                'height': self.height,
                'fps': self.fps_actual,
                'start_time': self.rec_start_epoch,
                'num_frames': len(self.frame_timestamps),
                'session_video': self.session_video_path,
                'frames': self.frame_timestamps
            }
            with open(self.json_filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved index data to {self.json_filepath}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_recording(self, recording, base_dir, stage_name=None):
        self.rec_base_dir = base_dir
        self.rec_requested = recording
        self.stage_name = stage_name or ""

    def msleep(self, ms):
        time.sleep(ms / 1000.0)

    def _stop_gst_recording(self): # Compatibility
        self._stop_external_recording()

    def _start_gst_recording(self): # Compatibility
        self._start_external_recording()

class AudioLevelMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setOrientation(Qt.Horizontal)
        self.bar.setFixedWidth(200)
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 2px;
                background-color: #EEE;
            }
            QProgressBar::chunk {
                background-color: #00cc00;
            }
        """)
        
        self.layout.addWidget(QLabel("Mic:"))
        self.layout.addWidget(self.bar)

        self.audio_input = None
        self.device = None
        self.init_audio()
    
    def init_audio(self):
        format = QAudioFormat()
        format.setSampleRate(8000)
        format.setChannelCount(1)
        format.setSampleSize(16)
        format.setCodec("audio/pcm")
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setSampleType(QAudioFormat.SignedInt)

        info = QAudioDeviceInfo.defaultInputDevice()
        if not info.isFormatSupported(format):
            format = info.nearestFormat(format)

        self.audio_input = QAudioInput(info, format)
        self.device = self.audio_input.start()
        self.device.readyRead.connect(self.process_audio)

    def process_audio(self):
        if not self.audio_input:
            return
            
        bytes_ready = self.audio_input.bytesReady()
        if bytes_ready:
            data = self.device.read(bytes_ready)
            # Calculate RMS amplitude
            # Convert bytes to 16-bit integers
            samples = np.frombuffer(data, dtype=np.int16)
            if len(samples) > 0:
                # Simple peak or rms
                rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
                # Normalize to 0-100 (assuming 16-bit max is 32768)
                # Use log scale for better visualization
                # db = 20 * log10(rms / 32768)
                # Let's map 0 to 32768 to 0-100 linearly for simplicity first, or close
                val = min(100, int((rms / 10000) * 100)) # Gain factor
                self.bar.setValue(val)


class AudioRecorder:
    def __init__(self):
        self.recorder = None

    def start(self, filepath):
        self.recorder = QAudioRecorder()
        
        # Configure for WAV recording (PCM)
        settings = QAudioEncoderSettings()
        settings.setCodec("audio/pcm")
        settings.setQuality(QMultimedia.HighQuality)
        
        self.recorder.setEncodingSettings(settings)
        
        # Set container format explicitly to WAV
        # On Linux/GStreamer backend this corresponds to wavenc
        self.recorder.setContainerFormat("audio/x-wav") 
        
        # Ensure absolute path
        abs_path = os.path.abspath(filepath)
        self.recorder.setOutputLocation(QUrl.fromLocalFile(abs_path))
        
        # Start
        self.recorder.record()
        
        # Check for immediate errors
        if self.recorder.error() != QAudioRecorder.NoError:
             print(f"Error starting audio recording: {self.recorder.errorString()}")
        else:
             print(f"Starting audio recording (Qt): {abs_path}")

    def stop(self):
        if self.recorder:
            print("Stopping audio recording (Qt)...")
            self.recorder.stop()
            self.recorder = None


class VideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
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

class RecordWindow(QMainWindow):
    def __init__(self, config_files, joy_topic=None, external_node=None):
        super().__init__()
        self.setWindowTitle("Video Record (OpenCV/GStreamer)")
        
        self.ros_node = external_node
        
        # Pre-detect best encoder to avoid delay on first record
        VideoThread.get_best_encoder()

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
        
        self.audio_recorder = AudioRecorder()

        # ROS2 Setup
        # Cache whether the installed `ros2 bag record` supports the `--topics` flag
        self._rosbag_topics_flag_cache = None
        # Cache whether mcap storage is available
        self._rosbag_mcap_available_cache = None
        self.init_ros()

        # Update Loop
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_loop)
        self.update_timer.start(50) # 20Hz
        self.base_dir = self.config.get("data_directory", "data")
        if not os.path.exists(self.base_dir):
            try:
                os.makedirs(self.base_dir)
            except OSError as e:
                print(f"Error creating directory {self.base_dir}: {e}")

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Box 1: Data Directory ---
        grp_directory = QGroupBox("Data Directory")
        dir_layout = QHBoxLayout()
        
        self.txt_directory = QLineEdit(self.base_dir)
        self.txt_directory.setReadOnly(True)
        dir_layout.addWidget(self.txt_directory)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.btn_browse)
        
        # Disk space indicator
        self.lbl_disk_space = QLabel("Disk: Checking...")
        self.lbl_disk_space.setStyleSheet("color: gray; margin-left: 10px;")
        dir_layout.addWidget(self.lbl_disk_space)
        
        grp_directory.setLayout(dir_layout)
        main_layout.addWidget(grp_directory)
        
        # --- Box 2: Audio Settings ---
        grp_audio = QGroupBox("Audio Settings")
        audio_layout = QHBoxLayout()
        
        self.chk_audio = QCheckBox("Record Audio (Default Mic)")
        self.chk_audio.setToolTip("Creates a separate 'audio_<timestamp>.wav' file.")
        self.chk_audio.setChecked(self.config.get("record_audio", False))
        audio_layout.addWidget(self.chk_audio)
        
        # Audio Meter
        self.audio_meter = AudioLevelMeter()
        audio_layout.addWidget(self.audio_meter)
        
        audio_layout.addStretch()
        grp_audio.setLayout(audio_layout)
        main_layout.addWidget(grp_audio)

        # --- Box 3: Video Streams ---
        grp_video = QGroupBox("Video Streams")
        video_layout = QHBoxLayout()

        # Grid for videos
        self.grid_layout = QGridLayout()
        video_layout.addLayout(self.grid_layout, 4) # Give more weight to videos
        
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
            video_layout.addWidget(stages_container)
        
        grp_video.setLayout(video_layout)
        main_layout.addWidget(grp_video, 1) # This box expands
        
        # --- Box 4: Controls ---
        grp_controls = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setMinimumWidth(200)
        self.btn_record.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.btn_record)
        
        self.lbl_bag_status = QLabel("Rosbag: Idle")
        self.lbl_bag_status.setStyleSheet("color: gray; margin-left: 10px;")
        controls_layout.addWidget(self.lbl_bag_status)
        
        controls_layout.addStretch()
        
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)
        controls_layout.addWidget(self.btn_quit)
        
        grp_controls.setLayout(controls_layout)
        main_layout.addWidget(grp_controls)
        
        self.init_videos()
        self.update_record_button_state()

    def update_record_button_state(self):
        # Check if any video is selected for recording
        any_video_record = any(entry['chk_record'].isChecked() for entry in self.threads)
        
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
        try:
            if not rclpy.ok():
                rclpy.init(args=[])
        except Exception as e:
            print(f"Failed to initialize rclpy: {e}")

        if not self.ros_node:
            try:
                self.ros_node = rclpy.create_node('record')
            except Exception as e:
                print(f"Failed to create ROS node: {e}")
                return
        
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

    def _rosbag_supports_topics(self):
        """Detect whether the installed `ros2 bag record` command accepts a
        `--topics` option. Result is cached on the instance.
        """
        if self._rosbag_topics_flag_cache is not None:
            return self._rosbag_topics_flag_cache

        try:
            # Run help and inspect output for the `--topics` option
            p = subprocess.run(["ros2", "bag", "record", "--help"], capture_output=True, text=True)
            out = (p.stdout or "") + (p.stderr or "")
            self._rosbag_topics_flag_cache = ('--topics' in out)
        except Exception:
            # If anything goes wrong, assume older style (positional topics)
            self._rosbag_topics_flag_cache = False

        return self._rosbag_topics_flag_cache

    def _rosbag_get_storage(self):
        """Detect available storage format for ros2 bag.
        Prefers mcap if available, otherwise uses sqlite3.
        Result is cached on the instance.
        """
        if self._rosbag_mcap_available_cache is not None:
            return self._rosbag_mcap_available_cache

        try:
            # Run help and check available storage plugins
            p = subprocess.run(["ros2", "bag", "record", "--help"], capture_output=True, text=True)
            out = (p.stdout or "") + (p.stderr or "")
            # Check if mcap is listed as a storage option
            if 'mcap' in out:
                self._rosbag_mcap_available_cache = 'mcap'
                print("Using mcap storage for rosbag")
            else:
                self._rosbag_mcap_available_cache = 'sqlite3'
                print("Using sqlite3 storage for rosbag (mcap not available)")
        except Exception:
            # Fallback to sqlite3
            self._rosbag_mcap_available_cache = 'sqlite3'

        return self._rosbag_mcap_available_cache

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
        
        # Update disk space indicator
        free_gb = self.get_disk_space(self.base_dir)
        if free_gb is not None:
            if free_gb < 1.0:
                # Critical - red
                self.lbl_disk_space.setText(f"Disk: {free_gb:.1f} GB (CRITICAL!)")
                self.lbl_disk_space.setStyleSheet("color: red; font-weight: bold; margin-left: 10px;")
                # Auto-stop recording if critically low
                if self.is_recording:
                    print(f"WARNING: Critically low disk space ({free_gb:.1f} GB). Stopping recording.")
                    self.toggle_recording()
            elif free_gb < 5.0:
                # Warning - orange
                self.lbl_disk_space.setText(f"Disk: {free_gb:.1f} GB (Low)")
                self.lbl_disk_space.setStyleSheet("color: orange; font-weight: bold; margin-left: 10px;")
            else:
                # OK - green
                self.lbl_disk_space.setText(f"Disk: {free_gb:.1f} GB")
                self.lbl_disk_space.setStyleSheet("color: green; margin-left: 10px;")
        else:
            self.lbl_disk_space.setText("Disk: Unknown")
            self.lbl_disk_space.setStyleSheet("color: gray; margin-left: 10px;")
        
        # Update performance stats for each video stream
        for entry in self.threads:
            th = entry['thread']
            lbl_stats = entry['lbl_stats']
            
            fps_str = f"{th.fps_actual:.1f}" if th.fps_actual > 0 else "--"
            stats_text = f"FPS: {fps_str} | Frames: {th.frame_count} | Dropped: {th.dropped_frames}"
            
            # Color code based on dropped frames
            if th.dropped_frames == 0:
                lbl_stats.setStyleSheet("color: green;")
            elif th.dropped_frames < 10:
                lbl_stats.setStyleSheet("color: orange;")
            else:
                lbl_stats.setStyleSheet("color: red;")
            
            lbl_stats.setText(stats_text)

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

    def get_disk_space(self, path):
        """Get available disk space in GB for the given path."""
        try:
            stat = shutil.disk_usage(path)
            return stat.free / (1024**3)  # Convert to GB
        except Exception as e:
            print(f"Error checking disk space: {e}")
            return None

    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory", self.base_dir)
        if dir_path:
            self.base_dir = dir_path
            self.txt_directory.setText(dir_path)

    def load_configs(self, paths):
        merged_config = {
            "data_directory": "data",
            "record_audio": False,
            "videos": [],
            "ros_topics": [],
            "stages": []
        }
        
        # Track loaded files to avoid circular dependencies
        loaded_files = set()
        
        # Track master config directory for search path
        master_config_dir = None
        
        # Helper to find video by name in list
        def get_video_index(name, videos):
            for i, p in enumerate(videos):
                if p.get("name") == name:
                    return i
            return -1
        
        # Helper to load a single config file recursively
        def load_config_recursive(path, base_dir=None):
            nonlocal master_config_dir
            
            # Resolve relative paths - try base_dir first, then master_config_dir
            if not os.path.isabs(path):
                if base_dir:
                    path = os.path.join(base_dir, path)
                elif master_config_dir:
                    path = os.path.join(master_config_dir, path)
            
            # Normalize path to detect duplicates
            normalized_path = os.path.normpath(os.path.abspath(path))
            
            # Check for circular dependency
            if normalized_path in loaded_files:
                print(f"Skipping already loaded config: {path}")
                return
            
            if not os.path.exists(path):
                print(f"Warning: Configuration file not found: {path}")
                return
            
            loaded_files.add(normalized_path)
            config_dir = os.path.dirname(normalized_path)
            
            try:
                with open(path, 'r') as f:
                    cfg = json.load(f)
                    
                    # First, recursively load any referenced configuration files
                    if "configuration_files" in cfg and isinstance(cfg["configuration_files"], list):
                        for ref_file in cfg["configuration_files"]:
                            load_config_recursive(ref_file, config_dir)
                    
                    # Merge data_directory (last one wins)
                    if "data_directory" in cfg:
                        merged_config["data_directory"] = cfg["data_directory"]
                    
                    # Merge record_audio (OR)
                    if "record_audio" in cfg:
                        merged_config["record_audio"] = merged_config["record_audio"] or bool(cfg["record_audio"])
                    
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

                    # Append videos, renaming duplicates
                    if "videos" in cfg and isinstance(cfg["videos"], list):
                        for p_new in cfg["videos"]:
                            if "name" not in p_new:
                                merged_config["videos"].append(p_new)
                                continue
                                
                            original_name = p_new["name"]
                            name = original_name
                            counter = 2
                            
                            # Check if name exists in currently merged videos and find a unique name
                            while get_video_index(name, merged_config["videos"]) >= 0:
                                name = f"{original_name}_{counter}"
                                counter += 1
                                
                            p_new["name"] = name
                            merged_config["videos"].append(p_new)
                                
            except Exception as e:
                print(f"Error loading config {path}: {e}")
        
        # If a single string is passed, wrap it in a list
        if isinstance(paths, str):
            paths = [paths]
        
        # Load all config files recursively
        for path in paths:
            # Set master config directory from the first config file
            if master_config_dir is None and os.path.exists(path):
                master_config_dir = os.path.dirname(os.path.abspath(path))
            load_config_recursive(path)

        return merged_config

    def init_videos(self):
        print("DEBUG: init_videos() started", flush=True)
        cols = 2
        for i, p_config in enumerate(self.config.get('videos', [])):
            print(f"DEBUG: Processing video {i}: {p_config.get('name')}", flush=True)
            row = i // cols
            col = i % cols
            
            stream_name = p_config.get('name', f"Stream {i}")
            # Label and Checkbox
            lbl = QLabel(stream_name)
            lbl.setStyleSheet("font-weight: bold;")
            
            chk_preview = QCheckBox("Preview")
            chk_preview.setChecked(True)
            chk_record = QCheckBox("Record")

            # Header layout
            header_layout = QHBoxLayout()
            header_layout.addWidget(lbl)
            header_layout.addWidget(chk_record)
            header_layout.addWidget(chk_preview)
            header_layout.addStretch()

            # Video Widget
            vw = VideoWidget()
            self.video_widgets.append(vw)
            
            # Performance stats label
            lbl_stats = QLabel("FPS: -- | Frames: 0 | Dropped: 0")
            lbl_stats.setStyleSheet("color: gray;")
            lbl_stats.setAlignment(Qt.AlignLeft)

            # Thread
            th = VideoThread(p_config)
            th.change_pixmap_signal.connect(vw.update_image)
            th.start()
            
            # Store thread and associated control checkboxes
            self.threads.append({
                'thread': th,
                'chk_record': chk_record,
                'lbl_stats': lbl_stats
            })

            # Connect checkboxes
            chk_preview.toggled.connect(vw.set_preview_enabled)
            chk_preview.toggled.connect(lambda checked, t=th: setattr(t, 'preview_enabled', checked))
            
            # Set initial state from config
            chk_record.setChecked(p_config.get('record', True))
            
            # Container
            v_layout = QVBoxLayout()
            v_layout.addLayout(header_layout)
            v_layout.addWidget(vw, 1)  # Stretch factor of 1 to expand
            v_layout.addWidget(lbl_stats)
            
            container = QWidget()
            container.setLayout(v_layout)
            container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
                # If everything is fine, just log to console instead of showing a popup
                print(msg_text)
        except Exception as e:
            QMessageBox.critical(self, "Rosbag Error", f"Failed to verify rosbag at {bag_path}: {e}")

    def save_index(self):
        if not hasattr(self, 'current_stage_dir') or not self.current_stage_dir:
            return
            
        try:
            videos = []
            for entry in self.threads:
                th = entry['thread']
                if th.last_video_name:
                    v_duration = 0.0
                    # Compute duration from the sidecar JSON file
                    if th.last_json_path and os.path.exists(th.last_json_path):
                        try:
                            with open(th.last_json_path, 'r') as f:
                                v_data = json.load(f)
                                # Check for new format (frames list with 'ts')
                                if "frames" in v_data and len(v_data["frames"]) > 1:
                                    v_duration = v_data["frames"][-1]["ts"] - v_data["frames"][0]["ts"]
                                # Check for old format (timestamps_ns)
                                elif "timestamps_ns" in v_data and len(v_data["timestamps_ns"]) > 1:
                                    v_duration = (v_data["timestamps_ns"][-1] - v_data["timestamps_ns"][0]) / 1e9
                                else:
                                    v_duration = th.last_video_duration
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
            # Check disk space before starting
            free_gb = self.get_disk_space(self.base_dir)
            if free_gb is not None:
                if free_gb < 1.0:
                    QMessageBox.critical(
                        self,
                        "Insufficient Disk Space",
                        f"Cannot start recording. Only {free_gb:.2f} GB available.\n\n"
                        "Please free up disk space or select a different directory."
                    )
                    return
                elif free_gb < 5.0:
                    reply = QMessageBox.warning(
                        self,
                        "Low Disk Space",
                        f"Warning: Only {free_gb:.2f} GB available.\n\n"
                        "Recording may fail if session is long.\n\n"
                        "Continue anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
            
            # Start
            self.btn_browse.setEnabled(False)
            self.list_stages.setEnabled(False)
            self.chk_audio.setEnabled(False)
            
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
            try:
                os.makedirs(stage_dir, exist_ok=True)
                print(f"Created directory: {stage_dir}")
            except OSError as e:
                print(f"Error creating directory {stage_dir}: {e}")
                # Fallback to base_dir if stage_dir creation fails
                stage_dir = self.base_dir
            
            # Set current_stage_dir after potential fallback
            self.current_stage_dir = stage_dir

            # Start Audio Recorder if enabled
            if self.chk_audio.isChecked():
                audio_filename = f"audio_{stage_timestamp}.wav"
                audio_path = os.path.join(stage_dir, audio_filename)
                self.audio_recorder.start(audio_path)

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
                    
                    # Some ros2 distributions accept a `--topics` option, older
                    # ones expect topics as positional arguments. Detect which
                    # style is available and build the command accordingly.
                    storage = self._rosbag_get_storage()
                    if self._rosbag_supports_topics():
                        cmd = ["ros2", "bag", "record", "-o", bag_path, "-s", storage, "--topics"] + ros_topics
                    else:
                        cmd = ["ros2", "bag", "record", "-o", bag_path, "-s", storage] + ros_topics
                    print(f"Starting rosbag: {' '.join(cmd)}")
                    # Prevent ros2 bag from capturing stdin (avoids pausing on SPACE)
                    self.bag_process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL) 
                except Exception as e:
                    print(f"Failed to start rosbag: {e}")

            # Start video recordings
            for entry in self.threads:
                th = entry['thread']
                chk_rec = entry['chk_record']
                
                # Disable checkboxes
                chk_rec.setEnabled(False)
                
                # Update record config based on checkbox
                th.video_config['record'] = chk_rec.isChecked()
                th.set_recording(True, stage_dir, stage_name)

            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: red; color: white;")
            self.is_recording = True
        else:
            # Stop Audio
            self.audio_recorder.stop()

            # Stop
            for entry in self.threads:
                th = entry['thread']
                chk_rec = entry['chk_record']
                
                th.set_recording(False, self.base_dir)
                chk_rec.setEnabled(True)
            
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
                any_recording = any(entry['thread'].is_recording for entry in self.threads)
                if not any_recording:
                    break
                time.sleep(0.1)

            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("")
            self.is_recording = False
            self.btn_browse.setEnabled(True)
            self.list_stages.setEnabled(True)
            self.chk_audio.setEnabled(True)
            
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
        # Stop recording gracefully if it's currently active
        if self.is_recording:
            print("Stopping recording before exit...")
            self.toggle_recording()

        # 1. Signal all threads to stop (without waiting yet)
        for entry in self.threads:
            entry['thread']._run_flag = False
            
        # 2. Stop ROS Bag if it's still running for some reason
        if self.bag_process:
            print("Stopping rosbag during exit...")
            self.bag_process.send_signal(signal.SIGINT)
            try:
                self.bag_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.bag_process.kill()
            self.bag_process = None
        
        # 3. Wait for all threads to finish
        for entry in self.threads:
            entry['thread'].wait()

        # 4. Final ROS cleanup
        if self.ros_node:
            self.ros_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        event.accept()

def main():
    print("DEBUG: main() started. Creating global node...", flush=True)
    try:
        # Create node after all imports but before Qt
        global_ros_node = rclpy.create_node('video_recorder_node')
        print("DEBUG: Global ROS node created.", flush=True)
    except Exception as e:
        print(f"CRITICAL: ROS2 node creation failed: {e}", flush=True)
        global_ros_node = None

    parser = argparse.ArgumentParser(description="Video Record")
    parser.add_argument("-c", "--config", help="Path to JSON configuration file", action="append", required=True)
    parser.add_argument("-p", "--joy-topic", help="ROS Joy topic for recording control (button 0)", default=None)
    args = parser.parse_args()

    # Resolve config file paths relative to current working directory
    config_files = [os.path.abspath(cfg) for cfg in args.config]

    app = QApplication(sys.argv)
    
    # Allow Ctrl+C to close the application correctly
    # We need a timer to periodically yield control to the Python interpreter 
    # so it can process signals.
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(100)
    
    window = None

    def handle_sigint(sig, frame):
        print("\nInterrupt received, shutting down...")
        if window:
            window.close() # This will trigger closeEvent and its cleanup
        app.quit()
        app.quit()
        
    signal.signal(signal.SIGINT, handle_sigint)
    
    window = RecordWindow(config_files, joy_topic=args.joy_topic, external_node=global_ros_node)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
