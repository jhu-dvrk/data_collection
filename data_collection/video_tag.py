#!/usr/bin/env python3
"""
Video Annotator with Frame-Accurate Seeking
Uses PyQt5 for UI and GStreamer for video playback
"""

import sys
import os
import argparse
import json
import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QFileDialog, QStyle, QScrollArea, QSizePolicy, 
                             QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo, GLib

# Initialize GStreamer
Gst.init(None)



class VideoTag(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Tag with Frame Seeking")
        self.setGeometry(100, 100, 1400, 800)
        
        # GStreamer pipeline
        self.player = None
        self.pipeline = None
        self.video_sink = None
        self.duration = 0
        self.fps = 30.0  # Default FPS, will be updated from stream
        self.current_frame = 0
        self.total_frames = 0
        self.frame_stepping = False  # Flag to preserve frame when stepping
        
        # Stages and tagging
        self.stages = []
        self.tags = []  # Additional tags loaded from config
        self.frame_tags = {}  # Dictionary to track frame -> [tags] mapping
        self.video_path = None  # Track the current video path for saving
        self.unsaved_changes = False  # Track if there are unsaved changes        self.shortcut_map = {}  # Map for numeric shortcuts        
        # Setup UI
        self.init_ui()
        
        # Timer for updating position
        self.timer = QTimer()
        self.timer.setInterval(100)  # Update every 100ms
        self.timer.timeout.connect(self.update_position)
        
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (horizontal: video on left, stages on right)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side - video and controls
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)  # Stretch factor 1 for main content
        
        # Video display area
        self.video_widget = QWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        self.video_widget.setMinimumSize(400, 300)
        # Set attribute for proper embedding
        self.video_widget.setAttribute(Qt.WA_NativeWindow)
        left_layout.addWidget(self.video_widget, 1)  # Stretch factor 1
        
        # Info label
        self.info_label = QLabel("No video loaded")
        self.info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.info_label, 0)
        
        # Timeline slider
        timeline_layout = QHBoxLayout()
        self.timeline_label = QLabel("00:00:00")
        timeline_layout.addWidget(self.timeline_label)
        
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.sliderMoved.connect(self.seek_to_position)
        self.timeline_slider.sliderPressed.connect(self.slider_pressed)
        self.timeline_slider.sliderReleased.connect(self.slider_released)
        timeline_layout.addWidget(self.timeline_slider)
        
        self.duration_label = QLabel("00:00:00")
        timeline_layout.addWidget(self.duration_label)
        
        left_layout.addLayout(timeline_layout)
        
        # Frame control slider
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1000)
        self.frame_slider.sliderMoved.connect(self.seek_to_frame)
        self.frame_slider.sliderPressed.connect(self.slider_pressed)
        self.frame_slider.sliderReleased.connect(self.slider_released)
        frame_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(120)
        frame_layout.addWidget(self.frame_label)
        
        left_layout.addLayout(frame_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        # Previous frame
        self.prev_frame_btn = QPushButton("Previous Frame")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)
        controls_layout.addWidget(self.prev_frame_btn)
        
        # Play/Pause button
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        # Next frame
        self.next_frame_btn = QPushButton("Next Frame")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        controls_layout.addWidget(self.next_frame_btn)
        
        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        # Playback speed control
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.05x", "0.1x", "0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.set_playback_speed)
        self.speed_combo.setMaximumWidth(80)
        controls_layout.addWidget(self.speed_combo)
        
        controls_layout.addStretch()
        left_layout.addLayout(controls_layout)
        
        # Right side - stages list with frame buttons
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Tags:"))
        
        # Create a scrollable area for stages and frame buttons
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(300)
        self.stages_container = QWidget()
        self.stages_layout = QVBoxLayout(self.stages_container)
        self.stages_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(self.stages_container)
        right_layout.addWidget(scroll_area)
        
        # Dictionary to store tag checkboxes and buttons
        self.tag_buttons = {}  # {tag_name: {'checkbox': btn, 'frames': {frame: btn}}}
        
        # Save tags button
        self.save_tags_btn = QPushButton("Save Tags")
        self.save_tags_btn.clicked.connect(self.save_tags)
        self.save_tags_btn.setEnabled(False)
        right_layout.addWidget(self.save_tags_btn)
        
        # Add right layout with no stretch (fixed size based on content)
        main_layout.addLayout(right_layout, 0)
        
    def load_stages(self, filepath):
        """Load stages and tags from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.stages = data.get('stages', [])
                self.tags = data.get('tags', [])
                self.populate_stages_list()
        except Exception as e:
            print(f"Error loading stages file: {e}")
    
    def populate_stages_list(self):
        """Populate the stages list with checkbox buttons and frame buttons"""
        # Clear existing widgets
        while self.stages_layout.count():
            item = self.stages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Recursively delete layout items if needed, though takeAt usually suffices for top level
                # But here we added layouts directly using addLayout.
                # When addLayout is used, the layout becomes a child.
                # wait, addLayout puts the layout into the items list.
                pass 
                # PyQt handling of addLayout is tricky for deletion. 
                # It's better if we wrapped each row in a QWidget, but current code uses addLayout directly.
                # However, earlier code does:
                # tag_layout = QHBoxLayout() ... self.stages_layout.addLayout(tag_layout)
                # To clear a layout that has sub-layouts, one must delete the items.
                
        # Actually, standard way to clear layout with sub-layouts:
        while self.stages_layout.count():
            item = self.stages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # If we added a layout directly, we need to delete its widgets
                sublayout = item.layout()
                while sublayout.count():
                    subitem = sublayout.takeAt(0)
                    if subitem.widget():
                        subitem.widget().deleteLater()
                sublayout.deleteLater()

        self.tag_buttons.clear()
        self.shortcut_map = {}
        shortcut_idx = 0
        shortcuts = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        
        # Create buttons for each stage/variant
        for stage in self.stages:
            for variant in ['start', 'end']:
                tag_name = f"{stage} {variant}"
                shortcut = shortcuts[shortcut_idx] if shortcut_idx < len(shortcuts) else None
                self.add_tag_row(tag_name, shortcut)
                if shortcut:
                    self.shortcut_map[shortcut] = tag_name
                    shortcut_idx += 1
        
        # Create buttons for each direct tag (loaded as-is)
        for tag_name in self.tags:
            shortcut = shortcuts[shortcut_idx] if shortcut_idx < len(shortcuts) else None
            self.add_tag_row(tag_name, shortcut)
            if shortcut:
                self.shortcut_map[shortcut] = tag_name
                shortcut_idx += 1
        
        self.stages_layout.addStretch()
        self.update_stages_display()

    def add_tag_row(self, tag_name, shortcut=None):
        """Helper to create a single tag row UI"""
        # Create a horizontal layout for this tag
        tag_layout = QHBoxLayout()
        tag_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toggle button for the tag
        display_name = f"({shortcut}) {tag_name}" if shortcut is not None else tag_name
        tag_btn = QPushButton(display_name)
        tag_btn.setCheckable(True)
        tag_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        tag_btn.clicked.connect(lambda checked, t=tag_name: self.toggle_tag(t))
        tag_layout.addWidget(tag_btn)
        
        # Container for frame buttons
        frames_widget = QWidget()
        frames_layout = QHBoxLayout(frames_widget)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(3)
        
        self.tag_buttons[tag_name] = {
            'checkbox': tag_btn,
            'frames': {},
            'frames_layout': frames_layout
        }
        
        tag_layout.addWidget(frames_widget)
        tag_layout.addStretch()
        
        # Add to main layout
        self.stages_layout.addLayout(tag_layout)
    
    def toggle_tag(self, tag_name):
        """Toggle a tag for the current frame"""
        if tag_name in self.tag_buttons:
            is_checked = self.tag_buttons[tag_name]['checkbox'].isChecked()
            
            if is_checked:
                # Add tag to current frame
                if self.current_frame not in self.frame_tags:
                    self.frame_tags[self.current_frame] = []
                if tag_name not in self.frame_tags[self.current_frame]:
                    self.frame_tags[self.current_frame].append(tag_name)
            else:
                # Remove tag from current frame
                if self.current_frame in self.frame_tags:
                    if tag_name in self.frame_tags[self.current_frame]:
                        self.frame_tags[self.current_frame].remove(tag_name)
                    if not self.frame_tags[self.current_frame]:
                        del self.frame_tags[self.current_frame]
            
            # Update display (save happens only on explicit Save Tags button click)
            self.unsaved_changes = True
            self.update_stages_display()
        
    def update_stages_display(self):
        """Update display of stages with their tagged frame indices as buttons"""
        # Build a mapping of tag -> list of frames
        tag_to_frames = {}
        for frame_num, tags in self.frame_tags.items():
            for tag in tags:
                if tag not in tag_to_frames:
                    tag_to_frames[tag] = []
                tag_to_frames[tag].append(frame_num)
        
        # Sort frames for each tag
        for tag in tag_to_frames:
            tag_to_frames[tag].sort()
        
        # Update button states and frame buttons
        for tag_name, button_info in self.tag_buttons.items():
            # Update checkbox state
            is_tagged = tag_name in tag_to_frames
            button_info['checkbox'].setChecked(is_tagged)
            
            # Clear old frame buttons
            for frame_btn in button_info['frames'].values():
                frame_btn.deleteLater()
            button_info['frames'].clear()
            
            # Add new frame buttons
            if is_tagged:
                frames = tag_to_frames[tag_name]
                for frame_num in frames:
                    frame_btn = QPushButton(str(frame_num))
                    frame_btn.setMaximumWidth(50)
                    frame_btn.setMaximumHeight(25)
                    frame_btn.clicked.connect(lambda checked, f=frame_num: self.seek_to_frame(f))
                    button_info['frames'][frame_num] = frame_btn
                    button_info['frames_layout'].insertWidget(len(button_info['frames']) - 1, frame_btn)
        
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All Files (*)"
        )
        
        if filename:
            self.load_video(filename)
            
    def load_video(self, filepath):
        # Store the video path for saving tags later
        self.video_path = filepath
        self.save_tags_btn.setEnabled(True)
        
        # Stop existing pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Reset playback speed to default
        self.speed_combo.blockSignals(True)
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.blockSignals(False)
            
        # Create new pipeline
        self.pipeline = Gst.Pipeline.new("video-player")
        
        # Create elements
        source = Gst.ElementFactory.make("filesrc", "source")
        source.set_property("location", filepath)
        
        decodebin = Gst.ElementFactory.make("decodebin", "decoder")
        
        # Video sink - use platform-appropriate sink
        videoconvert = Gst.ElementFactory.make("videoconvert", "converter")
        videoscale = Gst.ElementFactory.make("videoscale", "scaler")
        
        # Use autovideosink for better Qt integration
        self.video_sink = Gst.ElementFactory.make("autovideosink", "sink")
        
        # Add elements to pipeline
        self.pipeline.add(source)
        self.pipeline.add(decodebin)
        self.pipeline.add(videoconvert)
        self.pipeline.add(videoscale)
        self.pipeline.add(self.video_sink)
        
        # Link elements (decodebin will be linked dynamically)
        source.link(decodebin)
        videoconvert.link(videoscale)
        videoscale.link(self.video_sink)
        
        # Connect decodebin pad-added signal
        decodebin.connect("pad-added", self.on_pad_added, videoconvert)
        
        # Set up bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        bus.enable_sync_message_emission()
        bus.connect("sync-message::element", self.on_sync_message)
        
        # Start playing to get metadata
        self.pipeline.set_state(Gst.State.PAUSED)
        
        # Wait for state change
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        
        # Get duration and FPS
        self.get_video_info()
        
        # Load existing tags for this video
        self.load_tags_from_file()
        
        # Update UI
        self.info_label.setText(f"Loaded: {os.path.basename(filepath)}")
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
        
        # Update duration label
        self.update_duration_label()
        
    def on_pad_added(self, element, pad, sink):
        """Handle dynamic pad creation from decodebin"""
        # Only link video pads
        caps = pad.query_caps(None)
        structure = caps.get_structure(0)
        if structure.get_name().startswith('video'):
            sink_pad = sink.get_static_pad("sink")
            if not sink_pad.is_linked():
                pad.link(sink_pad)
    
    def on_sync_message(self, bus, message):
        """Handle sync messages for video overlay"""
        if message.get_structure() is None:
            return
        
        message_name = message.get_structure().get_name()
        if message_name == 'prepare-window-handle':
            # Get the native window ID
            win_id = self.video_widget.winId()
            message.src.set_window_handle(win_id)
            
    def on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.EOS:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.timer.stop()
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            
    def get_video_info(self):
        """Get video duration and frame rate"""
        # Get duration
        success, self.duration = self.pipeline.query_duration(Gst.Format.TIME)
        
        if success and self.duration > 0:
            # Try to get FPS from video stream
            # Iterate through pipeline elements
            it = self.pipeline.iterate_elements()
            while True:
                result, element = it.next()
                if result != Gst.IteratorResult.OK:
                    break
                    
                if isinstance(element, Gst.Bin):
                    continue
                    
                # Check all pads for video capabilities
                pad_it = element.iterate_pads()
                while True:
                    pad_result, pad = pad_it.next()
                    if pad_result != Gst.IteratorResult.OK:
                        break
                        
                    caps = pad.get_current_caps()
                    if caps:
                        for i in range(caps.get_size()):
                            structure = caps.get_structure(i)
                            if structure.get_name().startswith('video'):
                                # Try to get framerate
                                has_field = structure.has_field('framerate')
                                if has_field:
                                    success, fps_num, fps_denom = structure.get_fraction('framerate')
                                    if success and fps_denom > 0:
                                        self.fps = fps_num / fps_denom
                                        break
            
            # Calculate total frames
            self.total_frames = int((self.duration / Gst.SECOND) * self.fps)
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))
            
    def toggle_playback(self):
        """Toggle between play and pause"""
        if not self.pipeline:
            return
            
        state = self.pipeline.get_state(0)[1]
        
        if state == Gst.State.PLAYING:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
        else:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start()
    
    def set_playback_speed(self, speed_str):
        """Set playback speed via GStreamer"""
        if not self.pipeline:
            return
        
        # Parse speed string (e.g., "1.5x" -> 1.5)
        try:
            speed = float(speed_str.rstrip('x'))
        except ValueError:
            return
        
        # Get current position
        success, position = self.pipeline.query_position(Gst.Format.TIME)
        if not success:
            return
        
        # Set playback rate using seek with speed
        self.pipeline.seek(
            speed,
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
            Gst.SeekType.SET,
            position,
            Gst.SeekType.NONE,
            -1
        )
            
    def stop(self):
        """Stop playback and reset to beginning"""
        if not self.pipeline:
            return
        
        # Reset playback speed to default
        self.speed_combo.blockSignals(True)
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.blockSignals(False)
            
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.seek_simple(Gst.Format.TIME, 
                                 Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 
                                 0)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.timer.stop()
        self.update_position()
        
    def slider_pressed(self):
        """Pause updates when slider is being dragged"""
        self.timer.stop()
        
    def slider_released(self):
        """Resume updates when slider is released"""
        # Force an immediate update after seek completes
        QTimer.singleShot(50, self.update_position)
        state = self.pipeline.get_state(0)[1]
        if state == Gst.State.PLAYING:
            self.timer.start()
            
    def _execute_seek(self, seek_time):
        """Execute seek operation and update UI"""
        self.pipeline.seek(
            1.0,
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
            Gst.SeekType.SET,
            seek_time,
            Gst.SeekType.NONE,
            -1
        )
        
        # Update UI immediately with target values while seek completes
        pos_sec = seek_time / Gst.SECOND
        self.timeline_label.setText(self.format_time(pos_sec))
        
        # Determine frame from seek time if we have FPS
        if self.fps > 0:
            self.current_frame = int(round((seek_time / Gst.SECOND) * self.fps))
            
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"{self.current_frame} / {self.total_frames}")
        
        if self.duration > 0:
            slider_pos = int((seek_time / self.duration) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(slider_pos)
            self.timeline_slider.blockSignals(False)
            
        # Schedule a delayed update after seek completes
        QTimer.singleShot(100, self.update_position)

    def seek_to_position(self, position):
        """Seek to position based on timeline slider"""
        if not self.pipeline or self.duration <= 0:
            return
            
        seek_time = int((position / 1000.0) * self.duration)
        self._execute_seek(seek_time)
        
    def seek_to_frame(self, frame_num):
        """Seek to specific frame number"""
        if not self.pipeline or self.total_frames <= 0 or self.fps <= 0:
            return
        
        # Calculate seek time from frame number with rounding for accuracy
        seek_time = int(round((frame_num / self.fps) * Gst.SECOND))
        self._execute_seek(seek_time)
        
    def prev_frame(self):
        """Go to previous frame and pause"""
        if not self.pipeline:
            return
            
        # Ensure playback is paused when stepping
        self.pipeline.set_state(Gst.State.PAUSED)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.timer.stop()
        
        if self.current_frame > 0:
            new_frame = self.current_frame - 1
            self.frame_stepping = True
            self.seek_to_frame(new_frame)
            
    def next_frame(self):
        """Go to next frame and pause"""
        if not self.pipeline:
            return
            
        # Ensure playback is paused when stepping
        self.pipeline.set_state(Gst.State.PAUSED)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.timer.stop()

        if self.current_frame < self.total_frames - 1:
            new_frame = self.current_frame + 1
            self.frame_stepping = True
            self.seek_to_frame(new_frame)
            
    def update_position(self):
        """Update position indicators"""
        if not self.pipeline:
            return
            
        success, position = self.pipeline.query_position(Gst.Format.TIME)
        
        if success:
            # Update timeline slider
            if self.duration > 0:
                slider_pos = int((position / self.duration) * 1000)
                self.timeline_slider.blockSignals(True)
                self.timeline_slider.setValue(slider_pos)
                self.timeline_slider.blockSignals(False)
                
            # Update time label
            pos_sec = position / Gst.SECOND
            self.timeline_label.setText(self.format_time(pos_sec))
            
            # Update frame counter - only if not frame stepping
            # Frame stepping preserves the exact frame we sought to
            if not self.frame_stepping:
                calculated_frame = int((position / Gst.SECOND) * self.fps)
                # Only update if it's noticeably different (avoid jitter)
                if abs(calculated_frame - self.current_frame) > 0:
                    self.current_frame = calculated_frame
            
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f"{self.current_frame} / {self.total_frames}")
            self.frame_stepping = False  # Reset flag after update
            
            # Restore tags for this frame
            self.restore_tags_for_frame(self.current_frame)
            
    def update_duration_label(self):
        """Update duration label"""
        if self.duration > 0:
            dur_sec = self.duration / Gst.SECOND
            self.duration_label.setText(self.format_time(dur_sec))
            
    def format_time(self, seconds):
        """Format time in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def restore_tags_for_frame(self, frame_num):
        """Restore tag selections for the current frame"""
        # Update all tag buttons based on current frame
        for tag_name, button_info in self.tag_buttons.items():
            is_tagged = frame_num in self.frame_tags and tag_name in self.frame_tags[frame_num]
            button_info['checkbox'].blockSignals(True)
            button_info['checkbox'].setChecked(is_tagged)
            button_info['checkbox'].blockSignals(False)
    
    def validate_stage_pairs(self):
        """Validate that all stages have matching start/end pairs in order"""
        # Check each stage in the stages list
        for stage in self.stages:
            start_tag = f"{stage} start"
            end_tag = f"{stage} end"
            
            # Get frames where each tag appears (in sorted order)
            start_frames = sorted([frame_num for frame_num, tags in self.frame_tags.items() if start_tag in tags])
            end_frames = sorted([frame_num for frame_num, tags in self.frame_tags.items() if end_tag in tags])
            
            # Skip validation if there are no tags for this stage
            if not start_frames and not end_frames:
                continue
            
            # Check for mismatched counts
            if len(start_frames) != len(end_frames):
                error_msg = f"Stage '{stage}' has mismatched start/end count.\n"
                error_msg += f"  - Start tags ({len(start_frames)}): {start_frames}\n"
                error_msg += f"  - End tags ({len(end_frames)}): {end_frames}"
                self.info_label.setText(error_msg)
                return False
            
            # Check that each end frame >= corresponding start frame (in order)
            for start_frame, end_frame in zip(start_frames, end_frames):
                if end_frame < start_frame:
                    error_msg = f"Stage '{stage}' has invalid pair.\n"
                    error_msg += f"  - Start frame: {start_frame}\n"
                    error_msg += f"  - End frame: {end_frame} (must be >= start)"
                    self.info_label.setText(error_msg)
                    return False
        
        return True
    
    def save_tags(self):
        """Save tags to file when user clicks Save Tags button"""
        # Validate stages have matching start/end pairs
        if not self.validate_stage_pairs():
            return
        self.write_tags_to_file()
    
    def write_tags_to_file(self):
        """Write frame tags to a JSON file with new format"""
        if not self.video_path:
            return
        
        # Create output filename based on video file
        base_path = os.path.splitext(self.video_path)[0]
        tags_file = f"{base_path}_tags.json"
        
        # Build the stages data in new format
        stages_data = []
        for stage in self.stages:
            start_tag = f"{stage} start"
            end_tag = f"{stage} end"
            
            # Get frames where start and end tags appear
            start_frames = sorted([frame for frame in self.frame_tags if start_tag in self.frame_tags[frame]])
            end_frames = sorted([frame for frame in self.frame_tags if end_tag in self.frame_tags[frame]])
            
            # Using the same logic as validation: pair the i-th start with the i-th end
            # We assume validation has passed, so lengths are equal and end >= start
            for start_frame, end_frame in zip(start_frames, end_frames):
                stages_data.append({
                    "name": stage,
                    "start": start_frame,
                    "end": end_frame
                })
        
        # Build the tags data (for direct tags, not stage-based)
        # Format: "tag_name": [frame1, frame2, ...]
        tags_data = {}
        for frame_num, tags in self.frame_tags.items():
            # Filter out stage-based tags
            direct_tags = [tag for tag in tags if tag not in [f"{s} start" for s in self.stages] 
                          and tag not in [f"{s} end" for s in self.stages]]
            
            for tag in direct_tags:
                if tag not in tags_data:
                    tags_data[tag] = []
                tags_data[tag].append(frame_num)
        
        # Sort frame numbers for each tag
        for tag in tags_data:
            tags_data[tag].sort()
        
        try:
            output_data = {}
            if stages_data:
                output_data["stages"] = stages_data
            if tags_data:
                output_data["tags"] = tags_data
                
            with open(tags_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            self.unsaved_changes = False
            self.info_label.setText(f"Tags saved to {os.path.basename(tags_file)}")
        except Exception as e:
            self.info_label.setText(f"Error saving tags: {e}")
    
    def load_tags_from_file(self):
        """Load frame tags from JSON file if it exists"""
        if not self.video_path:
            return
        
        base_path = os.path.splitext(self.video_path)[0]
        tags_file = f"{base_path}_tags.json"
        
        # Reset current tags
        self.frame_tags = {}
        self.unsaved_changes = False
        
        if os.path.exists(tags_file):
            try:
                with open(tags_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load stages in new format
                    for stage_entry in data.get("stages", []):
                        if isinstance(stage_entry, dict) and "name" in stage_entry:
                            stage_name = stage_entry["name"]
                            start_frame = stage_entry.get("start")
                            end_frame = stage_entry.get("end")
                            
                            if start_frame is not None:
                                if start_frame not in self.frame_tags:
                                    self.frame_tags[start_frame] = []
                                start_tag = f"{stage_name} start"
                                if start_tag not in self.frame_tags[start_frame]:
                                    self.frame_tags[start_frame].append(start_tag)
                            
                            if end_frame is not None:
                                if end_frame not in self.frame_tags:
                                    self.frame_tags[end_frame] = []
                                end_tag = f"{stage_name} end"
                                if end_tag not in self.frame_tags[end_frame]:
                                    self.frame_tags[end_frame].append(end_tag)
                    
                    # Load direct tags (format: tag_name -> [frame_indices])
                    tags_section = data.get("tags", {})
                    for tag_name, frames in tags_section.items():
                        for frame_num in frames:
                            if frame_num not in self.frame_tags:
                                self.frame_tags[frame_num] = []
                            if tag_name not in self.frame_tags[frame_num]:
                                self.frame_tags[frame_num].append(tag_name)
                    
                    # Update display with loaded tags
                    self.update_stages_display()
            except Exception as e:
                print(f"Error loading tags: {e}")
        
        # Always update display to show current state (empty or loaded)
        self.update_stages_display()
        
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key_text = event.text().lower()
        key_code = event.key()
        modifiers = event.modifiers()
        
        # Ctrl+C handler
        if key_code == Qt.Key_C and modifiers & Qt.ControlModifier:
            self.close()
            return

        if key_text == 'a':
            # Start from beginning
            self.stop()
        elif key_text == 's':
            # Previous frame
            self.prev_frame()
        elif key_text == 'd':
            # Play/Pause
            self.toggle_playback()
        elif key_text == 'f':
            # Next frame
            self.next_frame()
        elif key_text == 'q':
            # Quit
            self.close()
        elif key_text in self.shortcut_map:
            # Toggle tag via numeric shortcut
            tag_name = self.shortcut_map[key_text]
            if tag_name in self.tag_buttons:
                # Toggle the button state (this will trigger toggle_tag via signal)
                btn = self.tag_buttons[tag_name]['checkbox']
                btn.setChecked(not btn.isChecked())
                self.toggle_tag(tag_name)
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Clean up on close with save prompt"""
        if self.unsaved_changes:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                       "You have unsaved changes. Do you want to save them before quitting?",
                                       QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)

            if reply == QMessageBox.Save:
                # Validate and save
                if self.validate_stage_pairs():
                    self.write_tags_to_file()
                    # After saving, we can close
                else:
                    # Validation failed, don't close, user needs to fix
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
            # Discard case falls through to close
            
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        event.accept()


def main():
    # Create QApplication FIRST before any QWidget instantiation
    app = QApplication(sys.argv)
    
    # Handle Ctrl+C (SIGINT) from terminal gracefully
    signal.signal(signal.SIGINT, lambda *args: QApplication.quit())
    
    # Use a timer to allow Python signal handler to run
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None) 
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video Tag with Frame-Accurate Seeking')
    parser.add_argument('-v', '--video', type=str, help='Path to video file to open on startup')
    parser.add_argument('-c', '--config', type=str, help='Path to JSON file with tags configuration')
    args = parser.parse_args()
    
    # Now create the player widget
    player = VideoTag()
    
    # Handle Ctrl+C (SIGINT) from terminal by closing the window
    signal.signal(signal.SIGINT, lambda *args: player.close())
    
    player.show()
    
    # Load stages if specified
    if args.config:
        player.load_stages(args.config)
    
    # If a video file is specified, load it
    if args.video:
        player.load_video(args.video)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
