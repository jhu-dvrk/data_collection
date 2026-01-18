# Video Data Recorder

A multi-stream video recorder application using Python, PyQt5, OpenCV, and GStreamer. It allows creating flexible video streams via JSON configuration, provides live previews, and supports synchronized recording with timestamps.  ROS topics can also be recorded along the videos.  Note that the videos are recorded directly from the source using GStreamer and don't rely on ROS topics.

The application also integrates with ROS2 for remote control and status monitoring.

## Prerequisites

*   **ROS2**: The application is a ROS2 node and requires a working ROS2 installation (Humble, Jazzy, etc.).
*   **GStreamer 1.0**: (plugins-base, plugins-good, plugins-bad, plugins-ugly, libav)
*   **System Tools**: `gst-inspect-1.0` (required for hardware encoder discovery)
*   **System Packages**:
    ```bash
    sudo apt install python3-opencv python3-numpy python3-pyqt5
    ```

## Installation & Environment

Before running the application, ensure your ROS2 environment is sourced:

```bash
source /opt/ros/YOUR_DISTRO/setup.bash
# or if using a workspace
source install/setup.bash
```

## Usage

### 1. Configuration

Define your video sources in a JSON file. The configuration format is defined in the [JSON schema](share/data_recorder.schema.json).

**Example `config.json`:**
```json
{
  "data_directory": "data",
  "ros_topics": [ 
    "/PSM1/measured_cp",
    "/PSM1/measured_cv",
    "/PSM1/jaw/measured_js"
  ],
  "stages": [
    "calibration",
    "exercise_1",
    "exercise_2"
  ],
  "videos": [
    {
      "name": "camera_1",
      "stream": "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1",
      "record": true,
      "popup": true,
      "encoding": {
        "width": 320,
        "height": 240,
        "bitrate": 5000
      }
    },
    {
      "name": "test_pattern",
      "stream": "videotestsrc pattern=smpte75",
      "record": false,
      "popup": false
    }
  ]
}
```

#### Configuration File Composition

Configuration files can reference other configuration files using the `configuration_files` field. This allows you to organize and reuse configurations across different setups.

**Example with `configuration_files`:**
```json
{
  "data_directory": "data",
  "configuration_files": [
    "devices/PSM1.json",
    "devices/PSM2.json",
    "cameras/stereo.json"
  ]
}
```

**How it works:**
- Referenced configuration files are loaded and merged recursively
- Paths in `configuration_files` are resolved relative to the current config file's directory
- If a file is not found relative to the current config, it falls back to searching relative to the master config file's directory
- All `videos`, `ros_topics`, and `stages` from referenced files are combined (deduplicated)
- The `data_directory` from the last processed file is used
- Circular dependencies are automatically detected and prevented

This approach enables modular configurations where you can maintain separate files for each device or component and combine them as needed for different experimental setups.

### 2. Running the Recorder

#### Option A: Using ros2 run (Recommended)

After building your workspace, run the recorder using `ros2 run`:

```bash
ros2 run data_collection data_recorder -c config.json
```

Multiple configuration files can be loaded and merged. You can collect multiple video streams and multiple ROS topics defined in existing files (e.g. `PSM1.json`). This allows users to re-use configuration files for each component used for a given experimental setup.

```bash
ros2 run data_collection data_recorder -c PSM1.json -c PSM2.json -c SUJ.json -c video_config.json
```

**Note**: Configuration file paths can be relative to your current working directory or absolute paths.

#### Option B: Direct Script Execution

Alternatively, run the script directly:

```bash
./data_recorder.py -c config.json
```

Multiple configuration files:

```bash
./data_recorder.py -c PSM1.json -c PSM2.json -c SUJ.json -c video_config.json
```

### 3. Stages Feature

If the `stages` field is provided in the configuration, a "Stages" list will appear on the right side of the GUI.

*   **File Naming**: When a stage is selected, its name is appended to the session directory and all recorded files. The naming convention for video files is `camera_name_YYMMDD_HHMMSS_stage.mp4`.
*   **Auto-Advancement**: After stopping a recording, the application automatically selects the next stage in the list.
*   **Looping**: When the last stage is completed, it wraps back to the first stage.
*   **Manual Override**: Users can click any stage in the list to select it for the next recording (selection is disabled while recording is in progress).
*   **Hardware-Accelerated Encoding**: Automatically detects and uses available hardware encoders (NVENC, VAAPI) to minimize CPU usage.
*   **Nanosecond Precision**: All video frames are timestamped in nanoseconds since epoch, ensuring perfect alignment with ROS2 bags.
*   **Session Metadata**: An `index.json` file is created in each session directory, storing video/bag durations and metadata.

### 4. Data Post-processing

The `extract_frames.py` script identifies all data in a session directory and:
1.  Extracts individual frames from all recorded `.mp4` files using sidecar `.json` nanosecond timestamps.
2.  Converts all recorded ROS bag topics into individual `.csv` files.

#### Using ros2 run (Recommended)

To process a recorded session directory:

```bash
ros2 run data_collection extract_frames -d 20260117_153206
```

To list the videos in a session without processing:

```bash
ros2 run data_collection extract_frames -d 20260117_153206 -l
```

#### Direct Script Execution

Alternatively, run the script directly:

```bash
./extract_frames.py -d 20260117_153206
```

To list the videos in a session without processing:

```bash
./extract_frames.py -d 20260117_153206 -l
```

## ROS2 Integration

The recorder functions as a ROS2 node named `data_recorder`.

### Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/data_recorder/record` | `std_msgs/msg/Bool` | Subscriber | Send `true` to start recording, `false` to stop. |
| `/data_recorder/recording` | `std_msgs/msg/Bool` | Publisher | Publishes actual recording state (`true` if recording). |

### Command Line Examples

Ensure you have sourced your ROS2 environment (e.g., `source /opt/ros/humble/setup.bash`).

**Start Recording:**
```bash
ros2 topic pub /data_recorder/record std_msgs/msg/Bool "{data: true}" --once
```

**Stop Recording:**
```bash
ros2 topic pub /data_recorder/record std_msgs/msg/Bool "{data: false}" --once
```

**Monitor Status:**
```bash
ros2 topic echo /data_recorder/recording
```
