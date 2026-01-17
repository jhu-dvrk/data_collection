# Video Data Recorder

A multi-stream video recorder application using Python, PyQt5, OpenCV, and GStreamer. It allows creating flexible video pipelines via JSON configuration, provides live previews, and supports synchronized recording with timestamps.  ROS topics can also be recorded along the videos.  Note that the videos are recorded directly from the source using GStreamer and don't rely on ROS topics.

The application also integrates with ROS2 for remote control and status monitoring.

## Prerequisites

*   **ROS2**: The application is a ROS2 node and requires a working ROS2 installation (Humble, Jazzy, etc.).
*   **GStreamer 1.0**: (plugins-base, plugins-good, plugins-bad, plugins-ugly, libav)
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
  "pipelines": [
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
      "stream": "videotestsrc pattern=ball",
      "record": false,
      "popup": false
    }
  ]
}
```

### 2. Running the Recorder

Run the script passing one or more configuration files:

```bash
./data_recorder.py -c config.json
```

Multiple configuration files can be loaded and merged.  You can collect multiple video streams and multiple ROS topics defined in existing files (e.g. `PSM1.json`).  This allows users to re-use configuration files for each component used for a given experimental setup.

```bash
./data_recorder.py -c config1.json -c config2.json
```

### 3. Data Post-processing

The `extract_frames.py` script identifies all data in a session directory and:
1.  Extracts individual frames from all recorded `.mp4` files.
2.  Converts all recorded ROS bag topics into individual `.csv` files.

To process a recorded session directory:

```bash
./extract_frames.py 20260117_153206
```

To list the videos in a session without processing:

```bash
./extract_frames.py 20260117_153206 -l
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
