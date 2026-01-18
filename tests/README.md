# Test Suite for Video Data Recorder

This directory contains test configurations, scripts, and verification tools to ensure the data recorder is functioning correctly.

## Prerequisites

To run the full test suite, including the 1KHz ROS2 publisher and the timestamp verification tool, you need the following:

### 1. ROS2 Environment
Ensure your ROS2 environment (e.g., Humble or Jazzy) is sourced:
```bash
source /opt/ros/YOUR_DISTRO/setup.bash
```

### 2. Python Environment & Packages
It is recommended to use a virtual environment for the verification tools:
```bash
# Create a virtual environment that preserves ROS2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Install required packages
pip install pytesseract
```

### 3. System Packages
The timestamp verification script requires the Tesseract OCR engine:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

## Available Tools

### 1. High-Frequency Publisher ([ros_publisher.py](ros_publisher.py))
A simple ROS2 node that publishes a `std_msgs/Float64` timestamp to `/test_topic` at **1000Hz (1KHz)**. Use this to verify high-frequency data capture and bagging.

### 2. Test Configuration ([capture_config.json](capture_config.json))
A complete configuration file for testing. It includes:
*   A video pipeline with a burned-in clock (`timeoverlay`).
*   Subscription to `/test_topic`.

### 3. Timestamp Verifier ([check_timestamps.py](check_timestamps.py))
A script that uses OCR to compare the time burned into the video frames against the sidecar `.json` timestamps. It checks for jitter or drift introduced during recording.

## End-to-End Test Procedure

1.  **Start the ROS2 Topic**:
    ```bash
    python3 tests/ros_publisher.py
    ```

2.  **Run the Recorder**:
    ```bash
    ./data_recorder.py -c tests/capture_config.json
    ```
    *   Click **Start Recording**.
    *   Wait for a few seconds.
    *   Click **Stop Recording**.

3.  **Extract Data**:
    Identify the session directory (e.g., `test_data/20260118_120000_test_run`) and run:
    ```bash
    python3 data_collection/extract_frames.py -d test_data/YOUR_SESSION_FOLDER
    ```

4.  **Verify Timestamps**:
    ```bash
    python3 tests/check_timestamps.py test_data/YOUR_SESSION_FOLDER/extracted
    ```
    The script will output the average and maximum error between text-in-video and sidecar metadata.
