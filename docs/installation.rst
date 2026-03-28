Installation & Prerequisites
============================

Before starting your data collection pipeline, follow these steps to set up your environment.

The application is a ROS2 node and requires a working ROS2 installation (Humble, Jazzy, etc.).

System Dependencies
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    sudo apt update
    sudo apt install build-essential cmake pkg-config \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
        libgtkmm-3.0-dev gstreamer1.0-gtk3 libjsoncpp-dev \
        python3-opencv python3-numpy

Python Dependencies
~~~~~~~~~~~~~~~~~~~

- **GStreamer (gi)**: Core pipeline handling for video extraction.
- **opencv-python**: Used by the extract and post-processing scripts.
- **rclpy**, **rosbag2_py**: For ROS 2 integration and bag processing.

NVIDIA Hardware Acceleration
----------------------------

To use NVIDIA GPUs for hardware-accelerated video encoding (NVENC), you need to have the NVIDIA drivers and the GStreamer ``nvcodec`` plugin installed. This is often provided by the NVIDIA DeepStream SDK.

.. code-block:: bash

    sudo apt update
    sudo apt install libnvidia-encode-*

To verify availability, run:

.. code-block:: bash

    gst-inspect-1.0 nvh264enc

If you just installed DeepStream or new GStreamer plugins and they are not appearing in ``gst-inspect-1.0``, you may need to clear the GStreamer registry cache:

.. code-block:: bash

    rm -rf ~/.cache/gstreamer-1.0

Installation & Environment
--------------------------

Before running the application, ensure your ROS2 environment is sourced:

.. code-block:: bash

    source /opt/ros/YOUR_DISTRO/setup.bash
    # or if using a workspace
    source install/setup.bash

To build the workspace:

.. code-block:: bash

    colcon build --symlink-install
