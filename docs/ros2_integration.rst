ROS2 Integration
================

The record functions as a ROS2 node named ``record``.

Video Streaming
---------------

Any video stream can be published as a ROS 2 image topic by setting ``ros_camera_name`` in its configuration:

.. code-block:: json

    {
      "videos": [
        {
          "name": "main_camera",
          "stream": "v4l2src device=/dev/video0",
          "ros_camera_name": "main",
          "encoding": { "frame_rate": 30, "width": 640, "height": 480 }
        }
      ]
    }

This publishes the video frames on ``main/image_raw`` (``sensor_msgs/msg/Image``, RGB8 encoding) and ``main/camera_info`` (``sensor_msgs/msg/CameraInfo``) using ``image_transport::CameraPublisher``. The video is captured and encoded directly by GStreamer — the ROS topic is an additional output branch that does not affect recording performance.

To view the stream:

.. code-block:: bash

    # Using rqt_image_view (click refresh to discover topics)
    ros2 run rqt_image_view rqt_image_view
    # Or specify the topic directly
    ros2 run image_view image_view --ros-args -r image:=/main/image_raw

Control Topics
--------------

==================== ==================== ============ ==================================================
Topic                Type                 Direction    Description
==================== ==================== ============ ==================================================
``record/record``    ``std_msgs/msg/Bool`` Subscriber   Send ``true`` to start recording, ``false`` to stop.
``record/recording`` ``std_msgs/msg/Bool`` Publisher    Publishes actual recording state (``true`` if recording).
==================== ==================== ============ ==================================================

Command Line Examples
---------------------

Ensure you have sourced your ROS2 environment (e.g., ``source /opt/ros/humble/setup.bash``).

**Start Recording:**

.. code-block:: bash

    ros2 topic pub /record/record std_msgs/msg/Bool "{data: true}" --once

**Stop Recording:**

.. code-block:: bash

    ros2 topic pub /record/record std_msgs/msg/Bool "{data: false}" --once

**Monitor Status:**

.. code-block:: bash

    ros2 topic echo /record/recording
