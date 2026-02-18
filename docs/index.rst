Introduction
============

A multi-stream video record application using C++, GStreamer, and GTKmm. It allows creating flexible video streams via JSON configuration, provides live previews, and supports synchronized recording with timestamps. ROS topics can also be recorded along the videos. Note that the videos are recorded directly from the source using GStreamer and don't rely on ROS topics.

.. image:: _static/live_preview.png
   :alt: Multi-stream Video Data Record Preview
   :align: center
   :width: 600

The application also integrates with ROS2 for remote control and status monitoring.

Workflow and Shared Files
-------------------------

The suite of tools provided in this repository is designed to work together in a sequence. This section describes the data flow and how shared files connect the different programs.

Data Flow Overview
~~~~~~~~~~~~~~~~~~

1. **Configuration**: A master JSON configuration file defines the hardware setup and recording parameters.
2. **Recording**: The :doc:`record` application uses the configuration to capture video, audio, and ROS topics.
3. **Curation**: The :doc:`video_tag` tool loads the session to add temporal labels and frame-accurate tags.
4. **Extraction**: The :doc:`extract` tool processes the recorded videos and ROS bags to generate training-ready data (images and CSVs).
5. **Validation**: The :doc:`latency` tools verify syncing and measuring system performance.

Shared Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``config.json``: The primary configuration file used by :doc:`record <record>`, :doc:`video_tag <video_tag>`, and ``video_latency`` (in :doc:`latency`). It ensures all tools use the same video stream definitions and labels.
- ``JSON Schema``: Defines the structure of the configuration files, facilitating reuse and composition.

Session Directory Output
~~~~~~~~~~~~~~~~~~~~~~~~

When :doc:`record <record>` completes a session, it creates a directory (named by its timestamp, e.g., ``2026/02/18_025457/``) containing:

- **Video Files**: Recorded ``.mp4`` files for each enabled camera.
- **Sidecar Timestamps**: JSON files named ``camera_name_YYMMDD_HHMMSS_timestamps.json`` containing nanosecond-accurate epoch timestamps for every frame.
- **ROS bags**: Recorded data topics stored in ROS2 bag format.
- **index.json**: A summary of durations, metadata, and data paths.

Downstream Consumption
~~~~~~~~~~~~~~~~~~~~~~

The session directory and its contents are consumed by:

- :doc:`video_tag <video_tag>`: Loads the video files and uses ``index.json`` to identify available streams. It generates or updates a ``tags.json`` file in the session directory.
- :doc:`extract <extract>`: Uses the sidecar timestamps to extract frames from ``.mp4`` at the exact original nanoseconds and converts ROS bags into easy-to-use CSVs.
- :doc:`check_timestamps <latency>`: Uses extracted frames to verify synchronization against burned-in overlays.

.. toctree::
   :maxdepth: 2
   :caption: Documentation sections:

   installation
   record
   video_tag
   extract
   latency

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
