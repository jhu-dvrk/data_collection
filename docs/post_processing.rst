Data Post-processing
====================

The ``extract.py`` script identifies all data in a session directory and:

1. Extracts individual frames from all recorded ``.mp4`` files using sidecar ``.json`` nanosecond timestamps.
2. Converts all recorded ROS bag topics into individual ``.csv`` files.

To process a recorded session directory:

.. code-block:: bash

    ros2 run data_collection extract -d 20260117_153206

To list the videos in a session without processing:

.. code-block:: bash

    ros2 run data_collection extract -d 20260117_153206 -l

Synchronization Verification
----------------------------

The ``check_timestamps.py`` script (located in the ``tests/`` directory) verifies the synchronization between the recording's filenames (based on system time) and the burned-in GStreamer timestamps. It uses Tesseract OCR to read the "Timestamp overlay" strip.

To verify timestamps in extracted frames:

.. code-block:: bash

    python3 tests/check_timestamps.py -d 20260117_153206/extracted

**Key Features:**

- **Automatic ROI:** Specifically targets the bottom 30px black timestamp overlay strip for speed and accuracy.
- **Latency Analysis:** Calculates the average difference between system capture time and the video's internal clock.
- **Jitter Measurement:** Calculates the standard deviation of latency across all frames.
- **Transition Detection:** Identifies frame boundaries where the integer second changes to estimate sub-second precision.
- **Validation:** Filters out OCR misreads and handles logical day-wrapping.

Video Tag Tool
--------------

The ``video_tag`` C++ application is designed for post-recording data curation. It allows users to review recorded videos with frame-accurate precision and assign temporal labels (stages) or discrete frame tags.

.. code-block:: bash

    ros2 run data_collection video_tag -v video.mp4 -c config.json

**Features:**

- **Frame-Accurate Navigation:** Seek by time or frame number with exact visualization.
- **Stage Support:** Labels defined in the config are automatically shown as toggleable ranges.
- **Tag Search:** Jump quickly between frame tags using dropdown menus.
- **Missing Tag Validation:** Automatically detects tags in the JSON file that are missing from the current configuration and asks to "Accept" them.
- **Speed Control:** Variable playback speeds (0.1x to 2.0x).
- **Session Sync:** Can automatically load tags from the session's ``tags.json`` to overlay existing annotations.

Encord Integration
------------------

The ``encord_to_tags`` script allows importing labels from the Encord platform into the project's native tag format.

.. code-block:: bash

    ros2 run data_collection encord_to_tags --encord path/to/encord.json --sidecar path/to/video_sidecar.json

This ensures that cloud-based annotations can be used seamlessly with the local extraction and curation tools.
