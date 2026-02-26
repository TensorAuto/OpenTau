.. _human_demo:

Collecting Human Demos
=======================

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

OpenTau supports training VLAs on human demonstration data collected in LeRobot format. There are two ways to collect human demos:

1. **RecordHuman VR app** (recommended) — record hand and head poses directly from a PICO VR headset with 3D tracking.
2. **MediaPipe video conversion** — extract poses from ordinary MP4 videos using MediaPipe landmark detection.

.. note::
   RecordHuman is recommended because it captures a full 7-D pose (3D position + quaternion orientation) for every hand joint in camera space and tracks head movement through space, giving richer action representations.
   MediaPipe, by contrast, only provides 3D positions relative to the hand's own center, so it cannot capture how the hand moves through the scene.

.. _human_demo_vr:

Option 1: RecordHuman VR App (Recommended)
------------------------------------------

The `RecordHuman <https://github.com/TensorAuto/RecordHuman>`_ Unity app runs on a PICO VR headset and records ego-perspective video together with hand and head pose data.

Step 1: Install RecordHuman on a PICO headset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the setup instructions in the `RecordHuman README <https://github.com/TensorAuto/RecordHuman#readme>`_ to install the app on your PICO VR headset.

Step 2: Record demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Launch RecordHuman on the headset and perform the task you want to demonstrate. The app saves a video file and a JSON pose file for each recording.

Step 3: Convert to LeRobot format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``recordhuman_to_lerobot.py`` to convert the recorded data into a LeRobot dataset that OpenTau can train on.

**Basic usage:**

.. code-block:: bash

   python -m opentau.scripts.recordhuman_to_lerobot \
       --video recording.mp4 \
       --poses recording.json \
       --output ./datasets/my_vr_dataset \
       --prompt "Pick up the snack bag"

**Specify a target FPS** (e.g. 10 Hz). The overlay video, if requested, still uses the original video FPS:

.. code-block:: bash

   python -m opentau.scripts.recordhuman_to_lerobot \
       --video recording.mp4 \
       --poses recording.json \
       --output ./datasets/my_vr_dataset \
       --prompt "Pick up the snack bag" \
       --fps 10

**Generate a skeleton overlay video** for visual inspection:

.. code-block:: bash

   python -m opentau.scripts.recordhuman_to_lerobot \
       --video recording.mp4 \
       --poses recording.json \
       --output ./datasets/my_vr_dataset \
       --prompt "Pick up the snack bag" \
       --overlay overlay.mp4

The conversion script produces a dataset with:

- Frames as ``observation.images.camera``
- Camera-space hand joint poses (364-D) as ``observation.state``
- Next-step hand state + delta head pose (371-D) as ``action``
- The task prompt you provide

**Full list of options:**

.. code-block:: text

   --video            Path to input video file (required)
   --poses            Path to JSON pose data file (required)
   --output           LeRobot dataset output directory; must not exist (required)
   --prompt           Task description for the episode (required)
   --overlay          Write a skeleton overlay video to this path
   --fov              Vertical FOV in degrees for projection, overlay only (default: 90)
   --time-offset      Seconds added to video timestamps for pose alignment (default: 0)
   --tracking-origin  XR origin offset subtracted from head_pos (default: -0.32 2.0276 0)
   --eye-offset       Horizontal eye offset in meters (default: 0.06)
   --fps              Output dataset FPS; defaults to the video's FPS
   --overlay-codec    FourCC codec for the overlay video (default: mp4v)


.. _human_demo_mediapipe:

Option 2: MediaPipe Video Conversion
-------------------------------------

If you don't have a VR headset, you can convert ordinary MP4 videos of human demonstrations into LeRobot datasets. The ``human_video_to_lerobot.py`` script uses MediaPipe for pose (third-person / exo) or hand (first-person / ego) landmark detection and writes frames, 3D landmarks as state, and next-step landmarks as action.

Each video becomes one episode with:

- Frames as ``observation.images.camera``
- 3D pose or hand landmarks as ``observation.state``
- Next-step landmarks as ``action``
- A task prompt you provide (e.g. "Pick up the cup")

Converting videos
^^^^^^^^^^^^^^^^^

From the project root, run the conversion script. The **output path is the LeRobot dataset root** and must not exist yet.

**Single video (exo — third-person pose):**

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \
       /path/to/demo.mp4 \
       ./datasets/my_exo_dataset \
       --prompt "Pick up the red block"

**Single video (ego — first-person hands):**

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \
       /path/to/ego_demo.mp4 \
       ./datasets/my_ego_dataset \
       --prompt "Open the drawer" \
       --mode ego

**Use a specific FPS for the dataset** (e.g. 10 Hz). The overlay video (if requested) still uses the original video FPS:

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \
       /path/to/demo.mp4 \
       ./datasets/my_dataset \
       --prompt "Place the cup on the table" \
       --fps 10

**Save a landmark-overlay video** for inspection:

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \
       /path/to/demo.mp4 \
       ./datasets/my_dataset \
       --prompt "Pick up the cup" \
       --overlay /path/to/overlay.mp4
