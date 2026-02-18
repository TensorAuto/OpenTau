.. _human_demo:

Converting Human Demos to LeRobot Format
========================================

This tutorial describes how to convert human demonstration videos into LeRobot-format datasets for training VLAs. The script uses MediaPipe for pose (third-person / exo) or hand (first-person / ego) landmark detection and writes frames, 3D landmarks as state, and next-step landmarks as action.

Overview
--------

Run ``human_video_to_lerobot.py`` on one or more MP4s. Each video becomes one episode with:

- Frames as ``observation.images.camera``
- 3D pose or hand landmarks as ``observation.state``
- Next-step landmarks as ``action``
- A task prompt you provide (e.g. "Pick up the cup")

Prerequisites
------------

- OpenTau installed (see :doc:`/installation`).
- One or more MP4 videos of human demonstrations (exo = full body in frame, ego = hand(s) in frame).

Converting videos
-----------------

From the project root, run the conversion script. The **output path is the LeRobot dataset root** and must not exist yet.

**Single video (exo — third-person pose):**

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \\
       /path/to/demo.mp4 \\
       ./datasets/my_exo_dataset \\
       --prompt "Pick up the red block"

**Single video (ego — first-person hands):**

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \\
       /path/to/ego_demo.mp4 \\
       ./datasets/my_ego_dataset \\
       --prompt "Open the drawer" \\
       --mode ego

**Use a specific FPS for the dataset** (e.g. 10 Hz). The overlay video (if requested) still uses the original video FPS:

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \\
       /path/to/demo.mp4 \\
       ./datasets/my_dataset \\
       --prompt "Place the cup on the table" \\
       --fps 10

**Save a landmark-overlay video** for inspection:

.. code-block:: bash

   python -m opentau.scripts.human_video_to_lerobot \\
       /path/to/demo.mp4 \\
       ./datasets/my_dataset \\
       --prompt "Pick up the cup" \\
       --overlay /path/to/overlay.mp4
