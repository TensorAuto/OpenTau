Overview
========

AutoX Humanoid VLA is a codebase for training and evaluating robot policies, based on the LeRobot library.

LeRobot: State-of-the-art AI for real-world robotics
----------------------------------------------------

LeRobot aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning.

It provides a set of pretrained models, datasets with human collected demonstrations, and simulation environments to get started without assembling a robot.

Key Features
------------

* **Pretrained Models**: Access to state-of-the-art pretrained policies.
* **Datasets**: Tools to visualize and load datasets in standard formats.
* **Simulation**: Support for environments like ALOHA, PushT, and xArm.
* **Training**: Scripts for training policies using imitation learning.
* **Evaluation**: Tools for evaluating policies in simulation and on real robots.

Structure
---------

The repository is structured as follows:

* ``lerobot/configs``: Configuration classes for policies and environments.
* ``lerobot/common``: Core classes including datasets, environments, policies, and robot devices.
* ``lerobot/scripts``: Command-line scripts for training, evaluation, and data processing.
* ``examples``: Examples demonstrating how to use the library.
