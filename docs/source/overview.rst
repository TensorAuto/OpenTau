Overview
========

OpenTau is a PyTorch toolkit for training vision-language-action (VLA) models while fully leveraging the key techniques introduced in the :math:`\pi`-series papers.

Motivation
----------

The :math:`\pi` series policies are by far the most popular models in today's VLA community. While many training techniques have been described in the :math:`\pi` papers, only a small fraction of these techniques have been implemented in open-source repositories.

Whether you use the official OpenPi codebase or LeRobot's reimplementation, you are missing out on a lot of the key components that make the :math:`\pi` models perform so well.

OpenTau (:math:`\tau`) is a tool developed by `Tensor <https://www.tensor.auto>`_ to bridge this gap.

Key Features
------------

OpenTau aims to make it easy to train VLAs on your own datasets while including the techniques that matter, such as:

- Co-training on an adjustable mixture of heterogeneous datasets
- Discrete actions for fast VLM convergence in :math:`\pi_{0.5}`
- Knowledge insulation between the VLM backbone and the action expert
- Response prediction in :math:`\pi_{0.5}` for training on VQA datasets and high level planning while inferencing.
- Dropout layers in the VLM that prevent overfitting
- A reinforcement learning pipeline described in :math:`\pi^*_{0.6}`
- Multi-node and multi-GPU training
- Simulation environments for evaluating models

Quick Start
-----------

If you are familiar with LeRobot, getting started with OpenTau is very easy. Because OpenTau is a fork of the popular LeRobot repository, any LeRobot-compliant policy and dataset can be used directly with OpenTau.

Checkpoints
-----------

We provide fully functioning :math:`\pi_{0.5}` checkpoints trained on the LIBERO dataset with high success rates.

- `TensorAuto/tPi0.5-libero <https://huggingface.co/TensorAuto/tPi0.5-libero>`_: A :math:`\pi_{0.5}` model checkpoint trained on the LIBERO dataset with discrete actions and knowledge insulation.

Acknowledgements
----------------

We would like to thank the original authors of the :math:`\pi` series `papers <https://www.pi.website/blog>`_ for their groundbreaking work in the VLA field.

We also acknowledge the contributions of the open-source community, especially `LeRobot <https://huggingface.co/lerobot>`_, for their efforts in re-implementing the :math:`\pi` models and standardizing training infrastructure.

OpenTau builds upon these foundations to provide a more accessible and comprehensive tool for training vision-language agents.
