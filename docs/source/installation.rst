Installation
============

Requirements
------------

Supported Operating Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Ubuntu 20.04 or newer is required.**
- Other Linux distributions and macOS may work but are not officially supported.
- Windows is **not supported**.

GPU Requirements
^^^^^^^^^^^^^^^^

- An NVIDIA GPU is required. Minimum recommended GPU/VRAM for each use case:

  +------------------------+------------------+-----------------------------------+
  |        **Mode**        | **Memory (VRAM)**|         **Example GPU**           |
  +========================+==================+===================================+
  | Inference              | > 8 GB           | RTX 3090                          |
  +------------------------+------------------+-----------------------------------+
  | Training               | > 70 GB          | A100 (80GB) / H100                |
  +------------------------+------------------+-----------------------------------+

- For most purposes, **training and inference require NVIDIA GPUs with recent CUDA support** (CUDA 11+, commonly available with driver version 450+).

- Multi-GPU setups (A100, H100, etc.) should be used for large-scale training.


Download Source Code
---------------------

Download the source code:

.. code-block:: bash

    git clone https://github.com/TensorAuto/OpenTau.git
    cd OpenTau


Environment Setup
-----------------

We recommend using `uv <https://docs.astral.sh/uv/>`_ for fast and simple Python dependency management.

1. **Install uv**
   Follow the `official uv installation instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.

2. **Install dependencies**
   Sync all required dependencies and optional extras with:

   .. code-block:: bash

      uv sync --extra tau0 --extra test --extra video_benchmark --extra accelerate --extra dev --extra feetech --extra openai --extra onnx --extra smolvla --extra libero --extra metaworld

3. **Activate the virtual environment**

   .. code-block:: bash

      source .venv/bin/activate


Docker Installation (Optional)
-------------------------------------------

You can also use Docker to install and run OpenTau.

1. **Build the Docker image**

   .. code-block:: bash

      docker build -t opentau .

2. **Run the Docker container**

   .. code-block:: bash

      docker run -it --gpus all opentau /bin/bash

   Note: The ``--gpus all`` flag requires the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.


Experiment Tracking
-------------------

To use `Weights and Biases <https://docs.wandb.ai/quickstart>`_ for experiment tracking, log in with:

.. code-block:: bash

    wandb login


Distributed Training Configuration
----------------------------------

Configure accelerate for your distributed training setup:

.. code-block:: bash

    accelerate config

This will create an accelerate config file at `~/.cache/huggingface/accelerate/default_config.yaml`. We are currently using DeepSpeed ZeRO2 for model parallelism distributed training. For an accelerate config example, see `this config file <../../examples/accelerate_ci_config.yaml>`_ used for our CI pipelines.
