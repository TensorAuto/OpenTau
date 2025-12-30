Inference
=========

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Running inference with a trained model
--------------------------------------

To run inference on a trained model, you will need the saved checkpoint folder from training that contains at least ``train_config.json`` and ``model.safetensors`` files. If you ran the checkpointing and resuming tutorial, you should be able to find the checkpoint config file at ``outputs/train/tau0/checkpoints/000040/train_config.json``. Make sure you ran the ``zero_to_fp32.py`` and ``bin_to_safetensors.py`` scripts (or the ``convert_checkpoint.sh`` script) to convert the sharded model checkpoint files into a single ``model.safetensors`` file.

To run inference with the entire model on the same device, run:

.. code-block:: bash

    $ python lerobot/scripts/unified_model_inference.py --config_path=outputs/train/tau0/checkpoints/000040/train_config.json

Running inference on the TAU0 model takes less than 8 GB of GPU memory.

For an example of how to run inference with the VLM in the cloud and the action expert on the robot, run:

.. code-block:: bash

    $ python lerobot/scripts/cloud_robot_inference.py --config_path=outputs/train/tau0/checkpoints/000040/train_config.json


Zero Shot inferencing smolvla
-----------------------------

To run inference with the entire model on the same device, run:

.. code-block:: bash

    $ python lerobot/scripts/unified_model_inference.py --config_path=examples/train_config_smolvla.json

Download the smolvla weights using huggingface cli:

.. code-block:: bash

    $ huggingface-cli download lerobot/smolvla_base

Download the SmolVLM2-500M-Video-Instruct using hugging face cli:

.. code-block:: bash

    $ huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct

