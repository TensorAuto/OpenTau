Training and Checkpointing
==========================

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

.. tip::
   If you're investigating slow training or low GPU utilization, see the
   :doc:`benchmarking` guide for diagnostic scripts that break a step down
   into forward / backward / optimizer / sync phases.

Distributed Training Configuration
----------------------------------

The recommended distributed setup for the pi05 reference policy (and other policies that fit in GPU memory) is plain DDP with bf16 mixed precision. For an example, see `configs/examples/accelerate_ddp_config.yaml <https://github.com/TensorAuto/OpenTau/blob/main/configs/examples/accelerate_ddp_config.yaml>`_.

A DeepSpeed ZeRO-2 example is also available at `configs/examples/accelerate_deepspeed_config.yaml <https://github.com/TensorAuto/OpenTau/blob/main/configs/examples/accelerate_deepspeed_config.yaml>`_ for memory-constrained scenarios; it is the config used by our CI pipelines so we keep coverage for that path. For mid-sized policies like pi05, DDP is significantly faster — see issue #177 for benchmarks.

To train a model, run the following command:

.. code-block:: bash

    opentau-train --config_path=examples/pi05_config.json

This uses the default accelerate config file at `~/.cache/huggingface/accelerate/default_config.yaml` which is set by running ``accelerate config``.

Optionally, to use a specific accelerate config file (instead of the default), run:

.. code-block:: bash

    opentau-train --accelerate-config=configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/pi05_config.json

.. note::
   For advanced users: ``opentau-train`` is a convenience wrapper that invokes ``accelerate launch`` and `src/opentau/scripts/train.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/scripts/train.py>`_. The command above is equivalent to running:

   .. code-block:: bash

       accelerate launch --config_file configs/examples/accelerate_ddp_config.yaml src/opentau/scripts/train.py --config_path=configs/examples/pi05_config.json

Checkpointing and Resuming Training
-----------------------------------

Start training and saving checkpoints:

.. code-block:: bash

    opentau-train --config_path=examples/pi05_config.json --output_dir=outputs/train/pi05 --steps 40 --log_freq 5 --save_freq 20

A checkpoint should be saved at step 40. The checkpoint should be saved in the directory ``outputs/train/pi05/checkpoints/000040/``.

Under DDP the model state is saved as a single ``model.safetensors`` file directly. Under DeepSpeed (ZeRO), the model state is saved as sharded files; to consolidate them into a single ``model.safetensors`` file, run:

.. code-block:: bash

    ./convert_checkpoint.sh outputs/train/pi05/checkpoints/000040/

This generates a ``model.safetensors`` file that can be used for inference or resuming training.

Training can be resumed by running:

.. code-block:: bash

    opentau-train --config_path=outputs/train/pi05/checkpoints/000040/train_config.json --resume=true --steps=100

.. note::
   When resuming training from a checkpoint, the training step count will continue from the checkpoint's step, but the dataloader will be reset.
