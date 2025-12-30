Inference
=========

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Running inference with a trained model
--------------------------------------

To run inference on a trained model, you will need the saved checkpoint folder from training that contains at least these two files: ``train_config.json`` and ``model.safetensors``.
If you ran the :doc:`checkpointing and resuming tutorial </tutorials/training>`, you should be able to find the checkpoint config file at ``outputs/train/pi05/checkpoints/000040/train_config.json``.

To run inference, run the following command:

.. code-block:: bash

    python lerobot/scripts/inference.py --config_path=outputs/train/pi05/checkpoints/000040/train_config.json
