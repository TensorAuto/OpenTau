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


Running inference with autoregressive response prediction
----------------------------------------------------------

To run inference with autoregressive response prediction, you will need to use the ``pi05_inference_config.json`` file.
Set the predict_response flag to true in the policy config.
Example of important config fields for inference with autoregressive response prediction:

.. code-block:: javascript

   {
    "dataset_mixture": {
        "datasets": [
            {
                "repo_id": "physical-intelligence/libero"
            }

        ],
        "weights": [
            1.0
        ],
        "action_freq": 10.0,
        "image_resample_strategy": "nearest",
        "vector_resample_strategy": "nearest"
    },
    "policy": {
        "type": "pi05",
        "pretrained_path": "TensorAuto/pi05_base",
        "n_obs_steps": 1,
        ...
        "predict_response": true,
        ...
    }
