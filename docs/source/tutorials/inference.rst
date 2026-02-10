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

    python src/opentau/scripts/inference.py --config_path=outputs/train/pi05/checkpoints/000040/train_config.json


Running inference with autoregressive response prediction
----------------------------------------------------------

To run inference with autoregressive response prediction, set the predict_response flag to true in the policy config as shown below.
For now, we only support greedy decoding for response prediction.
Example of important config fields for inference with autoregressive response prediction:

.. code-block:: javascript

   {
    ...
    "policy": {
        "type": "pi05",
        "pretrained_path": "TensorAuto/pi05_base",
        "n_obs_steps": 1,
        ...
        "predict_response": true,
        ...
    }
    ...
   }


Running inference with ONNX and TensorRT
----------------------------------------

You can export a trained PI05 policy to ONNX and run inference with ONNX Runtime or TensorRT for deployment or faster GPU inference.

**Prerequisites**

- A trained checkpoint with ``train_config.json`` and ``model.safetensors`` (same as for standard inference).
- For TensorRT inference: install the optional TensorRT extra: ``uv sync --extra trt``.

**Step 1: Export the model to ONNX**

Export uses the same train config as training. The ONNX model is written to the directory given by ``policy.pretrained_path`` in that config (typically your checkpoint directory). Two files are produced: ``model.onnx`` (graph) and ``model.onnx.data`` (weights; used for models with large weights).

.. code-block:: bash

    python3 src/opentau/scripts/export_to_onnx.py --config_path=outputs/train/pi05/checkpoints/000040/train_config.json

Notes:

- Only PI05 policies are supported for ONNX export.
- The script exports the core tensor operations with pre-tokenized inputs; tokenization and state discretization are done outside the ONNX graph (e.g. in the inference scripts).
- Export uses ``float32``. Since the PI05 model is larger than ONNX size limit (2 GiB), weights are stored in external data (``model.onnx.data``). Keep both files in the same directory when loading.

**Step 2: Run ONNX inference**

Use the directory that contains ``model.onnx`` and ``model.onnx.data`` as ``--checkpoint_dir``. You can optionally dump inputs and outputs of the first run to a pickle file for debugging.

.. code-block:: bash

    python3 src/opentau/scripts/onnx_inference.py --checkpoint_dir=outputs/train/pi05/checkpoints/000040/ --dump_path=/path/to/dump/inputs_and_outputs

Optional arguments (defaults in parentheses): ``--num_cams`` (2), ``--resolution_height`` / ``--resolution_width`` (224), ``--prompt_max_length`` (256), ``--n_action_steps`` (10), ``--max_action_dim`` (32), ``--max_state_dim`` (32), ``--delay`` (1), ``--predict_response`` (false), ``--prompt``, ``--n_repeats`` (10), ``--provider`` (CUDA/CPU auto), ``--seed`` (42). The script uses the ``google/paligemma-3b-pt-224`` tokenizer and reports latency statistics over the repeated runs.

**Step 3: Run TensorRT inference (GPU, FP16)**

TensorRT uses the same ONNX artifact but runs it with the TensorRT execution provider in FP16 for faster GPU inference. The first run can take several minutes while TensorRT builds and caches the engine.

.. code-block:: bash

    python3 src/opentau/scripts/tensorrt_inference.py --checkpoint_dir=outputs/train/pi05/checkpoints/000040/ --dump_path=/path/to/dump/inputs_and_outputs

Optional arguments inherit from the ONNX inference script; in addition you can set ``--engine_cache_dir`` to a directory to cache the TensorRT engine for faster subsequent loads.
