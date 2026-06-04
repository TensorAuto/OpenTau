Evaluation
==========

OpenTau supports evaluation in asynchronous vectorized simulation environments during training and creating validation dataset splits during training.

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Evaluating a policy in simulation
---------------------------------

Make sure to set the ``EnvConfig`` and ``EvalConfig`` in the training config file (example shown in :ref:`evaluating-libero`).
To evaluate a policy in simulation, you can launch the `src/opentau/scripts/eval.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/scripts/eval.py>`_ script with ``opentau-eval``.
To run simulation rollouts during training, set the ``eval_freq`` to a non-zero value in the training config file.
Each accelerate process will only work on its fraction of the tasks, improving throughput.
For example, to evaluate a policy on the LIBERO 10, run:

.. code-block:: bash

    opentau-eval --accelerate-config <ACCELERATE_CONFIG_PATH> --config_path=outputs/train/pi05/checkpoints/000040/train_config.json

.. note::
   You can't pass in an DeepSpeed accelerate config file to ``eval.py`` as DeepSpeed expects optimizer and dataloader during ``accelerator.prepare()``, which we do not provide during eval. It is recommended to pass in a DDP config.

.. note::
   Make sure that the ``EnvConfig`` and ``EvalConfig`` are set to the correct values for the simulation environment in your train config file.

.. _evaluating-libero:

Evaluating a policy in a LIBERO environment
-------------------------------------------

OpenTau supports simulated evaluation in both the `LIBERO benchmark <https://libero-project.github.io/main.html>`_ and the `RoboCasa365 <https://robocasa.ai/>`_ kitchen simulation. To evaluate a policy on LIBERO, add the following section to the training config:

.. code-block:: javascript

    {
        ...,
        "env": {
            "type": "libero",
            "task": "libero_spatial",
            "task_ids": [0, 2]
        },
        "eval": {
            "n_episodes": 8,
            "batch_size": 8
        },
        "eval_freq": 25,
        ...
    }

This will run the 0th task and 2nd task in ``libero_spatial``. Each task will run for 8 simulations in parallel.

.. _evaluating-robocasa:

Evaluating a policy in a RoboCasa environment
---------------------------------------------

OpenTau also evaluates in the `RoboCasa365 <https://robocasa.ai/>`_ kitchen
simulation, in-process and vectorized just like LIBERO (no external server
required). Install the simulator with the ``robocasa`` extra
(``uv sync --extra robocasa`` or ``uv sync --all-extras``); kitchen assets
auto-download on the first env build. See :doc:`/tutorials/robocasa` for setup
details and an external rollout-client alternative.

Set ``env.type`` to ``robocasa`` in the training config (see
``configs/examples/pi05_robocasa_eval_config.json`` for a complete example):

.. code-block:: javascript

    {
        ...,
        "env": {
            "type": "robocasa",
            "task": "CloseFridge",
            "camera_name": "robot0_agentview_left,robot0_eye_in_hand,robot0_agentview_right",
            "metadata": {
                "robot_type": "PandaOmron",
                "control_mode": "ee"
            }
        },
        "eval": {
            "n_episodes": 2,
            "batch_size": 2,
            "use_async_envs": true,
            "control_mode": "ee"
        },
        "eval_freq": 25,
        ...
    }

Run headless (e.g. on a GPU server) with ``MUJOCO_GL=egl``. As with LIBERO, each
RoboCasa task is its own group, so eval reports a per-task success rate and a
per-task video grid, and tasks shard across accelerate ranks.

Running validation during training
----------------------------------

To run validation during training, set the ``val_freq`` to a non-zero value in the training config file.
This will create a validation dataset split and run validation every ``val_freq`` steps.
You can specify the validation split ratio using ``val_split_ratio`` inside ``dataset_mixture``.
The ``val_split_ratio`` value will apply to all datasets in the mixture.
If ``val_freq`` is set to 0, a validation dataset will not be created and ``val_split_ratio`` will be ignored.

For example, to create a validation dataset split of 10% of the training dataset, set the ``val_freq`` to 1000 and ``val_split_ratio`` to 0.1 in the training config file.
This will run validation every 1000 steps and create a validation dataset split of 10% of the training dataset.

.. code-block:: javascript

    {
        ...,
        "val_freq": 1000,
        "dataset_mixture": {
            ...,
            "val_split_ratio": 0.1
        }
        ...
    }
