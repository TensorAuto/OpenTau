Evaluation
==========

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Evaluating a policy in Simulation
---------------------------------

To evaluate a policy in simulation, you can launch the ``lerobot/scripts/eval.py`` script with ``accelerate launch``.
Each accelerate process will only work on its fraction of the tasks, improving throughput.
For example, to evaluate a policy on the LIBERO 10, run:

.. code-block:: bash

    $ accelerate launch --config_file <ACCELERATE_CONFIG_PATH> lerobot/scripts/eval.py --config_path=outputs/train/tau0/checkpoints/000040/train_config.json

.. note::
   You can't pass in an DeepSpeed accelerate config file to ``eval.py`` as DeepSpeed expects optimizer and dataloader during ``accelerator.prepare()``, which we do not provide during eval. It is recommended to pass in a DDP config.

.. note::
   Make sure that the ``EnvConfig`` and ``EvalConfig`` are set to the correct values for the simulation environment in your train config file.

Evaluating policy in a libero environment
-----------------------------------------

To evaluate the policy on the LIBERO benchmark, add the following section to the training config:

.. code-block:: json

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

When launched with accelerate, each GPU process will only work on its fraction of the tasks, improving throughput.

Using Simulations
-----------------

Metaworld
^^^^^^^^^

When using Metaworld on Ubuntu machines with headless rendering, make sure to export these environment variables:

.. code-block:: bash

    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl

