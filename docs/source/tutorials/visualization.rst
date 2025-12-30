Visualization
=============

Using so100_visualization script
--------------------------------

The ``so100_visualization`` script imitates action from lerobot format dataset in simulator. To run ``bi-so100-block-manipulator`` dataset, pull the data from git lfs and move it under ``lerobot/lerobot`` directory. Install the simulator using:

.. code-block:: bash

    $ uv sync --extra tau0 --extra pusht --extra test --extra video_benchmark --extra accelerate --extra dev --extra feetech --extra openai --extra onnx --extra smolvla --extra so100
    $ source .venv/bin/activate

Then simply run the below script:

.. code-block:: bash

    python lerobot/scripts/so100_visualization.py --config_path=configs/so100/so100_viz_config.json

