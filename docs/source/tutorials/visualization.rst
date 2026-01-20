Visualization of Datasets
=========================

You can visualize a given episode of a LeRobot dataset using the ``opentau-dataset-viz`` command.

The following example shows how to visualize the first episode (index 0) of the ``lerobot/droid_100`` dataset:

.. code-block:: bash

   opentau-dataset-viz --repo-id lerobot/droid_100 --episode-index 0

This command will open a `rerun <https://rerun.io>`_ window displaying the selected episode, allowing you to explore the episode interactively.

OpenTau also supports visualizing a dataset with URDF models. To do this, you need to first install ``opentau`` with optional URDF support:

.. code-block:: bash

   pip install "opentau[urdf]"

or, if you are installing with uv from source:

.. code-block:: bash

    git clone https://github.com/TensorAuto/OpenTau.git
    cd OpenTau
    uv sync --extra urdf
    source .venv/bin/activate

Then, you can visualize an episode with URDF models using the following command:

.. code-block:: bash

   opentau-dataset-viz --repo-id lerobot/droid_100 --episode-index 0 --urdf </path/to/your/model.urdf>

If your URDF model comes with its own package of meshes, you can specify the package path using the ``--urdf-package-dir`` argument:

.. code-block:: bash

   opentau-dataset-viz --repo-id lerobot/droid_100 --episode-index 0 --urdf </path/to/your/model.urdf> --urdf-package-dir </root/of/your/urdf/package>

If no ``--urdf-package-dir`` is passed, the environment variable ``ROS_PACKAGE_PATH`` will be used to locate packages:

.. code-block:: bash

   export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:</root/of/your/urdf/package>
   opentau-dataset-viz --repo-id lerobot/droid_100 --episode-index 0 --urdf </path/to/your/model.urdf>
