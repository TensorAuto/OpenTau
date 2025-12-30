Datasets
========

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Building a dataset mixture 
--------------------------

You can define a dataset mixture in your configuration file using the ``dataset_mixture`` key. Here is an example:

.. code-block:: json

    {
        "dataset_mixture": {
            "datasets": [
                {
                    "repo_id": "physical-intelligence/libero"
                },
                {
                    "repo_id": "lerobot/droid_100"
                }
            ],
            "weights": [
                0.3,
                0.7
            ],
            "action_freq": 30.0,
        },
        ...
    }

For each new dataset, you must add an entry to ``opentau/datasets/standard_data_format_mapping.py`` to map the dataset features to the Standard Data Format (see the :ref:`Standard Data Format section <concepts/standard-data-format>` in the Concepts documentation).
Alternatively, you can provide a custom mapping in the dataset config using the ``data_features_name_mapping`` and ``loss_type_mapping`` keys.
For example:

.. code-block:: json

    {
        "dataset_mixture": {
            "datasets": [
                {
                    "repo_id": "physical-intelligence/libero"
                    "data_features_name_mapping": {
                        "camera0": "observation.images.exterior_image_1_left",
                        "camera1": "observation.images.exterior_image_2_left",
                    }
                    "loss_type_mapping": "MSE"
                },
                {
                    "repo_id": "lerobot/droid_100"
                }
            ],
            "weights": [
                0.3,
                0.7
            ],
            "action_freq": 30.0,
        },
        ...
    }

Computing max token length for dataset mixture
----------------------------------------------

Each training config should contain a dataset mixture definition. To evaluate the maximum token length for the dataset mixture, you can run the following command:

.. code-block:: bash

    python lerobot/scripts/compute_max_token_length.py \
        --target_cfg=<path/to/your/training/config.json>\
        --output_path=outputs/stats/token_count.json \
        --num_workers=10

This will output a token count for each language key in the dataset mixture, and save it to ``outputs/stats/token_count.json``.


