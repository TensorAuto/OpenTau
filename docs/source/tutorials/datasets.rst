Datasets
========

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Building a dataset mixture
--------------------------

You can define a dataset mixture in your configuration file using the ``dataset_mixture`` key. Here is an example:

.. code-block:: javascript

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

The ``weights`` field is optional. If you set ``"weights": null`` (or omit the field),
OpenTau infers weights from dataset lengths at runtime (``float(len(dataset))`` for each dataset).
Provide explicit ``weights`` only when you want custom sampling ratios.

For each new dataset, you must add an entry to `src/opentau/datasets/standard_data_format_mapping.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/datasets/standard_data_format_mapping.py>`_ to map the dataset features to the Standard Data Format.
Alternatively, you can provide a custom mapping in the dataset config using the ``data_features_name_mapping`` and ``loss_type_mapping`` keys.
For example:

.. code-block:: javascript

    {
        "dataset_mixture": {
            "datasets": [
                {
                    "repo_id": "physical-intelligence/libero",
                    "data_features_name_mapping": {
                        "camera0": "observation.images.exterior_image_1_left",
                        "camera1": "observation.images.exterior_image_2_left"
                    },
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

    python src/opentau/scripts/compute_max_token_length.py \
        --target_cfg=<path/to/your/training/config.json>\
        --output_path=outputs/stats/token_count.json \
        --num_workers=10

This will output a token count for each language key in the dataset mixture, and save it to ``outputs/stats/token_count.json``.

Adding subtask responses to a dataset
--------------------------------------

Some policies (e.g. π0.5) can be trained with per-frame subtask annotations that tell the model *what* sub-goal is active at each timestep.
The ``add_subtask_response`` script reads per-episode subtask JSON files, converts the time-based subtask boundaries to frame indices using the dataset FPS, and writes the active subtask string into a ``response`` column in each episode parquet file.

Prerequisites:

- Each dataset must have a ``subtask_path`` field in its ``meta/info.json`` that points to per-episode subtask JSON files
  (e.g. ``"subtask_path": "subtask/episode_{episode_index:06d}.json"``).
- Each subtask JSON is a list of objects with ``"time"`` (in seconds) and ``"subtask"`` (a string) keys:

  .. code-block:: javascript

      [
          {"time": 0.0, "subtask": "pick up the cup"},
          {"time": 2.5, "subtask": "pour water into the cup"},
          {"time": 5.1, "subtask": "place the cup on the table"}
      ]

Create a config file that lists the datasets you want to process, each with a local ``root`` path:

.. code-block:: javascript

    {
        "datasets": [
            {
                "repo_id": "TensorAuto/ice-lemonade",
                "root": "/path/to/local/dataset"
            }
        ]
    }

Then run the script:

.. code-block:: bash

    python src/opentau/scripts/add_subtask_response.py \
        --config_path=configs/examples/add_subtask_response.json

The script will:

1. Read ``meta/info.json`` for each dataset to determine the FPS and subtask file path template.
2. For each episode, load the subtask JSON, map time-based boundaries to frame indices, and
   assign the active subtask string to every frame in that range.
3. Write (or overwrite) the ``response`` column in the episode parquet file.
4. Add a ``response`` feature entry to ``meta/info.json`` if it doesn't already exist.

If a subtask JSON is missing for an episode, the ``response`` column is filled with empty strings and a warning is emitted.

To use the subtask responses during training, map the ``response`` key in your dataset config:

.. code-block:: javascript

    {
        "dataset_mixture": {
            "datasets": [
                {
                    "repo_id": "TensorAuto/IceLemonade_100",
                    "data_features_name_mapping": {
                        "camera0": "observation.images.rgb",
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                        "response": "response"
                    }
                }
            ],
            ...
        },
        ...
    }
