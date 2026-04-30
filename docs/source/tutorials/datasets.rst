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

Automatically annotating subtasks with a VLM
---------------------------------------------

``annotate_subtasks.py`` generates per-frame subtask labels for every episode in a dataset
mixture automatically, using ``claude-opus-4-7`` as a vision-language model.  It is the
recommended way to produce the ``response`` column required by policies such as π0.5.

How it works
^^^^^^^^^^^^

For each episode the script:

1. Samples **1 frame/second** from the episode video (configurable via ``--sample-fps``),
   giving a 30-50× reduction over the raw frame rate.
2. Resizes each frame to 640 px wide (configurable via ``--target-width``) before JPEG
   encoding, keeping image-token costs low.
3. Sends all sampled frames with their timestamps to ``claude-opus-4-7`` in a **single API
   call** and asks it to identify subtask transition times.
4. Saves the returned boundaries as a per-episode JSON file under
   ``{root}/subtasks/episode_{N:06d}.json``.
5. Writes a ``subtask_path`` field to ``meta/info.json``.
6. Expands the boundaries into a per-frame ``response`` column in each episode parquet
   (can be skipped with ``--no-write-response-column``).

Episodes whose subtask JSON already exists are skipped, making the script **fully resumable**
after a crash.

Prerequisites
^^^^^^^^^^^^^

Set your Anthropic API key:

.. code-block:: bash

    export ANTHROPIC_API_KEY="sk-ant-..."

Datasets with a ``root`` field are processed directly from disk.  Hub-only datasets (no
``root``) are automatically downloaded to
``~/.cache/huggingface/opentau_subtasks/`` via ``snapshot_download`` before processing.

.. note::
   Only LeRobot **v2.1** datasets are supported.  If a dataset on the Hub has been
   upgraded to v3.0, pin it to its ``v2.1`` tag using the ``revision`` field (see the
   example config below).

Running the script
^^^^^^^^^^^^^^^^^^

Create (or reuse) a dataset mixture config.  Both the full training-config format and the
simpler ``datasets``-only format are accepted:

.. code-block:: javascript

    // configs/examples/train_mixture_config.json
    {
        "dataset_mixture": {
            "datasets": [
                {
                    "repo_id": "lerobot/droid_100",
                    "revision": "v2.1"
                },
                {
                    "repo_id": "TensorAuto/my-local-dataset",
                    "root": "/path/to/local/dataset"
                }
            ]
        }
    }

Then run:

.. code-block:: bash

    python src/opentau/scripts/annotate_subtasks.py \
        --config-path configs/examples/train_mixture_config.json

For a dry run that annotates only 1 episode per dataset and skips the parquet update:

.. code-block:: bash

    python src/opentau/scripts/annotate_subtasks.py \
        --config-path configs/examples/train_mixture_config.json \
        --max-episodes-per-dataset 1 \
        --no-write-response-column

Full list of flags:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--config-path``
     - *(required)*
     - Path to dataset mixture config JSON.
   * - ``--sample-fps``
     - ``1.0``
     - Frames per second to sample from each episode video.  Lower values reduce API cost.
   * - ``--target-width``
     - ``640``
     - Resize frames to this width (px) before encoding as JPEG.
   * - ``--subtask-path-template``
     - ``subtasks/episode_{episode_index:06d}.json``
     - Template for per-episode subtask JSON paths, relative to the dataset root.
   * - ``--model``
     - ``claude-opus-4-7``
     - Anthropic model ID to use.
   * - ``--write-response-column`` / ``--no-write-response-column``
     - enabled
     - Whether to expand subtask boundaries into a ``response`` parquet column after annotation.
   * - ``--max-episodes-per-dataset``
     - *(none)*
     - Cap the number of episodes processed per dataset — useful for dry runs.
   * - ``--hub-cache-dir``
     - ``~/.cache/huggingface/opentau_subtasks``
     - Directory for caching Hub dataset downloads.

Output
^^^^^^

For each processed episode the script writes a JSON file at the ``subtask_path`` template:

.. code-block:: javascript

    [
        {"time": 0.0,  "subtask": "approach the marker on the table"},
        {"time": 4.0,  "subtask": "grasp the marker"},
        {"time": 6.0,  "subtask": "lift and move marker toward pot"},
        {"time": 10.0, "subtask": "place marker into the pot"}
    ]

``time`` is in seconds from the start of the episode.  When ``--write-response-column`` is
enabled (the default), the script also:

- Adds a ``response`` column to each episode parquet, where every frame row contains the
  subtask string active at that timestamp.
- Adds a ``response`` feature entry to ``meta/info.json``.

API cost estimate
^^^^^^^^^^^^^^^^^

At 1 fps sampling with frames resized to 640 px wide (≈ 410 image tokens each):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Scale
     - Episodes
     - Estimated cost
   * - Dry run (1 ep/dataset)
     - ~10
     - < $0.01
   * - Medium mixture
     - 500
     - ~$0.05
   * - Large mixture
     - 5 000
     - ~$0.50

Costs scale linearly with episode count × episode duration.  Use ``--sample-fps 0.5`` or
lower to halve/quarter costs on longer episodes.

Adding subtask responses to a dataset
--------------------------------------

Some policies (e.g. π0.5) can be trained with per-frame subtask annotations that tell the model *what* sub-goal is active at each timestep.
The ``add_subtask_response`` script reads per-episode subtask JSON files, converts the time-based subtask boundaries to frame indices using the dataset FPS, and writes the active subtask string into a ``response`` column in each episode parquet file.

.. note::
   If you used ``annotate_subtasks.py`` with ``--write-response-column`` (the default),
   this step has already been done for you.

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
