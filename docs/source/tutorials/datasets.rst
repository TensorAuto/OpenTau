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
mixture automatically, using a vision-language model.  Anthropic's ``claude-opus-4-7`` is
the default; Google's Gemini family — including ``gemini-robotics-er-1.6-preview`` — is
also supported via ``--model``.  This is the recommended way to produce the ``response``
column required by policies such as π0.5.

How it works
^^^^^^^^^^^^

For each episode the script:

1. Samples **1 frame/second** from the episode video (configurable via ``--sample-fps``),
   giving a 30-50× reduction over the raw frame rate.
2. Resizes each frame to 640 px wide (configurable via ``--target-width``) before JPEG
   encoding, keeping image-token costs low.
3. Sends all sampled frames with their timestamps to the selected model in a **single API
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

Set the API key for the provider you intend to use:

.. code-block:: bash

    # Anthropic (default)
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Gemini (when using --model gemini-* or --model robotics-er-*)
    export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

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

To annotate with Gemini Robotics-ER 1.6 instead of Claude:

.. code-block:: bash

    GEMINI_API_KEY=... python src/opentau/scripts/annotate_subtasks.py \
        --config-path configs/examples/train_mixture_config.json \
        --model gemini-robotics-er-1.6-preview

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
     - Model ID to use.  Anthropic IDs (e.g. ``claude-opus-4-7``) go through
       ``ANTHROPIC_API_KEY``; IDs starting with ``gemini`` or ``robotics-er``
       (e.g. ``gemini-robotics-er-1.6-preview``) go through ``GEMINI_API_KEY``
       (or ``GOOGLE_API_KEY``) via ``google-genai``.
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

At 1 fps sampling with frames resized to 640 px wide (≈ 410 image tokens each), using
``claude-opus-4-7``:

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

Automatically annotating mistakes with a VLM
---------------------------------------------

``annotate_mistakes.py`` adds a per-frame ``mistake`` column (``int64`` ∈ ``{0, 1}``) to
every episode parquet in a dataset mixture, by asking a VLM whether each subtask was
completed successfully. It runs **after** ``annotate_subtasks.py`` and reuses the same
mixture config format.

How it works
^^^^^^^^^^^^

For each episode the script:

1. Reads the per-frame ``response`` column from the episode parquet (written by
   ``annotate_subtasks.py``). Every contiguous run of identical ``response`` values is
   treated as one subtask segment.
2. Decodes the ``camera0`` video once (resolved with the same lookup chain as
   ``annotate_subtasks.py``: inline ``data_features_name_mapping``, then
   ``DATA_FEATURES_NAME_MAPPING``, then the first ``dtype=='video'`` feature) and
   pulls the **last frame of each contiguous run** — no temporal subsampling, just one
   frame per segment. Frames whose shorter side exceeds ``--target-size`` (default 448)
   are downsampled and center-cropped before JPEG encoding; smaller frames pass through
   unchanged.
3. Sends that single frame plus the segment's subtask string to the configured VLM
   (default: ``gemini-robotics-er-1.6-preview``; Anthropic Claude is supported via
   ``--model``) and asks for a ``{"success": bool, "reason": str}`` JSON verdict.
4. Sets every parquet row in the segment to ``mistake=1`` if the VLM reports failure,
   ``0`` otherwise. Any parse / API failure defaults to ``0`` (no mistake).
5. Atomically rewrites the episode parquet with the new ``mistake`` column and registers
   it in ``meta/info.json`` features the first time it is added to a dataset.

Episodes whose parquet already contains a ``mistake`` column are skipped (cheap O(1)
schema check), making the script **fully resumable**. Episodes whose parquet has no
``response`` column are skipped with a warning — run ``annotate_subtasks.py`` first.

Prerequisites
^^^^^^^^^^^^^

Set the API key for the provider you intend to use:

.. code-block:: bash

    # Gemini (default)
    export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

    # Anthropic (when using --model claude-*)
    export ANTHROPIC_API_KEY="sk-ant-..."

The dataset must already have been processed by ``annotate_subtasks.py`` so that each
episode parquet has a non-empty ``response`` column.

Running the script
^^^^^^^^^^^^^^^^^^

Reuse the same dataset mixture config you passed to ``annotate_subtasks.py``.
A minimal one-dataset example (with the Hub revision pinned to ``v2.1``, since
this script has only been tested against v2.1 datasets) is checked in at
``configs/examples/annotate_mistakes_example.json``:

.. code-block:: bash

    python src/opentau/scripts/annotate_mistakes.py \
        --config-path configs/examples/annotate_mistakes_example.json

For a dry run that processes only 1 episode per dataset:

.. code-block:: bash

    python src/opentau/scripts/annotate_mistakes.py \
        --config-path configs/examples/annotate_mistakes_example.json \
        --max-episodes-per-dataset 1

To annotate with Claude instead of Gemini:

.. code-block:: bash

    ANTHROPIC_API_KEY=... python src/opentau/scripts/annotate_mistakes.py \
        --config-path configs/examples/annotate_mistakes_example.json \
        --model claude-opus-4-7

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
   * - ``--target-size``
     - ``448``
     - Downsample frames whose shorter side exceeds this many pixels (then center-crop
       to a square). Frames at or below this size pass through unchanged — never upsamples.
   * - ``--model``
     - ``gemini-robotics-er-1.6-preview``
     - Model ID to use. IDs starting with ``gemini`` or ``robotics-er`` go through
       ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) via ``google-genai``; Anthropic IDs
       (e.g. ``claude-opus-4-7``) go through ``ANTHROPIC_API_KEY``.
   * - ``--max-episodes-per-dataset``
     - *(none)*
     - Cap the number of episodes processed per dataset — useful for dry runs.
   * - ``--max-api-retries``
     - ``8``
     - Anthropic SDK retry count for 429/5xx responses (ignored for Gemini).
   * - ``--hub-cache-dir``
     - ``~/.cache/huggingface/opentau_subtasks``
     - Directory for caching Hub dataset downloads. The default deliberately matches
       ``annotate_subtasks.py`` so this script reuses datasets already downloaded by
       the prior step — pass the same value here if you overrode it there.

Output
^^^^^^

For each processed episode the script:

- Adds a ``mistake`` column to the episode parquet, where every frame row contains
  ``0`` (subtask completed successfully, per the VLM) or ``1`` (subtask flagged as a
  failure). All frames within the same contiguous ``response`` run share the same value.
- Adds a ``mistake`` feature entry to ``meta/info.json``
  (``{"dtype": "int64", "shape": (1,), "names": None}``).

To force regeneration of the mistake labels, drop the ``mistake`` column from the
relevant episode parquets (or delete the cached dataset) before rerunning.

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
