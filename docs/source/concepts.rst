Concepts
========

This section explains the core concepts used in the OpenTau codebase.

Policies
--------
Policies map observations (e.g., camera images, robot proprioceptive states) to actions (or action chunks).
Policies are implemented as PyTorch modules and inherit from ``opentau.policies.pretrained.PreTrainedPolicy``.

Datasets
--------
Datasets are used to handle data loading and processing.
It supports downloading datasets from the Hugging Face Hub and loading them from local disk.
The dataset format is versioned (currently v2.1) and utilizes parquet files for data and mp4 files for videos to ensure efficiency and portability.
There are currently two types of datasets:

*   ``LeRobotDataset``: For robotic data.
*   ``VQADataset``: For VLM training datasets.

These datasets are used to train policies.

DatasetMixture
^^^^^^^^^^^^^^
To train policies on multiple datasets simultaneously, OpenTau uses ``opentau.datasets.dataset_mixture.WeightedDatasetMixture``.
This class:

*   Combines multiple ``LeRobotDataset`` and ``VQADataset`` instances.
*   Different weights can be assigned to each dataset to control the sampling frequency; if weights are omitted (or set to ``null`` in JSON), weights default to dataset lengths.
*   Aggregates statistics from all constituent datasets to ensure consistent normalization across the mixture.
*   Resamples the action output frequency to match the action frequency specified in the configuration.

Metadata
^^^^^^^^
Metadata is crucial for defining the structure and statistics of a dataset. Handled by ``LeRobotDatasetMetadata`` and ``DatasetMetadata``, it includes:

*   **Info**: Feature shapes, data types, FPS, and robot type.
*   **Stats**: Mean, standard deviation, min, and max values for each feature, used for normalization (e.g., standardizing images or normalizing action vectors).
*   **Tasks**: Natural language descriptions of the tasks contained in the dataset.

Metadata is stored in JSON files (``info.json``, ``stats.json``) and JSONL files (``tasks.jsonl``) within the dataset directory.

.. _standard-data-format:

Standard Data Format
--------------------
To ensure compatibility across different datasets and policies, OpenTau introduces the **Standard Data Format**.
The Standard Data Format is the expected data format returned by ``torch.utils.data.Dataset``'s ``__getitem__`` and the expected input to ``torch.nn.Module``'s ``forward`` method. Any new datasets, VLMs, or VLAs that get added to this repository need to adhere to this format. Data being passed to the model during inference should also adhere to this format.

Default format (``n_obs_history=None``):

.. code-block:: python

    {
        "camera0": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        "camera1": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        # ...
        "camera{num_cams-1}": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.

        "state": torch.Tensor,    # shape (max_state_dim,)
        "actions": torch.Tensor,  # shape (action_chunk, max_action_dim)
        "prompt": str,            # the task prompt, e.g. "Pick up the object and place it on the table."
        "response": str,          # the response from the VLM for vision QA tasks. For LeRobotDataset, this will be an empty string.
        "loss_type": str,         # the loss type to be applied to this sample (either "CE" for cross entropy or "MSE" for mean squared error)

        "img_is_pad": torch.BoolTensor,  # shape (num_cams,) with values 0 or 1, where 1 indicates that the camera image is a padded image.
        "action_is_pad": torch.BoolTensor,  # shape (action_chunk,) with values 0 or 1, where 1 indicates that the action is a padded action.
        "obs_history_is_pad": torch.BoolTensor,  # shape (1,) — always False when n_obs_history is None.
    }

With observation history (``n_obs_history=T``):

.. code-block:: python

    {
        "camera0": torch.Tensor,  # shape (T, C, H, W) — T historical steps for camera 0.
        # ...
        "camera{num_cams-1}": torch.Tensor,  # shape (T, C, H, W)

        "state": torch.Tensor,    # shape (T, max_state_dim)
        "actions": torch.Tensor,  # shape (action_chunk, max_action_dim)
        # ... (prompt, response, loss_type unchanged)

        "img_is_pad": torch.BoolTensor,  # shape (num_cams,) — camera slot availability.
        "action_is_pad": torch.BoolTensor,  # shape (action_chunk,)
        "obs_history_is_pad": torch.BoolTensor,  # shape (T,) — True for timesteps outside the episode boundary.
    }

When ``n_obs_history=T`` and ``history_interval=k``, observations are sampled at timesteps
:math:`t - (T-1)k,\; t - (T-2)k,\; \ldots,\; t` relative to the current timestep :math:`t`, where
the interval is measured in dataset steps (at the configured ``action_freq``). For timesteps that
fall before the start of the current episode, the observation is clamped to the first step of the
episode and the corresponding entry in ``obs_history_is_pad`` is set to ``True``.

The config file will have to provide the following information in TrainPipelineConfig:

*   ``H, W``: The height and width of the camera images. These should be the same for all cameras.
*   ``num_cams``: The number of cameras for the cloud VLM in the dataset.
*   ``max_state_dim``: The maximum dimension of the state vector.
*   ``max_action_dim``: The maximum dimension of the action vector.
*   ``action_chunk``: The number of actions in the action vector. This is usually 1 for single action tasks, but can be more for multi-action tasks.

The following fields are set in ``DatasetMixtureConfig``:

*   ``n_obs_history``: Number of historical observation steps to include. When ``None`` (default), the single-step format is used. When set to an integer ``T``, cameras and state gain a leading temporal dimension of size ``T``.
*   ``history_interval``: Step interval between historical observation steps. Defaults to ``1``. Only relevant when ``n_obs_history`` is set.

Cameras should be labeled in order of importance (e.g. camera0 is the most important camera, camera1 is the second most important camera, etc.). The model dataset will select the most important cameras to use if num_cams is less than the number of cameras in the dataset.

.. _standard-data-format-optional-keys:

Optional Standard-Format Keys
-----------------------------

On top of the core fields above, ``__getitem__`` emits several *optional*
keys when the dataset has been enriched with segment metadata (see
:doc:`tutorials/attach_metadata`) or for the subgoal images sampled from
future video frames. Each optional key is **always present**. Numeric
and image keys pair with an ``{key}_is_pad`` boolean flag — zero-filled
+ flag True means "unavailable or masked". String keys
(``response``, ``memory``, ``next_memory``, ``robot_type``,
``control_mode``) don't get a separate flag: the empty string ``""``
is itself the pad signal, which also keeps the default PyTorch collate
happy (list of strings, same length as batch).

``robot_type`` and ``control_mode`` are **dataset-level identifiers**
(constant for every sample within a given dataset, distinct across
datasets in a mixture batch) sourced directly from ``meta/info.json``.
Like ``speed``, ``mistake``, and ``quality``, they participate in the
``metadata_drop_all_prob`` / ``metadata_drop_each_prob`` dropout rolls —
see :ref:`Training-time dropout <standard-data-format-optional-keys-dropout>`.

.. code-block:: python

    {
        # ... core keys above ...

        "robot_type": str,         # e.g. "aloha", "panda", "human" — copied verbatim
                                   # from `meta/info.json["robot_type"]` (a standard
                                   # LeRobot v2 field). Empty string ("") when the key
                                   # is absent or null (e.g. VQA datasets).
        "control_mode": str,       # One of {"joint", "ee", "mixed"} when the dataset
                                   # opted in (see PR #183). Empty string ("") when
                                   # `meta/info.json["control_mode"]` is absent.

        "memory": str,             # Cumulative subtask summary for the current frame's segment.
                                   # Empty string ("") when memory_raw is absent
                                   # (legacy / unannotated dataset).
        "next_memory": str,        # Memory string for frame t+1 (same as `memory` within a
                                   # segment, differs at segment boundaries). Clipped at episode
                                   # end. Empty string when unavailable.

        "speed": torch.LongTensor,     # Scalar in {0, 10, 20, ..., 100}; **per-task decile rank** of this
                                       # episode's length-in-frames. Lower = faster, higher = slower.
                                       # Computed by grouping episodes from ``meta/episodes.jsonl`` by
                                       # task (``tasks[0]`` — the codebase assumes the list is length-1)
                                       # and bucketing each episode's length against the
                                       # ``[p5, p15, ..., p95]`` boundaries of *its own task's*
                                       # length distribution. Episodes shorter than p5 bucket to 0;
                                       # episodes at or above p95 bucket to 100; ties at a boundary
                                       # land in the upper bucket (``p_X <= length < p_Y``). Per-task
                                       # boundaries are persisted to ``meta/speed_percentiles.jsonl``;
                                       # delete that file to recompute (e.g. after appending episodes).
                                       # Tasks with fewer than 10 distinct episode lengths are treated
                                       # as sparse and bucket every episode to 50 (the median-equivalent
                                       # neutral default). The per-task framing makes the bucket carry
                                       # task-relative information: "this is a fast example of *this*
                                       # task" rather than "this episode lasted N seconds". See
                                       # :mod:`opentau.datasets.speed_percentiles` for the helpers.
                                       # Populated unconditionally — available on every LeRobotDataset
                                       # regardless of whether the dataset went through
                                       # ``attach_metadata``.
        "speed_is_pad": torch.BoolTensor,  # True only when the metadata drop rolls in
                                           # _emit_optional_keys fire at training time. The percentile
                                           # compute itself never produces a pad signal: sparse-task
                                           # episodes still get a (neutral) bucket value rather than
                                           # being padded.

        "mistake": torch.BoolTensor,   # Scalar; True iff the current segment's success flag is False.
        "mistake_is_pad": torch.BoolTensor,

        "quality": torch.LongTensor,   # Scalar in {1,2,3,4,5}; episode-level quality score.
        "quality_is_pad": torch.BoolTensor,

        "subgoal0": torch.Tensor,       # shape (3, H, W), values in [0,1]. A single future frame from
                                        # camera0 sampled either at end-of-segment (with probability
                                        # `subgoal_end_of_segment_prob`) or uniformly in [t, t+4 seconds].
        # ...
        "subgoal{num_cams-1}": torch.Tensor,
        "subgoal_is_pad": torch.BoolTensor,   # Single flag covering every `subgoalK`. Subgoals are either
                                              # all present (annotated dataset, not dropped this step) or
                                              # all padded (legacy dataset, or `subgoal_drop_prob` fired).

        # `response` (already in the core fields) may be replaced with ""
        # when `response_drop_prob` fires — consumers read "" as masked,
        # same convention as `memory` / `next_memory`.
    }

Subgoals are always rank-3 ``(3, H, W)`` regardless of
``n_obs_history`` — they represent a single future target frame, not a
temporal window. All camera slots share a single ``subgoal_is_pad``
flag because subgoals are all-or-none.

Subgoal image **paths** are read from ``meta/info.json`` under the
``subgoals`` key. When the key is absent (the state of every LeRobot
dataset today), ``_load_subgoal_frames`` returns ``{}`` and every
``subgoalK`` tensor comes out zero-filled with ``subgoal_is_pad=True``.
Datasets opt in to subgoals by adding the key; the loader then uses the
frame-selection machinery (end-of-segment vs. uniform ``[t, t+4 s]``)
described below.

.. _standard-data-format-optional-keys-dropout:

Training-time dropout
^^^^^^^^^^^^^^^^^^^^^

Six probability fields on ``DatasetMixtureConfig`` control how often
each optional key is masked during a single ``__getitem__`` call.
Masks
are independent per sample (each call rolls fresh). ``DataLoader``
workers seed their own torch RNG, so samples within a batch are
independent across workers; seed globally via ``torch.manual_seed(...)``
for reproducibility.

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Field
     - Default
     - Effect
   * - ``history_state_drop_prob``
     - ``0.3``
     - Zero-fills ``state`` and historical camera frames (when
       ``n_obs_history > 1``); sets ``obs_history_is_pad`` all True.
   * - ``subgoal_drop_prob``
     - ``0.75``
     - Zero-fills every ``subgoal{K}`` image together and sets the single
       shared ``subgoal_is_pad`` flag to True.
   * - ``subgoal_end_of_segment_prob``
     - ``0.25``
     - Probability that a *present* subgoal is sourced from the end of
       the current segment. Otherwise sampled uniformly in time from
       the current timestamp through ``t + 4 s`` (clipped at segment
       end, then episode end).
   * - ``response_drop_prob``
     - ``0.3``
     - Replaces ``response`` with the empty string. Only rolled when
       subgoals are NOT dropped (dropping both response and subgoals
       would remove the primary task signal).
   * - ``metadata_drop_all_prob``
     - ``0.15``
     - Masks ``speed``, ``mistake``, ``quality``, ``robot_type``, and
       ``control_mode`` together.
   * - ``metadata_drop_each_prob``
     - ``0.05``
     - Per-field independent mask roll for each of ``speed``,
       ``mistake``, ``quality``, ``robot_type``, ``control_mode``.
       Only rolled when the shared drop did not fire.
   * - ``val_enable_optional_key_dropout``
     - ``False``
     - Whether the five drop rolls above also fire on the **validation**
       split. Default is ``False`` so validation metrics aren't
       artificially noisy. Set to ``True`` if you want the validation
       distribution to match training. Subgoal *frame* selection
       (end-of-segment vs. uniform in the next 4 s) stays random either
       way — only the masking logic is gated.

``make_dataset`` enforces this by giving the validation subset its own
shallow-copied dataset instance with ``enable_optional_key_dropout``
flipped accordingly; the underlying ``meta`` / ``hf_dataset`` objects
are still shared with the training subset, so the extra copy is cheap.

Legacy datasets that have not been passed through
:mod:`opentau.scripts.attach_metadata` still load: every optional key
appears with a zero/empty value and ``_is_pad=True``, so policies that
consume these fields can train without gating on dataset provenance.

Configs
-------
Configuration management is handled using `Draccus <https://github.com/dlwh/draccus>`_.
The main configuration class is ``opentau.configs.train.TrainPipelineConfig``, which orchestrates training settings,
policy configuration, and environment setup. Configs can be loaded from pretrained checkpoints to reproduce experiments.

Environments
------------
Environments wrap simulation or real-robot interfaces compatible with OpenAI Gym/Gymnasium.
The factory `src/opentau/envs/factory.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/envs/factory.py>`_ creates vectorized environments for efficient training and evaluation.
Currently, only `Libero <https://libero-project.github.io/main.html>`_ is supported and it is configured via ``opentau.envs.configs.LiberoEnv``.
