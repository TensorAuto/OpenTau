#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default configuration classes for datasets, evaluation, and logging.

This module provides default configuration classes for:
- Dataset configuration and dataset mixtures
- Weights & Biases (wandb) logging configuration
- Evaluation settings and parameters
"""

from dataclasses import dataclass, field

import draccus
import numpy as np
from draccus.parsers.encoding import encode_dataclass

from opentau import (
    policies,  # noqa: F401
)
from opentau.datasets.transforms import ImageTransformsConfig
from opentau.datasets.video_utils import get_safe_default_codec

# --- Custom NumPy encoder registration ---
# For decoding from cmd/yaml
draccus.decode.register(np.ndarray, np.asarray)
# For encoding to yaml
draccus.encode.register(np.ndarray, lambda x: x.tolist())


@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    You may provide a list of datasets here. `train.py` creates them all and
    concatenates them. Note: only data keys common between the datasets are kept.
    Each dataset gets an additional transform that inserts the "dataset_index"
    into the returned item. The index mapping is made according to the order in
    which the datasets are provided.

    Args:
        repo_id: HuggingFace repository ID for the dataset. Exactly one of
            `repo_id` or `vqa` must be set.
        vqa: VQA dataset identifier. Exactly one of `repo_id` or
            `vqa` must be set.
        root: Root directory where the dataset will be stored (e.g. 'dataset/path').
            Defaults to None.
        episodes: List of episode indices to use from the dataset. If None, all
            episodes are used. Defaults to None.
        excluded_episodes: List of episode indices to drop from this dataset.
            Takes precedence over `episodes`: an index present in both is
            excluded. If None (default), no episodes are excluded. Note: on
            legacy v2.0 datasets (no per-episode stats) the listed episodes are
            dropped from training, but normalization stats stay the global
            (all-episode) aggregate — only v2.1+ can recompute them.
        image_transforms: Configuration for image transformations. Defaults to
            ImageTransformsConfig().
        revision: Git revision of the dataset repository to use. Defaults to None.
        use_imagenet_stats: Whether to use ImageNet statistics for normalization.
            Defaults to True.
        video_backend: Video codec backend to use. Defaults to a safe default codec.
        stats: Dictionary of statistics for normalization, keyed by feature name.
            Each value is a dictionary with 'mean' and 'std' arrays. Defaults to None.
        data_features_name_mapping: Optional mapping from dataset feature names to
            standard feature names. Defaults to None.
        robot_type: Optional override for the dataset's ``robot_type`` metadata
            field. When provided (including the empty string), takes precedence
            over the value loaded from ``meta/info.json``. ``None`` (default)
            leaves the loaded value untouched.
        control_mode: Optional override for the dataset's ``control_mode``
            metadata field. When provided (including the empty string), takes
            precedence over the value loaded from ``meta/info.json``. ``None``
            (default) leaves the loaded value untouched.
        tolerance_s: Optional per-dataset override for the timestamp-sync
            tolerance (in seconds) passed to ``LeRobotDataset``'s load-time
            ``check_timestamps_sync`` call. ``None`` (default) inherits the
            mixture-wide ``DatasetMixtureConfig.tolerance_s`` value. Set this
            to a larger value (e.g. ``1e-3``) when a single dataset in the
            mixture has slightly off-fps timestamps but you don't want to
            loosen the check for the others. Must be ``>= 0`` when set.
            This also applies to frames decoded from video files: the same
            value is used as the per-frame match tolerance in
            ``query_video_frames_*``, so loosening it widens the
            video-frame match window as well.
        skip_timestamp_check: Optional per-dataset override that bypasses the
            load-time ``check_timestamps_sync`` call entirely. ``None``
            (default) inherits the mixture-wide
            ``DatasetMixtureConfig.skip_timestamp_check``. ``True`` skips the
            check (a warning is logged); ``False`` forces the check on for this
            dataset even if the mixture default is ``True``. Does not affect
            the record-time check inside ``add_episode``.

    Raises:
        ValueError: If both or neither of `repo_id` and `vqa` are set, or
            if `data_features_name_mapping` is provided.
            is provided.
    """

    repo_id: str | None = None
    vqa: str | None = None
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    excluded_episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    stats: dict[str, dict[str, np.ndarray]] | None = None

    # optional standard data format mapping for the dataset if mapping is not already in standard_data_format_mapping.py
    data_features_name_mapping: dict[str, str] | None = None

    # Optional overrides for the metadata fields read from `meta/info.json`.
    # `None` means "do not override". Any string value (including "") is
    # written through to `dataset.meta.info[...]` after the dataset is built.
    robot_type: str | None = None
    control_mode: str | None = None

    # Per-dataset overrides for the load-time timestamp-sync check. `None`
    # inherits from `DatasetMixtureConfig.{tolerance_s, skip_timestamp_check}`.
    # A non-None value here wins over the mixture default for this dataset
    # only — useful when one dataset in a mixture has off-fps timestamps.
    tolerance_s: float | None = None
    skip_timestamp_check: bool | None = None

    # Per-dataset override for the train/validation split ratio. `None`
    # inherits from `DatasetMixtureConfig.val_split_ratio`. A non-None value
    # (including `0.0`, which opts this dataset out of validation) wins over the
    # mixture default for this dataset only — useful when one dataset in a
    # mixture wants a different validation fraction than the rest. Must be in
    # `[0, 1]` when set. Only consulted when `TrainPipelineConfig.val_freq > 0`.
    val_split_ratio: float | None = None

    def __post_init__(self):
        """Validate dataset configuration and register custom mappings if provided."""
        if (self.repo_id is None) == (self.vqa is None):
            raise ValueError("Exactly one of `repo_id` or `vqa` for Dataset config should be set.")

        if self.tolerance_s is not None and self.tolerance_s < 0:
            raise ValueError(
                f"`DatasetConfig.tolerance_s` must be >= 0 (or None to inherit), "
                f"got {self.tolerance_s} for {self.repo_id or self.vqa}."
            )

        if self.val_split_ratio is not None and not (0.0 <= self.val_split_ratio <= 1.0):
            raise ValueError(
                f"`DatasetConfig.val_split_ratio` must be in [0, 1] (or None to inherit), "
                f"got {self.val_split_ratio} for {self.repo_id or self.vqa}."
            )

        # If data_features_name_mapping is provided, upsert it into the global
        # DATA_FEATURES_NAME_MAPPING. Register under the plain repo_id (back-compat
        # fallback, last-wins) AND, when this entry carries a real control mode,
        # under the composite `repo_id::control_mode` key. The composite key lets
        # two entries that share a repo_id but use different action columns
        # (e.g. control_mode="joint" -> action_joint vs control_mode="ee" ->
        # action_ee) coexist instead of the second silently clobbering the first.
        if self.data_features_name_mapping is not None:
            from opentau.datasets.standard_data_format_mapping import (
                DATA_FEATURES_NAME_MAPPING,
                feature_mapping_key,
            )

            DATA_FEATURES_NAME_MAPPING[self.repo_id] = self.data_features_name_mapping
            if self.repo_id is not None:
                key = feature_mapping_key(self.repo_id, self.control_mode)
                if key != self.repo_id:
                    DATA_FEATURES_NAME_MAPPING[key] = self.data_features_name_mapping


@dataclass
class DatasetMixtureConfig:
    r"""Configuration for a mixture of multiple datasets.

    This configuration allows combining multiple datasets with specified weights
    for training. The datasets are sampled according to their weights during
    training, and features are resampled to a common action frequency.

    Args:
        datasets: List of dataset configs to be used in the mixture.
        weights: Optional list of weights for each dataset in the mixture. Must be the
            same length as `datasets` when provided. If None, weights are inferred
            from dataset lengths. Defaults to None.
        action_freq: Frequency at which actions from the dataset mixture are
            resampled, in Hz. ``None`` (default) disables resampling — each
            dataset is sampled at its native fps, so a single batch can mix
            samples from sources running at different rates (predicting
            ``chunk_size`` consecutive native frames per sample). Set a
            positive float to resample every dataset in the mixture to that
            common rate via nearest-neighbor frame selection. When using
            ``None``, prefer also setting ``emit_fps=True`` so the policy
            can condition on the per-sample rate.
        image_resample_strategy: Resample strategy for image features. Must be
            one of 'linear' or 'nearest'. Defaults to 'nearest'.
        vector_resample_strategy: Resample strategy for non-image features, such
            as action or state. Must be one of 'linear' or 'nearest'.
            Defaults to 'nearest'.
        val_split_ratio: Mixture-wide default fraction of each dataset reserved
            for the validation split (only used when
            ``TrainPipelineConfig.val_freq > 0``). A per-dataset
            ``DatasetConfig.val_split_ratio`` overrides this value for that
            dataset; ``None`` there inherits this mixture default. Must be in
            ``[0, 1]``. Defaults to 0.05.
        n_obs_history: Number of historical observation steps to include. When
            set to ``T``, each camera returns shape ``(T, C, H, W)`` and state
            returns shape ``(T, max_state_dim)``. When ``None``, the default
            single-step behavior is preserved with rank-3 camera tensors
            ``(C, H, W)`` and rank-1 state tensors ``(max_state_dim,)``.
            Note that ``n_obs_history=1`` produces rank-4/rank-2 tensors with
            a leading singleton dimension, so downstream consumers must handle
            both rank conventions. Defaults to ``None``. The temporal stride
            between sampled observations is read from the policy config's
            ``history_interval`` attribute (defaults to 1 when the policy
            doesn't define one), so observations are sampled at timesteps
            :math:`t - (T-1)k,\; t - (T-2)k,\; \ldots,\; t`.
        history_state_drop_prob: Probability of dropping the observation
            *history* during a single ``__getitem__`` call. When it fires, the
            historical steps are masked via ``obs_history_is_pad`` (set all True)
            and the historical camera frames are zeroed; the current step —
            current ``observation.state`` and current camera frame — is kept.
            ``state`` is deliberately NOT zeroed here: it is MEAN_STD-normalized
            downstream, so the dropped history is zeroed *after* normalization
            inside the policy (zeroing a raw state pre-normalization would map
            0 to ``-mean/std``, an out-of-distribution extreme). Must be in
            ``[0, 1]``. Defaults to 0.3.
        subgoal_drop_prob: Probability of dropping all subgoal images during a
            single ``__getitem__`` call. Must be in ``[0, 1]``. Defaults to 0.75.
        subgoal_end_of_segment_prob: Probability of sampling the subgoal frame
            at the end of the current segment (vs. uniformly in the next 4s of
            wall-clock time). Must be in ``[0, 1]``. Defaults to 0.25.
        response_drop_prob: Probability of dropping the ``response`` (subtask
            text) during a single ``__getitem__`` call. Only rolled when
            subgoals are not dropped. Must be in ``[0, 1]``. Defaults to 0.3.
        metadata_drop_all_prob: Probability of dropping ``speed``, ``mistake``,
            ``quality``, ``robot_type``, and ``control_mode`` together during a
            single ``__getitem__`` call. Must be in ``[0, 1]``. Defaults to 0.15.
        metadata_drop_each_prob: Per-field independent drop probability for
            ``speed``, ``mistake``, ``quality``, ``robot_type``, and
            ``control_mode``. Only rolled when ``metadata_drop_all_prob`` did not
            fire. Must be in ``[0, 1]``.
            Defaults to 0.05.
        val_enable_optional_key_dropout: Whether to apply the five
            ``*_drop_prob`` rolls above to the validation split. Defaults to
            ``False`` — validation evaluates on un-masked samples so metrics
            aren't polluted by training-time augmentation. Subgoal *frame*
            sampling (end-of-segment vs. uniform in the next 4s) stays active
            either way; only the masking logic is gated.
        require_non_empty_robot_type: If True, every dataset in the mixture
            must have a non-empty ``robot_type`` after the optional
            ``DatasetConfig.robot_type`` override has been applied. Defaults to
            ``False`` (empty / missing values are allowed).
        require_non_empty_control_mode: If True, every dataset in the mixture
            must have a non-empty ``control_mode`` after the optional
            ``DatasetConfig.control_mode`` override has been applied. Defaults
            to ``False`` (empty / missing values are allowed).
        emit_fps: Whether ``__getitem__`` returns the *effective*
            per-sample frame rate (``action_freq`` if set, else the
            dataset's native ``meta.fps``) as the ``fps`` metadata key
            (``torch.long`` scalar, paired with ``fps_is_pad=False``).
            Default ``False`` — fps conditioning is an opt-in feature so
            pre-PR checkpoints resume without the policy's metadata
            prefix gaining an unfamiliar ``FPS:`` segment. Flip to
            ``True`` for new training runs that want per-sample
            frame-rate conditioning (especially heterogeneous-frequency
            mixtures where ``action_freq=None`` lets each dataset run at
            its native rate). Unlike the other metadata fields, ``fps``
            is **not** rolled by ``metadata_drop_*_prob`` — it's an
            intrinsic property of the chunk, not a noisy label, so it
            is always present (non-pad) for LeRobot samples when
            ``emit_fps=True``. VQA samples (no temporal axis) emit
            ``fps=0, fps_is_pad=True`` regardless so heterogeneous
            VLA + VQA batches stay schema-aligned.
        tolerance_s: Mixture-wide default tolerance (in seconds) for the
            load-time ``check_timestamps_sync`` call inside
            ``LeRobotDataset.__init__``. Each dataset's frame-to-frame
            timestamp spacing must lie within ``1/fps +/- tolerance_s`` or the
            check raises. Defaults to ``1e-4``. A per-dataset
            ``DatasetConfig.tolerance_s`` overrides this value when set. Must
            be ``>= 0``. This also applies to frames decoded from video
            files: the same value is used as the per-frame match tolerance
            in ``query_video_frames_*``, so loosening it widens the
            video-frame match window as well.
        skip_timestamp_check: If True, bypass the load-time
            ``check_timestamps_sync`` call for every dataset in the mixture
            (a warning is logged per dataset). Useful as a debug knob when
            the timing data is known-bad but you still want the mixture to
            load. Defaults to ``False``. A per-dataset
            ``DatasetConfig.skip_timestamp_check`` overrides this value when
            set. Does not affect the record-time check inside ``add_episode``.

    Note:
        Dropout rolls use the default torch RNG. PyTorch DataLoader workers
        auto-seed each process's torch RNG from the base seed + worker id, so
        workers sample independently. For reproducibility the caller should
        seed via ``torch.manual_seed(...)`` in the main process before
        constructing the DataLoader.

    Raises:
        ValueError: If `weights` is provided and its length doesn't match
            `datasets`, if `action_freq` is not None and not positive, if
            resample strategies are invalid, or if any drop probability is
            outside ``[0, 1]``.
    """

    # List of dataset configs to be used in the mixture.
    datasets: list[DatasetConfig] = field(default_factory=list)
    # Optional list of weights for each dataset in the mixture.
    # Must be the same length as `datasets` when provided.
    weights: list[float] | None = None
    # Frequency at which the actions from dataset mixture are resampled, in Hz.
    # ``None`` disables resampling — each dataset is sampled at its native fps.
    action_freq: float | None = None
    # Resample strategy for image features
    image_resample_strategy: str = "nearest"
    # Resample strategy for non-image features, such as action or state
    vector_resample_strategy: str = "nearest"
    # Mixture-wide default ratio of each dataset to be used for validation.
    # If `val_freq` is set to 0, a validation dataset will not be created and this value will be ignored.
    # A per-dataset `DatasetConfig.val_split_ratio` overrides this for that
    # dataset (`None` there inherits this value).
    # Defaults to 0.05.
    val_split_ratio: float = 0.05
    # Number of historical observation steps. None preserves default single-step behavior.
    n_obs_history: int | None = None

    # Training-time dropout probabilities for optional sample keys.
    history_state_drop_prob: float = 0.3
    subgoal_drop_prob: float = 0.75
    subgoal_end_of_segment_prob: float = 0.25
    response_drop_prob: float = 0.3
    metadata_drop_all_prob: float = 0.15
    metadata_drop_each_prob: float = 0.05
    # Whether the above dropout rolls also fire on the validation split.
    # Default keeps validation deterministic-ish (no masking); subgoal frame
    # selection stays random either way.
    val_enable_optional_key_dropout: bool = False

    # When True, require every dataset in the mixture to have a non-empty
    # robot_type / control_mode after `DatasetConfig.{robot_type,control_mode}`
    # overrides have been applied. Defaults are False — empty values are
    # tolerated unless the caller opts in to the stricter check.
    require_non_empty_robot_type: bool = False
    require_non_empty_control_mode: bool = False

    # Whether `__getitem__` emits the effective per-sample fps as the `fps`
    # metadata key. Default `False` so pre-PR checkpoints resume cleanly
    # (no new `FPS:` segment in the policy's metadata prefix). Flip to
    # `True` for new training runs that want per-sample fps conditioning;
    # especially relevant for heterogeneous-frequency mixtures
    # (`action_freq=None`). Independent of `metadata_drop_*_prob` — fps
    # is intrinsic to the chunk, not a noisy label, so it is always
    # present (never padded) for LeRobot samples when this is True.
    emit_fps: bool = False

    # Mixture-wide defaults for the load-time timestamp-sync check. Each
    # dataset can override these via `DatasetConfig.{tolerance_s,
    # skip_timestamp_check}`. The default tolerance matches
    # `LeRobotDataset.__init__`'s historical default.
    tolerance_s: float = 1e-4
    skip_timestamp_check: bool = False

    def __post_init__(self):
        """Validate dataset mixture configuration."""
        if self.weights is not None and len(self.datasets) != len(self.weights):
            raise ValueError("The length of `weights` must match the length of `datasets`.")
        if self.action_freq is not None and self.action_freq <= 0:
            raise ValueError(f"`action_freq` must be a positive number or None, got {self.action_freq}.")
        if self.image_resample_strategy not in ["linear", "nearest"]:
            raise ValueError(
                f"`image_resample_strategy` must be one of ['linear', 'nearest'], got {self.image_resample_strategy}."
            )
        if self.vector_resample_strategy not in ["linear", "nearest"]:
            raise ValueError(
                f"`vector_resample_strategy` must be one of ['linear', 'nearest'], got {self.vector_resample_strategy}."
            )
        if self.val_split_ratio < 0 or self.val_split_ratio > 1:
            raise ValueError(f"`val_split_ratio` must be between 0 and 1, got {self.val_split_ratio}.")
        if self.tolerance_s < 0:
            raise ValueError(f"`tolerance_s` must be >= 0, got {self.tolerance_s}.")
        if self.n_obs_history is not None and (
            not isinstance(self.n_obs_history, int) or self.n_obs_history < 1
        ):
            raise ValueError(f"`n_obs_history` must be None or a positive integer, got {self.n_obs_history}.")
        for name in (
            "history_state_drop_prob",
            "subgoal_drop_prob",
            "subgoal_end_of_segment_prob",
            "response_drop_prob",
            "metadata_drop_all_prob",
            "metadata_drop_each_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must be in [0, 1], got {value}.")


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases (wandb) logging.

    Args:
        enable: Enable Weights & Biases logging. Defaults to False.
        entity: The entity name in Weights & Biases, e.g. your username or your
            team name. Defaults to None.
        project: The project name in Weights & Biases, e.g. "pi0". Defaults to "opentau".
        run_id: If provided, the run will be forked from this run ID. Defaults to None.
        name: Name of the run, shown in the UI. Defaults to None.
        notes: Description of the run, shown in the UI. If None and `enable` is True,
            will prompt the user for input. Defaults to None.
        tags: Tags to be added to the run in the UI, e.g. ["robot", "v1.0"].
            Defaults to empty list.
        group: Used to group runs in the UI, e.g. "experiment_1", "experiment_2".
            Defaults to None.
        job_type: Used to group runs in the UI, e.g. "train", "eval", "test".
            Defaults to None.
        mode: Allowed values: 'online', 'offline', 'disabled'. Defaults to None
            (which uses 'online').
        allow_resume: If True, resume the run from the last checkpoint when
            `run_id` is provided. Defaults to True.
        disable_artifact: Set to True to disable saving an artifact despite
            `training.save_checkpoint=True`. Defaults to False.
        disable_video: Set to True to skip logging eval grid-summary videos to
            wandb. Defaults to False (videos are logged).
    """

    enable: bool = False  # Enable Weights & Biases logging.
    entity: str | None = None  # The entity name in Weights & Biases, e.g. your username or your team name
    project: str = "opentau"  # The project name in Weights & Biases, e.g. "pi0"
    run_id: str | None = None  # If provided, the run will be forked from this run ID.
    name: str | None = None  # Name of the run, shown in the UI
    notes: str | None = None  # Description of the run, shown in the UI
    tags: list[str] = field(
        default_factory=list
    )  # Tags to be added to the run in the UI, e.g. ["robot", "v1.0"]
    group: str | None = None  # Used to group runs in the UI, e.g. "experiment_1", "experiment_2"
    job_type: str | None = None  # Used to group runs in the UI, e.g. "train", "eval", "test"
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'
    allow_resume: bool | None = True  # If True, resume the run from the last checkpoint.
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    # Set to true to skip logging eval grid-summary videos to wandb.
    disable_video: bool = False

    def __post_init__(self):
        """Prompt user for wandb notes if enabled and notes are not provided."""
        if not self.enable or self.notes is not None:
            return

        confirm = False
        while not confirm:
            self.notes = input("Please enter a description for wandb logging:\n")
            confirm = input("Confirm (y/N): ").strip().lower() == "y"

    def to_wandb_kwargs(self, step=None):
        """Convert configuration to keyword arguments for wandb.init().

        Args:
            step: Optional training step number. If provided along with `run_id`,
                used for resuming or forking runs. Defaults to None.

        Returns:
            Dictionary of keyword arguments suitable for passing to wandb.init().
        """
        kwargs = encode_dataclass(self)
        excluded_keys = ["enable", "disable_artifact", "disable_video", "project"]
        for ek in excluded_keys:
            kwargs.pop(ek)

        allow_resume = kwargs.pop("allow_resume")
        run_id = kwargs.pop("run_id", None)

        # If both `run_id` and `step` are provided, we handle the resuming or forking logic.
        if run_id is not None and step is not None:
            if allow_resume:
                # if `allow_resume`, we resume from the `run_id` if provided.
                kwargs["id"] = run_id
                kwargs["resume"] = "allow"
            else:
                # Without `allow_resume`, we create a new run,
                # and add information about the forked run in the notes.
                # TODO request `kwargs[fork_from]=f"{run_id}?_step={step}"` feature from wandb
                kwargs["notes"] += f"\nForked from run {run_id} at step {step}."

        return kwargs


@dataclass
class EvalConfig:
    """Configuration for evaluation settings.

    Args:
        n_episodes: Number of episodes to run during evaluation. Defaults to 16.
        batch_size: Number of environments to use in a gym.vector.VectorEnv.
            Only used for environments that are not already vectorized.
            Defaults to 16.
        use_async_envs: Whether to use asynchronous environments (multiprocessing).
            Defaults to True.
        max_episodes_rendered: Maximum number of episodes to render as videos.
            Defaults to 16.
        grid_size: Grid dimensions for video summary (rows, cols). If None, will
            be auto-calculated as a square grid. Defaults to None.
        recording_root: Root directory for saving evaluation recordings.
            Defaults to None.

    Raises:
        ValueError: If `batch_size` is greater than `n_episodes`.
    """

    n_episodes: int = 16
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv. (Only used for environments that are not already vectorized.)
    batch_size: int = 16
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = True
    max_episodes_rendered: int = 16
    # Grid dimensions for video summary (rows, cols). If None, will be auto-calculated as square grid.
    grid_size: tuple[int, int] | None = None

    recording_root: str | None = None

    # Which training-time norm head to use when calling `policy.select_action`
    # on eval observations. Either:
    #   - set both `robot_type` and `control_mode` to address the head by
    #     `(robot_type, control_mode)` (preferred for multi-head checkpoints
    #     trained against the new per-`(robot_type, control_mode)`
    #     aggregation),
    #   - or set `dataset_repo_id` to a training-time dataset name (the
    #     policy maps it to the norm head via its persisted
    #     `dataset_to_norm_index`; back-compat path that also works on
    #     legacy per-dataset checkpoints).
    # When all three are ``None`` (default), single-head policies fall back
    # to the `_resolve_dataset_index` zero-default; multi-head ones raise.
    # The robot_type / control_mode pair takes precedence over
    # `dataset_repo_id` when both are set. Plumbed by
    # `scripts/eval.py::rollout`.
    dataset_repo_id: str | None = None
    robot_type: str | None = None
    control_mode: str | None = None

    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )
