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
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING, LOSS_TYPE_MAPPING
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
            `repo_id` or `grounding` must be set.
        grounding: Grounding dataset identifier. Exactly one of `repo_id` or
            `grounding` must be set.
        root: Root directory where the dataset will be stored (e.g. 'dataset/path').
            Defaults to None.
        episodes: List of episode indices to use from the dataset. If None, all
            episodes are used. Defaults to None.
        image_transforms: Configuration for image transformations. Defaults to
            ImageTransformsConfig().
        revision: Git revision of the dataset repository to use. Defaults to None.
        use_imagenet_stats: Whether to use ImageNet statistics for normalization.
            Defaults to True.
        video_backend: Video codec backend to use. Defaults to a safe default codec.
        stats: Dictionary of statistics for normalization, keyed by feature name.
            Each value is a dictionary with 'mean' and 'std' arrays. Defaults to None.
        data_features_name_mapping: Optional mapping from dataset feature names to
            standard feature names. Must be provided together with `loss_type_mapping`.
            Defaults to None.
        loss_type_mapping: Optional loss type mapping for the dataset. Must be
            provided together with `data_features_name_mapping`. Defaults to None.

    Raises:
        ValueError: If both or neither of `repo_id` and `grounding` are set, or
            if only one of `data_features_name_mapping` and `loss_type_mapping`
            is provided.
    """

    repo_id: str | None = None
    grounding: str | None = None
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    stats: dict[str, dict[str, np.ndarray]] | None = None

    # optional standard data format mapping for the dataset if mapping is not already in standard_data_format_mapping.py
    data_features_name_mapping: dict[str, str] | None = None
    loss_type_mapping: str | None = None

    def __post_init__(self):
        """Validate dataset configuration and register custom mappings if provided."""
        if (self.repo_id is None) == (self.grounding is None):
            raise ValueError("Exactly one of `repo_id` or `grounding` for Dataset config should be set.")

        # data_features_name_mapping and loss_type_mapping have to be provided together
        if (self.data_features_name_mapping is None) != (self.loss_type_mapping is None):
            raise ValueError(
                "`data_features_name_mapping` and `loss_type_mapping` have to be provided together."
            )

        # add data_features_name_mapping and loss_type_mapping to standard_data_format_mapping.py if they are provided
        if self.data_features_name_mapping is not None and self.loss_type_mapping is not None:
            DATA_FEATURES_NAME_MAPPING[self.repo_id] = self.data_features_name_mapping
            LOSS_TYPE_MAPPING[self.repo_id] = self.loss_type_mapping


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of multiple datasets.

    This configuration allows combining multiple datasets with specified weights
    for training. The datasets are sampled according to their weights during
    training, and features are resampled to a common action frequency.

    Args:
        datasets: List of dataset configs to be used in the mixture.
        weights: List of weights for each dataset in the mixture. Must be the
            same length as `datasets`. Defaults to empty list.
        action_freq: Frequency at which actions from the dataset mixture are
            resampled, in Hz. Defaults to 30.0.
        image_resample_strategy: Resample strategy for image features. Must be
            one of 'linear' or 'nearest'. Defaults to 'nearest'.
        vector_resample_strategy: Resample strategy for non-image features, such
            as action or state. Must be one of 'linear' or 'nearest'.
            Defaults to 'nearest'.

    Raises:
        ValueError: If the length of `weights` doesn't match `datasets`, if
            `action_freq` is not positive, or if resample strategies are invalid.
    """

    # List of dataset configs to be used in the mixture.
    datasets: list[DatasetConfig] = field(default_factory=list)
    # List of weights for each dataset in the mixture. Must be the same length as `datasets`.
    weights: list[float] = field(default_factory=list)
    # Frequency at which the actions from dataset mixture are resampled, in Hz.
    action_freq: float = 30.0
    # Resample strategy for image features
    image_resample_strategy: str = "nearest"
    # Resample strategy for non-image features, such as action or state
    vector_resample_strategy: str = "nearest"

    def __post_init__(self):
        """Validate dataset mixture configuration."""
        if len(self.datasets) != len(self.weights):
            raise ValueError("The length of `weights` must match the length of `datasets`.")
        if self.action_freq <= 0:
            raise ValueError(f"`action_freq` must be a positive number, got {self.action_freq}.")
        if self.image_resample_strategy not in ["linear", "nearest"]:
            raise ValueError(
                f"`image_resample_strategy` must be one of ['linear', 'nearest'], got {self.image_resample_strategy}."
            )
        if self.vector_resample_strategy not in ["linear", "nearest"]:
            raise ValueError(
                f"`vector_resample_strategy` must be one of ['linear', 'nearest'], got {self.vector_resample_strategy}."
            )


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
        excluded_keys = ["enable", "disable_artifact", "project"]
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
