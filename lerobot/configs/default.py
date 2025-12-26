#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

import draccus
import numpy as np
from draccus.parsers.encoding import encode_dataclass

from lerobot.common import (
    policies,  # noqa: F401
)
from lerobot.common.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING, LOSS_TYPE_MAPPING
from lerobot.common.datasets.transforms import ImageTransformsConfig
from lerobot.common.datasets.video_utils import get_safe_default_codec

# --- Custom NumPy encoder registration ---
# For decoding from cmd/yaml
draccus.decode.register(np.ndarray, np.asarray)
# For encoding to yaml
draccus.encode.register(np.ndarray, lambda x: x.tolist())


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
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
    enable: bool = False  # Enable Weights & Biases logging.
    entity: str | None = None  # The entity name in Weights & Biases, e.g. your username or your team name
    project: str = "lerobot"  # The project name in Weights & Biases, e.g. "pi0"
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
        if not self.enable or self.notes is not None:
            return

        confirm = False
        while not confirm:
            self.notes = input("Please enter a description for wandb logging:\n")
            confirm = input("Confirm (y/N): ").strip().lower() == "y"

    def to_wandb_kwargs(self, step=None):
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
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )
