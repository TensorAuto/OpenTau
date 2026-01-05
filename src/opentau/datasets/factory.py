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

"""Factory functions for creating datasets and dataset mixtures.

This module provides factory functions to create individual datasets and
weighted dataset mixtures from configuration objects. It handles the setup
of delta timestamps, image transforms, and metadata configuration before
instantiating datasets.

The factory supports two types of datasets:
    1. LeRobot datasets: Standard robot learning datasets loaded from HuggingFace
       repositories with configurable delta timestamps for temporal alignment.
    2. Grounding datasets: Vision-language grounding datasets (CLEVR, COCO-QA,
       PIXMO, VSR, etc.) for multimodal learning tasks.

Key Features:
    - Delta timestamp resolution: Automatically configures temporal offsets
      for features based on policy latency settings (action decoder and
      cloud VLM latencies).
    - Image transform support: Applies configurable image transformations
      during dataset creation.
    - Imagenet stats override: Optionally replaces dataset statistics with
      ImageNet normalization statistics for camera features.
    - Grounding dataset registration: Supports extensible grounding dataset
      registration through side-effect imports.

Functions:
    make_dataset: Creates a single dataset instance from a DatasetConfig,
        handling delta timestamp setup, image transforms, and metadata
        configuration.
    make_dataset_mixture: Creates a WeightedDatasetMixture from a
        TrainPipelineConfig containing multiple dataset configurations.
    resolve_delta_timestamps: Resolves delta timestamps configuration based
        on TrainPipelineConfig settings, mapping features to temporal groups.

Constants:
    IMAGENET_STATS: ImageNet normalization statistics (mean, std, min, max)
        used for camera feature normalization when use_imagenet_stats is enabled.

Example:
    Create a single dataset:
        >>> dataset = make_dataset(dataset_cfg, train_cfg, return_advantage_input=False)

    Create a dataset mixture:
        >>> mixture = make_dataset_mixture(train_cfg, return_advantage_input=False)
        >>> dataloader = mixture.get_dataloader()
"""

import numpy as np

# NOTE: Don't delete; imported for side effects.
import opentau.datasets.grounding.clevr  # noqa: F401
import opentau.datasets.grounding.cocoqa  # noqa: F401
import opentau.datasets.grounding.dummy  # noqa: F401
import opentau.datasets.grounding.pixmo  # noqa: F401
import opentau.datasets.grounding.vsr  # noqa: F401
from opentau import available_grounding_datasets
from opentau.configs.default import DatasetConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.dataset_mixture import WeightedDatasetMixture
from opentau.datasets.lerobot_dataset import (
    BaseDataset,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from opentau.datasets.transforms import ImageTransforms

IMAGENET_STATS = {
    "min": [[[0.0]], [[0.0]], [[0.0]]],  # (c,1,1)
    "max": [[[1.0]], [[1.0]], [[1.0]]],  # (c,1,1)
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: TrainPipelineConfig, dataset_cfg: DatasetConfig, ds_meta: LeRobotDatasetMetadata
) -> tuple:
    """Resolves delta_timestamps by based on TrainPipelineConfig.

    Args:
        cfg (TrainPipelineConfig): The TrainPipelineConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        A 2-tuple containing:

            - At index 0, a 4-tuple containing delta timestamps mean, std, lower, and upper bounds for each group.
            - At index 1, a dictionary mapping feature names to their corresponding group and index.

        The delta timestamps and group mapping should follow the structure expected by LeRobotDataset.
    """
    group = "input_group"
    feature2group = {}
    # Delta timestamps are in seconds, and negative because they represent past timestamps.
    # Hence, lower and upper bounds correspond to -upper and -lower.
    delta_timestamps = {group: [-cfg.policy.action_decoder_latency_mean, -cfg.policy.cloud_vlm_latency_mean]}
    delta_timestamps_std = {group: [cfg.policy.action_decoder_latency_std, cfg.policy.cloud_vlm_latency_std]}
    delta_timestamps_lower = {
        group: [-cfg.policy.action_decoder_latency_upper, -cfg.policy.cloud_vlm_latency_upper]
    }
    delta_timestamps_upper = {
        group: [-cfg.policy.action_decoder_latency_lower, -cfg.policy.cloud_vlm_latency_lower]
    }
    action_freq = cfg.dataset_mixture.action_freq

    name_map = DATA_FEATURES_NAME_MAPPING[dataset_cfg.repo_id]
    reverse_name_map = {v: k for k, v in name_map.items()}
    for key in ds_meta.features:
        if key not in reverse_name_map:
            continue  # only process camera, state, and action features

        standard_key = reverse_name_map[key]
        if standard_key == "actions" and cfg.policy.action_delta_indices is not None:
            delta_timestamps[key] = [i / action_freq for i in cfg.policy.action_delta_indices]
            feature2group[key] = (key, None)
        if "camera" in standard_key:
            # Index 0 corresponds to action decoder latency and index 1 to cloud VLM latency.
            # Pick both indices. `_to_standard_data_format()` will separate the two.
            feature2group[key] = (group, [0, 1])
        elif standard_key == "state":
            # Pick index 0, which corresponds to latency of action decoder, and squeeze it to a scalar.
            feature2group[key] = (group, 0)

    return (
        delta_timestamps,
        delta_timestamps_std,
        delta_timestamps_lower,
        delta_timestamps_upper,
    ), feature2group


def make_dataset(
    cfg: DatasetConfig,
    train_cfg: TrainPipelineConfig,
    return_advantage_input: bool = False,
) -> BaseDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (DatasetConfig): A DatasetConfig used to create a LeRobotDataset.
        train_cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.
        return_advantage_input (bool): Whether the created dataset includes advantage inputs including "success",
            "episode_end_idx", "current_idx", "last_step", "episode_index", and "timestamp". Defaults to False.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        BaseDataset
    """
    image_transforms = ImageTransforms(cfg.image_transforms) if cfg.image_transforms.enable else None

    if isinstance(cfg.grounding, str) + isinstance(cfg.repo_id, str) != 1:
        raise ValueError("Exactly one of `cfg.grounding` and `cfg.repo_id` should be provided.")

    if isinstance(cfg.grounding, str):
        ds_cls = available_grounding_datasets.get(cfg.grounding)
        if ds_cls is None:
            raise ValueError(
                f"Unknown grounding dataset '{cfg.grounding}'. "
                f"Supported datasets are: {available_grounding_datasets.keys()}"
            )
        # TODO support dataset-specific arg / kwargs
        dataset = ds_cls(train_cfg)
    elif isinstance(cfg.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        (dt_mean, dt_std, dt_lower, dt_upper), f2g = resolve_delta_timestamps(train_cfg, cfg, ds_meta)
        dataset = LeRobotDataset(
            train_cfg,
            cfg.repo_id,
            root=cfg.root,
            episodes=cfg.episodes,
            delta_timestamps=dt_mean,
            delta_timestamps_std=dt_std,
            delta_timestamps_lower=dt_lower,
            delta_timestamps_upper=dt_upper,
            feature2group=f2g,
            image_transforms=image_transforms,
            revision=cfg.revision,
            video_backend=cfg.video_backend,
            image_resample_strategy=train_cfg.dataset_mixture.image_resample_strategy,
            vector_resample_strategy=train_cfg.dataset_mixture.vector_resample_strategy,
            return_advantage_input=return_advantage_input,
        )

    # TODO grounding datasets implement stats in original feature names, but camera_keys are standardized names
    if not isinstance(cfg.grounding, str) and "dummy" not in cfg.repo_id and cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                if key not in dataset.meta.stats:
                    dataset.meta.stats[key] = {}
                dataset.meta.stats[key][stats_type] = np.array(stats, dtype=np.float32)

    return dataset


def make_dataset_mixture(
    cfg: TrainPipelineConfig, return_advantage_input: bool = False
) -> WeightedDatasetMixture:
    """Creates a dataset mixture from the provided TrainPipelineConfig.

    Args:
        cfg (TrainPipelineConfig): The configuration containing the datasets to mix.
        return_advantage_input (bool): Whether the datasets should return advantage inputs including "success",
            "episode_end_idx", "current_idx", "last_step", "episode_index", and "timestamp". Defaults to False.

    Returns:
        WeightedDatasetMixture: An instance of WeightedDatasetMixture containing the datasets.
    """
    datasets = [
        make_dataset(dataset_cfg, cfg, return_advantage_input=return_advantage_input)
        for dataset_cfg in cfg.dataset_mixture.datasets
    ]
    return WeightedDatasetMixture(cfg, datasets, cfg.dataset_mixture.weights, cfg.dataset_mixture.action_freq)
