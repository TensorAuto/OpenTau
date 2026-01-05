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

"""Normalization and Unnormalization utilities for policies.

This module provides classes and functions to normalize and unnormalize data
(e.g., observations and actions) based on statistical properties (mean, std, min, max).
It handles different normalization modes and supports creating buffers for statistics.
"""

import sys

import numpy as np
import torch
from torch import Tensor, nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature

EPS = 1e-8  # Small epsilon value for numerical stability in normalization


def warn_missing_keys(features: dict[str, PolicyFeature], batch: dict[str, Tensor], mode: str) -> None:
    """Warns if expected features are missing from the batch.

    Args:
        features: Dictionary of expected policy features.
        batch: Dictionary containing the data batch.
        mode: The operation mode (e.g., "normalization" or "unnormalization") for the warning message.
    """
    for missing_key in set(features) - set(batch):
        red_seq = "\033[91m"
        reset_seq = "\033[0m"
        print(
            f"{red_seq}Warning: {missing_key} was missing from the batch during {mode}.{reset_seq}",
            file=sys.stderr,
        )


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args:
        features: Dictionary mapping feature names to PolicyFeature objects.
        norm_map: Dictionary mapping feature types to NormalizationMode.
        stats: Optional dictionary containing pre-computed statistics (mean, std, min, max)
            for each feature. If None, buffers are initialized with infinity.

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.

    Raises:
        ValueError: If stats contain types other than np.ndarray or torch.Tensor.
    """
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        shape = tuple(ft.shape)

        if ft.type is FeatureType.VISUAL:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )

        # TODO(aliberts, rcadene): harmonize this to only use one framework (np or torch)
        if stats:
            if isinstance(stats[key]["mean"], np.ndarray):
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = torch.from_numpy(stats[key]["mean"]).to(dtype=torch.float32)
                    buffer["std"].data = torch.from_numpy(stats[key]["std"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = torch.from_numpy(stats[key]["min"]).to(dtype=torch.float32)
                    buffer["max"].data = torch.from_numpy(stats[key]["max"]).to(dtype=torch.float32)
            elif isinstance(stats[key]["mean"], torch.Tensor):
                # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
                # tensors anywhere (for example, when we use the same stats for normalization and
                # unnormalization). See the logic here
                # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = stats[key]["mean"].clone().to(dtype=torch.float32)
                    buffer["std"].data = stats[key]["std"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = stats[key]["min"].clone().to(dtype=torch.float32)
                    buffer["max"].data = stats[key]["max"].clone().to(dtype=torch.float32)
            else:
                type_ = type(stats[key]["mean"])
                raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    """Returns an error message string for missing statistics.

    Args:
        name: Name of the statistic (e.g., "mean", "std").

    Returns:
        str: The error message string.
    """
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the Normalize module.

        Args:
            features: A dictionary where keys are input modalities (e.g. "observation.image") and values
                are their PolicyFeature definitions.
            norm_map: A dictionary where keys are feature types and values are their normalization modes.
            stats: A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Normalizes the batch data.

        Args:
            batch: Dictionary containing the data to normalize.

        Returns:
            dict[str, Tensor]: The normalized batch data.

        Raises:
            ValueError: If an unknown normalization mode is encountered.
        """
        warn_missing_keys(self.features, batch, "normalization")
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            if batch[key].numel() == 0:  # skip empty tensors, which won't broadcast well
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + EPS)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] - min) / (max - min + EPS)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(norm_mode)
        return batch


class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """Initializes the Unnormalize module.

        Args:
            features: A dictionary where keys are input modalities (e.g. "observation.image") and values
                are their PolicyFeature definitions.
            norm_map: A dictionary where keys are feature types and values are their normalization modes.
            stats: A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values. If provided,
                these statistics will overwrite the default buffers.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalizes the batch data.

        Args:
            batch: Dictionary containing the data to unnormalize.

        Returns:
            dict[str, Tensor]: The unnormalized batch data.

        Raises:
            ValueError: If an unknown normalization mode is encountered.
        """
        warn_missing_keys(self.features, batch, "unnormalization")
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            if batch[key].numel() == 0:  # skip empty tensors, which won't broadcast well
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * (std + EPS) + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min + EPS) + min
            else:
                raise ValueError(norm_mode)
        return batch
