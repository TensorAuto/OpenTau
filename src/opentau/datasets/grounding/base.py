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

"""Base class for vision-language grounding datasets.

This module provides the base class for all grounding datasets, which are used
for training vision-language-action models on image-text tasks without robot
actions. Grounding datasets provide images, prompts, and responses for tasks
like visual question answering, spatial reasoning, and object grounding.

The base class handles common functionality including:
    - Metadata creation with ImageNet statistics for images
    - Zero-padding of state and action features for compatibility
    - Standard data format conversion
    - Integration with the dataset mixture system

Classes:
    GroundingDataset: Abstract base class that all grounding datasets inherit
        from. Provides common functionality for metadata creation, data format
        conversion, and zero-padding of missing features.

Example:
    Create a custom grounding dataset:
        >>> from opentau import register_grounding_dataset
        >>> @register_grounding_dataset("my_dataset")
        >>> class MyGroundingDataset(GroundingDataset):
        ...     def __getitem_helper__(self, item):
        ...         return {"image": ..., "task": ..., "postfix": ...}
"""

from abc import abstractmethod
from copy import deepcopy
from typing import final

import torch

from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.lerobot_dataset import CODEBASE_VERSION, BaseDataset, GroundingDatasetMetadata


class GroundingDataset(BaseDataset):
    """Base class for vision-language grounding datasets.

    Grounding datasets are used for training vision-language-action models on
    image-text tasks without robot actions. They provide images, prompts, and
    responses for grounding tasks.

    Attributes:
        num_frames: Number of frames in the dataset.
        num_episodes: Number of episodes (always 1 for grounding datasets).
        meta: Dataset metadata containing features and statistics.
    """

    def __init__(self, cfg: TrainPipelineConfig, num_frames: int = 1, num_episodes: int = 1):
        super().__init__(cfg)
        self.num_frames = num_frames
        self.num_episodes = num_episodes
        self.meta = self.create_meta()

    def create_meta(self) -> GroundingDatasetMetadata:
        """Create metadata for the grounding dataset.

        Initializes metadata with ImageNet statistics for images and zero
        statistics for state and actions (since grounding datasets don't have them).

        Returns:
            GroundingDatasetMetadata object with initialized info and stats.
        """
        from opentau.datasets.factory import IMAGENET_STATS

        info = {
            "codebase_version": CODEBASE_VERSION,
            "features": {
                "camera0": {
                    "dtype": "image",
                    "shape": [3, 224, 224],
                    "names": ["channel", "height", "width"],
                },
            },
        }
        stats = {
            "image": {
                "min": [[[0.0]], [[0.0]], [[0.0]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "count": [len(self)],
                **deepcopy(IMAGENET_STATS),  # mean and std
            },
            "state": {
                "min": [0.0],
                "max": [0.0],
                "mean": [0.0],
                "std": [0.0],
                "count": [len(self)],
            },
            "actions": {
                "min": [0.0],
                "max": [0.0],
                "mean": [0.0],
                "std": [0.0],
                "count": [len(self)],
            },
        }
        metadata = GroundingDatasetMetadata(info=info, stats=stats)
        metadata.repo_id = self._get_feature_mapping_key()
        return metadata

    @abstractmethod
    def __getitem_helper__(self, item) -> dict:
        """Helper method to get a dataset item (to be implemented by subclasses).

        Args:
            item: Index of the item to retrieve.

        Returns:
            Dictionary containing the raw item data with keys like 'image',
            'task', 'postfix', 'task_type', 'prompt'.
        """
        pass

    @final
    def __getitem__(self, item):
        item = self.__getitem_helper__(item)

        # Grounding datasets don't have states or actions. 0-padding is used.
        item["state"] = torch.zeros(self.max_state_dim)
        item["actions"] = torch.zeros(self.action_chunk, self.max_action_dim)
        item["actions_is_pad"] = torch.ones(self.action_chunk, dtype=torch.bool)
        item = self._to_standard_data_format(item)
        item["return_bin_idx"] = torch.tensor(0, dtype=torch.long)
        item["return_continuous"] = torch.tensor(0, dtype=torch.float32)
        item["advantage"] = torch.tensor(0, dtype=torch.bfloat16)
        return item

    def _separate_image_in_time(self, item: dict) -> None:
        """Separate images in time (no-op for grounding datasets).

        Grounding datasets don't have temporal image sequences, so this is a no-op.

        Args:
            item: Item dictionary (unmodified).
        """
        # Grounding datasets has nothing to separate.
        pass
