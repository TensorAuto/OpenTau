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

"""Dummy grounding dataset for testing and development.

This module provides a simple synthetic grounding dataset for testing the
dataset infrastructure without requiring external data sources or network
access. The dataset contains three predefined items: black, white, and gray
images with corresponding question-answer pairs.

The dataset cycles through the predefined items, making it useful for
testing data loading pipelines, training loops, and debugging.

Classes:
    DummyGroundingDataset: Synthetic dataset class that provides simple test
        data with configurable length.

Constants:
    _data: List of three predefined dataset items (black, white, gray images)
        with corresponding tasks, postfixes, and prompts.

Example:
    Use dummy dataset for testing:
        >>> from opentau.configs.default import DatasetConfig
        >>> cfg = DatasetConfig(grounding="dummy")
        >>> dataset = make_dataset(cfg, train_cfg)
        >>> len(dataset)  # Returns 1000 by default
"""

import torch

from opentau import register_grounding_dataset
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.grounding.base import GroundingDataset

_data = [
    {
        "image": torch.zeros(3, 224, 224),
        "task": "What do you see in the image?",
        "postfix": "This is a black image",
        "task_type": "qa",
        "prompt": '{"task": "qa", "description": "What do you see in the image?"}',
    },
    {
        "image": torch.ones(3, 224, 224),
        "task": "What do you see in the image?",
        "postfix": "This is a white image",
        "task_type": "qa",
        "prompt": '{"task": "qa", "description": "What do you see in the image?"}',
    },
    {
        "image": torch.ones(3, 224, 224) * 0.5,
        "task": "What do you see in the image?",
        "postfix": "This is a gray image",
        "task_type": "qa",
        "prompt": '{"task": "qa", "description": "What do you see in the image?"}',
    },
]


@register_grounding_dataset("dummy")
class DummyGroundingDataset(GroundingDataset):
    """Dummy grounding dataset for testing purposes.

    Provides simple synthetic data with black, white, and gray images
    for testing the dataset infrastructure.
    """

    def __init__(self, cfg: TrainPipelineConfig, length: int = 1000):
        self.length = length
        super().__init__(cfg)

    def __len__(self):
        return self.length

    def __getitem_helper__(self, item) -> dict:
        """Get a dummy dataset item.

        Cycles through a small set of predefined items (black, white, gray images).

        Args:
            item: Index of the item to retrieve.

        Returns:
            Dictionary with image, task, postfix, task_type, and prompt.
        """
        return _data[item % len(_data)]

    def _get_feature_mapping_key(self) -> str:
        return "dummy"
