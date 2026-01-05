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

"""COCO-QA dataset for visual question answering and grounding tasks.

This module provides the COCO-QA dataset implementation for training
vision-language models on visual question answering tasks. The dataset is
filtered to only include 'where' questions, focusing on spatial reasoning
tasks that are relevant for robotic manipulation.

The dataset is loaded from HuggingFace (ThucPD/coco-qa-vi) and automatically
filtered to retain only spatial reasoning questions.

Classes:
    COCODataset: Dataset class that loads, filters, and formats COCO-QA data
        for grounding tasks.

Functions:
    _img_to_normalized_tensor: Convert PIL Image to normalized torch tensor
        with channel-first format and [0, 1] normalization.
    _filter_dataset: Filter dataset samples to only include 'where' questions
        for spatial reasoning tasks.

Example:
    Use COCO-QA dataset in training:
        >>> from opentau.configs.default import DatasetConfig
        >>> cfg = DatasetConfig(grounding="cocoqa")
        >>> dataset = make_dataset(cfg, train_cfg)
"""

import logging
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from opentau import register_grounding_dataset
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.grounding.base import GroundingDataset

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def _img_to_normalized_tensor(img: Image.Image, img_shape: tuple) -> torch.Tensor:
    """Convert a PIL Image to a normalized torch tensor.

    Resizes the image and converts it from (H, W, C) to (C, H, W) format,
    normalizing pixel values to [0, 1].

    Args:
        img: PIL Image to convert.
        img_shape: Target image shape (height, width).

    Returns:
        Normalized tensor of shape (C, H, W) with values in [0, 1].
    """
    img = img.resize(img_shape, Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _filter_dataset(dataset: List) -> List:
    """Filter dataset to only include samples with 'where' questions.

    Args:
        dataset: List of dataset samples.

    Returns:
        Filtered list containing only samples with 'where' in the question.
    """
    filtered_dataset = []
    for sd in dataset:
        if "where" in sd["question"]:
            filtered_dataset.append(sd)

    return filtered_dataset


@register_grounding_dataset("cocoqa")
class COCODataset(GroundingDataset):
    """COCO-QA dataset for visual question answering and grounding tasks.

    Loads the ThucPD/coco-qa-vi dataset from HuggingFace and filters it to
    only include 'where' questions for spatial reasoning tasks.
    """

    def __init__(self, cfg: TrainPipelineConfig):
        self.dataset = load_dataset("ThucPD/coco-qa-vi", split="train")

        self.filtered_dataset = _filter_dataset(self.dataset)
        super().__init__(cfg)

    def __len__(self):
        return len(self.filtered_dataset)

    def _get_feature_mapping_key(self) -> str:
        return "cocoqa"

    def __getitem_helper__(self, item) -> dict:
        """Get a COCO-QA dataset item.

        Args:
            item: Index of the item to retrieve.

        Returns:
            Dictionary with image, task, postfix, task_type, and prompt
            extracted from the COCO-QA dataset sample.
        """
        sample = self.filtered_dataset[item]
        img = sample["image"]

        return {
            "image": _img_to_normalized_tensor(img, self.resolution),
            "task": "grounding",
            "postfix": f"The answer is {sample['answer']}",
            "task_type": "grounding",
            "prompt": f'{{"task": "grounding", "description": "Using the Image, Answer the following question. \n  {sample["question"]}"}}',
        }
