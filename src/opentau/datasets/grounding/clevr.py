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

"""CLEVR dataset for visual reasoning and grounding tasks.

This module provides the CLEVR (Compositional Language and Elementary Visual
Reasoning) dataset implementation for training vision-language models on
compositional visual reasoning tasks. The dataset contains synthetic scenes
with geometric objects and questions requiring compositional reasoning.

The dataset is loaded from HuggingFace and formatted for grounding tasks,
providing images, questions, and answers for visual reasoning.

Classes:
    CLEVRDataset: Dataset class that loads and formats CLEVR data from
        MMInstruction/Clevr_CoGenT_TrainA_70K_Complex on HuggingFace.

Functions:
    _img_to_normalized_tensor: Convert PIL Image to normalized torch tensor
        with channel-first format and [0, 1] normalization.

Example:
    Use CLEVR dataset in training:
        >>> from opentau.configs.default import DatasetConfig
        >>> cfg = DatasetConfig(grounding="clevr")
        >>> dataset = make_dataset(cfg, train_cfg)
"""

import logging

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

    # pytorch uses (C, H, W) while PIL uses (H, W, C)
    return torch.from_numpy(np.array(img))[:, :, :3].permute(2, 0, 1).float() / 255.0


@register_grounding_dataset("clevr")
class CLEVRDataset(GroundingDataset):
    """CLEVR dataset for visual reasoning and grounding tasks.

    Loads the MMInstruction/Clevr_CoGenT_TrainA_70K_Complex dataset from
    HuggingFace and formats it for grounding tasks.
    """

    def __init__(self, cfg: TrainPipelineConfig, consecutive_bad_tolerance=100):
        self.dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex", split="train")
        super().__init__(cfg)

    def __len__(self):
        return len(self.dataset)

    def _get_feature_mapping_key(self) -> str:
        return "clevr"

    def __getitem_helper__(self, item) -> dict:
        """Get a CLEVR dataset item.

        Args:
            item: Index of the item to retrieve.

        Returns:
            Dictionary with image, task, postfix, task_type, and prompt
            extracted from the CLEVR dataset sample.
        """
        sample = self.dataset[item]
        img = sample["image"]

        return {
            "image": _img_to_normalized_tensor(img, self.resolution),
            "task": "grounding",
            "postfix": f"The answer is {sample['solution'].split('<answer>')[1].split('</answer>')[0]}",
            "task_type": "grounding",
            "prompt": f'{{"task": "grounding", "description": "Using the Image, Answer the following question. \n  {sample["problem"]}"}}',
        }
