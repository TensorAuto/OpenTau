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
    img = img.resize(img_shape, Image.BILINEAR)

    # pytorch uses (C, H, W) while PIL uses (H, W, C)
    return torch.from_numpy(np.array(img))[:, :, :3].permute(2, 0, 1).float() / 255.0


@register_grounding_dataset("clevr")
class CLEVRDataset(GroundingDataset):
    def __init__(self, cfg: TrainPipelineConfig, consecutive_bad_tolerance=100):
        self.dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex", split="train")
        super().__init__(cfg)

    def __len__(self):
        return len(self.dataset)

    def _get_feature_mapping_key(self) -> str:
        return "clevr"

    def __getitem_helper__(self, item):
        sample = self.dataset[item]
        img = sample["image"]

        return {
            "image": _img_to_normalized_tensor(img, self.resolution),
            "task": "grounding",
            "postfix": f"The answer is {sample['solution'].split('<answer>')[1].split('</answer>')[0]}",
            "task_type": "grounding",
            "prompt": f'{{"task": "grounding", "description": "Using the Image, Answer the following question. \n  {sample["problem"]}"}}',
        }
