import logging
from typing import List

import numpy as np
import torch
from PIL import Image

from datasets import load_dataset
from opentau import register_grounding_dataset
from opentau.datasets.grounding.base import GroundingDataset
from opentau.configs.train import TrainPipelineConfig

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def _img_to_normalized_tensor(img: Image.Image, img_shape: tuple) -> torch.Tensor:
    img = img.resize(img_shape, Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _filter_dataset(dataset: List) -> List:
    filtered_dataset = []
    for sd in dataset:
        if "where" in sd["question"]:
            filtered_dataset.append(sd)

    return filtered_dataset


@register_grounding_dataset("cocoqa")
class COCODataset(GroundingDataset):
    def __init__(self, cfg: TrainPipelineConfig):
        self.dataset = load_dataset("ThucPD/coco-qa-vi", split="train")

        self.filtered_dataset = _filter_dataset(self.dataset)
        super().__init__(cfg)

    def __len__(self):
        return len(self.filtered_dataset)

    def _get_feature_mapping_key(self) -> str:
        return "cocoqa"

    def __getitem_helper__(self, item):
        sample = self.filtered_dataset[item]
        img = sample["image"]

        return {
            "image": _img_to_normalized_tensor(img, self.resolution),
            "task": "grounding",
            "postfix": f"The answer is {sample['answer']}",
            "task_type": "grounding",
            "prompt": f'{{"task": "grounding", "description": "Using the Image, Answer the following question. \n  {sample["question"]}"}}',
        }
