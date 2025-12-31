import torch

from opentau import register_grounding_dataset
from opentau.datasets.grounding.base import GroundingDataset
from opentau.configs.train import TrainPipelineConfig

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
    def __init__(self, cfg: TrainPipelineConfig, length: int = 1000):
        self.length = length
        super().__init__(cfg)

    def __len__(self):
        return self.length

    def __getitem_helper__(self, item) -> dict:
        return _data[item % len(_data)]

    def _get_feature_mapping_key(self) -> str:
        return "dummy"
