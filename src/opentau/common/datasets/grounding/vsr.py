import logging
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from datasets import load_dataset
from lerobot import register_grounding_dataset
from lerobot.common.datasets.grounding.base import GroundingDataset
from lerobot.configs.train import TrainPipelineConfig

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

MAX_RETRIES = 1
HTTP_TIMEOUT = 1
LOG_EVERY_N_BAD = 1000

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
    ),
)


def _pil_from_url(url: str) -> Image.Image | None:
    """Download, decode, and resize an image using its URL. Returns None in case of failure."""
    try:
        r = _session.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        # TODO: Check against the hash in case the image somehow changed.
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def _img_to_normalized_tensor(img: Image.Image, img_shape: tuple) -> torch.Tensor:
    img = img.resize(img_shape, Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


@register_grounding_dataset("vsr")
class VSRDataset(GroundingDataset):
    def __init__(self, cfg: TrainPipelineConfig, consecutive_bad_tolerance=100):
        self.dataset = load_dataset("cambridgeltl/vsr_random", split="train")
        super().__init__(cfg)
        self.bad_ids = set()
        self.consecutive_bad_tolerance = consecutive_bad_tolerance
        self.mapping = {0: "False", 1: "True"}

    def __len__(self):
        return len(self.dataset)

    def _get_feature_mapping_key(self) -> str:
        return "vsr"

    def __getitem_helper__(self, item):
        for _ in range(self.consecutive_bad_tolerance):
            if item in self.bad_ids:
                item = np.random.randint(0, len(self.dataset))
                continue

            sample = self.dataset[item]
            img = _pil_from_url(sample["image_link"])
            if img is None:
                self.bad_ids.add(item)
                item = np.random.randint(0, len(self.dataset))
                continue

            return {
                "image": _img_to_normalized_tensor(img, self.resolution),
                "task": sample["label"],
                "postfix": f"The statement is {self.mapping[sample['label']]}",
                "task_type": "grounding",
                "prompt": f'{{"task": "grounding", "description": "Using the Image, Tell me if following statement is true or false. \n  {sample["caption"]}"}}',
            }

        raise RuntimeError("Too many consecutive bad items. Please check dataset or increase the tolerance.")
