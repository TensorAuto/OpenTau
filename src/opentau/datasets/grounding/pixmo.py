"""
Datasets for Image-Text Point Set grounding tasks.

Example usage:

    from time import sleep
    from datetime import datetime
    from torch.utils.data import DataLoader
    from opentau.datasets.grounding.pixmo import PixmoDataset

    pixmo_loader = DataLoader(
        PixmoDataset(),
        batch_size=16,
        num_workers=8,
        prefetch_factor=8,
    )
    now = datetime.now()
    for i, batch in enumerate(pixmo_loader):
        print(f"PixMo batch {i}: {batch['image'].shape=}, {batch['postfix']=}, {batch['task']=}")
        sleep(1)  # To simulate processing time
        print("time elapsed", datetime.now() - now)
        now = datetime.now()
"""

import json
import logging
import random
import warnings
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from datasets import load_dataset
from opentau import register_grounding_dataset
from opentau.datasets.grounding.base import GroundingDataset
from opentau.configs.train import TrainPipelineConfig

# TODO: add a config to filter the warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
    module=r"PIL\.Image",
)
warnings.filterwarnings(
    "ignore",
    message=r"image file could not be identified because AVIF support not installed",
    category=UserWarning,
    module=r"PIL\.Image",
)

IMG_SIZE = 224
POINT_GRID = 255
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


def _get_post_fix(label, points, orig_w, orig_h, max_points=16):
    r"""Map the points from pixel space to grid space, deduplicate, and return a postfix string (in json format)."""
    # use `dict` to deduplicate as `set` is not guaranteed to preserve order
    deduplicated = {
        (int(p["x"] * POINT_GRID / orig_w), int(p["y"] * POINT_GRID / orig_h)): None for p in points
    }
    if len(deduplicated) > max_points:
        deduplicated = random.choices(list(deduplicated), k=max_points)
    rows = [{"in_frame": True, "point": pair, "label": label} for pair in deduplicated]
    return json.dumps(rows)


def _img_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    # pytorch uses (C, H, W) while PIL uses (H, W, C)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


@register_grounding_dataset("pixmo")
class PixmoDataset(GroundingDataset):
    r"""Dataset for the iterable PixMo dataset implementation, recommended to be used together with PrefetchWrapper"""

    def __init__(self, cfg: TrainPipelineConfig, consecutive_bad_tolerance=100):
        # Self.ds is needed for metadata, which is computed in parent constructor
        self.ds = load_dataset("allenai/pixmo-points", split="train")
        super().__init__(cfg)
        self.bad_ids = set()
        self.consecutive_bad_tolerance = consecutive_bad_tolerance

    def __len__(self):
        return len(self.ds)

    def _get_feature_mapping_key(self) -> str:
        return "pixmo"

    def __getitem_helper__(self, item):
        for _ in range(self.consecutive_bad_tolerance):
            if item in self.bad_ids:
                item = np.random.randint(0, len(self.ds))
                continue
            ex = self.ds[item]
            img = _pil_from_url(ex["image_url"])
            if img is None:
                self.bad_ids.add(item)
                item = np.random.randint(0, len(self.ds))
                continue

            return {
                "image": _img_to_normalized_tensor(img),
                "task": ex["label"],
                "postfix": _get_post_fix(ex["label"], ex["points"], *img.size),
                "task_type": "part",
                "prompt": f'{{"task": "part", "description": "Find {ex["label"]} in the image"}}',
            }

        raise RuntimeError("Too many consecutive bad items. Please check dataset or increase the tolerance.")
