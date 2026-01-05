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
"""Datasets for Image-Text Point Set grounding tasks.

This module provides the PIXMO (Pixel-level Manipulation) dataset implementation
for training vision-language models on part localization and object grounding tasks.
"""

import json
import logging
import random
import warnings
from io import BytesIO

import numpy as np
import requests
import torch
from datasets import load_dataset
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from opentau import register_grounding_dataset
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.grounding.base import GroundingDataset

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


def _get_post_fix(label: str, points: list, orig_w: int, orig_h: int, max_points: int = 16) -> str:
    """Map points from pixel space to grid space and return a JSON postfix string.

    Converts pixel coordinates to a 255x255 grid, deduplicates points, and
    limits to max_points. Returns a JSON string with point coordinates and labels.

    Args:
        label: Label for the points (e.g., object class name).
        points: List of point dictionaries with 'x' and 'y' keys.
        orig_w: Original image width.
        orig_h: Original image height.
        max_points: Maximum number of points to include. Defaults to 16.

    Returns:
        JSON string containing point coordinates and labels.
    """
    # use `dict` to deduplicate as `set` is not guaranteed to preserve order
    deduplicated = {
        (int(p["x"] * POINT_GRID / orig_w), int(p["y"] * POINT_GRID / orig_h)): None for p in points
    }
    if len(deduplicated) > max_points:
        deduplicated = random.choices(list(deduplicated), k=max_points)
    rows = [{"in_frame": True, "point": pair, "label": label} for pair in deduplicated]
    return json.dumps(rows)


def _img_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a normalized torch tensor.

    Resizes the image to IMG_SIZE and converts it from (H, W, C) to (C, H, W)
    format, normalizing pixel values to [0, 1].

    Args:
        img: PIL Image to convert.

    Returns:
        Normalized tensor of shape (C, IMG_SIZE, IMG_SIZE) with values in [0, 1].
    """
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

    def __getitem_helper__(self, item) -> dict:
        """Get a PixMo dataset item.

        Downloads the image from URL and formats it for part localization tasks.
        Retries with random indices if image download fails.

        Args:
            item: Index of the item to retrieve.

        Returns:
            Dictionary with image, task, postfix, task_type, and prompt.

        Raises:
            RuntimeError: If too many consecutive items fail to load.
        """
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
