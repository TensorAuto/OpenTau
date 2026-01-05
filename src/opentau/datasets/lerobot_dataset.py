#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""LeRobot dataset implementation for robot learning data management.

This module provides the core dataset implementation for loading, creating, and
managing robot learning datasets. It supports both loading existing datasets from
the HuggingFace Hub or local disk, as well as creating new datasets for data
recording.

The dataset structure consists of:

    - Metadata: Info, statistics, tasks, and episode information stored as JSON
    - Data files: Episode data stored as Parquet files organized by chunks
    - Videos: Optional video files for camera observations stored as MP4 files

Key Features:

    - Temporal alignment: Supports delta timestamps for temporal feature
      alignment, enabling sampling of features at different time offsets with
      optional Gaussian noise for data augmentation.
    - Multi-modal support: Handles images, videos, state vectors, actions, and
      text prompts with automatic format conversion and standardization.
    - Version compatibility: Automatic version checking and backward compatibility
      handling for datasets created with older format versions.
    - Asynchronous image writing: Optional async image writer for high-frequency
      data recording without blocking the main process.
    - Statistics management: Per-episode and aggregated statistics for data
      normalization, with automatic computation and aggregation.
    - Video handling: Supports multiple video backends (torchcodec, pyav,
      video_reader) for efficient video encoding and decoding.

Classes:

    DatasetMetadata
        Base class for dataset metadata management.

    LeRobotDatasetMetadata
        Metadata manager for LeRobot datasets with Hub integration, version
        checking, and statistics loading.

    GroundingDatasetMetadata
        Metadata manager for grounding datasets.

    BaseDataset
        Base PyTorch Dataset class with common functionality.

    LeRobotDataset
        Main dataset class for robot learning data, supporting loading from
        Hub/local disk, temporal alignment, video/image handling, and data
        recording.

Functions:
    retry_random_on_failure
        Decorator to retry dataset item retrieval with random indices on failure.

Example:
    Load an existing dataset:
        >>> dataset = LeRobotDataset(cfg, repo_id="my-robot-dataset")
        >>> dataloader = DataLoader(dataset, batch_size=32)

    Create a new dataset for recording:
        >>> dataset = LeRobotDataset.create(
        ...     repo_id="my-new-dataset",
        ...     fps=30,
        ...     features={"state": {"shape": (7,), "dtype": "float32"}},
        ...     use_videos=True
        ... )
"""

import contextlib
import functools
import logging
import math
import shutil
import traceback
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils
from datasets import concatenate_datasets, load_dataset
from einops import rearrange
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from opentau.configs.train import TrainPipelineConfig
from opentau.constants import HF_OPENTAU_HOME
from opentau.datasets.compute_stats import aggregate_stats, compute_episode_stats
from opentau.datasets.image_writer import AsyncImageWriter, write_image
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING, LOSS_TYPE_MAPPING
from opentau.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices_soft,
    get_episode_data_index,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_advantages,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from opentau.datasets.video_utils import (
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from opentau.policies.value.configuration_value import ValueConfig
from opentau.policies.value.reward import (
    calculate_return_bins_with_equal_width,
)
from opentau.utils.utils import on_accelerate_main_proc


def retry_random_on_failure(f):
    """Decorator to retry dataset item retrieval with random indices on failure.

    When a dataset item fails to load, this decorator will retry with random
    indices up to `_total_rand_attempts` times before raising an error.

    Args:
        f: The `__getitem__` method to wrap.

    Returns:
        Wrapped function that retries on failure.
    """

    @functools.wraps(f)
    def wrapped(self, idx):
        g = getattr(self, "_rr_rng", None)
        total_attempts = getattr(self, "_total_rand_attempts", 0)
        if g is None:
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())  # different seed per DataLoader worker
            self._rr_rng = g

        n = len(self)
        cur = idx
        exceptions = []
        indices_tried = []
        for _ in range(total_attempts + 1):
            try:
                indices_tried.append(cur)
                return f(self, cur)
            except Exception as e:
                print(f"Encountered failure to load data at index {cur}; retrying with a different index.")
                cur = int(torch.randint(0, n, (1,), generator=g))
                exceptions.append(e)

        tb_strings = [
            f"Attempt {i}: trying to fetch index {item} ...\n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__))
            for i, (e, item) in enumerate(zip(exceptions, indices_tried, strict=False))
        ]
        tb_blob = "\n".join(tb_strings)
        raise RuntimeError(
            f"Failed to load data after {total_attempts + 1} attempt(s). "
            "Check the following traceback for each attempts made.\n\n"
            f"{tb_blob}"
        )

    return wrapped


CODEBASE_VERSION = "v2.1"


class DatasetMetadata:
    """Base class for dataset metadata containing info and statistics.

    Attributes:
        info: Dictionary containing dataset information (features, fps, etc.).
        stats: Dictionary containing dataset statistics for normalization.
        repo_id: Repository ID of the dataset (set by subclasses).
    """

    def __init__(self, *, info: dict = None, stats: dict = None):
        self.info = info or {"features": {}}
        self.stats = stats or {}

        for feature_name in self.stats:
            for metric in self.stats[feature_name]:
                if isinstance(self.stats[feature_name][metric], (list, tuple)):
                    self.stats[feature_name][metric] = np.array(self.stats[feature_name][metric])
                # TODO: check stats[feature_name][metric].shape is broadcastable with features[feature_name]["shape"]

        self.repo_id = None

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}


class GroundingDatasetMetadata(DatasetMetadata):
    """Metadata class for grounding datasets (vision-language datasets)."""

    pass


class LeRobotDatasetMetadata(DatasetMetadata):
    """Metadata manager for LeRobot datasets with Hub integration and version handling.

    This class manages all metadata for LeRobot datasets, including dataset info,
    statistics, episodes, tasks, and advantages. It handles loading from local disk
    or HuggingFace Hub, version compatibility checking, and provides utilities for
    accessing dataset files and information.

    The class automatically handles:
        - Loading metadata from local disk or downloading from HuggingFace Hub
        - Version compatibility checking and automatic version resolution
        - Backward compatibility with older dataset formats (v2.0 vs v2.1)
        - Episode and task management
        - Statistics aggregation (per-episode and global)

    Attributes:
        repo_id: Repository ID of the dataset on HuggingFace Hub.
        root: Local root directory where the dataset is stored.
        revision: Git revision (branch/tag/commit) of the dataset.
        info: Dictionary containing dataset information (features, fps, paths, etc.).
        stats: Aggregated statistics dictionary (mean, std, min, max, count).
        episodes_stats: Per-episode statistics dictionary.
        episodes: Dictionary mapping episode_index to episode information.
        tasks: Dictionary mapping task_index to task descriptions.
        task_to_task_index: Reverse mapping from task description to task_index.
        advantages: Dictionary mapping (episode_index, timestamp) to advantage values.

    Example:
        Load metadata from Hub:
            >>> meta = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet")
            >>> print(f"Total episodes: {meta.total_episodes}")

        Create new dataset metadata:
            >>> meta = LeRobotDatasetMetadata.create(
            ...     repo_id="my-dataset",
            ...     fps=30,
            ...     features={"state": {"dtype": "float32", "shape": (7,)}}
            ... )
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_OPENTAU_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self) -> None:
        """Load dataset metadata from disk.

        Loads info, tasks, episodes, statistics, and advantages from the
        dataset root directory. Handles version compatibility checks.
        """
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

        self.advantages = load_advantages(self.root)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        """Get the file path for a specific episode's parquet data file.

        Args:
            ep_index: Episode index.

        Returns:
            Path to the parquet file for the episode.
        """
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """Get the file path for a specific episode's video file.

        Args:
            ep_index: Episode index.
            vid_key: Video key/name (e.g., "camera0").

        Returns:
            Path to the video file for the episode.
        """
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for a given episode index.

        Episodes are grouped into chunks for efficient storage.

        Args:
            ep_index: Episode index.

        Returns:
            Chunk index containing this episode.
        """
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_OPENTAU_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        if features is None:
            raise ValueError("Dataset features must be explicitly passed upon creation.")
        else:
            # TODO(aliberts, rcadene): implement sanity check for features
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class BaseDataset(torch.utils.data.Dataset):
    """Base class for all robot learning datasets.

    This abstract base class provides common functionality for both LeRobotDataset
    and GroundingDataset, including data format standardization, image processing,
    and vector padding. It ensures all datasets conform to a standard format
    regardless of their source or structure.

    Key Features:
        - Standard data format conversion: Maps dataset-specific feature names
          to standard names (camera0, camera1, state, actions, etc.)
        - Image standardization: Resizes and pads images to target resolution
          while maintaining aspect ratio
        - Vector padding: Pads state and action vectors to maximum dimensions
        - Data type conversion: Converts floating-point tensors to bfloat16 for
          memory efficiency
        - String normalization: Ensures prompts and responses have consistent
          newline formatting

    Subclasses must implement:
        - `_get_feature_mapping_key()`: Returns the key used for feature name
          mapping (e.g., "lerobot/aloha_mobile_cabinet")
        - `_separate_image_in_time()`: Separates temporal image sequences into
          individual frames

    Attributes:
        resolution: Target image resolution (height, width).
        num_cams: Number of camera views in each sample.
        max_state_dim: Maximum dimension for state vectors.
        max_action_dim: Maximum dimension for action vectors.
        action_chunk: Number of actions processed in a chunk.

    Example:
        Create a custom dataset:
            >>> class MyDataset(BaseDataset):
            ...     def _get_feature_mapping_key(self):
            ...         return "my-dataset"
            ...     def _separate_image_in_time(self, item):
            ...         pass  # No temporal separation needed
    """

    def __init__(self, cfg: TrainPipelineConfig):
        super().__init__()
        # Standard Data Format parameters
        self.resolution = cfg.resolution  # resolution of images (H, W) in data sample
        self.num_cams = cfg.num_cams  # number of cameras in each data sample
        self.max_state_dim = cfg.max_state_dim  # maximum dimension of the state vector
        self.max_action_dim = cfg.max_action_dim  # maximum dimension of the action vector
        self.action_chunk = cfg.action_chunk  # number of actions to be processed in a chunk

    @abstractmethod
    def _get_feature_mapping_key(self) -> str:
        r"""Returns the key used for feature mapping"""
        pass

    @abstractmethod
    def _separate_image_in_time(self, item: dict):
        r"""Some keys correspond to 2 images, where the first is image at current timestamp and the second is the image
        from some time ago. We separate these 2 images into different keys by modifying the `item` dictionary.
        For example, {"image_key": torch.zeros(2, 3, 224, 224), "image_key_is_pad": [False, True] } will become
        {
            "image_key": torch.zeros(3, 224, 224),
            "image_key_is_pad: False,
        }.
        """
        raise NotImplementedError

    def _standardize_images(self, item, standard_item, n_cams, is_local) -> list[bool]:
        """Standardize image features to a common format.

        Resizes images to the target resolution with padding, and tracks
        which images are padded.

        Args:
            item: Input item dictionary with original image keys.
            standard_item: Output dictionary to populate with standardized images.
            n_cams: Number of cameras to process.
            is_local: Whether processing local (past) images.

        Returns:
            List of boolean values indicating which images are padded.
        """
        name_map = DATA_FEATURES_NAME_MAPPING[self._get_feature_mapping_key()]
        image_is_pad = []
        for cam_idx in range(n_cams):
            std_key = f"camera{cam_idx}"
            key = name_map.get(std_key)

            if key is None:
                standard_item[std_key] = torch.zeros((3, *self.resolution))
                image_is_pad.append(True)
            else:
                standard_item[std_key] = self.resize_with_pad(
                    item[key],
                    self.resolution[1],
                    self.resolution[0],
                    pad_value=0,
                )
                image_is_pad.append(item.get(key + "_is_pad", torch.tensor(False)).item())
            assert (
                len(standard_item[std_key].shape) == 3
                and standard_item[std_key].shape[0] == 3
                and standard_item[std_key].min() >= 0.0 - 1e-6  # bfloat16 results in precision loss
                and standard_item[std_key].max() <= 1.0 + 1e-6  # bfloat16 results in precision loss
            ), (
                f"Expected image {std_key} to have shape (3, H, W) with values in [0, 1], "
                f"Got shape {standard_item[std_key].shape}, "
                f"min={standard_item[std_key].min()}, "
                f"max={standard_item[std_key].max()}, "
                f"self={self._get_feature_mapping_key()}."
            )

        return image_is_pad

    def _to_standard_data_format(self, item: dict) -> dict:
        """Convert dataset item to standard data format.

        Standardizes feature names, separates images in time, pads vectors,
        and ensures consistent data types and formats.

        Args:
            item: Raw dataset item dictionary.

        Returns:
            Dictionary with standardized feature names and formats.
        """
        name_map = DATA_FEATURES_NAME_MAPPING[self._get_feature_mapping_key()]
        self._separate_image_in_time(item)

        standard_item = {}
        img_is_pad = self._standardize_images(item, standard_item, self.num_cams, False)

        for new_key, key in name_map.items():
            if new_key.startswith("camera"):
                continue
            standard_item[new_key] = item[key]

        # pad state and action vectors
        standard_item["state"] = self.pad_vector(standard_item["state"], self.max_state_dim)
        standard_item["actions"] = self.pad_vector(standard_item["actions"], self.max_action_dim)

        standard_item["img_is_pad"] = torch.tensor(img_is_pad, dtype=torch.bool)
        standard_item["action_is_pad"] = item[name_map["actions"] + "_is_pad"]

        # add loss type
        standard_item["loss_type"] = LOSS_TYPE_MAPPING[self._get_feature_mapping_key()]

        # cast all tensors in standard_item to bfloat16
        for key, value in standard_item.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                standard_item[key] = value.to(dtype=torch.bfloat16)

        # ensure that non-empty strings contain exactly one newline character at the end of the string
        for key in ["prompt", "response"]:
            if standard_item[key].endswith(
                "\n"
            ):  # ensure there isn't going to be an extra space at the end after calling replace
                standard_item[key] = standard_item[key][:-1]
            standard_item[key] = standard_item[key].replace("\n", " ") + "\n"

        return standard_item

    def resize_with_pad(self, img, width, height, pad_value=0) -> torch.Tensor:
        """Resize an image to target dimensions with padding.

        Maintains aspect ratio by resizing to fit within target dimensions,
        then pads on the left and top to reach exact target size.

        Args:
            img: Input image tensor of shape (C, H, W).
            width: Target width.
            height: Target height.
            pad_value: Value to use for padding. Defaults to 0.

        Returns:
            Resized and padded image tensor of shape (C, height, width).

        Raises:
            ValueError: If input image doesn't have 4 dimensions when reshaped.
        """
        # assume no-op when width height fits already
        img = rearrange(img, "c h w -> 1 c h w")
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)

        # rearrange back to (c, h, w)
        padded_img = rearrange(padded_img, "1 c h w -> c h w")
        return padded_img

    @staticmethod
    def pad_vector(vector, new_dim):
        """Only the last dimension of the vector is padded to 'new_dim' with zeros."""
        if vector.shape[-1] == new_dim:
            return vector
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector
        return new_vector


class LeRobotDataset(BaseDataset):
    """Main dataset class for loading and managing robot learning data.

    This class provides a PyTorch Dataset interface for robot learning datasets
    stored in the LeRobot format. It supports loading from HuggingFace Hub or
    local disk, handles temporal alignment with delta timestamps, manages video
    and image data, and provides data recording capabilities.

    The dataset structure consists of:
        - Metadata: JSON files containing info, statistics, episodes, tasks
        - Data files: Parquet files organized by chunks containing episode data
        - Videos: Optional MP4 files for camera observations

    Key Features:
        - Hub integration: Automatic download from HuggingFace Hub with version
          compatibility checking
        - Temporal alignment: Delta timestamps enable sampling features at
          different time offsets with optional Gaussian noise for augmentation
        - Video/image handling: Supports both video files and individual images
          with automatic frame extraction and synchronization
        - Episode filtering: Load specific episodes by index
        - Data recording: Create new datasets and add episodes programmatically
        - Statistics: Per-episode and aggregated statistics for normalization

    Two Usage Modes:
        1. Loading existing datasets: From local disk or HuggingFace Hub
        2. Creating new datasets: Using the `create()` classmethod for data
           recording

    Attributes:
        cfg: Training pipeline configuration.
        repo_id: Repository ID of the dataset.
        root: Local root directory for the dataset.
        meta: LeRobotDatasetMetadata instance containing all metadata.
        hf_dataset: HuggingFace Dataset containing parquet data.
        episodes: Dictionary mapping episode_index to episode info.
        image_transforms: Optional image transforms to apply.
        delta_timestamps_params: Processed delta timestamp parameters.
        feature2group: Mapping from features to temporal groups.
        video_backend: Backend used for video decoding.
        standardize: Whether to standardize data format.

    Example:
        Load dataset from Hub:
            >>> dataset = LeRobotDataset(cfg, repo_id="lerobot/aloha")
            >>> dataloader = DataLoader(dataset, batch_size=32)

        Load specific episodes:
            >>> dataset = LeRobotDataset(
            ...     cfg,
            ...     repo_id="lerobot/aloha",
            ...     episodes=[0, 1, 2, 5, 10]
            ... )

        Create new dataset for recording:
            >>> dataset = LeRobotDataset.create(
            ...     cfg,
            ...     repo_id="my-new-dataset",
            ...     fps=30,
            ...     features={"state": {"dtype": "float32", "shape": (7,)}}
            ... )
    """

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, np.ndarray | list[float]] | None = None,
        delta_timestamps_std: dict[str, np.ndarray | list[float]] | None = None,
        delta_timestamps_lower: dict[str, np.ndarray | list[float]] | None = None,
        delta_timestamps_upper: dict[str, np.ndarray | list[float]] | None = None,
        feature2group: dict[str, tuple[str, (list[int] | int | None)]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_resample_strategy: str = "nearest",
        vector_resample_strategy: str = "nearest",
        standardize: bool = True,
        return_advantage_input: bool = False,
    ):
        """Initialize LeRobotDataset.

        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:

            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.

        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.

        In terms of files, LeRobotDataset encapsulates 3 main things:

            - metadata:

                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.

            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.

            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path::

            .
            ├── data
            │   ├── chunk-000
            │   │   ├── episode_000000.parquet
            │   │   ├── episode_000001.parquet
            │   │   ├── episode_000002.parquet
            │   │   └── ...
            │   ├── chunk-001
            │   │   ├── episode_001000.parquet
            │   │   ├── episode_001001.parquet
            │   │   ├── episode_001002.parquet
            │   │   └── ...
            │   └── ...
            ├── meta
            │   ├── episodes.jsonl
            │   ├── info.json
            │   ├── stats.json
            │   └── tasks.jsonl
            └── videos
                ├── chunk-000
                │   ├── observation.images.laptop
                │   │   ├── episode_000000.mp4
                │   │   ├── episode_000001.mp4
                │   │   ├── episode_000002.mp4
                │   │   └── ...
                │   ├── observation.images.phone
                │   │   ├── episode_000000.mp4
                │   │   ├── episode_000001.mp4
                │   │   ├── episode_000002.mp4
                │   │   └── ...
                ├── chunk-001
                └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            cfg (TrainPipelineConfig): Training configuration object.
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the HF_OPENTAU_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/opentau'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): Dictionary where each key is a group name and its
                corresponding value is a list of delta timestamps in seconds. For example, {'group1': [0, 0.1]} means
                features of group1 will be returned as a chunk of 2, with the first element being the value at current
                time and the second element being the value at current time + 0.1 seconds. This will also add a key
                named '{feature}_is_pad' to the returned item, with a boolean type and a length of 2, indicating whether
                the feature is padded or not. Padding will happen when t + 0.1 is outside the episode time range.
                Defaults to None.
            delta_timestamps_std: (dict[list[float]] | None, optional): Similar to delta_timestamps, but specifies an
                optional standard deviation for the delta timestamps. If a key is absent, the delta timestamps for that
                key will be deterministic. If a key is present without corresponding delta_timestamps, it will be
                ignored. E.g., delta_timestamps={'group1': [0, 0.1]} and delta_timestamps_std={'group1': [0, 0.05]} will
                result in a chunk of 2, with the first element being the feature at current time and the second
                element at a time following a Gaussian distribution with N(t+0.1, 0.05^2). When it takes on a value
                outside the episode, the corresponding element in `{feature}_is_mask` will be set to True.
                Defaults to None.
            delta_timestamps_lower: (dict[list[float]] | None, optional): Similar to delta_timestamps_std, but specifies
                a minimum value for the delta timestamps. When specified, the delta timestamps will be lower-clipped
                accordingly. Defaults to None.
            delta_timestamps_upper: (dict[list[float]] | None, optional): Similar to delta_timestamps_std, but specifies
                a maximum value for the delta timestamps. When specified, the delta timestamps will be upper-clipped
                accordingly. Defaults to None.
            feature2group: (dict[str, tuple[str, (list[int] | int | None)]] | None, optional): Dictionary mapping every
                individual feature to a tuple of (group name, indices). Group names are keys passed to delta_timestamps.
                If `indices` is None, will use all indices in the group. If indices is a list, will use only those
                indices in the corresponding order, including duplicates if present. If indices is an int, will return
                that index only, resulting in a reduction in ndim by 1.
                For example, `feature2group={'action': ('group1', None), 'observation.state': ('group2', 0),
                'observation.images.left_hand': ('group2', [0, 1])}` means the feature `action` will use resolved
                `delta_timestamps` from `group1` and will return every index. Also, `observation.state` will pick the
                first element (index-0) of `group2`. `observation.images.left_hand` will pick the first and second
                elements (indices 0 and 1) of `group2` and return them as a chunk of 2 images. The first element of
                `observation.images.left_hand` and the state vector will always be sampled at the same timestamp despite
                having gaussian noise applied, because they are in the same group.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            image_resample_strategy: str: Resampling strategy to use for image features.
                If 'linear', it will use linear interpolation between two immediate timestamps.
                If 'nearest', it will use nearest neighbor interpolation.
                Defaults to 'nearest'.
            vector_resample_strategy: str: Resampling strategy to use for non-image features, such as action or state.
                If 'linear', it will use linear interpolation between two immediate timestamps.
                If 'nearest', it will use nearest neighbor interpolation.
                Defaults to 'nearest'.
            standardize (bool, Optional): Flag to enable standardization in `__getitem__`. Defaults to True.
            return_advantage_input (bool, Optional): Flag to return advantage inputs ("success", "episode_end_idx", "current_idx", "last_step", "episode_index", "timestamp", ). Defaults to False. Ignored if standardize is False.
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_OPENTAU_HOME / repo_id
        self.image_transforms = image_transforms
        if bool(delta_timestamps) ^ bool(feature2group):
            raise ValueError(
                "Either both delta_timestamps and feature2group should be provided, or neither of them."
            )
        # delta_timestamps_params is a 4 tuple (mean, std, lower, upper)
        self.delta_timestamps_params = self.compute_delta_params(
            delta_timestamps,
            delta_timestamps_std,
            delta_timestamps_lower,
            delta_timestamps_upper,
        )
        self.feature2group = feature2group or {}
        self._check_feature_group_mapping()
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()

        if image_resample_strategy not in ["linear", "nearest"]:
            raise ValueError(
                f"Invalid image resample strategy: {image_resample_strategy}. Choose 'linear' or 'nearest'."
            )
        if vector_resample_strategy not in ["linear", "nearest"]:
            raise ValueError(
                f"Invalid action resample strategy: {vector_resample_strategy}. Choose 'linear' or 'nearest'."
            )
        self.image_resample_strategy = image_resample_strategy
        self.vector_resample_strategy = vector_resample_strategy

        self.standardize = standardize
        if return_advantage_input and not standardize:
            print(
                "Warning: `return_advantage_input` is True while `standardize` is False. "
                "No advantage inputs will be returned."
            )
        self.return_advantage_input = return_advantage_input

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        if self.episodes is None:
            self.episodes = list(self.meta.episodes)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index, self.epi2idx = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        # If transform is set, with_transform will decode all columns of a row before returning the desired column(s).
        no_transform_ds = self.hf_dataset.with_transform(None).with_format("numpy")
        logging.info("Checking timestamps synchronization...")
        timestamps = np.asarray(no_transform_ds["timestamp"], dtype=np.float32)
        episode_indices = np.asarray(no_transform_ds["episode_index"], dtype=np.int64)
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

    @on_accelerate_main_proc(local=True, _sync=True)
    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    @on_accelerate_main_proc(local=True, _sync=True)
    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        """Get file paths for all selected episodes.

        Returns paths for both parquet data files and video files (if applicable)
        for all episodes in the dataset.

        Returns:
            List of file paths for episode data and videos.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        """Create an empty HuggingFace dataset with the correct features.

        Returns:
            Empty dataset with features matching the dataset specification.
        """
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices_soft(
        self, idx: int, ep_idx: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Get soft (float) indices for querying features with delta timestamps.

        Computes indices for features based on delta timestamps, accounting for
        episode boundaries. Returns both query indices and padding masks.

        Args:
            idx: Current data index.
            ep_idx: Current episode index.

        Returns:
            Tuple of (query_indices, padding):
                - query_indices: Dictionary mapping feature names to soft indices.
                - padding: Dictionary mapping feature names to boolean padding masks.
        """
        ep_start = self.episode_data_index["from"][self.epi2idx[ep_idx]].item()
        ep_end = self.episode_data_index["to"][self.epi2idx[ep_idx]].item()

        # Get the delta_indices by group
        delta_indices = get_delta_indices_soft(self.delta_timestamps_params, self.fps)
        # Map from group to feature
        delta_indices = {
            feature: delta_indices[group][
                slice(None) if indices is None else [indices] if isinstance(indices, int) else indices
            ]
            for feature, (group, indices) in self.feature2group.items()
        }
        query_indices = {
            key: np.clip(idx + delta_idx, ep_start, ep_end - 1) for key, delta_idx in delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor((idx + delta_idx < ep_start) | (idx + delta_idx >= ep_end))
            for key, delta_idx in delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Get query timestamps for video features.

        Converts soft indices to timestamps for video frame extraction.
        If query_indices is provided, uses them; otherwise uses current timestamp.

        Args:
            current_ts: Current timestamp in seconds.
            query_indices: Optional dictionary of soft indices for features.

        Returns:
            Dictionary mapping video keys to query timestamps.
        """
        if query_indices:
            # In case values are lists
            query_indices = {k: np.array(v, dtype=np.float32) for k, v in query_indices.items()}
            q_indices = next(iter(query_indices.values()))
            # Pick any (soft) row index, which is guaranteed to be within [ep_start, ep_end), then take the floor
            in_ep_row_idx = math.floor(q_indices[0])
            # Index of the episode (not index of row). E.g., episode_index = 36 for row index = 10000
            ep_idx = self.hf_dataset.select([in_ep_row_idx])["episode_index"][0].item()
            # Row index where the current episode start
            ep_start_row_idx = self.episode_data_index["from"][self.epi2idx[ep_idx]].item()
        else:
            ep_start_row_idx = None

        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = (query_indices[key] - ep_start_row_idx) / self.fps
            else:
                query_timestamps[key] = np.array([current_ts], dtype=np.float32)

        return query_timestamps

    def _query_hf_dataset_soft(self, soft_indices: dict[str, np.ndarray]) -> dict:
        """Query dataset using soft (float) indices with interpolation.

        Converts soft indices to hard indices based on resample strategy
        (linear interpolation or nearest neighbor).

        Args:
            soft_indices: Dictionary mapping feature names to soft (float) indices.

        Returns:
            Dictionary of feature values queried from the dataset.

        Raises:
            ValueError: If vector_resample_strategy is not 'linear' or 'nearest'.
        """
        # soft indices are float indices that need to be converted to hard (integer) indices
        if self.vector_resample_strategy == "linear":
            floor_indices = {k: np.floor(v).astype(int) for k, v in soft_indices.items()}
            dist2floor = {k: v - floor_indices[k] for k, v in soft_indices.items()}
            # In the unlikely case that the soft index is exactly (ep_end - 1), floor will (ep_end - 1), and (floor + 1)
            #  will be ep_end, which may be out of bounds (despite usually being the start of the next episode).
            #  Therefore, we add 0 instead of 1 whenever the distance to floor is 0.
            ceil_indices = {k: floor_indices[k] + (dist2floor[k] > 0.0) for k, v in soft_indices.items()}
            q_floor = self._query_hf_dataset(floor_indices)
            q_ceil = self._query_hf_dataset(ceil_indices)

            item = {}
            for k, d2f in dist2floor.items():
                if k not in q_floor:
                    continue
                d2f = torch.tensor(d2f)
                d2f = rearrange(d2f, f"n -> {'n' + ' 1' * (q_floor[k].ndim - 1)}")
                item[k] = (1.0 - d2f) * q_floor[k] + d2f * q_ceil[k]
            return item
        elif self.vector_resample_strategy == "nearest":
            hard_indices = {k: v.round().astype(int) for k, v in soft_indices.items()}
            return self._query_hf_dataset(hard_indices)

        raise ValueError(
            f"Unsupported vector_resample_strategy: {self.vector_resample_strategy}. Choose 'linear' or 'nearest'."
        )

    def _query_hf_dataset(self, hard_indices: dict[str, np.ndarray]) -> dict:
        """Query dataset using hard (integer) indices.

        Args:
            hard_indices: Dictionary mapping feature names to integer indices.

        Returns:
            Dictionary of feature values stacked as tensors.
        """
        # TODO(shuheng): look into optimization when using hf_dataset.select
        return {
            key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))
            for key, q_idx in hard_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, np.ndarray], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frame_indices_soft = query_ts * self.fps
            if self.image_resample_strategy == "linear":
                frame_indices_floor = np.floor(frame_indices_soft).astype(int)
                dist2floor = frame_indices_soft - frame_indices_floor
                frame_indices_ceil = np.floor(frame_indices_soft) + 1.0 * (dist2floor > 0.0)
                query_ts_floor = (frame_indices_floor / self.fps).tolist()
                query_ts_ceil = (frame_indices_ceil / self.fps).tolist()
                frames_floor = decode_video_frames(
                    video_path, query_ts_floor, self.tolerance_s, self.video_backend
                )
                frames_ceil = decode_video_frames(
                    video_path, query_ts_ceil, self.tolerance_s, self.video_backend
                )
                dist2floor = dist2floor[:, None, None, None]
                frames = frames_ceil * dist2floor + frames_floor * (1 - dist2floor)
            elif self.image_resample_strategy == "nearest":
                query_ts_rounded = (frame_indices_soft.round() / self.fps).tolist()
                frames = decode_video_frames(
                    video_path, query_ts_rounded, self.tolerance_s, self.video_backend
                )
            else:
                raise ValueError(
                    f"Unsupported image_resample_strategy: {self.image_resample_strategy}. Choose 'linear' or 'nearest'."
                )
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        """Add padding mask keys to the item dictionary.

        Args:
            item: Item dictionary to modify.
            padding: Dictionary mapping feature names to boolean padding masks.

        Returns:
            Modified item dictionary with padding keys added.
        """
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    @retry_random_on_failure
    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        if self.episode_data_index is not None and self.epi2idx is not None:
            ep_end = self.episode_data_index["to"][self.epi2idx[ep_idx]].item()

        episodes_info = self.meta.episodes[ep_idx]

        # Soft indices are floats instead of integers, which allows for different interpolation strategies such as
        # nearest neighbor or linear interpolation.
        query_indices_soft = None
        if self.delta_timestamps_params[0]:
            query_indices_soft, padding = self._get_query_indices_soft(idx, ep_idx)
            query_result = self._query_hf_dataset_soft(query_indices_soft)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices_soft)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        # If indices is an int, squeeze the feature
        for feature, (_, indices) in self.feature2group.items():
            if isinstance(indices, int):
                item[feature] = item[feature].squeeze(0)

        # The conversion script of AGI BOT dataset uses a dataloader to enumerate data and compute stats.
        # If we enable standardization, those stats will be computed under their mapped names, which is wrong.

        if self.standardize:
            # Add response as a string
            if "response" not in item:
                item["response"] = ""

            episode_index = item["episode_index"].item()
            # don't convert to timestamp to `float`, because torch.float64 is not supported on MPS
            timestamp = item["timestamp"]

            # change data naming to standard data format
            item = self._to_standard_data_format(item)

            if self.meta.advantages is not None:
                advantage = self.meta.advantages.get((episode_index, timestamp), 0)
                item["advantage"] = torch.tensor(advantage, dtype=torch.bfloat16)
            else:
                item["advantage"] = torch.tensor(0.0, dtype=torch.bfloat16)

            success = episodes_info.get("success", True)

            # only add the below fields to item when training or evaluating the value fns
            if isinstance(self.cfg.policy, ValueConfig):
                item["return_bin_idx"], item["return_continuous"] = calculate_return_bins_with_equal_width(
                    success,
                    self.cfg.policy.reward_config.number_of_bins,
                    ep_end,
                    self.cfg.policy.reward_config.reward_normalizer,
                    idx,
                    self.cfg.policy.reward_config.C_neg,
                )

                item["return_bin_idx"] = torch.tensor(item["return_bin_idx"], dtype=torch.long)
                item["return_continuous"] = torch.tensor(item["return_continuous"], dtype=torch.float32)
                # success, episode_end_idx and last step is required for calculating advantage
                if self.return_advantage_input:
                    item["success"] = success
                    item["episode_end_idx"] = ep_end
                    item["current_idx"] = idx
                    item["last_step"] = idx + self.cfg.policy.reward_config.N_steps_look_ahead >= ep_end
                    item["episode_index"] = episode_index
                    item["timestamp"] = timestamp
            else:
                item["return_bin_idx"] = torch.tensor(0, dtype=torch.long)
                item["return_continuous"] = torch.tensor(0, dtype=torch.float32)

            # sanity check for action chunk lengths
            assert item["actions"].shape[0] == self.cfg.action_chunk
            assert item["action_is_pad"].shape[0] == self.cfg.action_chunk

        return item

    def _get_feature_mapping_key(self) -> str:
        return self.repo_id

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index, _ = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

        self.episode_data_index, self.epi2idx = get_episode_data_index(self.meta.episodes, self.episodes)

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths

    def _separate_image_in_time(self, item: dict):
        name_map = DATA_FEATURES_NAME_MAPPING[self._get_feature_mapping_key()]
        cam_keys = {v for k, v in name_map.items() if k.startswith("camera")}
        for k in cam_keys:
            images = item.pop(k)
            assert len(images) == 2, (
                f"{k} in {self.__class__} is expected to have length 2, got shape={images.shape}"
            )
            item[k + "_local"], item[k] = images

            pads = item.pop(k + "_is_pad")
            assert len(pads) == 2, (
                f"{k} in {self.__class__} is expected to have length 2, got shape={pads.shape}"
            )
            item[k + "_local_is_pad"], item[k + "_is_pad"] = pads

    @staticmethod
    def compute_delta_params(
        mean: dict[str, np.ndarray | list[float]],
        std: dict[str, np.ndarray | list[float]],
        lower: dict[str, np.ndarray | list[float]],
        upper: dict[str, np.ndarray | list[float]],
    ):
        r"""Process the parameters `mean`, `std`, `lower` and `upper` for delta timestamps.
        Delta timestamps will be computed dynamically in `__getitem__` with `clip(dT, lower, upper)` where `dT` follows
        the gaussian distribution N(mean, std^2). Each parameter is a dictionary mapping group names to sequences of
        floats.

        For example, mean = {"group1": [-0.1, 0.0, 0.1], "group2": [0.0, 0.2]}. indicates that 3 delta timestamps for
        features in group1 will be sampled: time t-0.1, t, and t+0.1; and 2 delta timestamps for features in group2 will
        be sampled: time t and t+0.2, where t is the timestamp of the current data point.

        It is assumed that the `std`, `lower`, and `upper` have the same keys as `mean`, and matching keys have values
        of the same length. If a key absent from `std`, `lower`, or `upper`, it will be set to a default value.
        Namely, `std` will be set to all 0, `lower` will be set to all `-inf`, and `upper` will be set to `+inf`, with
        lengths equal to the length of sequences in `mean` for that key.
        If a key is absent from `mean` but present in `std`, `lower`, or `upper`, it will be ignored.

        After processing, the function returns four dictionaries: `mean`, `std`, `lower`, and `upper`, where each key
        is a feature name and each value is a numpy array of floats, satisfying the above conditions.
        """
        inf = float("inf")
        mean = mean or {}
        mean = {k: np.array(v) for k, v in mean.items()}

        std = std or {}
        std = {k: np.array(std.get(k) or np.zeros_like(v)) for k, v in mean.items()}

        lower = lower or {}
        lower = {k: np.array(lower.get(k) or (np.zeros_like(v) - inf)) for k, v in mean.items()}

        upper = upper or {}
        upper = {k: np.array(upper.get(k) or (np.zeros_like(v) + inf)) for k, v in mean.items()}

        for k in mean:
            if not (mean[k].shape == std[k].shape == lower[k].shape == upper[k].shape):
                raise ValueError(
                    f"Delta timestamps parameters for {k} have inconsistent shapes: "
                    f"mean={mean[k].shape}, std={std[k].shape}, lower={lower[k].shape}, upper={upper[k].shape}"
                )

        return mean, std, lower, upper

    def _check_feature_group_mapping(self):
        for feature, (group, indices) in self.feature2group.items():
            if group not in self.delta_timestamps_params[0]:
                raise ValueError(
                    f"Feature '{feature}' is mapped to group '{group}', which is not present in "
                    "delta_timestamps_params. Please check the mapping."
                )
            if indices is not None and not isinstance(indices, (int, list)):
                raise ValueError(
                    f"Indices for feature '{feature}' in group '{group}' should be a list, an int, or None"
                )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        image_resample_strategy: str = "nearest",
        vector_resample_strategy: str = "nearest",
        standardize: bool = True,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps_params = obj.compute_delta_params(None, None, None, None)
        obj.feature2group = {}
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.image_resample_strategy = image_resample_strategy
        obj.vector_resample_strategy = vector_resample_strategy
        obj.standardize = standardize
        obj.episode_data_index, obj.epi2idx = get_episode_data_index(obj.meta.episodes, obj.episodes)
        return obj
