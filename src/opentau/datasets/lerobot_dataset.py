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

    VQADatasetMetadata
        Metadata manager for vqa datasets.

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
import copy
import functools
import json
import logging
import math
import re
import shutil
import traceback
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import packaging.version
import PIL.Image
import pyarrow.dataset as pa_ds
import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils
from datasets import Dataset, DatasetInfo, concatenate_datasets
from einops import rearrange
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from opentau.configs.train import TrainPipelineConfig
from opentau.constants import HF_OPENTAU_HOME
from opentau.datasets.compute_stats import aggregate_stats, compute_episode_stats
from opentau.datasets.image_writer import AsyncImageWriter, write_image
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from opentau.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    DeltaTimestampInfo,
    DeltaTimestampParam,
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
    resample_and_trim_video,
)
from opentau.policies.value.configuration_value import ValueConfig
from opentau.policies.value.reward import (
    calculate_return_bins_with_equal_width,
)
from opentau.utils.accelerate_utils import get_proc_accelerator
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

# Set of repo_ids for which we've already emitted the "missing control_mode" warning.
# Keyed at module level so duplicates are suppressed across multiple LeRobotDataset
# instances within a single process (e.g., train + val constructed for the same repo).
_CONTROL_MODE_WARNED: set[str] = set()

# ``skip_timestamp_check`` is a mixture-wide decision (with optional per-dataset
# override) that produces an identical warning for every dataset in the mixture.
# For a 392-dataset pretraining run on 8 ranks, the naive per-dataset emission
# floods the run log with ~3K identical lines. This flag makes the warning fire
# once per process; combined with the rank-0 gate at the call site, that's once
# per run rather than 8 × num_datasets.
_SKIP_TIMESTAMP_WARNED: bool = False


def suppress_control_mode_warning(repo_id: str) -> None:
    """Mark ``repo_id`` as already-warned so the missing-``control_mode`` warning
    does not fire for it during subsequent ``LeRobotDataset.__init__`` calls.

    Intended for callers that supply an explicit ``control_mode`` override and
    therefore already know the on-disk metadata is missing the field. Must be
    invoked *before* the ``LeRobotDataset`` is constructed for ``repo_id``;
    once ``__init__`` has emitted the warning, further calls are a no-op.
    """
    _CONTROL_MODE_WARNED.add(repo_id)


_SPEED_BUCKET_SECONDS = 10


def speed_duration_bucket_s(num_frames: int, fps: float) -> int:
    """Bucket an episode's physical duration to the nearest 10 seconds.

    Used to compute the ``speed`` optional key emitted by
    ``LeRobotDataset.__getitem__``. Working in seconds (rather than native
    frames) makes the bucket invariant to the dataset's native FPS and to the
    mixture's ``cfg.dataset_mixture.action_freq`` resampling — see
    :ref:`standard-data-format-optional-keys` in ``docs/source/concepts.rst``.

    Uses Python's built-in ``round()`` (banker's rounding), e.g. a 25 s
    episode buckets to 20, not 30.
    """
    return int(round((num_frames / fps) / _SPEED_BUCKET_SECONDS) * _SPEED_BUCKET_SECONDS)


class DatasetMetadata:
    """Base class for dataset metadata containing info and statistics.

    Attributes:
        info: Dictionary containing dataset information (features, fps, etc.).
        stats: Dictionary containing dataset statistics for normalization.
        repo_id: Repository ID of the dataset (set by subclasses).
    """

    def __init__(self, *, info: dict | None = None, stats: dict | None = None):
        self.info = info or {"features": {}}
        self.stats = stats or {}

        for feature_name in self.stats:
            for metric in self.stats[feature_name]:
                if isinstance(self.stats[feature_name][metric], (list, tuple)):
                    self.stats[feature_name][metric] = np.array(self.stats[feature_name][metric])
                # TODO: check stats[feature_name][metric].shape is broadcastable with features[feature_name]["shape"]

        self.repo_id: str | None = None

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


class VQADatasetMetadata(DatasetMetadata):
    """Metadata class for vqa datasets (vision-language datasets)."""

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

            # In distributed training, only rank 0 downloads to avoid race conditions
            # where other ranks read metadata before the download has finished.
            acc = get_proc_accelerator()
            if acc is not None and acc.num_processes > 1:
                if acc.is_main_process:
                    (self.root / "meta").mkdir(exist_ok=True, parents=True)
                    self.pull_from_repo(allow_patterns="meta/")
                acc.wait_for_everyone()
            else:
                (self.root / "meta").mkdir(exist_ok=True, parents=True)
                self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self) -> None:
        """Load dataset metadata from disk.

        Loads info, tasks, episodes, statistics, and advantages from the
        dataset root directory. Handles version compatibility checks.
        """
        assert self.repo_id is not None
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            assert self.stats is not None
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
        assert self.repo_id is not None
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
        assert self.video_path is not None
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
    def control_mode(self) -> str:
        """Control-mode label from info.json. Defaults to 'unknown' when absent."""
        return self.info.get("control_mode", "unknown")

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

    def update_video_info(self, skip_keys: set[str] | None = None) -> None:
        """Update video metadata from the first episode's video files.

        Warning: this function writes info from first episode videos, implicitly
        assuming that all videos have been encoded the same way. Also, this means
        it assumes the first episode exists.

        Args:
            skip_keys: Optional set of video keys to skip (e.g. deferred video
                keys whose files don't exist yet).
        """
        for key in self.video_keys:
            if skip_keys and key in skip_keys:
                continue
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                if not video_path.is_file():
                    continue
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
    and VQADataset, including data format standardization, image processing,
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
    """

    def __init__(self, cfg: TrainPipelineConfig):
        super().__init__()
        # Standard Data Format parameters
        self.resolution = cfg.resolution  # resolution of images (H, W) in data sample
        self.num_cams = cfg.num_cams  # number of cameras in each data sample
        self.max_state_dim = cfg.max_state_dim  # maximum dimension of the state vector
        self.max_action_dim = cfg.max_action_dim  # maximum dimension of the action vector
        self.action_chunk = cfg.action_chunk  # number of actions to be processed in a chunk
        dm = cfg.dataset_mixture
        self.n_obs_history = dm.n_obs_history if dm else None
        # Optional-key dropout probabilities (all default to 0 when no mixture config is
        # provided, preserving legacy/VQA paths that don't use these keys).
        self.history_state_drop_prob = dm.history_state_drop_prob if dm else 0.0
        self.subgoal_drop_prob = dm.subgoal_drop_prob if dm else 0.0
        self.subgoal_end_of_segment_prob = dm.subgoal_end_of_segment_prob if dm else 0.0
        self.response_drop_prob = dm.response_drop_prob if dm else 0.0
        self.metadata_drop_all_prob = dm.metadata_drop_all_prob if dm else 0.0
        self.metadata_drop_each_prob = dm.metadata_drop_each_prob if dm else 0.0
        # Whether the above drop rolls actually fire. `make_dataset` flips this
        # off on the validation subset (unless `val_enable_optional_key_dropout`
        # is set). Subgoal *frame* selection (end-of-segment vs. uniform window)
        # stays random either way — this flag only gates masking/zero-fill.
        self.enable_optional_key_dropout = True

    def shallow_copy_with_dropout(self, *, enable_dropout: bool) -> "BaseDataset":
        """Return a shallow copy that only diverges in ``enable_optional_key_dropout``.

        Used by :func:`opentau.datasets.factory.make_dataset` to give the
        validation subset its own dataset instance so dropout can be toggled
        independently of the training subset. The copy shares ``meta``,
        ``hf_dataset``, ``episode_data_index``, cached segment tables, and
        every other instance attribute by reference — only
        ``enable_optional_key_dropout`` diverges. If a future refactor adds an
        instance attribute that needs to diverge between train and val, it
        must be set here too.
        """
        clone = copy.copy(self)
        clone.enable_optional_key_dropout = enable_dropout
        return clone

    @abstractmethod
    def _get_feature_mapping_key(self) -> str:
        r"""Returns the key used for feature mapping"""
        pass

    def _assert_image_in_unit_range(
        self, img: torch.Tensor, *, name: str, expect_temporal: bool | None = None
    ) -> None:
        """Assert an image tensor has the expected rank, 3-channel, and [0, 1] range.

        By default, the expected shape follows ``self.n_obs_history``: rank-3
        ``(3, H, W)`` when None, rank-4 ``(T, 3, H, W)`` otherwise. Pass
        ``expect_temporal`` explicitly to override — e.g. subgoals are always
        single-frame targets regardless of observation history.

        Args:
            img: Image tensor to validate.
            name: Human-readable key name for the error message.
            expect_temporal: If ``None``, defers to ``self.n_obs_history``. If
                ``True`` force-expects ``(T, 3, H, W)``. If ``False`` force-expects
                ``(3, H, W)``.
        """
        if expect_temporal is None:
            expect_temporal = self.n_obs_history is not None
        if expect_temporal:
            expected_ndim = 4
            expected_c_dim = 1
            shape_desc = "(T, 3, H, W)"
        else:
            expected_ndim = 3
            expected_c_dim = 0
            shape_desc = "(3, H, W)"
        assert (
            len(img.shape) == expected_ndim
            and img.shape[expected_c_dim] == 3
            and img.min() >= 0.0 - 1e-6
            and img.max() <= 1.0 + 1e-6
        ), (
            f"Expected image {name} to have shape {shape_desc} with values in [0, 1], "
            f"Got shape {img.shape}, "
            f"min={img.min()}, max={img.max()}, "
            f"self={self._get_feature_mapping_key()}."
        )

    def _standardize_images(self, item, standard_item, n_cams) -> list[bool]:
        """Standardize image features to a common format.

        Resizes images to the target resolution with padding, and tracks
        which camera slots are padded (absent cameras).

        When ``self.n_obs_history`` is set, camera tensors have shape
        ``(T, C, H, W)`` and each frame is resized individually.

        Args:
            item: Input item dictionary with original image keys.
            standard_item: Output dictionary to populate with standardized images.
            n_cams: Number of cameras to process.

        Returns:
            List of boolean values indicating which camera slots are padded.
        """
        name_map = DATA_FEATURES_NAME_MAPPING[self._get_feature_mapping_key()]
        image_is_pad = []
        for cam_idx in range(n_cams):
            std_key = f"camera{cam_idx}"
            key = name_map.get(std_key)

            if key is None:
                if self.n_obs_history is not None:
                    standard_item[std_key] = torch.zeros((self.n_obs_history, 3, *self.resolution))
                else:
                    standard_item[std_key] = torch.zeros((3, *self.resolution))
                image_is_pad.append(True)
            elif self.n_obs_history is not None:
                standard_item[std_key] = self.resize_with_pad(
                    item[key], self.resolution[1], self.resolution[0], pad_value=0
                )
                # Per-frame temporal padding info (from clamped episode boundaries) is
                # tracked by obs_history_is_pad on the state side. Camera _is_pad only
                # indicates whether the entire camera slot is absent, so we always mark
                # a present camera as not padded regardless of frame-level clamping.
                image_is_pad.append(False)
            else:
                standard_item[std_key] = self.resize_with_pad(
                    item[key],
                    self.resolution[1],
                    self.resolution[0],
                    pad_value=0,
                )
                image_is_pad.append(item.get(key + "_is_pad", torch.tensor(False)).item())

            self._assert_image_in_unit_range(standard_item[std_key], name=std_key)

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

        standard_item = {}
        img_is_pad = self._standardize_images(item, standard_item, self.num_cams)

        for new_key, key in name_map.items():
            if "camera" in new_key:
                continue
            standard_item[new_key] = item[key]

        # pad state and action vectors
        standard_item["state"] = self.pad_vector(standard_item["state"], self.max_state_dim)
        standard_item["actions"] = self.pad_vector(standard_item["actions"], self.max_action_dim)

        standard_item["img_is_pad"] = torch.tensor(img_is_pad, dtype=torch.bool)
        standard_item["action_is_pad"] = item[name_map["actions"] + "_is_pad"]

        state_raw_key = name_map.get("state")
        state_pad_key = f"{state_raw_key}_is_pad" if state_raw_key else None
        if state_pad_key and state_pad_key in item:
            standard_item["obs_history_is_pad"] = item[state_pad_key]
        elif self.n_obs_history is not None:
            standard_item["obs_history_is_pad"] = torch.zeros(self.n_obs_history, dtype=torch.bool)
        else:
            standard_item["obs_history_is_pad"] = torch.tensor([False], dtype=torch.bool)

        # Emit optional keys (memory, next_memory, speed, mistake, quality,
        # robot_type, control_mode, subgoalK) with their _is_pad siblings,
        # applying training-time dropout rolls.
        self._emit_optional_keys(item, standard_item)

        # cast all tensors in standard_item to bfloat16
        for key, value in standard_item.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                standard_item[key] = value.to(dtype=torch.bfloat16)

        return standard_item

    def _emit_optional_keys(self, item: dict, standard_item: dict) -> None:
        """Emit optional memory/subgoal/metadata keys with training-time dropout.

        All emitted keys are always present (zero/empty-filled when absent or
        masked). Numeric / image keys come with a parallel ``{key}_is_pad``
        bool; string keys (``response``, ``memory``, ``next_memory``,
        ``robot_type``, ``control_mode``) use the empty string itself as the
        pad signal — a consumer seeing ``""`` can assume the field was
        unavailable or was masked this step.

        Dropout order:
            1. ``history_state_drop_prob``: zero ``state`` and historical camera
               frames; mark ``obs_history_is_pad`` all True.
            2. ``subgoal_drop_prob``: zero every ``subgoalK`` and set the
               single ``subgoal_is_pad`` flag to True (subgoals are all-or-none).
            3. ``response_drop_prob``: only when subgoals were NOT dropped, mask
               ``response`` to the empty string.
            4. ``metadata_drop_all_prob``: mask {speed, mistake, quality,
               robot_type, control_mode} together. If this didn't fire,
               ``metadata_drop_each_prob`` rolls independently for each field.

        Dropout rolls use the default torch RNG (auto-seeded per worker).

        Args:
            item: Raw item dict (may include ``memory_raw``, ``next_memory_raw``,
                ``speed_raw``, ``mistake_raw``, ``quality_raw``, and
                ``subgoalK_raw`` tensors for K in ``[0, num_cams)``). Missing
                numeric/image entries produce zero + ``_is_pad=True``. Missing
                string entries produce the empty string.
            standard_item: Output dict populated by :meth:`_to_standard_data_format`.
                Mutated in place.
        """

        # When dropout is disabled (e.g. the validation subset), every drop
        # roll is suppressed but subgoal *frame* selection upstream stays
        # random. We do this by treating each drop roll as a Bernoulli that
        # always returns False.
        def _roll(prob: float) -> bool:
            if not self.enable_optional_key_dropout:
                return False
            return bool(torch.rand(()) < prob)

        # (1) History + observation.state drop.
        drop_hist = _roll(self.history_state_drop_prob)
        if drop_hist:
            standard_item["state"] = torch.zeros_like(standard_item["state"])
            if self.n_obs_history is not None:
                standard_item["obs_history_is_pad"] = torch.ones(self.n_obs_history, dtype=torch.bool)
                if self.n_obs_history > 1:
                    for k in range(self.num_cams):
                        cam_key = f"camera{k}"
                        if cam_key in standard_item:
                            standard_item[cam_key][:-1] = 0
            else:
                standard_item["obs_history_is_pad"] = torch.tensor([True], dtype=torch.bool)

        # (2) Subgoal drop. A single ``subgoal_is_pad`` flag covers every slot
        # because subgoals are either all present (annotated and not dropped)
        # or all missing (legacy / `_load_subgoal_frames` decided to drop).
        # The drop roll itself lives in ``_load_subgoal_frames`` so a dropped
        # subgoal skips video decoding entirely — here we just detect absence.
        pad_subgoals = not all(item.get(f"subgoal{k}_raw") is not None for k in range(self.num_cams))
        for k in range(self.num_cams):
            out_key = f"subgoal{k}"
            if pad_subgoals:
                standard_item[out_key] = torch.zeros((3, *self.resolution))
            else:
                standard_item[out_key] = self.resize_with_pad(
                    item[f"subgoal{k}_raw"],
                    self.resolution[1],
                    self.resolution[0],
                    pad_value=0,
                )
            # Subgoals are always single-frame regardless of n_obs_history.
            self._assert_image_in_unit_range(standard_item[out_key], name=out_key, expect_temporal=False)
        standard_item["subgoal_is_pad"] = torch.tensor(pad_subgoals)

        # (3) Response drop — only rolled when subgoals are actually present
        # (dropping both would remove the primary task signal at once). No
        # separate ``response_is_pad`` flag: an empty string IS the pad signal.
        if not pad_subgoals and _roll(self.response_drop_prob):
            standard_item["response"] = ""

        # (4) Memory pass-through (no drop probability). No separate pad flag —
        # consumers treat an empty string as "no memory available".
        memory_raw = item.get("memory_raw")
        standard_item["memory"] = memory_raw if isinstance(memory_raw, str) else ""
        next_memory_raw = item.get("next_memory_raw")
        standard_item["next_memory"] = next_memory_raw if isinstance(next_memory_raw, str) else ""

        # (5) Metadata drops: numeric fields get a _is_pad flag; string
        # identifier fields use "" as the pad signal (no separate flag).
        drop_meta_all = _roll(self.metadata_drop_all_prob)
        for key, dtype in (("speed", torch.long), ("mistake", torch.bool), ("quality", torch.long)):
            raw_key = f"{key}_raw"
            raw = item.get(raw_key)
            drop_this = drop_meta_all or _roll(self.metadata_drop_each_prob)
            if raw is None or drop_this:
                zero = False if dtype is torch.bool else 0
                standard_item[key] = torch.tensor(zero, dtype=dtype)
                standard_item[f"{key}_is_pad"] = torch.tensor(True)
            else:
                value = bool(raw) if dtype is torch.bool else int(raw)
                standard_item[key] = torch.tensor(value, dtype=dtype)
                standard_item[f"{key}_is_pad"] = torch.tensor(False)
        for key in ("robot_type", "control_mode"):
            val = self.meta.info.get(key) or ""
            drop_this = drop_meta_all or _roll(self.metadata_drop_each_prob)
            standard_item[key] = "" if drop_this else val

    def resize_with_pad(self, img, width, height, pad_value=0) -> torch.Tensor:
        """Resize an image to target dimensions with padding.

        Maintains aspect ratio by resizing to fit within target dimensions,
        then pads on the left and top to reach exact target size.

        Args:
            img: Input image tensor of shape (C, H, W) or (T, C, H, W).
            width: Target width.
            height: Target height.
            pad_value: Value to use for padding. Defaults to 0.

        Returns:
            Resized and padded image tensor of shape (C, height, width) or
            (T, C, height, width), matching the input rank.

        Raises:
            ValueError: If input is not 3D or 4D.
        """
        batched = img.ndim == 4
        if not batched:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            raise ValueError(f"Expected (C,H,W) or (T,C,H,W), got shape {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)

        if not batched:
            padded_img = padded_img.squeeze(0)
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
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_resample_strategy: str = "nearest",
        vector_resample_strategy: str = "nearest",
        standardize: bool = True,
        return_advantage_input: bool = False,
        skip_timestamp_check: bool = False,
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
            delta_timestamps (dict[list[float]] | None, optional): Dictionary mapping feature
                names to lists of delta timestamps in seconds. For example,
                ``{'state': [0.0], 'action': [0, 0.5, 1.0]}`` means state is sampled at the current
                time and action is sampled at three offsets. A ``{feature}_is_pad`` boolean mask of the
                same length is added to the returned item. Defaults to None.
            delta_timestamps_std: (dict[list[float]] | None, optional): Per-feature standard
                deviation for the delta timestamps. Absent keys are treated as deterministic (std=0).
                Defaults to None.
            delta_timestamps_lower: (dict[list[float]] | None, optional): Per-feature lower bound
                for the delta timestamps. Defaults to None.
            delta_timestamps_upper: (dict[list[float]] | None, optional): Per-feature upper bound
                for the delta timestamps. Defaults to None.
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
            skip_timestamp_check (bool, Optional): If True, bypass the load-time
                ``check_timestamps_sync`` call (a warning is logged). Frame-to-frame
                spacing is NOT verified against ``1/fps``; downstream
                ``delta_timestamps`` lookups may sample unintended frames. Defaults
                to False. Does not affect the record-time check inside
                ``add_episode``.
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_OPENTAU_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps_params = self.compute_delta_params(
            delta_timestamps,
            delta_timestamps_std,
            delta_timestamps_lower,
            delta_timestamps_upper,
        )
        # Sort episodes here so that hf_dataset row layout (sorted by
        # episode_index after the glob in load_hf_dataset) stays aligned with
        # episode_data_index["from"/"to"] (built in self.episodes order in
        # get_episode_data_index). Mismatched order would silently return
        # rows from the wrong episode for callers that pass an unsorted list.
        self.episodes = sorted(episodes) if episodes is not None else None
        self.tolerance_s = tolerance_s
        self.skip_timestamp_check = skip_timestamp_check
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
        self.skip_video_stats = False

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        # Overlay setup: when info.json declares a `videos.source_repo`, all video
        # reads are routed to that upstream repo. Fetch its info.json once (for
        # the path template + chunks_size) and cache under the canonical
        # HF_OPENTAU_HOME/<source_repo> location, so a future vanilla load of the
        # source repo reuses the same files. Must run before get_episodes_file_paths().
        self._overlay: dict | None = None
        videos_meta = self.meta.info.get("videos") or {}
        if videos_meta.get("source_repo"):
            src_repo = videos_meta["source_repo"]
            src_revision = videos_meta.get("source_revision")
            src_root = HF_OPENTAU_HOME / src_repo
            src_info_path = src_root / INFO_PATH
            if not src_info_path.is_file():
                acc = get_proc_accelerator()
                if acc is not None and acc.num_processes > 1:
                    if acc.is_main_process:
                        src_info_path.parent.mkdir(exist_ok=True, parents=True)
                        hf_hub_download(
                            repo_id=src_repo,
                            filename=INFO_PATH,
                            repo_type="dataset",
                            revision=src_revision,
                            local_dir=src_root,
                        )
                    acc.wait_for_everyone()
                else:
                    src_info_path.parent.mkdir(exist_ok=True, parents=True)
                    hf_hub_download(
                        repo_id=src_repo,
                        filename=INFO_PATH,
                        repo_type="dataset",
                        revision=src_revision,
                        local_dir=src_root,
                    )
            src_info = json.loads(src_info_path.read_text())
            self._overlay = {
                "source_repo": src_repo,
                "source_revision": src_revision,
                "source_root": src_root,
                "video_path": src_info["video_path"],
                "chunks_size": src_info["chunks_size"],
            }

        self.control_mode = self.meta.control_mode
        if "control_mode" not in self.meta.info and self.repo_id not in _CONTROL_MODE_WARNED:
            _CONTROL_MODE_WARNED.add(self.repo_id)
            logging.warning(
                "Dataset %r has no `control_mode` field in meta/info.json; defaulting to 'unknown'. "
                "Please add `control_mode` ∈ {'joint', 'ee', 'mixed'} to keep the mixture sampler honest.",
                self.repo_id,
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
        if self.skip_timestamp_check:
            # ``skip_timestamp_check`` is a mixture-wide decision and the
            # message is identical for every dataset, so emit it once per
            # process and only on the main rank. Naive per-dataset / per-rank
            # logging floods the run log with ``num_processes`` ×
            # ``num_datasets`` copies of the same line (392 × 8 ≈ 3K for a
            # wide pretraining mixture). Falls through to logging when no
            # Accelerator is set (single-process dev / tests).
            global _SKIP_TIMESTAMP_WARNED
            acc = get_proc_accelerator()
            if not _SKIP_TIMESTAMP_WARNED and (acc is None or acc.is_main_process):
                _SKIP_TIMESTAMP_WARNED = True
                logging.warning(
                    "skip_timestamp_check=True is in effect for one or more "
                    "datasets in this mixture (e.g. %s). Frame-to-frame "
                    "spacing is NOT verified against 1/fps; downstream "
                    "delta_timestamps lookups may sample unintended frames.",
                    self.repo_id,
                )
        else:
            no_transform_ds = self.hf_dataset.with_transform(None).with_format("numpy")
            logging.info("Checking timestamps synchronization...")
            timestamps = np.asarray(no_transform_ds["timestamp"], dtype=np.float32)
            episode_indices = np.asarray(no_transform_ds["episode_index"], dtype=np.int64)
            ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
            check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Per-episode caches used by the optional-key emission path. Populated
        # from meta/episodes.jsonl annotations when present, else filled with
        # safe defaults that cause every annotated key to emit as _is_pad=True.
        self.episode_lengths: dict[int, int] = {
            ep: int(info["length"]) for ep, info in self.meta.episodes.items()
        }
        self.segment_starts_by_episode: dict[int, np.ndarray] = {}
        for ep, info in self.meta.episodes.items():
            starts = info.get("segments")
            if starts is None or len(starts) == 0:
                starts = [0]
            self.segment_starts_by_episode[ep] = np.asarray(starts, dtype=np.int64)

        # One memory string per segment. Read once from the parquet's "memory"
        # column at segment-start indices so __getitem__ can resolve
        # ``next_memory`` without a secondary parquet query per sample. When the
        # dataset has no memory column (legacy), every segment gets "". Only
        # the rows we need (segment starts across selected episodes) are
        # materialized, which matters for multi-million-frame datasets.
        has_memory_column = "memory" in self.meta.features
        self.segment_memories_by_episode: dict[int, list[str]] = {}
        if has_memory_column:
            indices: list[int] = []
            offsets: list[int] = []  # cumulative offsets into `indices`, one per episode
            for ep in self.episodes:
                offsets.append(len(indices))
                starts = self.segment_starts_by_episode[ep]
                ep_start = int(self.episode_data_index["from"][self.epi2idx[ep]].item())
                indices.extend(int(ep_start + s) for s in starts)
            if indices:
                memory_rows = self.hf_dataset.with_format("arrow").select(indices)["memory"].to_pylist()
            else:
                memory_rows = []
            for ep, off in zip(self.episodes, offsets, strict=True):
                n = len(self.segment_starts_by_episode[ep])
                self.segment_memories_by_episode[ep] = [str(m) for m in memory_rows[off : off + n]]
        else:
            for ep in self.episodes:
                starts = self.segment_starts_by_episode[ep]
                self.segment_memories_by_episode[ep] = [""] * len(starts)

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

    def get_episodes_file_paths(self) -> list[str]:
        """Get file paths for all selected episodes.

        Returns paths for both parquet data files and video files (if applicable)
        for all episodes in the dataset.

        Returns:
            List of file paths for episode data and videos.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # Overlay repos don't carry videos themselves — those are pulled lazily
        # from the source repo by _resolve_video_path on first access.
        if len(self.meta.video_keys) > 0 and self._overlay is None:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        # Derive the parquet glob from the meta data_path template so that
        # datasets with a non-default `info["data_path"]` (deeper nesting,
        # flat layout, etc.) keep working. Default template is
        # "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        # which yields the glob "data/chunk-*/episode_*.parquet". Assumes the
        # template uses simple `{name}` / `{name:fmt}` placeholders and no
        # literal `{{`/`}}` escapes — true for every in-repo writer.
        glob_pattern = re.sub(r"\{[^}]+\}", "*", self.meta.data_path)
        paths = sorted(self.root.glob(glob_pattern))
        if not paths:
            raise FileNotFoundError(f"No parquet files matching {glob_pattern!r} under {self.root}")
        features = get_hf_features_from_features(self.meta.features)
        # Read parquet directly via pyarrow.dataset and wrap the resulting
        # pa.Table in a HF Dataset. Going through `load_dataset("parquet", ...)`
        # or `Dataset.from_parquet(...)` both route through ParquetDatasetBuilder,
        # which rewrites the parquet bytes into an uncompressed Arrow cache at
        # $HF_HOME/datasets/parquet/ — 1-5x the source size (compression-dependent)
        # and one cache entry per distinct (paths, filter) combo. Issue #277 has
        # the empirical numbers; verified on physical-intelligence/libero.
        #
        # Trade-off: `to_table(filter=...)` materializes the filtered rows into
        # RAM rather than mmapping a disk-backed Arrow cache. RAM cost scales
        # with `len(filtered rows) × avg-row-size`; concretely:
        # ~350 MB for physical-intelligence/libero with episodes=[0..9],
        # ~46 GB for humanoid-everyday-A-overlay with episodes=None (full corpus).
        # Narrow `episodes=` picks are fine; an episodes=None load on a multi-GB
        # image-heavy repo will OOM on a small dev box — pass a manageable
        # subset, or restore a mmap'd Arrow cache via tmp pa.ipc files if RAM
        # ever becomes the binding constraint.
        #
        # The `Dataset(table, info=DatasetInfo(features=features))` constructor
        # signature has been stable since datasets 2.x; the project pin is
        # `datasets>=2.19.0`, so we're well inside the supported window.
        pa_dataset = pa_ds.dataset(list(map(str, paths)), format="parquet")
        filter_expr = pa_ds.field("episode_index").isin(self.episodes) if self.episodes is not None else None
        table = pa_dataset.to_table(filter=filter_expr)
        hf_dataset = Dataset(table, info=DatasetInfo(features=features))
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        """Create an empty HuggingFace dataset with the correct features.

        Returns:
            Empty dataset with features matching the dataset specification.
        """
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split=datasets.Split.TRAIN)

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
    ) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
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
        query_indices = {
            key: np.clip(idx + delta_idx, ep_start, ep_end - 1) for key, delta_idx in delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.tensor(
                (idx + delta_idx < ep_start) | (idx + delta_idx >= ep_end), dtype=torch.bool
            )
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

    def _resolve_video_path(self, ep_idx: int, vid_key: str) -> Path:
        """Resolve the on-disk video file path for a given episode + video key.

        For vanilla datasets, returns the local-layout path under ``self.root``.
        For overlay datasets, formats the source repo's ``video_path`` template
        (using ``original_episode_index`` from ``meta/episodes.jsonl`` when
        present, else the dense ``ep_idx``) and lazily downloads the file from
        the source repo into ``HF_OPENTAU_HOME/<source_repo>/videos/...`` so
        the cache is shared with future vanilla loads of the source repo.
        """
        if self._overlay is None:
            return self.root / self.meta.get_video_file_path(ep_idx, vid_key)

        src_ep = self.meta.episodes[ep_idx].get("original_episode_index", ep_idx)
        ovl = self._overlay
        chunk = src_ep // ovl["chunks_size"]
        filename = ovl["video_path"].format(episode_chunk=chunk, video_key=vid_key, episode_index=src_ep)
        local_path = ovl["source_root"] / filename
        if not local_path.is_file():
            hf_hub_download(
                repo_id=ovl["source_repo"],
                filename=filename,
                repo_type="dataset",
                revision=ovl["source_revision"],
                local_dir=ovl["source_root"],
            )
        return local_path

    def _query_videos(self, query_timestamps: dict[str, np.ndarray], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self._resolve_video_path(ep_idx, vid_key)
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

    def _lookup_segment_index(self, ep_idx: int, frame_in_ep: int) -> int:
        """Return the zero-based segment index containing ``frame_in_ep``.

        Uses the cached ``segment_starts_by_episode`` table. Legacy datasets
        with a single placeholder segment always return 0.
        """
        starts = self.segment_starts_by_episode[ep_idx]
        idx = int(np.searchsorted(starts, frame_in_ep, side="right")) - 1
        return max(idx, 0)

    def _segment_end_in_ep(self, ep_idx: int, seg_idx: int) -> int:
        """Exclusive end-frame index of segment ``seg_idx`` within its episode."""
        starts = self.segment_starts_by_episode[ep_idx]
        if seg_idx + 1 < len(starts):
            return int(starts[seg_idx + 1])
        return self.episode_lengths[ep_idx]

    def _lookup_next_memory(self, ep_idx: int, frame_in_ep: int) -> str:
        """Return the memory string for frame ``frame_in_ep + 1``, clipped at episode end."""
        ep_length = self.episode_lengths[ep_idx]
        next_frame = min(frame_in_ep + 1, ep_length - 1)
        seg = self._lookup_segment_index(ep_idx, next_frame)
        memories = self.segment_memories_by_episode.get(ep_idx, [""])
        if not memories:
            return ""
        return memories[min(seg, len(memories) - 1)]

    def _sample_subgoal_frame(self, ep_idx: int, frame_in_ep: int, *, at_end_of_segment: bool) -> int:
        """Pick a future frame index (within-episode) to serve as the subgoal source.

        When ``at_end_of_segment`` is True, returns the last frame of the
        current segment (clipped to the episode's last frame). Otherwise samples
        a timestamp uniformly in ``[t, t + 4s]`` (wall-clock) and converts it to
        a frame index, clipping to the current segment end and the episode end.

        Episodes that have no ``segments`` annotation in ``episodes.jsonl``
        skip segment-aware clipping entirely and fall back to a fixed
        ~4-seconds-ahead subgoal frame (clipped to the episode end). This
        keeps subgoal supervision available on legacy datasets that never
        wrote per-episode segment boundaries.
        """
        ep_length = self.episode_lengths[ep_idx]
        window_frames = int(round(4.0 * self.fps))
        if "segments" not in self.meta.episodes[ep_idx]:
            return min(frame_in_ep + window_frames, ep_length - 1)
        seg_idx = self._lookup_segment_index(ep_idx, frame_in_ep)
        seg_end_excl = self._segment_end_in_ep(ep_idx, seg_idx)
        upper = min(seg_end_excl, ep_length) - 1  # inclusive upper bound.
        upper = max(upper, frame_in_ep)
        if at_end_of_segment:
            return upper
        top = min(frame_in_ep + window_frames, upper)
        if top <= frame_in_ep:
            return frame_in_ep
        offset = int(torch.randint(low=0, high=top - frame_in_ep + 1, size=()).item())
        return frame_in_ep + offset

    def _load_subgoal_frames(self, ep_idx: int, frame_in_ep: int) -> dict[str, torch.Tensor]:
        """Decode subgoal frames — one per camera slot — for this sample.

        Subgoal supervision is always-on for any dataset that exposes camera
        keys; the dedicated ``subgoals`` info.json declaration that the older
        pi07_paligemma path required is no longer consulted. Datasets without
        any cameras (``self.num_cams == 0`` or empty
        ``self.meta.camera_keys``) still return ``{}``, which lets
        :meth:`BaseDataset._emit_optional_keys` emit ``subgoal_is_pad=True``
        for every slot.

        Behavior for camera-bearing datasets:
        - The at-end-of-segment vs uniform sampling roll happens ONCE per
          ``__getitem__`` call (shared across all camera slots); each slot
          fetches the frame from its own source — video file for ``video``
          dtype features, parquet row for ``image`` dtype features (the
          latter share the same within-episode frame index).
        - Drop-roll is short-circuited here so a dropped subgoal skips the
          per-camera decode/lookup. When
          ``self.enable_optional_key_dropout`` is False (e.g. the validation
          subset), drop is never rolled — the frame-selection randomness
          stays live because it's about which future frame to read, not
          masking.
        - Episodes with no ``segments`` entry in ``episodes.jsonl`` fall
          back to a fixed ~4 s lookahead inside ``_sample_subgoal_frame``
          rather than skipping subgoal loading, so legacy datasets without
          segment annotations still get supervision.
        """
        if self.num_cams <= 0 or len(self.meta.camera_keys) == 0:
            return {}
        # Roll drop before any video decoding — at `subgoal_drop_prob=0.75` the
        # old ordering threw away 75% of decodes.
        if self.enable_optional_key_dropout and bool(torch.rand(()) < self.subgoal_drop_prob):
            return {}
        name_map = DATA_FEATURES_NAME_MAPPING[self._get_feature_mapping_key()]
        at_end = bool(torch.rand(()) < self.subgoal_end_of_segment_prob)
        subgoal_frame = self._sample_subgoal_frame(ep_idx, frame_in_ep, at_end_of_segment=at_end)
        ts = subgoal_frame / self.fps
        ep_start = int(self.episode_data_index["from"][self.epi2idx[ep_idx]].item())
        out: dict[str, torch.Tensor] = {}
        # ``hf_dataset[idx]`` runs ``hf_transform_to_torch`` over every column of the row, so
        # it must be done at most once per ``__getitem__`` even when multiple image-dtype
        # cameras live in the same row.
        for k in range(self.num_cams):
            cam_key = name_map.get(f"camera{k}")
            if cam_key is None:
                continue
            if cam_key in self.meta.video_keys:
                frames = self._query_videos({cam_key: np.array([ts])}, ep_idx)
                out[f"subgoal{k}_raw"] = frames[cam_key]
            elif cam_key in self.meta.image_keys:
                # Image-dtype cameras are stored per-frame in the parquet
                # rows of ``hf_dataset``. The within-episode index returned
                # by ``_sample_subgoal_frame`` maps directly to the absolute
                # row ``ep_start + subgoal_frame``.
                out[f"subgoal{k}_raw"] = self.hf_dataset[ep_start + subgoal_frame][cam_key]
        return out

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

        if self.episode_data_index is None or self.epi2idx is None:
            raise RuntimeError(
                "episode_data_index and epi2idx must be set before calling __getitem__. "
                "This usually means the dataset was not properly initialized."
            )
        ep_end = int(self.episode_data_index["to"][self.epi2idx[ep_idx]].item())

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

        # Squeeze the temporal dimension for features with a single delta timestamp
        for feature, mean in self.delta_timestamps_params[0].items():
            if len(mean) == 1 and feature in item:
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

            # Attach raw optional fields (stripped to _raw suffix) so
            # BaseDataset._emit_optional_keys can apply dropout uniformly.
            ep_start = int(self.episode_data_index["from"][self.epi2idx[ep_idx]].item())
            frame_in_ep = idx - ep_start
            if "memory" in item:
                item["memory_raw"] = str(item["memory"])
            if "mistake" in item:
                item["mistake_raw"] = int(item["mistake"])
            item["next_memory_raw"] = self._lookup_next_memory(ep_idx, frame_in_ep)
            quality = self.meta.episodes[ep_idx].get("quality")
            if quality is not None:
                item["quality_raw"] = int(quality)
            item["speed_raw"] = speed_duration_bucket_s(self.episode_lengths[ep_idx], self.fps)
            item.update(self._load_subgoal_frames(ep_idx, frame_in_ep))

            item = self._to_standard_data_format(item)

            if self.meta.advantages is not None:
                advantage = self.meta.advantages.get((episode_index, timestamp.item()), 0)
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
        deferred = getattr(self, "deferred_video_keys", set())
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            if key in deferred:
                continue
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

        Video features listed in ``deferred_video_keys`` (set via
        :meth:`create`) may be omitted from the frame; their observations will
        be attached later via :meth:`attach_video`.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        deferred = getattr(self, "deferred_video_keys", set())
        validate_frame(frame, self.features, deferred_features=deferred or None)

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

        Note:
            When ``deferred_video_keys`` are configured, the corresponding video
            features are excluded from validation, encoding, and existence
            checks. Use :meth:`attach_video` after saving episodes to supply
            the video files.
        """
        episode_buffer = self.episode_buffer if episode_data is None else episode_data
        if episode_buffer is None:
            raise RuntimeError("No episode data provided and no episode buffer exists. Call add_frame first.")

        deferred = getattr(self, "deferred_video_keys", set())
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features, deferred or None)

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
            if key in deferred:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        # Deferred keys are excluded from effective_features below, so only
        # honor the explicit dataset-level setting for skipping video stats.
        skip_video = getattr(self, "skip_video_stats", False)
        # Build a features dict that excludes deferred keys so compute_episode_stats
        # does not try to read image paths that don't exist, while still
        # computing stats for non-deferred image/video features.
        effective_features = {k: v for k, v in self.features.items() if k not in deferred}
        ep_stats = compute_episode_stats(episode_buffer, effective_features, skip_video_stats=skip_video)

        # Add placeholder stats for deferred video keys so downstream code
        # (standardization, dataset mixing) never encounters missing keys.
        for key in deferred:
            shape = self.features[key]["shape"]
            names = self.features[key].get("names") or []
            # Locate channel axis; LeRobot v2.1's default image convention is
            # (H, W, C) with names ["height", "width", "channel"], but some
            # callers declare (C, H, W). Look it up by name, falling back to
            # the CHW convention that aggregate_stats's (3, 1, 1) assertion
            # ultimately expects.
            if "channel" in names:
                c = shape[names.index("channel")]
            elif "channels" in names:
                c = shape[names.index("channels")]
            elif len(shape) >= 3:
                c = shape[0]
            else:
                c = 3
            ep_stats[key] = {
                "min": np.zeros((c, 1, 1), dtype=np.float64),
                "max": np.ones((c, 1, 1), dtype=np.float64),
                "mean": np.full((c, 1, 1), 0.5, dtype=np.float64),
                "std": np.full((c, 1, 1), 0.5, dtype=np.float64),
                "count": np.array([episode_length]),
            }

        # Encode videos for non-deferred video keys only
        non_deferred_video_keys = [k for k in self.meta.video_keys if k not in deferred]
        if non_deferred_video_keys:
            video_paths = self.encode_episode_videos(episode_index, skip_keys=deferred)
            for key in non_deferred_video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index, _ = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        timestamps = np.asarray(episode_buffer["timestamp"]).reshape(-1)
        episode_indices = np.full(episode_length, episode_index)
        check_timestamps_sync(
            timestamps,
            episode_indices,
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        expected_episodes = self.meta.total_episodes
        missing_videos: list[str] = []
        for ep_idx in range(expected_episodes):
            for vid_key in self.meta.video_keys:
                if vid_key in deferred:
                    continue
                video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                if not video_path.is_file():
                    missing_videos.append(str(video_path))
        assert not missing_videos, "Missing expected encoded videos:\n" + "\n".join(missing_videos)

        missing_parquet: list[str] = []
        for ep_idx in range(expected_episodes):
            parquet_path = self.root / self.meta.get_data_file_path(ep_idx)
            if not parquet_path.is_file():
                missing_parquet.append(str(parquet_path))
        assert not missing_parquet, "Missing expected parquet episode files:\n" + "\n".join(missing_parquet)

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

        self.episode_data_index, self.epi2idx = get_episode_data_index(self.meta.episodes, self.episodes)

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(
            episode_dict, features=self.hf_features, split=datasets.Split.TRAIN
        )
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        if self.episode_buffer is None:
            return
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
        skip_keys = getattr(self, "deferred_video_keys", None)
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx, skip_keys=skip_keys)

    def encode_episode_videos(self, episode_index: int, skip_keys: set[str] | None = None) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.

        Args:
            episode_index: Index of the episode to encode.
            skip_keys: Optional set of video keys to skip (e.g. deferred video keys).
        """
        video_paths = {}
        for key in self.meta.video_keys:
            if skip_keys and key in skip_keys:
                continue
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

    def attach_video(
        self,
        episode_index: int,
        video_key: str,
        input_video_path: str | Path,
        overwrite: bool = False,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        g: int | None = 2,
        crf: int | None = 30,
        start_time: float | None = None,
    ) -> Path:
        """Attach a pre-recorded MP4 video to an episode with deferred video observations.

        The source video is resampled to the dataset's FPS and trimmed so it
        contains exactly as many frames as the episode.  The resulting file is
        placed in the standard video path for the dataset.

        This method is meant to be called **after** :meth:`save_episode` for
        episodes whose video observations were deferred (see the
        ``deferred_video_keys`` parameter of :meth:`create`).

        After attaching videos for all deferred keys and episodes, you should
        call :meth:`update_video_info` to update the dataset metadata with the
        actual video properties (resolution, codec, etc.).

        Args:
            episode_index: Index of the episode to attach the video to.
            video_key: The video feature key (e.g. ``"observation.images.top"``).
            input_video_path: Path to the source MP4 file.
            overwrite: Whether to overwrite an existing video at the target path.
            vcodec: Video codec for re-encoding. Defaults to "libsvtav1".
            pix_fmt: Pixel format. Defaults to "yuv420p".
            g: GOP size. Defaults to 2.
            crf: Constant Rate Factor. Defaults to 30.
            start_time: Optional start offset in seconds into the source video.
                Useful when only a portion of the recording overlaps with the
                episode data. Defaults to None (start from the beginning).

        Returns:
            Path to the written video file inside the dataset.

        Raises:
            ValueError: If ``video_key`` is not a declared video feature, or the
                episode index does not exist.
            FileNotFoundError: If ``input_video_path`` does not exist.
        """
        if video_key not in self.meta.video_keys:
            raise ValueError(
                f"'{video_key}' is not a video feature. Available video keys: {self.meta.video_keys}"
            )
        if episode_index not in self.meta.episodes:
            raise ValueError(
                f"Episode {episode_index} does not exist. Total episodes: {self.meta.total_episodes}"
            )

        episode_length = self.meta.episodes[episode_index]["length"]
        output_path = self.root / self.meta.get_video_file_path(episode_index, video_key)

        if output_path.is_file() and not overwrite:
            logging.info(
                "Video already exists at %s, skipping (use overwrite=True to replace).",
                output_path,
            )
            return output_path

        resample_and_trim_video(
            input_path=input_video_path,
            output_path=output_path,
            target_fps=self.fps,
            num_frames=episode_length,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            overwrite=overwrite,
            start_time=start_time,
        )

        logging.info(
            "Attached video for episode %d, key '%s': %s -> %s (%d frames @ %d fps)",
            episode_index,
            video_key,
            input_video_path,
            output_path,
            episode_length,
            self.fps,
        )
        return output_path

    def update_video_info(self) -> None:
        """Update video metadata from the first episode's video files.

        Call this after attaching all deferred videos to populate the ``info``
        field of each video feature with actual video properties (resolution,
        codec, etc.) and persist the updated metadata to disk.
        """
        self.meta.update_video_info()
        write_info(self.meta.info, self.meta.root)

    @staticmethod
    def compute_delta_params(
        mean: dict[str, np.ndarray | list[float]] | None,
        std: dict[str, np.ndarray | list[float]] | None,
        lower: dict[str, np.ndarray | list[float]] | None,
        upper: dict[str, np.ndarray | list[float]] | None,
    ) -> DeltaTimestampInfo:
        r"""Process the parameters `mean`, `std`, `lower` and `upper` for delta timestamps.

        Delta timestamps are computed dynamically in ``__getitem__`` with
        ``clip(dT, lower, upper)`` where ``dT ~ N(mean, std^2)``.  Each
        parameter is a dictionary mapping **feature names** to sequences of
        floats.

        For example, ``mean = {"state": [0.0], "action": [0.0, 0.5, 1.0]}``
        means the ``state`` feature will be sampled at ``t + 0.0`` and the
        ``action`` feature will be sampled at three offsets ``t``, ``t+0.5``
        and ``t+1.0``.

        Keys present in ``std`` / ``lower`` / ``upper`` but absent from
        ``mean`` are ignored.  Keys absent from ``std`` / ``lower`` /
        ``upper`` but present in ``mean`` receive sensible defaults (0 for
        std, -inf / +inf for bounds).
        """
        inf = float("inf")
        mean_np: DeltaTimestampParam = {k: np.array(v) for k, v in (mean or {}).items()}

        std_raw = std or {}
        std_np: DeltaTimestampParam = {
            k: np.array(std_raw.get(k) or np.zeros_like(v)) for k, v in mean_np.items()
        }

        lower_raw = lower or {}
        lower_np: DeltaTimestampParam = {
            k: np.array(lower_raw.get(k) or (np.zeros_like(v) - inf)) for k, v in mean_np.items()
        }

        upper_raw = upper or {}
        upper_np: DeltaTimestampParam = {
            k: np.array(upper_raw.get(k) or (np.zeros_like(v) + inf)) for k, v in mean_np.items()
        }

        for k in mean_np:
            if not (mean_np[k].shape == std_np[k].shape == lower_np[k].shape == upper_np[k].shape):
                raise ValueError(
                    f"Delta timestamps parameters for {k} have inconsistent shapes: "
                    f"mean={mean_np[k].shape}, std={std_np[k].shape}, "
                    f"lower={lower_np[k].shape}, upper={upper_np[k].shape}"
                )

        return mean_np, std_np, lower_np, upper_np

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
        skip_video_stats: bool = False,
        deferred_video_keys: set[str] | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data.

        Args:
            deferred_video_keys: Optional set of video feature keys whose image
                observations will be provided later via :meth:`attach_video`
                instead of being passed to :meth:`add_frame`. When set, these
                keys are omitted from frame validation, episode video encoding,
                and post-save video existence checks. After all episodes are
                recorded, call :meth:`attach_video` for each episode to supply
                an MP4 that will be resampled to the dataset FPS and trimmed to
                the episode length.
        """
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

        # Deferred video keys: video features whose observations are attached later
        obj.deferred_video_keys = set(deferred_video_keys) if deferred_video_keys else set()
        if obj.deferred_video_keys:
            unknown = obj.deferred_video_keys - set(obj.meta.video_keys)
            if unknown:
                raise ValueError(
                    f"deferred_video_keys {unknown} are not declared as video features. "
                    f"Available video keys: {obj.meta.video_keys}"
                )

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps_params = obj.compute_delta_params(None, None, None, None)
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.image_resample_strategy = image_resample_strategy
        obj.vector_resample_strategy = vector_resample_strategy
        obj.standardize = standardize
        obj.skip_video_stats = skip_video_stats
        obj.episode_data_index, obj.epi2idx = get_episode_data_index(obj.meta.episodes, obj.episodes)
        return obj
