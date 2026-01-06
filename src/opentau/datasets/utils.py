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
"""Utility functions for dataset management, I/O, and validation.

This module provides a comprehensive set of utility functions for working with
LeRobot datasets, including file I/O operations, metadata management, data
validation, version compatibility checking, and HuggingFace Hub integration.

The module is organized into several functional areas:

* Dictionary manipulation: Flattening/unflattening nested dictionaries
* File I/O: JSON and JSONL reading/writing with automatic directory creation
* Metadata management: Loading and saving dataset info, statistics, episodes,
    tasks, and advantages
* Data validation: Frame and episode buffer validation with detailed error
    messages

Key Features:
    * Automatic serialization: Converts tensors and arrays to JSON-compatible
        formats.
    * Comprehensive validation: Validates frames and episodes.
    * Path management: Standard paths for dataset structure (meta/, data/).

Constants:
    DEFAULT_CHUNK_SIZE: Maximum number of episodes per chunk (1000).
    ADVANTAGES_PATH, INFO_PATH, EPISODES_PATH, STATS_PATH: Standard paths.

Classes:
    IterableNamespace: Namespace object supporting both dictionary iteration
        and dot notation access.

Functions:
    Dictionary manipulation:
        flatten_dict: Flatten nested dictionaries with separator-based keys.
        unflatten_dict: Expand flattened keys into nested dictionaries.
        serialize_dict: Convert tensors/arrays to JSON-serializable format.

    File I/O:
        load_json, write_json: JSON file operations.
        load_jsonlines, write_jsonlines: JSONL operations.

    Data validation:
        validate_frame: Validate frame data against feature specifications.
        validate_episode_buffer: Validate episode buffer before adding.

    (Note: Truncated for brevity, apply the same flat indentation to the rest)

Example:
    Load dataset metadata::

        >>> info = load_info(Path("my_dataset"))
        >>> stats = load_stats(Path("my_dataset"))
        >>> episodes = load_episodes(Path("my_dataset"))

    Validate a frame::

        >>> features = {"state": {"dtype": "float32", "shape": (7,)}}
        >>> frame = {"state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])}
        >>> validate_frame(frame, features)
"""

import contextlib
import importlib.resources
import json
import logging
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import datasets
import jsonlines
import numpy as np
import packaging.version
import torch
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError
from PIL import Image as PILImage
from torchvision import transforms

from opentau.configs.types import DictLike, FeatureType, PolicyFeature
from opentau.datasets.backward_compatibility import (
    V21_MESSAGE,
    BackwardCompatibilityError,
    ForwardCompatibilityError,
)
from opentau.utils.utils import is_valid_numpy_dtype_string

DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk

ADVANTAGES_PATH = "meta/advantages.json"
INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"

DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"

DATASET_CARD_TEMPLATE = """
---
# Metadata will go there
---
This dataset was created using [OpenTau](https://github.com/TensorAuto/OpenTau).

## {}

"""

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example::

        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary by expanding keys with separators into nested dictionaries.

    Args:
        d: Dictionary with flattened keys (e.g., {"a/b": 1, "a/c/d": 2}).
        sep: Separator used to split keys. Defaults to "/".

    Returns:
        Nested dictionary structure (e.g., {"a": {"b": 1, "c": {"d": 2}}}).

    Example:
        >>> dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        >>> print(unflatten_dict(dct))
        {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def get_nested_item(obj: DictLike, flattened_key: str, sep: str = "/") -> Any:
    """Get a nested item from a dictionary-like object using a flattened key.

    Args:
        obj: Dictionary-like object to access.
        flattened_key: Flattened key path (e.g., "a/b/c").
        sep: Separator used in the flattened key. Defaults to "/".

    Returns:
        The value at the nested path specified by the flattened key.

    Example:
        >>> dct = {"a": {"b": {"c": 42}}}
        >>> get_nested_item(dct, "a/b/c")
        42
    """
    split_keys = flattened_key.split(sep)
    getter = obj[split_keys[0]]
    if len(split_keys) == 1:
        return getter

    for key in split_keys[1:]:
        getter = getter[key]

    return getter


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """Serialize a dictionary containing tensors and arrays to JSON-serializable format.

    Converts torch.Tensor and np.ndarray to lists, and np.generic to Python scalars.
    The dictionary structure is preserved through flattening and unflattening.

    Args:
        stats: Dictionary containing statistics with tensor/array values.

    Returns:
        Dictionary with serialized (list/scalar) values in the same structure.

    Raises:
        NotImplementedError: If a value type is not supported for serialization.
    """
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    """Embed image bytes into the dataset table before saving to parquet.

    Converts the dataset to arrow format, embeds image storage, and restores
    the original format.

    Args:
        dataset: HuggingFace dataset containing images.

    Returns:
        Dataset with embedded image bytes, ready for parquet serialization.
    """
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def load_json(fpath: Path) -> Any:
    """Load JSON data from a file.

    Args:
        fpath: Path to the JSON file.

    Returns:
        Parsed JSON data (dict, list, or primitive type).
    """
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist. Uses 4-space indentation
    and allows non-ASCII characters.

    Args:
        data: Dictionary or other JSON-serializable data to write.
        fpath: Path where the JSON file will be written.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path) -> list[Any]:
    """Load JSON Lines (JSONL) data from a file.

    Args:
        fpath: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line in the file.
    """
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def write_jsonlines(data: dict, fpath: Path) -> None:
    """Write data to a JSON Lines (JSONL) file.

    Creates parent directories if they don't exist. Writes each item in the
    data iterable as a separate line.

    Args:
        data: Iterable of dictionaries to write (one per line).
        fpath: Path where the JSONL file will be written.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def append_jsonlines(data: dict, fpath: Path) -> None:
    """Append a single dictionary to a JSON Lines (JSONL) file.

    Creates parent directories if they don't exist. Appends the data as a
    new line to the existing file.

    Args:
        data: Dictionary to append as a new line.
        fpath: Path to the JSONL file (will be created if it doesn't exist).
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def write_info(info: dict, local_dir: Path) -> None:
    """Write dataset info dictionary to the standard info.json file.

    Args:
        info: Dataset info dictionary to write.
        local_dir: Root directory of the dataset where meta/info.json will be written.
    """
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Path) -> dict:
    """Load dataset info from the standard info.json file.

    Converts feature shapes from lists to tuples for consistency.

    Args:
        local_dir: Root directory of the dataset containing meta/info.json.

    Returns:
        Dataset info dictionary with feature shapes as tuples.
    """
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def write_stats(stats: dict, local_dir: Path) -> None:
    """Write dataset statistics to the standard stats.json file.

    Serializes tensors and arrays to JSON-compatible format before writing.

    Args:
        stats: Dictionary containing dataset statistics (may contain tensors/arrays).
        local_dir: Root directory of the dataset where meta/stats.json will be written.
    """
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    """Convert statistics dictionary values to numpy arrays.

    Flattens the dictionary, converts all values to numpy arrays, then
    unflattens to restore the original structure.

    Args:
        stats: Dictionary with statistics (values may be lists or other types).

    Returns:
        Dictionary with the same structure but all values as numpy arrays.
    """
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load dataset statistics from the standard stats.json file.

    Args:
        local_dir: Root directory of the dataset containing meta/stats.json.

    Returns:
        Dictionary with statistics as numpy arrays, or None if the file doesn't exist.
    """
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def load_advantages(local_dir: Path) -> dict:
    """Load advantage values from the advantages.json file.

    Advantages are keyed by (episode_index, timestamp) tuples in the JSON file
    as comma-separated strings, which are converted to tuple keys.

    Args:
        local_dir: Root directory of the dataset containing meta/advantages.json.

    Returns:
        Dictionary mapping (episode_index, timestamp) tuples to advantage values,
        or None if the file doesn't exist.
    """
    if not (local_dir / ADVANTAGES_PATH).exists():
        return None
    advantages = load_json(local_dir / ADVANTAGES_PATH)
    return {(int(k.split(",")[0]), float(k.split(",")[1])): v for k, v in advantages.items()}


def write_task(task_index: int, task: dict, local_dir: Path) -> None:
    """Write a task entry to the tasks.jsonl file.

    Args:
        task_index: Integer index of the task.
        task: Task description dictionary.
        local_dir: Root directory of the dataset where meta/tasks.jsonl will be written.
    """
    task_dict = {
        "task_index": task_index,
        "task": task,
    }
    append_jsonlines(task_dict, local_dir / TASKS_PATH)


def load_tasks(local_dir: Path) -> tuple[dict, dict]:
    """Load tasks from the tasks.jsonl file.

    Args:
        local_dir: Root directory of the dataset containing meta/tasks.jsonl.

    Returns:
        Tuple of (tasks_dict, task_to_index_dict):
            - tasks_dict: Dictionary mapping task_index to task description.
            - task_to_index_dict: Dictionary mapping task description to task_index.
    """
    tasks = load_jsonlines(local_dir / TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def write_episode(episode: dict, local_dir: Path) -> None:
    """Write an episode entry to the episodes.jsonl file.

    Args:
        episode: Episode dictionary containing episode_index, tasks, length, etc.
        local_dir: Root directory of the dataset where meta/episodes.jsonl will be written.
    """
    append_jsonlines(episode, local_dir / EPISODES_PATH)


def load_episodes(local_dir: Path) -> dict:
    """Load episodes from the episodes.jsonl file.

    Args:
        local_dir: Root directory of the dataset containing meta/episodes.jsonl.

    Returns:
        Dictionary mapping episode_index to episode information dictionary.
    """
    episodes = load_jsonlines(local_dir / EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def write_episode_stats(episode_index: int, episode_stats: dict, local_dir: Path) -> None:
    """Write episode statistics to the episodes_stats.jsonl file.

    Serializes tensors and arrays in the stats before writing.

    Args:
        episode_index: Index of the episode.
        episode_stats: Dictionary containing statistics for the episode (may contain tensors/arrays).
        local_dir: Root directory of the dataset where meta/episodes_stats.jsonl will be written.
    """
    # We wrap episode_stats in a dictionary since `episode_stats["episode_index"]`
    # is a dictionary of stats and not an integer.
    episode_stats = {"episode_index": episode_index, "stats": serialize_dict(episode_stats)}
    append_jsonlines(episode_stats, local_dir / EPISODES_STATS_PATH)


def load_episodes_stats(local_dir: Path) -> dict:
    """Load episode statistics from the episodes_stats.jsonl file.

    Args:
        local_dir: Root directory of the dataset containing meta/episodes_stats.jsonl.

    Returns:
        Dictionary mapping episode_index to statistics dictionary (with numpy arrays).
    """
    episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def backward_compatible_episodes_stats(
    stats: dict[str, dict[str, np.ndarray]], episodes: list[int]
) -> dict[str, dict[str, np.ndarray]]:
    """Create episode-level statistics from global statistics for backward compatibility.

    In older dataset versions, statistics were stored globally rather than per-episode.
    This function creates per-episode statistics by assigning the same global stats
    to each episode.

    Args:
        stats: Global statistics dictionary.
        episodes: List of episode indices.

    Returns:
        Dictionary mapping episode_index to the same statistics dictionary.
    """
    return dict.fromkeys(episodes, stats)


def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    """Load an image file as a numpy array.

    Args:
        fpath: Path to the image file.
        dtype: Data type for the array. Defaults to np.float32.
        channel_first: If True, return array in (C, H, W) format; otherwise (H, W, C).
            Defaults to True.

    Returns:
        Image as numpy array. If dtype is floating point, values are normalized to [0, 1].
        Otherwise, values are in [0, 255].
    """
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array


def hf_transform_to_torch(items_dict: dict[torch.Tensor | None]):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def is_valid_version(version: str) -> bool:
    """Check if a version string is valid and can be parsed.

    Args:
        version: Version string to validate.

    Returns:
        True if the version string is valid, False otherwise.
    """
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    """Check compatibility between a dataset version and the current codebase version.

    Args:
        repo_id: Repository ID of the dataset.
        version_to_check: Version of the dataset to check.
        current_version: Current codebase version.
        enforce_breaking_major: If True, raise error for major version mismatches.
            Defaults to True.

    Raises:
        BackwardCompatibilityError: If the dataset version is too old (major version mismatch).
    """
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(V21_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    """Returns available valid versions (branches and tags) on given repo."""
    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:
    """
    Returns the version if available on repo or the latest compatible one.
    Otherwise, will throw a `CompatibilityError`.
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def get_hf_features_from_features(features: dict) -> datasets.Features:
    """Convert dataset features dictionary to HuggingFace Features object.

    Maps feature types and shapes to appropriate HuggingFace feature types
    (Image, Value, Sequence, Array2D, Array3D, Array4D, Array5D).

    Args:
        features: Dictionary mapping feature names to feature specifications
            with 'dtype' and 'shape' keys.

    Returns:
        HuggingFace Features object compatible with the dataset library.

    Raises:
        ValueError: If a feature shape is not supported (more than 5 dimensions).
    """
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features to policy feature format.

    Maps dataset features to policy feature types (VISUAL, ENV, STATE, ACTION)
    based on feature names and data types.

    Args:
        features: Dictionary mapping feature names to feature specifications.

    Returns:
        Dictionary mapping feature names to PolicyFeature objects.

    Raises:
        ValueError: If a visual feature doesn't have 3 dimensions.
    """
    # TODO(aliberts): Implement "type" in dataset features and simplify this
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key == "state":
            type = FeatureType.STATE
        elif key == "actions":
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    robot_type: str,
    features: dict,
    use_videos: bool,
) -> dict:
    """Create an empty dataset info dictionary with default values.

    Args:
        codebase_version: Version of the codebase used to create the dataset.
        fps: Frames per second used during data collection.
        robot_type: Type of robot used (can be None).
        features: Dictionary of feature specifications.
        use_videos: Whether videos are used for visual modalities.

    Returns:
        Dictionary containing dataset metadata with initialized counters and paths.
    """
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    """Compute data indices for episodes in a flattened dataset.

    Calculates start and end indices for each episode in a concatenated dataset,
    and creates a mapping from episode index to position in the episodes list.

    Args:
        episode_dicts: Dictionary mapping episode_index to episode info dicts
            containing 'length' keys.
        episodes: Optional list of episode indices to include. If None, uses all
            episodes from episode_dicts.

    Returns:
        Tuple of (episode_data_index, ep2idx):
            - episode_data_index: Dictionary with 'from' and 'to' tensors indicating
              start and end indices for each episode.
            - ep2idx: Dictionary mapping episode_index to position in the episodes list.
    """
    # `episodes_dicts` are not necessarily sorted, or starting with episode_index 0.
    episode_lengths = {edict["episode_index"]: edict["length"] for edict in episode_dicts.values()}

    if episodes is None:
        episodes = list(episode_lengths.keys())

    episode_lengths = [episode_lengths[ep_idx] for ep_idx in episodes]
    cumulative_lengths = list(accumulate(episode_lengths))
    start = [0] + cumulative_lengths[:-1]
    end = cumulative_lengths
    ep2idx = {ep_idx: i for i, ep_idx in enumerate(episodes)}
    return {"from": torch.LongTensor(start), "to": torch.LongTensor(end)}, ep2idx


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """
    This check is to make sure that each timestamp is separated from the next by (1/fps) +/- tolerance
    to account for possible numerical error.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        episode_indices (np.ndarray): Array indicating the episode index for each timestamp.
        episode_data_index (dict[str, np.ndarray]): A dictionary that includes 'to',
            which identifies indices for the end of each episode.
        fps (int): Frames per second. Used to check the expected difference between consecutive timestamps.
        tolerance_s (float): Allowed deviation from the expected (1/fps) difference.
        raise_value_error (bool): Whether to raise a ValueError if the check fails.

    Returns:
        bool: True if all checked timestamp differences lie within tolerance, False otherwise.

    Raises:
        ValueError: If the check fails and `raise_value_error` is True.
    """
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    # Consecutive differences
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask to ignore differences at the boundaries between episodes
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = episode_data_index["to"][:-1] - 1  # indices at the end of each episode
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    # Check if all remaining diffs are within tolerance
    if not np.all(filtered_within_tolerance):
        # Track original indices before masking
        original_indices = np.arange(len(diffs))
        filtered_indices = original_indices[mask]
        outside_tolerance_filtered_indices = np.nonzero(~filtered_within_tolerance)[0]
        outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]

        outside_tolerances = []
        for idx in outside_tolerance_indices:
            entry = {
                "timestamps": [timestamps[idx], timestamps[idx + 1]],
                "diff": diffs[idx],
                "episode_index": episode_indices[idx].item()
                if hasattr(episode_indices[idx], "item")
                else episode_indices[idx],
            }
            outside_tolerances.append(entry)

        if raise_value_error:
            raise ValueError(
                f"""One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues during data collection.
                \n{pformat(outside_tolerances)}"""
            )
        return False

    return True


DeltaTimestampParam = dict[str, np.ndarray]
DeltaTimestampInfo = tuple[DeltaTimestampParam, DeltaTimestampParam, DeltaTimestampParam, DeltaTimestampParam]


def get_delta_indices_soft(delta_timestamps_info: DeltaTimestampInfo, fps: int) -> DeltaTimestampParam:
    r"""Returns soft indices (not necessarily integer) for delta timestamps based on the provided information.
    Soft indices are computed by sampling from a normal distribution defined by the mean and standard deviation
    and clipping the values to the specified lower and upper bounds.
    Note: Soft indices can be converted to integer indices by either rounding or interpolation.
    """
    soft_indices = {}
    mean, std, lower, upper = delta_timestamps_info
    for key in mean:
        dT = np.random.normal(mean[key], std[key]).clip(lower[key], upper[key])  # noqa: N806
        soft_indices[key] = dT * fps

    return soft_indices


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_branch(repo_id, *, branch: str, repo_type: str | None = None) -> None:
    """Create a branch on a existing Hugging Face repo. Delete the branch if it already
    exists before creating it.
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    """
    Keyword arguments will be used to replace values in `src/opentau/datasets/card_template.md`.
    Note: If specified, license must be one of https://huggingface.co/docs/hub/repositories-licenses.
    """
    card_tags = ["OpenTau"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("opentau.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


class IterableNamespace(SimpleNamespace):
    """
    A namespace object that supports both dictionary-like iteration and dot notation access.
    Automatically converts nested dictionaries into IterableNamespaces.

    This class extends SimpleNamespace to provide:
    - Dictionary-style iteration over keys
    - Access to items via both dot notation (obj.key) and brackets (obj["key"])
    - Dictionary-like methods: items(), keys(), values()
    - Recursive conversion of nested dictionaries

    Args:
        dictionary: Optional dictionary to initialize the namespace
        **kwargs: Additional keyword arguments passed to SimpleNamespace

    Examples:
        >>> data = {"name": "Alice", "details": {"age": 25}}
        >>> ns = IterableNamespace(data)
        >>> ns.name
        'Alice'
        >>> ns.details.age
        25
        >>> list(ns.keys())
        ['name', 'details']
        >>> for key, value in ns.items():
        ...     print(f"{key}: {value}")
        name: Alice
        details: IterableNamespace(age=25)
    """

    def __init__(self, dictionary: dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is not None:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, IterableNamespace(value))
                else:
                    setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()

    def keys(self):
        return vars(self).keys()


def validate_frame(frame: dict, features: dict) -> None:
    """Validate that a frame dictionary matches the expected features.

    Checks that all required features are present, no unexpected features exist,
    and that feature types and shapes match the specification.

    Args:
        frame: Dictionary containing frame data to validate.
        features: Dictionary of expected feature specifications.

    Raises:
        ValueError: If the frame doesn't match the feature specifications.
    """
    optional_features = {"timestamp"}
    expected_features = (set(features) - set(DEFAULT_FEATURES.keys())) | {"task"}
    actual_features = set(frame.keys())

    error_message = validate_features_presence(actual_features, expected_features, optional_features)

    if "task" in frame:
        error_message += validate_feature_string("task", frame["task"])

    common_features = actual_features & (expected_features | optional_features)
    for name in common_features - {"task"}:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(
    actual_features: set[str], expected_features: set[str], optional_features: set[str]
) -> str:
    """Validate that required features are present and no unexpected features exist.

    Args:
        actual_features: Set of feature names actually present.
        expected_features: Set of feature names that must be present.
        optional_features: Set of feature names that may be present but aren't required.

    Returns:
        Error message string (empty if validation passes).
    """
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - (expected_features | optional_features)

    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"

    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
) -> str:
    """Validate that a feature value matches its expected dtype and shape.

    Routes to appropriate validation function based on feature type.

    Args:
        name: Name of the feature being validated.
        feature: Feature specification dictionary with 'dtype' and 'shape' keys.
        value: Actual value to validate.

    Returns:
        Error message string (empty if validation passes).

    Raises:
        NotImplementedError: If the feature dtype is not supported.
    """
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(f"The feature dtype '{expected_dtype}' is not implemented yet.")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
) -> str:
    """Validate that a numpy array feature matches expected dtype and shape.

    Args:
        name: Name of the feature being validated.
        expected_dtype: Expected numpy dtype as a string.
        expected_shape: Expected shape as a list of integers.
        value: Actual numpy array to validate.

    Returns:
        Error message string (empty if validation passes).
    """
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"

        if actual_shape != expected_shape:
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
) -> str:
    """Validate that an image or video feature matches expected shape.

    Supports both channel-first (C, H, W) and channel-last (H, W, C) formats.

    Args:
        name: Name of the feature being validated.
        expected_shape: Expected shape as [C, H, W].
        value: Actual image/video value (PIL Image or numpy array).

    Returns:
        Error message string (empty if validation passes).

    Note:
        Pixel value range validation ([0,1] for float, [0,255] for uint8) is
        performed by the image writer threads, not here.
    """
    # Note: The check of pixels range ([0,1] for float and [0,255] for uint8) is done by the image writer threads.
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_string(name: str, value: str) -> str:
    """Validate that a feature value is a string.

    Args:
        name: Name of the feature being validated.
        value: Actual value to validate.

    Returns:
        Error message string (empty if validation passes).
    """
    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:
    """Validate that an episode buffer is properly formatted.

    Checks that required keys exist, episode_index matches total_episodes,
    buffer is not empty, and all features are present.

    Args:
        episode_buffer: Dictionary containing episode data to validate.
        total_episodes: Total number of episodes already in the dataset.
        features: Dictionary of expected feature specifications.

    Raises:
        ValueError: If the buffer is missing required keys, is empty, or has
            mismatched features.
        NotImplementedError: If episode_index doesn't match total_episodes.
    """
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:
        # TODO(aliberts): Add option to use existing episode_index
        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset. This is not supported for now."
        )

    if episode_buffer["size"] == 0:
        raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")

    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )
