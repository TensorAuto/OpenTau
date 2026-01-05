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

"""Statistics computation and aggregation for dataset features.

This module provides functionality to compute statistical measures (min, max,
mean, standard deviation, and count) for dataset features, with special
handling for image and video data. It supports per-episode statistics
computation and aggregation across multiple episodes or datasets using
weighted averaging.

The module handles two main use cases:
    1. Computing statistics for individual episodes: Samples images efficiently,
       downsamples large images to reduce memory usage, and computes statistics
       for all feature types (images, vectors, etc.).
    2. Aggregating statistics across multiple episodes/datasets: Combines
       statistics using weighted mean and variance computation, taking global
       min/max values.

Key Features:
    - Memory-efficient image sampling: Uses heuristic-based sampling to
      estimate optimal number of samples based on dataset size.
    - Automatic image downsampling: Reduces large images (>300px) to ~150px
      for faster processing.
    - Weighted aggregation: Supports custom weights or uses episode counts
      as weights for aggregating statistics.
    - Parallel variance algorithm: Uses efficient algorithm for computing
      weighted variance across multiple statistics.

Functions:
    estimate_num_samples
        Heuristic to estimate optimal number of samples based on dataset size.
    sample_indices
        Generate evenly spaced sample indices from a dataset.
    auto_downsample_height_width
        Automatically downsample large images.
    sample_images
        Load and downsample a subset of images from file paths.
    get_feature_stats
        Compute statistical measures for an array.
    compute_episode_stats
        Compute statistics for a single episode.
    aggregate_feature_stats
        Aggregate statistics for a feature across multiple episodes.
    aggregate_stats
        Aggregate statistics from multiple episodes/datasets.

Example:
    Compute statistics for a single episode:
        >>> episode_data = {"state": state_array, "camera0": image_paths}
        >>> features = {"state": {"dtype": "float32"}, "camera0": {"dtype": "image"}}
        >>> stats = compute_episode_stats(episode_data, features)

    Aggregate statistics across multiple episodes:
        >>> stats_list = [episode1_stats, episode2_stats, episode3_stats]
        >>> weights = [100, 200, 150]  # Optional: custom weights
        >>> aggregated = aggregate_stats(stats_list, weights=weights)
"""

from typing import Optional

import numpy as np

from opentau.datasets.utils import load_image_as_numpy


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    """Generate evenly spaced sample indices from a dataset.

    Uses estimate_num_samples to determine how many samples to take,
    then returns evenly spaced indices across the dataset length.

    Args:
        data_len: Total length of the dataset.

    Returns:
        List of evenly spaced integer indices.
    """
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(
    img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300
) -> np.ndarray:
    """Automatically downsample an image if it exceeds size threshold.

    If the image's maximum dimension is below the threshold, returns it unchanged.
    Otherwise, downsamples by an integer factor to bring the larger dimension
    close to the target size.

    Args:
        img: Input image array of shape (C, H, W).
        target_size: Target size for the larger dimension after downsampling.
            Defaults to 150.
        max_size_threshold: Maximum size before downsampling is applied.
            Defaults to 300.

    Returns:
        Downsampled image array, or original if no downsampling needed.
    """
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    """Load and downsample a subset of images from file paths.

    Samples images using evenly spaced indices, loads them as uint8 arrays,
    and automatically downsamples large images to reduce memory usage.

    Args:
        image_paths: List of file paths to image files.

    Returns:
        Array of shape (num_samples, C, H, W) containing sampled images as uint8.
    """
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    """Compute statistical measures (min, max, mean, std, count) for an array.

    Args:
        array: Input numpy array to compute statistics over.
        axis: Axes along which to compute statistics.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Dictionary containing 'min', 'max', 'mean', 'std', and 'count' statistics.
    """
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    """Compute statistics for a single episode.

    For image/video features, samples and downsamples images before computing stats.
    For other features, computes stats directly on the array data.

    Args:
        episode_data: Dictionary mapping feature names to their data (arrays or image paths).
        features: Dictionary of feature specifications with 'dtype' keys.

    Returns:
        Dictionary mapping feature names to their statistics (min, max, mean, std, count).
        Image statistics are normalized to [0, 1] range.
    """
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict]]) -> None:
    """Validate that statistics dictionaries have correct types and shapes.

    Checks that all values are numpy arrays, have at least 1 dimension,
    count has shape (1,), and image stats have shape (3, 1, 1).

    Args:
        stats_list: List of statistics dictionaries to validate.

    Raises:
        ValueError: If any statistic has incorrect type or shape.
    """
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(
    stats_ft_list: list[dict[str, dict]], weights: Optional[list[float]] = None
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate statistics for a single feature across multiple episodes/datasets.

    Computes weighted mean and variance using the parallel algorithm for variance
    computation. Min and max are taken as the global min/max across all stats.

    Args:
        stats_ft_list: List of statistics dictionaries for the same feature.
        weights: Optional weights for each statistics entry. If None, uses
            count values as weights.

    Returns:
        Aggregated statistics dictionary with min, max, mean, std, and count.
    """
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])

    # if weights are provided, use them to compute the weighted mean and variance
    # otherwise, use episode counts as weights
    if weights is not None:
        counts = np.stack(weights)
        total_count = counts.sum(axis=0)
    else:
        counts = np.stack([s["count"] for s in stats_ft_list])
        total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(
    stats_list: list[dict[str, dict]], weights: Optional[list[float]] = None
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key, weights)

    return aggregated_stats
