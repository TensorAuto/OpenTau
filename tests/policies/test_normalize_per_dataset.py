#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Per-dataset Normalize / Unnormalize: stacked buffers indexed per sample."""

import pytest
import torch

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.normalize import EPS, Normalize, Unnormalize


def _build_per_dataset_stats(num_datasets: int) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Build distinct stats per dataset so per-sample indexing is observable."""
    out = []
    for d in range(num_datasets):
        out.append(
            {
                "observation.state": {
                    "mean": torch.full((4,), float(d)),
                    "std": torch.full((4,), float(d + 1)),
                    "min": torch.full((4,), -float(d + 1)),
                    "max": torch.full((4,), float(d + 1)),
                },
                "action": {
                    "mean": torch.full((3,), float(d) * 0.1),
                    "std": torch.full((3,), float(d + 1) * 0.5),
                    "min": torch.full((3,), -float(d + 1)),
                    "max": torch.full((3,), float(d + 1)),
                },
            }
        )
    return out


def test_per_dataset_normalize_mean_std_picks_right_row():
    """Each sample's mean/std must be drawn from the row matching its dataset_index."""
    num_ds = 3
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    norm_map = {"STATE": NormalizationMode.MEAN_STD}

    per_dataset_stats = _build_per_dataset_stats(num_ds)
    norm = Normalize(features, norm_map, per_dataset_stats=per_dataset_stats)

    # B=6 batch with two samples per dataset row.
    dataset_index = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    state = torch.full((6, 4), 7.0)
    out = norm({"observation.state": state}, dataset_index)
    # For sample i: (7 - d) / (d+1 + EPS) where d = dataset_index[i]
    expected = torch.stack([(7.0 - d) / (d + 1.0 + EPS) * torch.ones(4) for d in dataset_index.tolist()])
    torch.testing.assert_close(out["observation.state"], expected)


def test_per_dataset_unnormalize_inverts_normalize():
    """Unnormalize must invert Normalize per-sample within MEAN_STD."""
    num_ds = 4
    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    norm_map = {"ACTION": NormalizationMode.MEAN_STD}

    per_dataset_stats = _build_per_dataset_stats(num_ds)
    norm = Normalize(features, norm_map, per_dataset_stats=per_dataset_stats)
    unnorm = Unnormalize(features, norm_map, per_dataset_stats=per_dataset_stats)

    dataset_index = torch.tensor([3, 0, 2, 1], dtype=torch.long)
    original = torch.randn(4, 3)
    normalized = norm({"action": original}, dataset_index)["action"]
    recovered = unnorm({"action": normalized}, dataset_index)["action"]
    # Some tolerance because EPS shifts both directions.
    torch.testing.assert_close(recovered, original, atol=1e-5, rtol=1e-4)


def test_per_dataset_min_max_normalizes_to_signed_unit_range():
    """MIN_MAX mode normalizes each sample's value into [-1, 1] using its own dataset's range."""
    num_ds = 3
    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    norm_map = {"ACTION": NormalizationMode.MIN_MAX}

    per_dataset_stats = _build_per_dataset_stats(num_ds)
    norm = Normalize(features, norm_map, per_dataset_stats=per_dataset_stats)

    # For dataset d, [min, max] = [-(d+1), d+1]. Feed the midpoint -> 0.
    midpoints = torch.zeros(num_ds, 3)
    dataset_index = torch.arange(num_ds, dtype=torch.long)
    out = norm({"action": midpoints}, dataset_index)["action"]
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0.0)

    # Feed each dataset's max -> +1 (modulo EPS slop).
    maxes = torch.stack([torch.full((3,), float(d + 1)) for d in range(num_ds)])
    out_max = norm({"action": maxes}, dataset_index)["action"]
    torch.testing.assert_close(out_max, torch.ones_like(out_max), atol=1e-4, rtol=0.0)


def test_per_dataset_broadcast_with_temporal_axis():
    """The forward must broadcast (B, *feat_shape) buffers over an extra T axis."""
    num_ds = 2
    t_dim = 5
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    norm_map = {"STATE": NormalizationMode.MEAN_STD}

    per_dataset_stats = _build_per_dataset_stats(num_ds)
    norm = Normalize(features, norm_map, per_dataset_stats=per_dataset_stats)

    dataset_index = torch.tensor([0, 1], dtype=torch.long)
    state = torch.full((2, t_dim, 4), 4.0)
    out = norm({"observation.state": state}, dataset_index)["observation.state"]
    assert out.shape == (2, t_dim, 4)
    # Sample 0 uses (mean=0, std=1); sample 1 uses (mean=1, std=2).
    expected_row0 = (4.0 - 0.0) / (1.0 + EPS)
    expected_row1 = (4.0 - 1.0) / (2.0 + EPS)
    torch.testing.assert_close(out[0], torch.full((t_dim, 4), expected_row0))
    torch.testing.assert_close(out[1], torch.full((t_dim, 4), expected_row1))


def test_per_dataset_image_buffer_uses_channels_only():
    """Image-feature stats live as (D, C, 1, 1) and broadcast to full (B, C, H, W)."""
    h_dim = w_dim = 8
    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h_dim, w_dim)),
    }
    norm_map = {"VISUAL": NormalizationMode.MEAN_STD}

    per_dataset_stats = [
        {
            "observation.image": {
                "mean": torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1, 1),
                "std": torch.tensor([1.0, 1.0, 1.0]).reshape(3, 1, 1),
            }
        },
        {
            "observation.image": {
                "mean": torch.tensor([10.0, 11.0, 12.0]).reshape(3, 1, 1),
                "std": torch.tensor([2.0, 2.0, 2.0]).reshape(3, 1, 1),
            }
        },
    ]

    norm = Normalize(features, norm_map, per_dataset_stats=per_dataset_stats)
    image = torch.zeros(2, 3, h_dim, w_dim)
    image[1] = 12.0  # match dataset-1 mean per channel later
    dataset_index = torch.tensor([0, 1], dtype=torch.long)
    out = norm({"observation.image": image}, dataset_index)["observation.image"]
    assert out.shape == (2, 3, h_dim, w_dim)
    # Sample 1 fed value 12 with means [10, 11, 12] and std 2 across channels
    # -> per-channel result [(12-10)/2, (12-11)/2, (12-12)/2] = [1.0, 0.5, 0.0]
    torch.testing.assert_close(out[1, 0], torch.full((h_dim, w_dim), 1.0), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(out[1, 1], torch.full((h_dim, w_dim), 0.5), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(out[1, 2], torch.full((h_dim, w_dim), 0.0), atol=1e-4, rtol=0.0)


def test_per_dataset_stats_length_mismatch_raises():
    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    norm_map = {"ACTION": NormalizationMode.MEAN_STD}
    per_dataset_stats = _build_per_dataset_stats(2)
    with pytest.raises(ValueError, match="must have the same length"):
        Normalize(features, norm_map, per_dataset_stats=per_dataset_stats, dataset_names=["only-one"])
