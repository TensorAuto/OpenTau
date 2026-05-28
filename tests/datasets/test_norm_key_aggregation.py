#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for per-(robot_type, control_mode) normalization-head aggregation.

Covers:
  - `compute_norm_key` rules (happy path, fallback triggers).
  - `DatasetMixtureMetadata` grouping by `(robot_type, control_mode)` and
    pooling stats via `aggregate_stats`.
  - Closed-form pooled-mean / pooled-variance correctness for groups
    sharing a head.
  - Dimensional-incompatibility hard-failure.
  - Logging: single aggregated fallback warning + the per-head INFO summary.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from opentau.datasets.dataset_mixture import DatasetMixtureMetadata, compute_norm_key
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING


class TestComputeNormKey:
    """`compute_norm_key`: rules for joining or falling back."""

    def test_both_present(self):
        key, fallback = compute_norm_key("franka_arm", "joint_position", "foo/bar")
        assert key == "franka_arm::joint_position"
        assert fallback is False

    def test_robot_type_empty_falls_back(self):
        key, fallback = compute_norm_key("", "joint_position", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True

    def test_robot_type_none_falls_back(self):
        key, fallback = compute_norm_key(None, "joint_position", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True

    def test_control_mode_empty_falls_back(self):
        key, fallback = compute_norm_key("franka_arm", "", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True

    def test_control_mode_unknown_falls_back(self):
        # "unknown" is the LeRobotDatasetMetadata sentinel for missing
        # info.json["control_mode"]; treat it as missing.
        key, fallback = compute_norm_key("franka_arm", "unknown", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True

    def test_whitespace_only_falls_back(self):
        key, fallback = compute_norm_key("   ", "joint", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True
        key, fallback = compute_norm_key("franka", "\t  ", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True

    def test_explicit_tags_strip_whitespace(self):
        # Trim surrounding whitespace but keep the inner content.
        key, fallback = compute_norm_key("  franka_arm ", "  joint ", "foo/bar")
        assert key == "franka_arm::joint"
        assert fallback is False

    def test_robot_type_unknown_falls_back(self):
        # Symmetric with control_mode: `robot_type="unknown"` is the
        # common stand-in for missing info.json["robot_type"] and must
        # not silently anchor an "unknown::<mode>" shared head.
        key, fallback = compute_norm_key("unknown", "joint", "foo/bar")
        assert key == "foo/bar"
        assert fallback is True

    def test_unknown_is_case_insensitive(self):
        for variant in ("Unknown", "UNKNOWN", "uNkNoWn"):
            key, fallback = compute_norm_key("franka", variant, "foo/bar")
            assert key == "foo/bar"
            assert fallback is True
            key, fallback = compute_norm_key(variant, "joint", "foo/bar")
            assert key == "foo/bar"
            assert fallback is True


# -- DatasetMixtureMetadata fixtures ---------------------------------------


def _make_stats(state_dim: int, action_dim: int, *, value: float, count: int) -> dict:
    """Build a stats dict matching the standardized layout that
    `DatasetMixtureMetadata` consumes (state, actions, single camera).
    """
    return {
        "state": {
            "mean": np.full((state_dim,), float(value), dtype=np.float32),
            "std": np.full((state_dim,), float(value) * 0.1 + 0.5, dtype=np.float32),
            "min": np.full((state_dim,), float(value) - 1.0, dtype=np.float32),
            "max": np.full((state_dim,), float(value) + 1.0, dtype=np.float32),
            "count": np.array([count], dtype=np.int64),
        },
        "actions": {
            "mean": np.full((action_dim,), float(value) * 2.0, dtype=np.float32),
            "std": np.full((action_dim,), float(value) * 0.1 + 0.5, dtype=np.float32),
            "min": np.full((action_dim,), float(value) - 1.0, dtype=np.float32),
            "max": np.full((action_dim,), float(value) + 1.0, dtype=np.float32),
            "count": np.array([count], dtype=np.int64),
        },
        "camera0": {
            "mean": np.full((3, 1, 1), float(value) * 0.01, dtype=np.float32),
            "std": np.full((3, 1, 1), 0.5, dtype=np.float32),
            "min": np.zeros((3, 1, 1), dtype=np.float32),
            "max": np.ones((3, 1, 1), dtype=np.float32),
            "count": np.array([count], dtype=np.int64),
        },
    }


def _make_metadata(repo_id: str, info: dict, stats: dict) -> SimpleNamespace:
    """Minimal stand-in for ``LeRobotDatasetMetadata`` — only the attributes
    `DatasetMixtureMetadata.__init__` reads.
    """
    return SimpleNamespace(repo_id=repo_id, info=info, stats=stats)


def _make_cfg(num_cams: int = 1, max_state_dim: int = 32, max_action_dim: int = 32) -> SimpleNamespace:
    """`TrainPipelineConfig` stand-in. Only the attributes that
    `_to_standard_data_format` reads need to be present.
    """
    return SimpleNamespace(
        num_cams=num_cams,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        resolution=(224, 224),
    )


def _patch_name_mapping(repo_ids: list[str]):
    """Register an identity (state/actions/camera0) name map for each repo."""
    for repo_id in repo_ids:
        DATA_FEATURES_NAME_MAPPING[repo_id] = {
            "state": "state",
            "actions": "actions",
            "camera0": "camera0",
        }


class TestDatasetMixtureMetadataNormAggregation:
    """`DatasetMixtureMetadata`: per-(robot_type, control_mode) grouping."""

    def test_two_datasets_same_robot_and_mode_share_head(self):
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg()
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(6, 7, value=1.0, count=100),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 200},
            stats=_make_stats(6, 7, value=3.0, count=200),
        )
        meta = DatasetMixtureMetadata(
            cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
        )
        assert meta.norm_keys == ["franka::joint"]
        assert len(meta.per_norm_key_stats) == 1
        assert meta.dataset_to_norm_index == {"repo/a": 0, "repo/b": 0}
        assert meta.norm_key_to_dataset_names == {"franka::joint": ["repo/a", "repo/b"]}
        assert len(meta.per_dataset_stats) == 2  # per-dataset list still around
        assert meta.dataset_names == ["repo/a", "repo/b"]

    def test_same_robot_different_control_mode_distinct_rows(self):
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg()
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(6, 7, value=1.0, count=100),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": "franka", "control_mode": "ee", "total_frames": 100},
            stats=_make_stats(6, 7, value=2.0, count=100),
        )
        meta = DatasetMixtureMetadata(
            cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
        )
        assert set(meta.norm_keys) == {"franka::joint", "franka::ee"}
        assert meta.dataset_to_norm_index["repo/a"] != meta.dataset_to_norm_index["repo/b"]
        assert len(meta.per_norm_key_stats) == 2

    def test_missing_robot_type_falls_back_with_single_warning(self, caplog):
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg()
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(6, 7, value=1.0, count=100),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": None, "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(6, 7, value=2.0, count=100),
        )
        with caplog.at_level(logging.WARNING):
            meta = DatasetMixtureMetadata(
                cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
            )
        # repo/b falls back to its own name as the norm key.
        assert meta.dataset_to_norm_index == {"repo/a": 0, "repo/b": 1}
        assert meta.norm_keys[1] == "repo/b"
        # Exactly one warning for the whole mixture, mentioning the
        # fallback dataset.
        fallback_warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "fallback" in r.getMessage().lower()
        ]
        assert len(fallback_warnings) == 1
        assert "repo/b" in fallback_warnings[0].getMessage()

    def test_pooled_stats_closed_form(self):
        """Pooled mean/variance per the Chan-style weighted formula."""
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg(max_state_dim=4, max_action_dim=4)
        n1, n2 = 100, 300
        m1_val, m2_val = 1.0, 5.0
        s1_val, s2_val = 0.5, 0.8

        # Build raw 4-dim stats with explicit, scalar-broadcast mean/std/count.
        def stats_for(mean: float, std: float, n: int) -> dict:
            return {
                "state": {
                    "mean": np.full((4,), mean, dtype=np.float32),
                    "std": np.full((4,), std, dtype=np.float32),
                    "min": np.full((4,), mean - 5.0, dtype=np.float32),
                    "max": np.full((4,), mean + 5.0, dtype=np.float32),
                    "count": np.array([n], dtype=np.int64),
                },
                "actions": {
                    "mean": np.full((4,), mean * 2.0, dtype=np.float32),
                    "std": np.full((4,), std, dtype=np.float32),
                    "min": np.full((4,), mean - 5.0, dtype=np.float32),
                    "max": np.full((4,), mean + 5.0, dtype=np.float32),
                    "count": np.array([n], dtype=np.int64),
                },
                "camera0": {
                    "mean": np.full((3, 1, 1), mean * 0.01, dtype=np.float32),
                    "std": np.full((3, 1, 1), std, dtype=np.float32),
                    "min": np.zeros((3, 1, 1), dtype=np.float32),
                    "max": np.ones((3, 1, 1), dtype=np.float32),
                    "count": np.array([n], dtype=np.int64),
                },
            }

        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": n1},
            stats=stats_for(m1_val, s1_val, n1),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": n2},
            stats=stats_for(m2_val, s2_val, n2),
        )
        meta = DatasetMixtureMetadata(
            cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
        )
        head = meta.per_norm_key_stats[0]
        # Pooled mean and variance match the Chan-style closed form when
        # weights are sample counts (the aggregator's contract when
        # weights= is passed). Implementation passes `total_frames` as
        # weights, so use the same numerator/denominator pair here.
        n_total = n1 + n2
        expected_mean = (n1 * m1_val + n2 * m2_val) / n_total
        expected_var = (
            n1 * (s1_val**2 + (m1_val - expected_mean) ** 2)
            + n2 * (s2_val**2 + (m2_val - expected_mean) ** 2)
        ) / n_total
        np.testing.assert_allclose(head["state"]["mean"][0], expected_mean, atol=1e-6)
        np.testing.assert_allclose(head["state"]["std"][0], np.sqrt(expected_var), atol=1e-6)
        # min/max are the global element-wise min/max.
        np.testing.assert_allclose(head["state"]["min"][0], m1_val - 5.0, atol=1e-6)
        np.testing.assert_allclose(head["state"]["max"][0], m2_val + 5.0, atol=1e-6)

    def test_pooled_stats_closed_form_three_datasets(self):
        """3-dataset pooling with mismatched per-dataset counts. Catches
        any off-by-one in the weights-wiring path that 2-dataset cases
        would not expose (e.g. dropping a dataset's contribution silently).
        """
        _patch_name_mapping(["repo/a", "repo/b", "repo/c"])
        cfg = _make_cfg(max_state_dim=4, max_action_dim=4)
        # Mismatched counts on purpose so the weighted pool is sensitive
        # to each contributor's weight; means / stds are also distinct.
        params = [
            {"n": 100, "mean": 0.0, "std": 0.25},
            {"n": 250, "mean": 4.0, "std": 0.5},
            {"n": 650, "mean": -1.0, "std": 1.5},
        ]

        def stats_for(mean: float, std: float, n: int) -> dict:
            return {
                "state": {
                    "mean": np.full((4,), mean, dtype=np.float32),
                    "std": np.full((4,), std, dtype=np.float32),
                    "min": np.full((4,), mean - 5.0, dtype=np.float32),
                    "max": np.full((4,), mean + 5.0, dtype=np.float32),
                    "count": np.array([n], dtype=np.int64),
                },
                "actions": {
                    "mean": np.full((4,), mean, dtype=np.float32),
                    "std": np.full((4,), std, dtype=np.float32),
                    "min": np.full((4,), mean - 5.0, dtype=np.float32),
                    "max": np.full((4,), mean + 5.0, dtype=np.float32),
                    "count": np.array([n], dtype=np.int64),
                },
                "camera0": {
                    "mean": np.full((3, 1, 1), mean * 0.01, dtype=np.float32),
                    "std": np.full((3, 1, 1), std, dtype=np.float32),
                    "min": np.zeros((3, 1, 1), dtype=np.float32),
                    "max": np.ones((3, 1, 1), dtype=np.float32),
                    "count": np.array([n], dtype=np.int64),
                },
            }

        metadatas = [
            _make_metadata(
                f"repo/{name}",
                info={
                    "robot_type": "franka",
                    "control_mode": "joint",
                    "total_frames": p["n"],
                },
                stats=stats_for(p["mean"], p["std"], p["n"]),
            )
            for name, p in zip("abc", params, strict=True)
        ]
        meta = DatasetMixtureMetadata(
            cfg,
            metadatas,
            dataset_weights=[1 / 3, 1 / 3, 1 / 3],
            dataset_names=["repo/a", "repo/b", "repo/c"],
        )
        assert meta.norm_keys == ["franka::joint"]
        head = meta.per_norm_key_stats[0]

        n_total = sum(p["n"] for p in params)
        expected_mean = sum(p["n"] * p["mean"] for p in params) / n_total
        expected_var = (
            sum(p["n"] * (p["std"] ** 2 + (p["mean"] - expected_mean) ** 2) for p in params) / n_total
        )
        np.testing.assert_allclose(head["state"]["mean"][0], expected_mean, atol=1e-5)
        np.testing.assert_allclose(head["state"]["std"][0], np.sqrt(expected_var), atol=1e-5)

    def test_norm_head_summary_logged(self, caplog):
        _patch_name_mapping(["repo/a", "repo/b", "repo/c"])
        cfg = _make_cfg()
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(6, 7, value=1.0, count=100),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(6, 7, value=2.0, count=100),
        )
        m3 = _make_metadata(
            "repo/c",
            info={"robot_type": "ur5", "control_mode": "ee", "total_frames": 100},
            stats=_make_stats(6, 7, value=3.0, count=100),
        )
        with caplog.at_level(logging.INFO):
            DatasetMixtureMetadata(
                cfg,
                [m1, m2, m3],
                dataset_weights=[0.3, 0.3, 0.4],
                dataset_names=["repo/a", "repo/b", "repo/c"],
            )
        summaries = [r.getMessage() for r in caplog.records if "Norm-head aggregation" in r.getMessage()]
        assert len(summaries) == 1
        summary = summaries[0]
        assert "2 heads over 3 datasets" in summary
        assert "franka::joint" in summary
        assert "ur5::ee" in summary
        assert "repo/a" in summary and "repo/b" in summary and "repo/c" in summary

    def test_dim_mismatch_in_shared_head_raises(self):
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg()
        # Both datasets carry the same (robot, control) tag but raw state
        # dims differ. This is the failure mode the user wants surfaced.
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(state_dim=6, action_dim=7, value=1.0, count=100),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(state_dim=8, action_dim=7, value=2.0, count=100),
        )
        with pytest.raises(ValueError, match="incompatible raw"):
            DatasetMixtureMetadata(
                cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
            )

    def test_dim_mismatch_in_fallback_is_allowed(self):
        """A fallback-keyed dataset is by construction a singleton head, so
        a dim mismatch with any other dataset cannot trigger the check."""
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg()
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(state_dim=6, action_dim=7, value=1.0, count=100),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": None, "control_mode": "joint", "total_frames": 100},
            stats=_make_stats(state_dim=8, action_dim=7, value=2.0, count=100),
        )
        # Must not raise.
        meta = DatasetMixtureMetadata(
            cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
        )
        assert meta.dataset_to_norm_index["repo/a"] != meta.dataset_to_norm_index["repo/b"]


class TestTaggedDatasetIndexIsNormHead:
    """The `_TaggedDataset` index emitted into samples is the norm-head row
    (computed via `dataset_to_norm_index`), not the per-dataset enumerate
    position. Verifies the rewiring done in `WeightedDatasetMixture`.
    """

    def test_shared_norm_key_yields_same_index(self):
        _patch_name_mapping(["repo/a", "repo/b"])
        cfg = _make_cfg()
        m1 = _make_metadata(
            "repo/a",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 50},
            stats=_make_stats(6, 7, value=1.0, count=50),
        )
        m2 = _make_metadata(
            "repo/b",
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 50},
            stats=_make_stats(6, 7, value=2.0, count=50),
        )
        meta = DatasetMixtureMetadata(
            cfg, [m1, m2], dataset_weights=[0.5, 0.5], dataset_names=["repo/a", "repo/b"]
        )
        assert meta.dataset_to_norm_index["repo/a"] == meta.dataset_to_norm_index["repo/b"] == 0
