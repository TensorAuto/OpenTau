#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for the dual-split feature-mapping collision.

Two mixture entries can share a ``repo_id`` while declaring different
``data_features_name_mapping`` values — the same robot exposed as
``control_mode="joint"`` (``actions -> action_joint``) and
``control_mode="ee"`` (``actions -> action_ee``). The global
``DATA_FEATURES_NAME_MAPPING`` used to be keyed by ``repo_id`` alone, so the
second ``DatasetConfig.__post_init__`` clobbered the first and BOTH control
modes silently read ``action_ee``. These tests pin the fix: the two mappings
coexist under composite keys and each control mode resolves to its own column,
so ``DatasetMixtureMetadata`` produces distinct joint vs ee action stats.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from opentau.configs.default import DatasetConfig
from opentau.datasets.dataset_mixture import DatasetMixtureMetadata
from opentau.datasets.standard_data_format_mapping import (
    DATA_FEATURES_NAME_MAPPING,
    feature_mapping_key,
    resolve_feature_mapping,
)


@pytest.fixture(autouse=True)
def _restore_global_mapping():
    """Snapshot/restore the process-global mapping so a test's registrations
    don't leak into other tests."""
    snapshot = copy.deepcopy(DATA_FEATURES_NAME_MAPPING)
    try:
        yield
    finally:
        DATA_FEATURES_NAME_MAPPING.clear()
        DATA_FEATURES_NAME_MAPPING.update(snapshot)


def test_missing_control_mode_sentinels_stay_in_sync():
    """Column resolution (`feature_mapping_key`) and the norm-head split
    (`compute_norm_key`) must agree on what counts as "no control mode". If the
    two sentinel sets drift, the column the data is read from and the head its
    stats land in can silently disagree -- re-opening exactly the dual-split
    collision this fixes. The sets are hand-mirrored across modules, so guard it.
    """
    from opentau.datasets.dataset_mixture import _NORM_KEY_MISSING_VALUES
    from opentau.datasets.standard_data_format_mapping import _MISSING_CONTROL_MODE_VALUES

    assert _MISSING_CONTROL_MODE_VALUES == _NORM_KEY_MISSING_VALUES


class TestFeatureMappingKey:
    """`feature_mapping_key`: composite key construction + sentinels."""

    def test_real_control_mode_makes_composite(self):
        assert feature_mapping_key("repo/x", "joint") == "repo/x::joint"
        assert feature_mapping_key("repo/x", "ee") == "repo/x::ee"

    def test_strips_whitespace(self):
        assert feature_mapping_key("repo/x", "  ee ") == "repo/x::ee"

    @pytest.mark.parametrize("cm", [None, "", "   ", "unknown", "Unknown", "UNKNOWN"])
    def test_missing_or_unknown_falls_back_to_repo_id(self, cm):
        assert feature_mapping_key("repo/x", cm) == "repo/x"


class TestResolveFeatureMapping:
    """`resolve_feature_mapping`: composite-first lookup with repo_id fallback."""

    def test_prefers_composite(self):
        DATA_FEATURES_NAME_MAPPING["repo/x"] = {"actions": "action_ee"}
        DATA_FEATURES_NAME_MAPPING["repo/x::joint"] = {"actions": "action_joint"}
        assert resolve_feature_mapping("repo/x", "joint")["actions"] == "action_joint"
        assert resolve_feature_mapping("repo/x", "ee")["actions"] == "action_ee"  # falls back to repo_id

    def test_falls_back_to_repo_id_when_no_composite(self):
        DATA_FEATURES_NAME_MAPPING["repo/x"] = {"actions": "action"}
        assert resolve_feature_mapping("repo/x", "joint")["actions"] == "action"
        assert resolve_feature_mapping("repo/x", None)["actions"] == "action"

    def test_missing_raises(self):
        with pytest.raises(KeyError):
            resolve_feature_mapping("repo/does-not-exist", "joint")


class TestDualSplitRegistration:
    """`DatasetConfig.__post_init__`: both mappings coexist after construction."""

    def test_joint_and_ee_coexist(self):
        repo = "TensorAuto/dual-split-demo"
        # Order matches the real configs: joint registered first, ee second.
        DatasetConfig(
            repo_id=repo,
            control_mode="joint",
            data_features_name_mapping={"state": "observation.state", "actions": "action_joint"},
        )
        DatasetConfig(
            repo_id=repo,
            control_mode="ee",
            data_features_name_mapping={"state": "observation.state", "actions": "action_ee"},
        )

        # Both composite keys present and correct (the bug: only one survived).
        assert DATA_FEATURES_NAME_MAPPING[f"{repo}::joint"]["actions"] == "action_joint"
        assert DATA_FEATURES_NAME_MAPPING[f"{repo}::ee"]["actions"] == "action_ee"
        # Plain repo_id retained (last-wins) purely as a back-compat fallback.
        assert DATA_FEATURES_NAME_MAPPING[repo]["actions"] == "action_ee"
        # The resolver returns the correct column per mode — this is what the fix buys.
        assert resolve_feature_mapping(repo, "joint")["actions"] == "action_joint"
        assert resolve_feature_mapping(repo, "ee")["actions"] == "action_ee"


# -- DatasetMixtureMetadata end-to-end -------------------------------------


def _feat_stats(dim: int, *, value: float, count: int = 100) -> dict:
    return {
        "mean": np.full((dim,), float(value), dtype=np.float32),
        "std": np.full((dim,), 0.5, dtype=np.float32),
        "min": np.full((dim,), float(value) - 1.0, dtype=np.float32),
        "max": np.full((dim,), float(value) + 1.0, dtype=np.float32),
        "count": np.array([count], dtype=np.int64),
    }


def _make_cfg() -> SimpleNamespace:
    # num_cams=0 -> `_to_standard_data_format` processes only state + actions.
    return SimpleNamespace(num_cams=0, max_state_dim=32, max_action_dim=32, resolution=(224, 224))


class TestDualSplitMixtureMetadata:
    """`DatasetMixtureMetadata`: joint vs ee heads get their OWN action stats."""

    def test_joint_and_ee_heads_have_distinct_action_stats(self):
        repo = "TensorAuto/dual-split-demo"
        DATA_FEATURES_NAME_MAPPING[repo] = {"state": "observation.state", "actions": "action_ee"}
        DATA_FEATURES_NAME_MAPPING[f"{repo}::joint"] = {
            "state": "observation.state",
            "actions": "action_joint",
        }
        DATA_FEATURES_NAME_MAPPING[f"{repo}::ee"] = {
            "state": "observation.state",
            "actions": "action_ee",
        }

        # Raw stats carry BOTH action columns with clearly distinct values.
        raw_stats = {
            "observation.state": _feat_stats(7, value=0.0),
            "action_joint": _feat_stats(7, value=1.0),
            "action_ee": _feat_stats(7, value=2.0),
        }
        m_joint = SimpleNamespace(
            repo_id=repo,
            info={"robot_type": "franka", "control_mode": "joint", "total_frames": 100},
            stats=copy.deepcopy(raw_stats),
        )
        m_ee = SimpleNamespace(
            repo_id=repo,
            info={"robot_type": "franka", "control_mode": "ee", "total_frames": 100},
            stats=copy.deepcopy(raw_stats),
        )

        meta = DatasetMixtureMetadata(
            _make_cfg(),
            [m_joint, m_ee],
            dataset_weights=[1.0, 1.0],
            dataset_names=[f"{repo}#0", f"{repo}#1"],
        )

        assert set(meta.norm_keys) == {"franka::joint", "franka::ee"}
        joint = meta.per_norm_key_stats[meta.norm_key_to_index["franka::joint"]]["actions"]
        ee = meta.per_norm_key_stats[meta.norm_key_to_index["franka::ee"]]["actions"]

        # The fix: joint head reads action_joint (mean 1.0), ee head reads action_ee (mean 2.0).
        # Pre-fix both resolved to action_ee, so these would have been identical.
        np.testing.assert_allclose(joint["mean"][:7], 1.0)
        np.testing.assert_allclose(ee["mean"][:7], 2.0)
        assert not np.allclose(joint["mean"][:7], ee["mean"][:7])
