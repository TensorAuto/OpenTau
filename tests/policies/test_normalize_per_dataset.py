#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Per-dataset Normalize / Unnormalize: stacked buffers indexed per sample."""

import logging

import pytest
import torch

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.normalize import _WARNED_SNAP_KEYS, EPS, Normalize, Unnormalize


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


# -- _resolve_dataset_index dispatch ---------------------------------------


class _TinyConfig:
    """Stand-in for `PreTrainedConfig` exposing only what
    `PreTrainedPolicy.__init__` reads. Real configs are dataclasses; this
    avoids pulling a concrete subclass (and its policy class wiring) into
    a normalize-only test."""

    def __init__(self, dataset_names, dataset_to_norm_index=None):
        self.dataset_names = dataset_names
        self.dataset_to_norm_index = dataset_to_norm_index


def _make_policy_with_normalize(dataset_names, dataset_to_norm_index, *, num_rows):
    """Build a minimal `nn.Module` carrying the policy's name maps plus
    one `Normalize` so `_resolve_dataset_index` has a buffer to sniff
    (`_stacked_num_datasets`).
    """
    from opentau.configs.policies import PreTrainedConfig
    from opentau.policies.pretrained import PreTrainedPolicy

    class _DummyConfig(PreTrainedConfig):
        @property
        def observation_delta_indices(self):
            return None

        @property
        def action_delta_indices(self):
            return None

        @property
        def reward_delta_indices(self):
            return None

        def get_optimizer_preset(self):
            raise NotImplementedError

        def get_scheduler_preset(self):
            return None

        def validate_features(self):
            return None

    class _DummyPolicy(PreTrainedPolicy):
        config_class = _DummyConfig
        name = "dummy-test-policy"

        def __init__(self, config):
            super().__init__(config)
            features = {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            }
            norm_map = {"STATE": NormalizationMode.MEAN_STD}
            stats = [
                {
                    "observation.state": {
                        "mean": torch.full((2,), float(i)),
                        "std": torch.ones(2),
                    }
                }
                for i in range(num_rows)
            ]
            self.normalize_inputs = Normalize(
                features, norm_map, per_dataset_stats=stats, dataset_names=dataset_names
            )

        def get_optim_params(self):
            return {}

        def reset(self):
            pass

        def forward(self, batch):
            raise NotImplementedError

        def select_action(self, batch):
            raise NotImplementedError

    cfg = _DummyConfig()
    cfg.dataset_names = list(dataset_names) if dataset_names is not None else None
    cfg.dataset_to_norm_index = dict(dataset_to_norm_index) if dataset_to_norm_index is not None else None
    return _DummyPolicy(cfg)


class TestResolveDatasetIndex:
    """`_resolve_dataset_index`: training, repo-id, and (robot,control) routes."""

    def test_dataset_index_passthrough(self):
        policy = _make_policy_with_normalize(
            ["franka::joint", "ur5::ee"],
            {"franka_repo": 0, "ur5_repo": 1},
            num_rows=2,
        )
        idx = policy._resolve_dataset_index({"dataset_index": torch.tensor([0, 1], dtype=torch.long)})
        assert torch.equal(idx.cpu(), torch.tensor([0, 1], dtype=torch.long))

    def test_dataset_repo_id_maps_via_dataset_to_norm_index(self):
        policy = _make_policy_with_normalize(
            ["franka::joint", "ur5::ee"],
            # Two dataset names route to the same head; another to a
            # different head — exercises the new mapping faithfully.
            {"franka_repo_a": 0, "franka_repo_b": 0, "ur5_repo": 1},
            num_rows=2,
        )
        idx = policy._resolve_dataset_index(
            {
                "dataset_repo_id": ["franka_repo_a", "franka_repo_b", "ur5_repo"],
                "observation.state": torch.zeros(3, 2),
            }
        )
        assert torch.equal(idx.cpu(), torch.tensor([0, 0, 1], dtype=torch.long))

    def test_robot_type_control_mode_route(self):
        policy = _make_policy_with_normalize(
            ["franka::joint", "ur5::ee"],
            {"franka_repo": 0, "ur5_repo": 1},
            num_rows=2,
        )
        idx = policy._resolve_dataset_index(
            {
                "robot_type": ["franka", "ur5"],
                "control_mode": ["joint", "ee"],
                "observation.state": torch.zeros(2, 2),
            }
        )
        assert torch.equal(idx.cpu(), torch.tensor([0, 1], dtype=torch.long))

    def test_robot_type_control_mode_unknown_raises(self):
        policy = _make_policy_with_normalize(
            ["franka::joint", "ur5::ee"],
            {"franka_repo": 0, "ur5_repo": 1},
            num_rows=2,
        )
        # "unknown" triggers fallback to "", which is not in
        # `_norm_key_to_index` — raises rather than silently routing.
        with pytest.raises(ValueError, match="not in this policy's training"):
            policy._resolve_dataset_index(
                {
                    "robot_type": ["franka"],
                    "control_mode": ["unknown"],
                    "observation.state": torch.zeros(1, 2),
                }
            )

    def test_legacy_config_without_dataset_to_norm_index(self):
        """Old checkpoint: only `dataset_names` is persisted. The policy
        synthesizes the identity mapping so dataset_repo_id lookups still
        resolve."""
        policy = _make_policy_with_normalize(
            ["repo/foo", "repo/bar"],
            dataset_to_norm_index=None,  # legacy
            num_rows=2,
        )
        idx = policy._resolve_dataset_index(
            {"dataset_repo_id": ["repo/bar", "repo/foo"], "observation.state": torch.zeros(2, 2)}
        )
        assert torch.equal(idx.cpu(), torch.tensor([1, 0], dtype=torch.long))

    def test_robot_type_control_mode_takes_precedence_over_dataset_repo_id(self):
        """When both inference keys are supplied (e.g. by a caller that
        bypasses the eval / gRPC scripts and hands-builds a batch),
        `(robot_type, control_mode)` wins — matches the precedence
        documented on `EvalConfig` / `ServerConfig`."""
        policy = _make_policy_with_normalize(
            ["franka::joint", "ur5::ee"],
            # Crafted so the two routes resolve to different rows: the
            # repo_id would say row 0, the (robot, control) pair says row 1.
            {"franka_repo": 0, "ur5_repo": 1},
            num_rows=2,
        )
        idx = policy._resolve_dataset_index(
            {
                "dataset_repo_id": ["franka_repo"],
                "robot_type": ["ur5"],
                "control_mode": ["ee"],
                "observation.state": torch.zeros(1, 2),
            }
        )
        assert torch.equal(idx.cpu(), torch.tensor([1], dtype=torch.long))

    def test_multirow_unidentified_batch_raises(self):
        policy = _make_policy_with_normalize(
            ["franka::joint", "ur5::ee"],
            {"franka_repo": 0, "ur5_repo": 1},
            num_rows=2,
        )
        with pytest.raises(KeyError, match="requires one of"):
            policy._resolve_dataset_index({"observation.state": torch.zeros(1, 2)})

    def test_single_head_legacy_no_maps_robot_type_routes_to_row0(self):
        """Backward-compat: a checkpoint predating per-(robot_type, control_mode)
        normalization has one norm head and no name maps (`dataset_names=None`). An
        eval batch tagged with (robot_type, control_mode) must route to row 0 rather
        than raising "loaded without dataset_names" — one head, nothing to resolve."""
        policy = _make_policy_with_normalize(None, None, num_rows=1)
        assert policy._norm_key_to_index is None  # genuinely the legacy single-head case
        idx = policy._resolve_dataset_index(
            {
                "robot_type": ["PandaOmron"],
                "control_mode": ["ee"],
                "observation.state": torch.zeros(1, 2),
            }
        )
        assert torch.equal(idx.cpu(), torch.tensor([0], dtype=torch.long))

    def test_single_head_legacy_no_maps_repo_id_routes_to_row0(self):
        """Same single-head fallback on the `dataset_repo_id` route."""
        policy = _make_policy_with_normalize(None, None, num_rows=1)
        idx = policy._resolve_dataset_index(
            {"dataset_repo_id": ["whatever/repo"], "observation.state": torch.zeros(1, 2)}
        )
        assert torch.equal(idx.cpu(), torch.tensor([0], dtype=torch.long))

    def test_multihead_no_maps_robot_type_still_raises(self):
        """The fallback must NOT swallow a genuinely-unresolvable case: a *multi-head*
        policy with no name map can't be routed from (robot_type, control_mode)."""
        policy = _make_policy_with_normalize(None, None, num_rows=2)
        assert policy._norm_key_to_index is None
        with pytest.raises(RuntimeError, match="without `dataset_names`"):
            policy._resolve_dataset_index(
                {
                    "robot_type": ["franka"],
                    "control_mode": ["joint"],
                    "observation.state": torch.zeros(1, 2),
                }
            )


# -- zero-variance guard ----------------------------------------------------


class TestZeroVarianceGuard:
    """The std-snap guard keeps a value deviating from a constant (std=0) dim finite,
    stays bit-identical for std>0 dims, and preserves the Normalize/Unnormalize round-trip."""

    features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,))}
    norm_map = {"STATE": NormalizationMode.MEAN_STD}

    @staticmethod
    def _stats(mean, std):
        return [{"observation.state": {"mean": torch.tensor(mean), "std": torch.tensor(std)}}]

    def test_deviating_value_is_bounded_not_exploded(self):
        m = 0.761
        norm = Normalize(
            self.features,
            self.norm_map,
            per_dataset_stats=self._stats([0.0, 0.0, 0.0, m], [1.0, 1.0, 1.0, 0.0]),
        )
        idx = torch.zeros(1, dtype=torch.long)
        # dim 3 is "constant" (std=0) but the fed value is 0 — i.e. it deviates from mean=m.
        out = norm({"observation.state": torch.zeros(1, 4)}, idx)["observation.state"]
        # Snapped: (0 - m)/(1+EPS) ~ -m, bounded. Without the guard it would be -m/EPS ~ -7.6e7.
        torch.testing.assert_close(out[0, 3], torch.tensor(-m), atol=1e-3, rtol=0.0)
        assert out[0, 3].abs().item() < 1.0
        torch.testing.assert_close(out[0, :3], torch.zeros(3))

    def test_matching_value_normalizes_to_zero(self):
        m = 0.761
        norm = Normalize(
            self.features,
            self.norm_map,
            per_dataset_stats=self._stats([0.0, 0.0, 0.0, m], [1.0, 1.0, 1.0, 0.0]),
        )
        idx = torch.zeros(1, dtype=torch.long)
        # The dim is genuinely constant: the fed value equals the mean -> maps to 0 (snap is a no-op).
        out = norm({"observation.state": torch.tensor([[0.0, 0.0, 0.0, m]])}, idx)["observation.state"]
        torch.testing.assert_close(out[0, 3], torch.tensor(0.0), atol=1e-6, rtol=0.0)

    def test_nonzero_std_is_bit_identical_to_legacy(self):
        mean = [0.0, 1.0, 2.0, 3.0]
        std = [1.0, 2.0, 3.0, 4.0]
        norm = Normalize(self.features, self.norm_map, per_dataset_stats=self._stats(mean, std))
        idx = torch.zeros(2, dtype=torch.long)
        x = torch.randn(2, 4)
        out = norm({"observation.state": x}, idx)["observation.state"]
        legacy = (x - torch.tensor(mean)) / (torch.tensor(std) + EPS)
        assert torch.equal(out, legacy)  # the guard is a strict no-op for std >= EPS

    def test_unnormalize_round_trips_on_zero_std_dim(self):
        m = 0.761
        stats = self._stats([0.0, 1.0, 2.0, m], [1.0, 2.0, 3.0, 0.0])
        norm = Normalize(self.features, self.norm_map, per_dataset_stats=stats)
        unnorm = Unnormalize(self.features, self.norm_map, per_dataset_stats=stats)
        idx = torch.zeros(1, dtype=torch.long)
        original = torch.tensor([[1.0, 2.0, 3.0, 0.5]])  # dim 3 deviates from the "constant" m
        z = norm({"observation.state": original}, idx)["observation.state"]
        rec = unnorm({"observation.state": z}, idx)["observation.state"]
        torch.testing.assert_close(rec, original, atol=1e-5, rtol=1e-4)

    def test_min_max_zero_range_is_bounded_and_round_trips(self):
        features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
        norm_map = {"ACTION": NormalizationMode.MIN_MAX}
        c = 2.0  # dim 3 is constant: min == max == c (a zero range)
        stats = [
            {"action": {"min": torch.tensor([-1.0, -1.0, -1.0, c]), "max": torch.tensor([1.0, 1.0, 1.0, c])}}
        ]
        norm = Normalize(features, norm_map, per_dataset_stats=stats)
        unnorm = Unnormalize(features, norm_map, per_dataset_stats=stats)
        idx = torch.zeros(1, dtype=torch.long)
        x = torch.tensor([[0.0, 0.0, 0.0, 5.0]])  # dim 3 = 5 deviates from the constant c
        z = norm({"action": x}, idx)["action"]
        assert z[0, 3].abs().item() < 100.0  # bounded; without the snap ~ (5 - 2)/EPS ~ 3e8
        rec = unnorm({"action": z}, idx)["action"]
        torch.testing.assert_close(rec, x, atol=1e-4, rtol=1e-4)


class TestSnapWarning:
    """The guard warns loudly — but only when the snap actually changed the result
    (a value deviating from a dim the stats call constant). A truly-constant dim is silent."""

    features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,))}
    norm_map = {"STATE": NormalizationMode.MEAN_STD}

    @pytest.fixture(autouse=True)
    def _clear_snap_warned(self):
        # The once-per-process dedup set is module-global; clear it around each case.
        _WARNED_SNAP_KEYS.clear()
        yield
        _WARNED_SNAP_KEYS.clear()

    def _norm(self, mean, std):
        stats = [{"observation.state": {"mean": torch.tensor(mean), "std": torch.tensor(std)}}]
        return Normalize(self.features, self.norm_map, per_dataset_stats=stats)

    def test_silent_when_value_matches_constant(self, caplog):
        m = 0.761
        norm = self._norm([0.0, 0.0, 0.0, m], [1.0, 1.0, 1.0, 0.0])
        idx = torch.zeros(1, dtype=torch.long)
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.tensor([[0.0, 0.0, 0.0, m]])}, idx)
        assert not any("zero-variance guard" in r.getMessage() for r in caplog.records)

    def test_warns_once_on_deviation(self, caplog):
        m = 0.761
        norm = self._norm([0.0, 0.0, 0.0, m], [1.0, 1.0, 1.0, 0.0])
        idx = torch.zeros(1, dtype=torch.long)
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.zeros(1, 4)}, idx)  # dim 3 deviates from m
        msgs = [r.getMessage() for r in caplog.records if "zero-variance guard" in r.getMessage()]
        assert len(msgs) == 1
        assert "observation.state" in msgs[0]
        assert "dim(s)=[3]" in msgs[0]

    def test_deduped_across_steps(self, caplog):
        m = 0.761
        norm = self._norm([0.0, 0.0, 0.0, m], [1.0, 1.0, 1.0, 0.0])
        idx = torch.zeros(1, dtype=torch.long)
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.zeros(1, 4)}, idx)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.zeros(1, 4)}, idx)  # same (feature,dim) offender again
        assert not any("zero-variance guard" in r.getMessage() for r in caplog.records)

    def test_silent_when_no_zero_variance_dim(self, caplog):
        norm = self._norm([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
        idx = torch.zeros(1, dtype=torch.long)
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.full((1, 4), 100.0)}, idx)  # large value, but std > 0
        assert not any("zero-variance guard" in r.getMessage() for r in caplog.records)
        assert norm._snapping_possible() is False

    def test_rewarns_only_on_larger_deviation(self, caplog):
        m = 0.761
        norm = self._norm([0.0, 0.0, 0.0, m], [1.0, 1.0, 1.0, 0.0])
        idx = torch.zeros(1, dtype=torch.long)
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.zeros(1, 4)}, idx)  # |0 - m| = m
        assert sum("zero-variance guard" in r.getMessage() for r in caplog.records) == 1
        caplog.clear()
        bigger = torch.zeros(1, 4)
        bigger[0, 3] = -10.0  # |-10 - m| ~ 10.76 > m -> re-warns
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": bigger}, idx)
        assert sum("zero-variance guard" in r.getMessage() for r in caplog.records) == 1
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            norm({"observation.state": torch.zeros(1, 4)}, idx)  # |dev|=m < 10.76 -> suppressed
        assert not any("zero-variance guard" in r.getMessage() for r in caplog.records)

    def test_min_max_zero_range_warns_on_deviation(self, caplog):
        features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
        norm_map = {"ACTION": NormalizationMode.MIN_MAX}
        c = 2.0
        stats = [
            {"action": {"min": torch.tensor([-1.0, -1.0, -1.0, c]), "max": torch.tensor([1.0, 1.0, 1.0, c])}}
        ]
        norm = Normalize(features, norm_map, per_dataset_stats=stats)
        idx = torch.zeros(1, dtype=torch.long)
        # value == min == max -> no deviation -> silent
        with caplog.at_level(logging.WARNING):
            norm({"action": torch.tensor([[0.0, 0.0, 0.0, c]])}, idx)
        assert not any("zero-variance guard" in r.getMessage() for r in caplog.records)
        caplog.clear()
        # deviating value at the zero-range dim -> warns naming dim 3
        with caplog.at_level(logging.WARNING):
            norm({"action": torch.tensor([[0.0, 0.0, 0.0, 5.0]])}, idx)
        msgs = [r.getMessage() for r in caplog.records if "zero-variance guard" in r.getMessage()]
        assert len(msgs) == 1 and "dim(s)=[3]" in msgs[0] and "action" in msgs[0]
