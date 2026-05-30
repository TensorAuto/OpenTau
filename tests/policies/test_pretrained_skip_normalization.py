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

"""Cross-policy tests for the ``skip_normalization_weights`` knob on
:class:`~opentau.configs.policies.PreTrainedConfig`.

Pins three guarantees:

1. Every policy config in the registry inherits the field with a ``False``
   default — so adding a new policy without explicitly opting in cannot
   accidentally start stripping buffers at load time.
2. The field is keyword-settable on every config — catches an accidental
   override that shadows the parent field or renames the keyword.
3. The shared strip / inf-guard helpers on
   :class:`~opentau.policies.pretrained.PreTrainedPolicy` behave as the
   per-policy ``from_pretrained`` overrides expect — single source of
   truth for the integration contract.
"""

import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi05.configuration_pi05 import PI05Config, PI05ContinuousStateConfig
from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig as PI07HighLevelConfig,
)
from opentau.policies.pi07.low_level.configuration_pi07_low_level import PI07LowLevelConfig
from opentau.policies.pi07_paligemma.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig as PI07PaligemmaHighLevelConfig,
)
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)
from opentau.policies.pretrained import PreTrainedPolicy, is_norm_buffer_key
from opentau.policies.value.configuration_value import ValueConfig

# Parallel list of (cls, display_id) — display IDs disambiguate the two
# ``PI07HighLevelPlannerConfig`` classes (one each in ``pi07`` and
# ``pi07_paligemma``, same class name) which would otherwise render as
# ``PI07HighLevelPlannerConfig0`` / ``PI07HighLevelPlannerConfig1`` under
# pytest's default ``ids=lambda c: c.__name__`` and obscure which sibling
# a failure came from.
_POLICY_CONFIG_CASES = [
    (PI0Config, "PI0Config"),
    (PI05Config, "PI05Config"),
    (PI05ContinuousStateConfig, "PI05ContinuousStateConfig"),
    (PI05MemConfig, "PI05MemConfig"),
    (PI06Config, "PI06Config"),
    (PI07LowLevelConfig, "PI07LowLevelConfig"),
    (PI07HighLevelConfig, "PI07HighLevelConfig"),
    (PI07PaligemmaLowLevelConfig, "PI07PaligemmaLowLevelConfig"),
    (PI07PaligemmaHighLevelConfig, "PI07PaligemmaHighLevelConfig"),
    (ValueConfig, "ValueConfig"),
]
ALL_POLICY_CONFIGS = [c for c, _ in _POLICY_CONFIG_CASES]
_POLICY_CONFIG_IDS = [name for _, name in _POLICY_CONFIG_CASES]


@pytest.mark.parametrize("config_cls", ALL_POLICY_CONFIGS, ids=_POLICY_CONFIG_IDS)
def test_inherited_default_is_false(config_cls):
    """Every policy config inherits ``skip_normalization_weights = False``
    from :class:`PreTrainedConfig`. A regression here means a subclass
    accidentally shadowed the parent field or the parent's default flipped.
    """
    cfg = config_cls()
    assert cfg.skip_normalization_weights is False


@pytest.mark.parametrize("config_cls", ALL_POLICY_CONFIGS, ids=_POLICY_CONFIG_IDS)
def test_can_be_set_true(config_cls):
    """The field is keyword-settable on every policy config. Catches a
    draccus-side renaming, a subclass overriding the field with a
    different type, or the parent field being made private.
    """
    cfg = config_cls(skip_normalization_weights=True)
    assert cfg.skip_normalization_weights is True


def _make_fake_module_with_normalize_buffers(
    *, stats: dict | None
) -> tuple[nn.Module, frozenset[str], frozenset[str]]:
    """Build a small ``nn.Module`` that holds real
    :py:class:`Normalize` / :py:class:`Unnormalize` submodules under the
    canonical attribute names used by every policy. Returns the module
    plus the two key sets the strip helper is expected to act on.
    """
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    parent = nn.Module()
    # New per-dataset normalization API (#336): pass a list of stat dicts via
    # ``per_dataset_stats=`` (or just ``num_datasets=`` to leave buffers at
    # the inf sentinel for the inf-guard test).
    norm_kwargs = {"per_dataset_stats": [stats]} if stats is not None else {"num_datasets": 1}
    parent.normalize_inputs = Normalize(features, norm_map, **norm_kwargs)
    parent.unnormalize_outputs = Unnormalize(features, norm_map, **norm_kwargs)
    # An unrelated parameter that the strip must NOT touch.
    parent.unrelated_layer = nn.Linear(2, 2)

    full_sd = parent.state_dict()
    normalize_keys = frozenset(k for k in full_sd if k.startswith("normalize_inputs."))
    unnormalize_keys = frozenset(k for k in full_sd if k.startswith("unnormalize_outputs."))
    assert normalize_keys, "test fixture: no normalize_inputs.* keys in state_dict"
    assert unnormalize_keys, "test fixture: no unnormalize_outputs.* keys in state_dict"
    return parent, normalize_keys, unnormalize_keys


class _FakeConfig:
    """Stand-in for :class:`PreTrainedConfig` for tests that exercise just
    the strip helper — the helper only reads ``skip_normalization_weights``
    and writes it back, so a duck-typed object is enough and avoids
    instantiating a concrete (abstract-method-requiring) subclass.
    """

    def __init__(self, skip_normalization_weights: bool):
        self.skip_normalization_weights = skip_normalization_weights


class TestStripNormalizationBuffersFromStateDict:
    """Integration tests for the shared strip helper used by every policy."""

    def test_no_op_when_flag_off(self):
        """Default path: flag is off ⇒ helper returns the dict unchanged
        and an empty stripped-keys set. The config flag is not mutated.
        """
        parent, _, _ = _make_fake_module_with_normalize_buffers(stats=None)
        sd = dict(parent.state_dict())
        original_sd = dict(sd)
        config = _FakeConfig(skip_normalization_weights=False)

        out_sd, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert out_sd is sd, "no-op path must return the input dict unchanged (no copy)"
        assert dict(out_sd) == original_sd
        assert stripped == frozenset()
        assert config.skip_normalization_weights is False  # unchanged

    def test_strips_only_normalize_buffer_keys_and_resets_flag(self):
        """Flag on with real buffer keys ⇒ helper drops every
        ``normalize_*`` / ``unnormalize_*`` key, leaves unrelated keys
        alone, and resets the flag to False (one-shot)."""
        parent, normalize_keys, unnormalize_keys = _make_fake_module_with_normalize_buffers(
            stats={
                "observation.state": {
                    "mean": np.zeros(4, dtype=np.float32),
                    "std": np.ones(4, dtype=np.float32),
                },
                "action": {
                    "mean": np.zeros(3, dtype=np.float32),
                    "std": np.ones(3, dtype=np.float32),
                },
            }
        )
        sd = dict(parent.state_dict())
        unrelated_keys = frozenset(k for k in sd if k.startswith("unrelated_layer."))
        assert unrelated_keys, "test fixture: no unrelated_layer.* keys"
        config = _FakeConfig(skip_normalization_weights=True)

        out_sd, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert stripped == normalize_keys | unnormalize_keys
        assert set(out_sd.keys()) == set(sd.keys()) - stripped
        # Unrelated weights survive.
        for key in unrelated_keys:
            assert key in out_sd
            assert torch.equal(out_sd[key], sd[key])
        # One-shot reset of the config flag.
        assert config.skip_normalization_weights is False

    def test_warning_when_flag_set_but_no_keys_match(self, caplog):
        """Flag on with no matching keys in the dict ⇒ helper emits a
        WARNING (not an INFO with "dropped 0 keys") so a user who flipped
        the flag *expecting* the source-mixture stats to be voided sees
        that the flag was dead this load."""
        config = _FakeConfig(skip_normalization_weights=True)
        sd: dict[str, torch.Tensor] = {"some.other.weight": torch.zeros(3)}

        with caplog.at_level(logging.WARNING):
            out_sd, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert stripped == frozenset()
        assert out_sd == sd  # unchanged dict
        assert config.skip_normalization_weights is False  # still consumed
        assert any("had no effect" in rec.getMessage() for rec in caplog.records), (
            f"expected a WARNING mentioning 'had no effect'; got {[r.getMessage() for r in caplog.records]!r}"
        )

    def test_info_lists_dropped_count(self, caplog):
        """Flag on with matching keys ⇒ helper logs an INFO listing the
        dropped key count so operators can grep the startup log."""
        parent, normalize_keys, unnormalize_keys = _make_fake_module_with_normalize_buffers(stats=None)
        sd = dict(parent.state_dict())
        config = _FakeConfig(skip_normalization_weights=True)

        with caplog.at_level(logging.INFO):
            _, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(sd, config)

        assert len(stripped) == len(normalize_keys | unnormalize_keys)
        assert any(f"dropped {len(stripped)} saved" in rec.getMessage() for rec in caplog.records), (
            "expected an INFO line listing the dropped key count; "
            f"got {[r.getMessage() for r in caplog.records]!r}"
        )

    def test_main_process_gates_logging(self, caplog):
        """``is_main_process=False`` ⇒ no log lines, even when keys are
        dropped. Prevents distributed runs from duplicating the message
        on every rank."""
        parent, _, _ = _make_fake_module_with_normalize_buffers(stats=None)
        sd = dict(parent.state_dict())
        config = _FakeConfig(skip_normalization_weights=True)

        with caplog.at_level(logging.INFO):
            _, stripped = PreTrainedPolicy._strip_normalization_buffers_from_state_dict(
                sd, config, is_main_process=False
            )

        assert stripped, "fixture must produce at least one stripped key"
        assert not any("skip_normalization_weights" in rec.getMessage() for rec in caplog.records), (
            "no skip_normalization_weights log lines should appear on non-main ranks"
        )


class TestAssertNormalizeBuffersInitialized:
    """Integration tests for the post-load inf-buffer guard."""

    def test_no_op_when_stripped_keys_empty(self):
        """``stripped_keys`` empty ⇒ guard returns without inspecting the
        model. This is the default-load path; running the inf-check
        unconditionally would crash bare-config flows that legitimately
        load weights before ``dataset_stats`` arrives.
        """
        parent, _, _ = _make_fake_module_with_normalize_buffers(stats=None)
        # parent still has inf buffers — but stripped_keys is empty, so
        # the guard must NOT raise.
        PreTrainedPolicy._assert_normalize_buffers_initialized(parent, stripped_keys=frozenset())

    def test_raises_when_inf_buffers_remain_after_strip(self):
        """``stripped_keys`` non-empty + at least one Normalize buffer at
        ``inf`` ⇒ ValueError pointing the user at ``per_dataset_stats``
        (the actual fix), not at the misleading "use a pretrained model"
        assertion that ``Normalize.forward`` would raise on the next
        batch.
        """
        parent, normalize_keys, _ = _make_fake_module_with_normalize_buffers(stats=None)

        with pytest.raises(ValueError, match=r"skip_normalization_weights=True requires `per_dataset_stats`"):
            PreTrainedPolicy._assert_normalize_buffers_initialized(parent, stripped_keys=normalize_keys)

    def test_passes_when_stats_passed_through(self):
        """``stripped_keys`` non-empty but ``dataset_stats`` did fill in
        the buffers ⇒ no inf params ⇒ guard returns silently. This is
        the success path the strip is designed for.
        """
        parent, _, unnormalize_keys = _make_fake_module_with_normalize_buffers(
            stats={
                "observation.state": {
                    "mean": np.zeros(4, dtype=np.float32),
                    "std": np.ones(4, dtype=np.float32),
                },
                "action": {
                    "mean": np.zeros(3, dtype=np.float32),
                    "std": np.ones(3, dtype=np.float32),
                },
            }
        )
        # No exception raised.
        PreTrainedPolicy._assert_normalize_buffers_initialized(parent, stripped_keys=unnormalize_keys)


# ---------------------------------------------------------------------------
# Multi-head (per-(robot_type, control_mode)) skip round-trip.
#
# The helper tests above pin the strip / inf-guard in isolation, but only
# with single-row buffers. This section pins the interaction introduced by
# the per-(robot_type, control_mode) normalization heads (#347): stacked
# ``(D, *feat_shape)`` Normalize/Unnormalize buffers, and loading a
# checkpoint whose saved heads number N into a fresh policy whose finetuning
# mixture has M heads with ``skip_normalization_weights=True`` and N != M.
#
# That head-count change is exactly the documented use case for the knob
# (finetuning a checkpoint whose saved stats were aggregated over a
# *different* dataset mixture). The strip is keyed by buffer *name*
# (`is_norm_buffer_key`), not shape, so it is oblivious to the stacked
# leading axis — these tests lock that in end-to-end, and guard the
# contrast: WITHOUT the strip, N != M is a hard `load_state_dict` error
# (the legacy promotion shim only rescues single-row -> (1, *feat), not an
# N -> M re-stack), which is *why* skip is the right knob here.
#
# These exercise the genuine `PreTrainedPolicy` helpers
# (`_promote_legacy_norm_buffers_in_state_dict`, `_strip_*`, `_inject_stats`,
# `_assert_normalize_buffers_initialized`, `_resolve_dataset_index`) on a
# real (backbone-free) policy subclass, replaying the normalization-relevant
# steps of a per-policy `_load_as_safetensor` override (see e.g.
# `PI0Policy._load_as_safetensor`). The base `_load_as_safetensor` does NOT
# strip — that lives in each override — so the replay is the faithful way to
# test the shared-helper contract without a multi-GB backbone.
# ---------------------------------------------------------------------------

# Per-head buffer values are tagged by a base float so a test can assert
# *which* mixture's stats ended up in the buffer. The checkpoint's sentinel
# base must never survive a skip-strip.
_CKPT_BASE = -999.0
_NEW_BASE = 100.0


def _build_head_stats(num_heads: int, base: float) -> list[dict]:
    """Per-head stats with row-distinct, base-tagged means (unit std).

    Head ``h`` gets ``mean = base + h`` across every state/action element, so
    a test can check both that the right *number* of rows landed and that the
    rows carry the *expected mixture's* values (not the other mixture's).
    """
    stats = []
    for h in range(num_heads):
        val = float(base + h)
        stats.append(
            {
                "observation.state": {
                    "mean": np.full((4,), val, dtype=np.float32),
                    "std": np.ones(4, dtype=np.float32),
                },
                "action": {
                    "mean": np.full((3,), val, dtype=np.float32),
                    "std": np.ones(3, dtype=np.float32),
                },
            }
        )
    return stats


def _make_multihead_policy(
    norm_keys: list[str],
    *,
    base: float,
    skip: bool,
    dataset_to_norm_index: dict[str, int] | None = None,
) -> PreTrainedPolicy:
    """Build a real ``PreTrainedPolicy`` with ``len(norm_keys)`` stacked norm
    heads, distinct per-head buffer values tagged by ``base``, plus a non-norm
    ``proj`` weight that must round-trip untouched through a skip-strip load.

    Carries all three of the common NORM_MODULE_NAMES a low-level policy
    attaches (``normalize_inputs`` over state, ``normalize_targets`` and
    ``unnormalize_outputs`` over action) so the strip / inject / assert walk
    multiple modules at once. Defined inline (à la
    ``test_normalize_per_dataset``) to use the genuine helpers without a real
    backbone.
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
        name = "dummy-skip-multihead-policy"

        def __init__(self, config, stats):
            super().__init__(config)
            in_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,))}
            out_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
            norm_map = {"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
            names = list(config.dataset_names)
            self.normalize_inputs = Normalize(
                in_features, norm_map, per_dataset_stats=stats, dataset_names=names
            )
            self.normalize_targets = Normalize(
                out_features, norm_map, per_dataset_stats=stats, dataset_names=names
            )
            self.unnormalize_outputs = Unnormalize(
                out_features, norm_map, per_dataset_stats=stats, dataset_names=names
            )
            # A non-norm weight: must survive a skip-strip load (the strip only
            # drops normalize_*/unnormalize_* buffers, never other params).
            self.proj = nn.Linear(4, 3)

        def get_optim_params(self):
            return {}

        def reset(self):
            pass

        def forward(self, batch):
            raise NotImplementedError

        def select_action(self, batch):
            raise NotImplementedError

    cfg = _DummyConfig()
    cfg.dataset_names = list(norm_keys)
    cfg.dataset_to_norm_index = dict(dataset_to_norm_index) if dataset_to_norm_index is not None else None
    cfg.skip_normalization_weights = skip
    return _DummyPolicy(cfg, _build_head_stats(len(norm_keys), base))


def _replay_norm_load(
    policy: PreTrainedPolicy, ckpt_state_dict: dict[str, torch.Tensor], *, strict: bool
) -> frozenset[str]:
    """Replay the normalization-relevant steps of a policy's
    ``_load_as_safetensor`` override against ``ckpt_state_dict``.

    Mirrors e.g. ``PI0Policy._load_as_safetensor`` for exactly the steps that
    touch Normalize/Unnormalize buffers — promote -> strip ->
    ``load_state_dict(effective_strict)`` -> assert — so this exercises the
    shared helper contract every override depends on. Policy-specific key
    transforms / linear-weight tiling are irrelevant to normalization and are
    omitted. Returns the ``stripped_keys`` frozenset.
    """
    sd = dict(ckpt_state_dict)
    policy._promote_legacy_norm_buffers_in_state_dict(sd)
    sd, stripped_keys = type(policy)._strip_normalization_buffers_from_state_dict(
        sd, policy.config, is_main_process=True
    )
    # Same formula as the real overrides: a fired strip forces non-strict so
    # the deliberately-dropped buffer keys do not surface as missing-key errors.
    effective_strict = strict and not stripped_keys
    policy.load_state_dict(sd, strict=effective_strict)
    type(policy)._assert_normalize_buffers_initialized(policy, stripped_keys=stripped_keys)
    return stripped_keys


class TestSkipNormalizationMultiHeadRoundTrip:
    """``skip_normalization_weights=True`` across per-(robot_type, control_mode)
    norm heads, including a head-count change between checkpoint and mixture."""

    @pytest.mark.parametrize(
        "ckpt_keys, new_keys",
        [
            (["a::1", "b::2", "c::3"], ["franka::joint", "ur5::ee"]),  # N=3 -> M=2 (shrink)
            (["a::1", "b::2"], ["x::1", "y::2", "z::3"]),  # N=2 -> M=3 (grow)
        ],
        ids=["shrink_3_to_2", "grow_2_to_3"],
    )
    def test_skip_strips_mismatched_head_count_and_keeps_new_stats(self, ckpt_keys, new_keys):
        """A checkpoint with N heads loads into an M-head (N != M) finetuning
        policy under ``skip_normalization_weights=True``: the strip drops the
        saved stacked buffers (no size-mismatch crash), the freshly-built
        M-head buffers keep the NEW mixture's values, the flag resets, and the
        name map reflects the new heads."""
        # Checkpoint: N heads tagged with the sentinel base.
        ckpt = _make_multihead_policy(ckpt_keys, base=_CKPT_BASE, skip=False)
        ckpt_sd = ckpt.state_dict()
        # Sanity: the saved norm buffers really carry the sentinel rows and the
        # N-head leading dim — i.e. this is a genuine stacked multi-head ckpt.
        ckpt_state_mean = ckpt_sd["normalize_inputs.buffer_observation_state.mean"]
        assert ckpt_state_mean.shape[0] == len(ckpt_keys)
        assert torch.isclose(ckpt_state_mean[0], torch.full((4,), _CKPT_BASE)).all()

        # New finetuning mixture: M heads, skip on.
        policy = _make_multihead_policy(new_keys, base=_NEW_BASE, skip=True)

        # strict=True from the caller: the fired strip must still force a clean
        # (non-strict) load rather than raising on the dropped buffer keys.
        stripped = _replay_norm_load(policy, ckpt_sd, strict=True)

        # The strip fired and touched only norm buffer keys.
        assert stripped, "expected the saved norm buffer keys to be stripped"
        assert all(is_norm_buffer_key(k) for k in stripped)
        # One-shot reset of the in-memory flag.
        assert policy.config.skip_normalization_weights is False

        # The freshly-built M-head buffers survive with the NEW mixture's
        # per-head values; the checkpoint's N-head sentinel rows are gone.
        for module_attr, buffer_attr, dim in [
            ("normalize_inputs", "buffer_observation_state", 4),
            ("normalize_targets", "buffer_action", 3),
            ("unnormalize_outputs", "buffer_action", 3),
        ]:
            mean = getattr(getattr(policy, module_attr), buffer_attr)["mean"]
            assert mean.shape[0] == len(new_keys), module_attr
            for h in range(len(new_keys)):
                torch.testing.assert_close(mean[h], torch.full((dim,), _NEW_BASE + h))
            assert not torch.isclose(mean, torch.full_like(mean, _CKPT_BASE)).any(), (
                f"{module_attr}: a checkpoint sentinel row leaked through the strip"
            )

        # Name map reflects the new heads, so inference routing targets them.
        assert policy._norm_key_to_index == {k: i for i, k in enumerate(new_keys)}

        # Non-norm weights DID round-trip from the checkpoint (proof the load
        # ran and only the norm buffers were withheld).
        torch.testing.assert_close(policy.proj.weight.detach(), ckpt.proj.weight.detach())

    def test_inject_stats_after_skip_refreshes_buffers_and_map(self):
        """``make_policy`` calls ``_inject_stats`` after the load; replicate
        that final step and confirm it is consistent on the multi-head skip
        path (cross-check passes, buffers hold the new values, map rebuilt)."""
        ckpt = _make_multihead_policy(["a::1", "b::2", "c::3"], base=_CKPT_BASE, skip=False)
        new_keys = ["franka::joint", "ur5::ee"]
        new_stats = _build_head_stats(len(new_keys), _NEW_BASE)
        policy = _make_multihead_policy(new_keys, base=_NEW_BASE, skip=True)

        _replay_norm_load(policy, ckpt.state_dict(), strict=True)
        # cfg.dataset_names was already set to new_keys at construction, so the
        # cross-check inside _inject_stats must agree (no reorder corruption).
        policy._inject_stats(new_stats, dataset_names=new_keys)

        mean = policy.normalize_inputs.buffer_observation_state["mean"]
        for h in range(len(new_keys)):
            torch.testing.assert_close(mean[h], torch.full((4,), _NEW_BASE + h))
        assert policy._norm_key_to_index == {k: i for i, k in enumerate(new_keys)}

    def test_non_skip_mismatched_head_count_raises_size_mismatch(self):
        """Without the skip knob, loading an N-head checkpoint into an M-head
        (N != M) policy is a hard ``load_state_dict`` size mismatch — the
        stacked buffer keys are present in both and their leading dims
        disagree. The mismatch raises regardless of ``strict`` (shape errors
        bypass strict), which is *why* ``skip_normalization_weights`` is the
        required knob when the mixture's head count changes."""
        ckpt = _make_multihead_policy(["a::1", "b::2", "c::3"], base=_CKPT_BASE, skip=False)
        policy = _make_multihead_policy(["x::1", "y::2"], base=_NEW_BASE, skip=False)

        with pytest.raises(RuntimeError, match="size mismatch"):
            _replay_norm_load(policy, ckpt.state_dict(), strict=False)

    def test_skip_roundtrip_then_resolves_new_heads_at_inference(self):
        """After a skip round-trip into a new mixture, inference-time
        ``_resolve_dataset_index`` routes the *new* (robot_type, control_mode)
        keys to the right rows, and the checkpoint's old keys are no longer
        routable — i.e. the rebuilt name map is live, not the checkpoint's."""
        ckpt = _make_multihead_policy(["old_a::m", "old_b::m", "old_c::m"], base=_CKPT_BASE, skip=False)
        new_keys = ["franka::joint", "ur5::ee"]
        policy = _make_multihead_policy(
            new_keys,
            base=_NEW_BASE,
            skip=True,
            dataset_to_norm_index={"franka_repo": 0, "ur5_repo": 1},
        )
        _replay_norm_load(policy, ckpt.state_dict(), strict=True)

        # Training-path index passthrough still works against the M heads.
        idx = policy._resolve_dataset_index({"dataset_index": torch.tensor([0, 1], dtype=torch.long)})
        assert torch.equal(idx.cpu(), torch.tensor([0, 1], dtype=torch.long))

        # (robot_type, control_mode) resolves to the NEW heads (order-swapped
        # to prove it is a real lookup, not positional).
        idx = policy._resolve_dataset_index(
            {
                "robot_type": ["ur5", "franka"],
                "control_mode": ["ee", "joint"],
                "observation.state": torch.zeros(2, 4),
            }
        )
        assert torch.equal(idx.cpu(), torch.tensor([1, 0], dtype=torch.long))

        # An old checkpoint key is gone — its head was replaced, so routing it
        # raises rather than silently picking a wrong row.
        with pytest.raises(ValueError, match="not in this policy's training set"):
            policy._resolve_dataset_index(
                {
                    "robot_type": ["old_a"],
                    "control_mode": ["m"],
                    "observation.state": torch.zeros(1, 4),
                }
            )
