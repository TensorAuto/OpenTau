#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Per-(robot_type, control_mode) state/action projections for pi07_paligemma.

Covers the checkpoint-compat remap (the 5 forward/backward-compat cases), the
state-dict wiring through a stub policy, and the per-sample routing through the
inner model's `_embed_item` / `_apply_proj`.
"""

import pytest
import torch
from torch import nn

from opentau.policies.layers import PerGroupLinear
from opentau.policies.pretrained import PreTrainedPolicy, ProjectionRemapError

_rec = PreTrainedPolicy._reconcile_projection_tensor


# --- _reconcile_projection_tensor: the compat case table --------------------


def test_reconcile_legacy_single_linear_tiles_to_all_groups():
    """compat #1: a legacy nn.Linear weight (out, in) duplicates into D rows."""
    out, inn, d = 6, 4, 3
    legacy = torch.randn(out, inn)
    r = _rec("model.state_proj.weight", legacy, torch.empty(d, out, inn), None, None)
    assert r.shape == (d, out, inn)
    assert r.is_contiguous()
    for i in range(d):
        assert torch.equal(r[i], legacy)


def test_reconcile_legacy_bias_tiles():
    out, d = 5, 3
    r = _rec("model.state_proj.bias", torch.randn(out), torch.empty(d, out), None, None)
    assert r.shape == (d, out)


def test_reconcile_single_row_tiles_to_all_groups():
    out, inn, d = 6, 4, 3
    src = torch.randn(1, out, inn)
    r = _rec("model.action_in_proj.weight", src, torch.empty(d, out, inn), ["a"], ["a", "b", "c"])
    assert r.shape == (d, out, inn)
    for i in range(d):
        assert torch.equal(r[i], src[0])


def test_reconcile_same_order_is_identity():
    """compat #3: identical group set & order loads unchanged (same object)."""
    src = torch.randn(3, 6, 4)
    r = _rec("model.state_proj.weight", src, torch.empty(3, 6, 4), ["a", "b", "c"], ["a", "b", "c"])
    assert r is src


def test_reconcile_name_remap_reorder_and_add_new_group():
    """compat #5: existing groups carried by name; a new group copies row 0."""
    out, inn = 2, 2
    src = torch.stack([torch.zeros(out, inn), torch.ones(out, inn)])  # a -> 0s, b -> 1s
    r = _rec("model.state_proj.weight", src, torch.empty(3, out, inn), ["a", "b"], ["b", "a", "c"])
    assert torch.equal(r[0], src[1])  # b
    assert torch.equal(r[1], src[0])  # a
    assert torch.equal(r[2], src[0])  # c -> reference row 0
    assert r.is_contiguous()


def test_reconcile_downgrade_raises():
    """A per-group checkpoint into a non-per-group (flag-off) policy must raise."""
    with pytest.raises(ProjectionRemapError, match="per-group axis"):
        _rec("model.state_proj.weight", torch.randn(3, 6, 4), torch.empty(6, 4), ["a", "b", "c"], None)


def test_reconcile_unknown_ordering_multirow_raises():
    with pytest.raises(ProjectionRemapError, match="ordering is unknown"):
        _rec("model.state_proj.weight", torch.randn(2, 6, 4), torch.empty(3, 6, 4), None, None)


def test_reconcile_flag_off_normal_load_is_identity():
    src = torch.randn(6, 4)
    r = _rec("model.state_proj.weight", src, torch.empty(6, 4), None, None)
    assert r is src


def test_reconcile_trailing_shape_mismatch_raises():
    with pytest.raises(ProjectionRemapError, match="incompatible"):
        _rec(
            "model.state_proj.weight",
            torch.randn(3, 9, 4),
            torch.empty(3, 6, 4),
            ["a", "b", "c"],
            ["a", "b", "c"],
        )


# --- _remap_..._in_state_dict: wiring through a stub policy ------------------


def _make_stub_policy(*, num_groups, per_group, dataset_names):
    """Minimal PreTrainedPolicy carrying the three (Per)GroupLinear projections.

    Mirrors the `_DummyPolicy` harness in test_normalize_per_dataset.py.
    """
    from opentau.configs.policies import PreTrainedConfig

    class _Cfg(PreTrainedConfig):
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

    class _Stub(PreTrainedPolicy):
        config_class = _Cfg
        name = "stub-per-group-proj"
        _PER_GROUP_PROJECTION_PREFIXES = ("state_proj.", "action_in_proj.", "action_out_proj.")

        def __init__(self, config):
            super().__init__(config)

            def mk(i, o):
                return PerGroupLinear(i, o, num_groups=num_groups) if per_group else nn.Linear(i, o)

            self.state_proj = mk(4, 6)
            self.action_in_proj = mk(4, 8)
            self.action_out_proj = mk(8, 4)

        def get_optim_params(self):
            return {}

        def reset(self):
            pass

        def forward(self, batch):
            raise NotImplementedError

        def select_action(self, batch):
            raise NotImplementedError

    cfg = _Cfg()
    cfg.dataset_names = list(dataset_names) if dataset_names is not None else None
    cfg.dataset_to_norm_index = None
    return _Stub(cfg)


def test_remap_tiles_legacy_state_dict_into_per_group_policy():
    """compat #1 end-to-end: a single-Linear checkpoint loads into a flag-on policy."""
    legacy = _make_stub_policy(num_groups=1, per_group=False, dataset_names=["a"])
    target = _make_stub_policy(num_groups=3, per_group=True, dataset_names=["a", "b", "c"])
    sd = dict(legacy.state_dict())  # 2-D projection weights
    target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a"])
    missing, unexpected = target.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    for i in range(3):
        assert torch.equal(target.state_proj.weight[i], legacy.state_proj.weight)
        assert torch.equal(target.action_out_proj.weight[i], legacy.action_out_proj.weight)


def test_remap_name_remaps_existing_and_adds_new_group():
    """compat #5 end-to-end: reorder + add a new group."""
    src = _make_stub_policy(num_groups=2, per_group=True, dataset_names=["a", "b"])
    target = _make_stub_policy(num_groups=3, per_group=True, dataset_names=["b", "a", "c"])
    sd = dict(src.state_dict())
    target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a", "b"])
    target.load_state_dict(sd, strict=True)
    assert torch.equal(target.state_proj.weight[0], src.state_proj.weight[1])  # b
    assert torch.equal(target.state_proj.weight[1], src.state_proj.weight[0])  # a
    assert torch.equal(target.state_proj.weight[2], src.state_proj.weight[0])  # c -> row 0


def test_remap_same_groups_round_trips():
    """compat #3 end-to-end: identical groups load unchanged."""
    src = _make_stub_policy(num_groups=2, per_group=True, dataset_names=["a", "b"])
    target = _make_stub_policy(num_groups=2, per_group=True, dataset_names=["a", "b"])
    sd = dict(src.state_dict())
    target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a", "b"])
    target.load_state_dict(sd, strict=True)
    torch.testing.assert_close(target.state_proj.weight, src.state_proj.weight)


def test_remap_downgrade_raises():
    src = _make_stub_policy(num_groups=2, per_group=True, dataset_names=["a", "b"])
    target = _make_stub_policy(num_groups=1, per_group=False, dataset_names=["a"])
    sd = dict(src.state_dict())
    with pytest.raises(ProjectionRemapError):
        target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a", "b"])


def test_remap_noop_when_policy_declares_no_prefixes():
    """A policy that hasn't opted in (empty prefixes) is untouched."""
    stub = _make_stub_policy(num_groups=1, per_group=False, dataset_names=["a"])
    stub._PER_GROUP_PROJECTION_PREFIXES = ()  # simulate a non-opted-in policy
    sd = dict(stub.state_dict())
    before = {k: v.clone() for k, v in sd.items()}
    stub._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=None)
    for k in sd:
        assert torch.equal(sd[k], before[k])


# --- inner-model threading: group_index reaches the projection --------------


def test_apply_proj_dispatches_on_module_type():
    """_apply_proj passes the index only to PerGroupLinear; other callables get x."""
    from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
        PI07PaligemmaLowLevelFlowMatching as Model,
    )

    x = torch.randn(2, 3)
    idx = torch.tensor([0, 1], dtype=torch.long)
    pgl = PerGroupLinear(3, 3, num_groups=2)
    lin = nn.Linear(3, 3)
    torch.testing.assert_close(Model._apply_proj(pgl, x, idx), pgl(x, idx))
    torch.testing.assert_close(Model._apply_proj(lin, x, idx), lin(x))
    # The 1-arg lambda / Identity monkeypatch contract: called as proj(x).
    assert torch.equal(Model._apply_proj(lambda z: z * 2, x, idx), x * 2)


def test_embed_item_routes_state_projection_per_group():
    """`_embed_item` applies the per-sample group row to the state projection."""
    from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
        ContextItem,
        PI07PaligemmaLowLevelFlowMatching,
    )

    model = object.__new__(PI07PaligemmaLowLevelFlowMatching)
    nn.Module.__init__(model)  # set up _modules so we can attach a submodule
    state_dim, h = 4, 5
    pgl = PerGroupLinear(state_dim, h, num_groups=2)
    with torch.no_grad():
        pgl.weight[0].zero_()  # group 0 -> zeros
        pgl.bias[0].zero_()
        pgl.weight[1].fill_(1.0)  # group 1 -> sum over the (all-ones) inputs
        pgl.bias[1].zero_()
    model.state_proj = pgl

    b, t = 2, 1
    item = ContextItem(
        data=torch.ones(b, t, state_dim),
        item_type="state",
        pad_mask=torch.ones(b, t, dtype=torch.bool),
    )
    idx = torch.tensor([0, 1], dtype=torch.long)
    emb, mask = model._embed_item(item, group_index=idx)
    emb = emb.float()
    assert emb.shape == (b, t, h)
    assert torch.allclose(emb[0], torch.zeros(t, h))
    assert torch.allclose(emb[1], torch.full((t, h), float(state_dim)))
    assert torch.equal(mask, item.pad_mask)
