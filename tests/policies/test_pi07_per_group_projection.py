#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Per-(robot_type, control_mode) state/action projections for primary pi07.

Ports the pi07_paligemma coverage (see ``test_per_group_projection.py``) to the
primary ``pi07`` low-level policy: the ``_apply_proj`` dispatch, the state-dict
remap wired through pi07's ``model.*`` projection prefixes, the per-sample
routing through the inner model's ``embed_suffix`` (pi07 embeds inline -- there
is no ``_embed_item``), and the train==eval head-consistency invariant the
per-group feature relies on (a given ``(robot_type, control_mode)`` resolves to
the same head whether tagged by the training ``dataset_index`` or by the eval
``(robot_type, control_mode)`` pair). All CPU-only -- no Gemma3 backbone.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opentau.policies.layers import PerGroupLinear
from opentau.policies.pretrained import PreTrainedPolicy, ProjectionRemapError

# --- a stub policy carrying pi07's nested `model.<proj>` projections ----------


def _make_pi07_stub_policy(*, num_groups, per_group, dataset_names):
    """Minimal PreTrainedPolicy whose projections live on ``self.model`` so the
    state-dict keys are ``model.state_proj.*`` etc. -- exactly pi07's layout and
    :attr:`PI07LowLevelPolicy._PER_GROUP_PROJECTION_PREFIXES`.

    Mirrors the ``_make_stub_policy`` harness in ``test_per_group_projection.py``
    but nests the projections one level deeper (under ``model``).
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
        name = "stub-pi07-per-group-proj"
        _PER_GROUP_PROJECTION_PREFIXES = (
            "model.state_proj.",
            "model.action_in_proj.",
            "model.action_out_proj.",
        )

        def __init__(self, config):
            super().__init__(config)

            def mk(i, o):
                return PerGroupLinear(i, o, num_groups=num_groups) if per_group else nn.Linear(i, o)

            self.model = nn.Module()
            self.model.state_proj = mk(4, 6)
            self.model.action_in_proj = mk(4, 8)
            self.model.action_out_proj = mk(8, 4)

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


# --- the policy opts in with the right prefixes -------------------------------


def test_pi07_policy_declares_per_group_prefixes():
    from opentau.policies.pi07.low_level.modeling_pi07_low_level import PI07LowLevelPolicy

    assert PI07LowLevelPolicy._PER_GROUP_PROJECTION_PREFIXES == (
        "model.state_proj.",
        "model.action_in_proj.",
        "model.action_out_proj.",
    )


# --- _remap_..._in_state_dict wired through pi07's prefixes -------------------


def test_pi07_remap_tiles_legacy_state_dict_into_per_group_policy():
    """compat #1 end-to-end: a single-Linear checkpoint loads into a flag-on policy."""
    legacy = _make_pi07_stub_policy(num_groups=1, per_group=False, dataset_names=["a"])
    target = _make_pi07_stub_policy(num_groups=3, per_group=True, dataset_names=["a", "b", "c"])
    sd = dict(legacy.state_dict())  # 2-D projection weights under model.*
    target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a"])
    missing, unexpected = target.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    for i in range(3):
        assert torch.equal(target.model.state_proj.weight[i], legacy.model.state_proj.weight)
        assert torch.equal(target.model.action_out_proj.weight[i], legacy.model.action_out_proj.weight)


def test_pi07_remap_name_remaps_existing_and_adds_new_group():
    """compat #5 end-to-end: reorder + add a new group (new group copies row 0)."""
    src = _make_pi07_stub_policy(num_groups=2, per_group=True, dataset_names=["a", "b"])
    target = _make_pi07_stub_policy(num_groups=3, per_group=True, dataset_names=["b", "a", "c"])
    sd = dict(src.state_dict())
    target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a", "b"])
    target.load_state_dict(sd, strict=True)
    assert torch.equal(target.model.state_proj.weight[0], src.model.state_proj.weight[1])  # b
    assert torch.equal(target.model.state_proj.weight[1], src.model.state_proj.weight[0])  # a
    assert torch.equal(target.model.state_proj.weight[2], src.model.state_proj.weight[0])  # c -> row 0


def test_pi07_remap_downgrade_raises():
    """A per-group checkpoint loaded into a flag-off pi07 policy must raise."""
    src = _make_pi07_stub_policy(num_groups=2, per_group=True, dataset_names=["a", "b"])
    target = _make_pi07_stub_policy(num_groups=1, per_group=False, dataset_names=["a"])
    sd = dict(src.state_dict())
    with pytest.raises(ProjectionRemapError):
        target._remap_per_group_projection_weights_in_state_dict(sd, old_dataset_names=["a", "b"])


# --- inner-model threading: group_index reaches the projection ---------------


def test_pi07_apply_proj_dispatches_on_module_type():
    """_apply_proj passes the index only to PerGroupLinear; other callables get x."""
    from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
        PI07LowLevelFlowMatching as Model,
    )

    x = torch.randn(2, 3)
    idx = torch.tensor([0, 1], dtype=torch.long)
    pgl = PerGroupLinear(3, 3, num_groups=2)
    lin = nn.Linear(3, 3)
    torch.testing.assert_close(Model._apply_proj(pgl, x, idx), pgl(x, idx))
    torch.testing.assert_close(Model._apply_proj(lin, x, idx), lin(x))
    # The 1-arg lambda / Identity monkeypatch contract: called as proj(x).
    assert torch.equal(Model._apply_proj(lambda z: z * 2, x, idx), x * 2)


def test_pi07_embed_suffix_routes_action_in_proj_per_group(monkeypatch):
    """`embed_suffix` applies the per-sample group row to ``action_in_proj``.

    pi07 has no ``_embed_item``; the action projection lives inside
    ``embed_suffix`` (shared by both the training ``forward`` and the inference
    ``denoise_step``). Build a backbone-free model carrying only the suffix's
    own submodules and confirm ``group_index`` selects the right row.
    """
    import opentau.policies.pi07.low_level.modeling_pi07_low_level as mod
    from opentau.policies.pi07.low_level.modeling_pi07_low_level import PI07LowLevelFlowMatching

    # Keep everything float32 so the float32 time_mlp doesn't clash with the
    # default bf16 _preferred_dtype the suffix path casts inputs to.
    monkeypatch.setattr(mod, "_preferred_dtype", lambda: torch.float32)

    model = object.__new__(PI07LowLevelFlowMatching)
    nn.Module.__init__(model)
    action_dim, proj_width, chunk = 4, 8, 3
    model.config = SimpleNamespace(proj_width=proj_width, chunk_size=chunk)

    pgl = PerGroupLinear(action_dim, proj_width, num_groups=2)
    with torch.no_grad():
        pgl.weight[0].zero_()  # group 0 -> zeros
        pgl.bias[0].zero_()
        pgl.weight[1].fill_(1.0)  # group 1 -> sum over the (all-ones) inputs
        pgl.bias[1].zero_()
    model.action_in_proj = pgl
    model.time_mlp_in = nn.Linear(proj_width, proj_width)
    model.time_mlp_out = nn.Linear(proj_width, proj_width)

    noisy = torch.ones(2, chunk, action_dim)
    timestep = torch.zeros(2, chunk)
    idx = torch.tensor([0, 1], dtype=torch.long)
    embs, pad_masks, _att_masks, _adarms = model.embed_suffix(noisy, timestep, group_index=idx)

    assert embs.shape == (2, chunk, proj_width)
    assert torch.allclose(embs[0], torch.zeros(chunk, proj_width))  # routed to group 0
    assert torch.allclose(embs[1], torch.full((chunk, proj_width), float(action_dim)))  # group 1
    assert pad_masks.shape == (2, chunk)


# --- train==eval head consistency (the property the feature relies on) --------


def test_pi07_train_and_eval_resolve_same_head_and_projection_row():
    """The same (robot_type, control_mode) selects the same head in both paths.

    Training tags each sample with the norm-head ``dataset_index``; eval supplies
    ``(robot_type, control_mode)``. Both flow through the shared
    ``_resolve_dataset_index`` and must yield the same index -- and feeding that
    index to the per-group ``state_proj`` must pick the same projection row, so
    the head used at train time matches the head used at eval time.
    """
    # dataset_names ARE the norm keys "robot::control"; group 0 = panda::joint,
    # group 1 = ur5::ee.
    stub = _make_pi07_stub_policy(num_groups=2, per_group=True, dataset_names=["panda::joint", "ur5::ee"])

    state = torch.randn(2, 4)
    # Training-style batch: explicit per-sample norm-head row (sample 0 -> ur5::ee,
    # sample 1 -> panda::joint).
    train_batch = {"state": state, "dataset_index": torch.tensor([1, 0], dtype=torch.long)}
    # Eval-style batch: the matching (robot_type, control_mode) pairs, same order.
    eval_batch = {
        "state": state,
        "robot_type": ["ur5", "panda"],
        "control_mode": ["ee", "joint"],
    }

    i_train = stub._resolve_dataset_index(train_batch)
    i_eval = stub._resolve_dataset_index(eval_batch)

    assert torch.equal(i_train, torch.tensor([1, 0]))
    assert torch.equal(i_train, i_eval)

    # And the resolved index drives the projection identically across paths.
    x = torch.randn(2, 4)
    out_train = stub.model.state_proj(x, i_train)
    out_eval = stub.model.state_proj(x, i_eval)
    torch.testing.assert_close(out_train, out_eval)
    # Per-sample distinct rows actually fired (sanity: the two samples differ).
    assert not torch.allclose(out_train[0], out_train[1])
