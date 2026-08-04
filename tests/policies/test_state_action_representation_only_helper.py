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

"""Unit tests for the ``train_state_action_representation_only`` helpers.

The flag trains only the dataset-specific state/action representation of an
otherwise-generic VLA checkpoint: the discrete-action embedding table plus its
logit head (inner, on the ``*WithExpertModel``) and the state/action projections
(outer, on the flow-matching module). Everything else — VLM, vision encoder,
multimodal projector, action expert, time MLPs, modality embeddings, the RLDX
``motion_module`` — is frozen.

These tests use hand-rolled fakes so they pin the helpers' *semantics* rather
than any one policy's wiring; the per-policy end-to-end coverage lives in
``test_state_action_representation_only_policies.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from opentau.policies.utils import (
    freeze_policy_level_params_for_state_action_representation_only,
    freeze_with_expert_params_for_state_action_representation_only,
    set_with_expert_train_mode_for_state_action_representation_only,
    validate_state_action_representation_only_config,
)


class _FakeWithExpert(nn.Module):
    """Stands in for a ``*WithExpertModel``: VLM + vision + expert + discrete heads."""

    def __init__(self, *, discrete: bool = True):
        super().__init__()
        self.vision_tower = nn.Linear(4, 4)
        self.multi_modal_projector = nn.Linear(4, 4)
        self.language_model = nn.Linear(4, 4)
        self.expert = nn.Linear(4, 4)
        self.dropout = nn.Dropout(0.5)
        if discrete:
            self.discrete_action_embedding = nn.Embedding(8, 4)
            self.da_head = nn.Linear(4, 8)


class _FakeVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_module = nn.Linear(4, 4)


class _FakePolicy(nn.Module):
    def __init__(self, *, discrete: bool = True, state_proj: bool = True):
        super().__init__()
        self.with_expert = _FakeWithExpert(discrete=discrete)
        self.video_encoder = _FakeVideoEncoder()
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)
        if state_proj:
            self.state_proj = nn.Linear(4, 4)
        # Frozen by the flag: hidden-size-only, fed by flow-matching time.
        self.time_mlp_in = nn.Linear(4, 4)
        self.time_mlp_out = nn.Linear(4, 4)
        self.modality_embedding = nn.Embedding(5, 4)


def _configure(model: _FakePolicy) -> None:
    """Run both halves of the flag, in the order the policies run them."""
    freeze_with_expert_params_for_state_action_representation_only(model.with_expert)
    freeze_policy_level_params_for_state_action_representation_only(model, model.with_expert)


# --------------------------------------------------------------------------- inner half


def test_with_expert_keeps_only_the_discrete_action_representation():
    with_expert = _FakeWithExpert()

    kept = freeze_with_expert_params_for_state_action_representation_only(with_expert)

    assert kept == ["da_head.bias", "da_head.weight", "discrete_action_embedding.weight"]
    assert all(p.requires_grad for p in with_expert.discrete_action_embedding.parameters())
    assert all(p.requires_grad for p in with_expert.da_head.parameters())
    for frozen in (with_expert.vision_tower, with_expert.language_model, with_expert.expert):
        assert not any(p.requires_grad for p in frozen.parameters())


def test_with_expert_freezes_the_multimodal_projector():
    """``freeze_vision_encoder`` freezes only the tower and leaves the projector
    trainable; the default-deny helper must cover the projector too."""
    with_expert = _FakeWithExpert()

    freeze_with_expert_params_for_state_action_representation_only(with_expert)

    assert not any(p.requires_grad for p in with_expert.multi_modal_projector.parameters())


def test_with_expert_without_discrete_modules_freezes_everything():
    """pi0 / cosmos3 have no discrete-action pathway: the wrapper ends fully frozen."""
    with_expert = _FakeWithExpert(discrete=False)

    kept = freeze_with_expert_params_for_state_action_representation_only(with_expert)

    assert kept == []
    assert not any(p.requires_grad for p in with_expert.parameters())


def test_with_expert_freeze_is_idempotent():
    with_expert = _FakeWithExpert()

    first = freeze_with_expert_params_for_state_action_representation_only(with_expert)
    second = freeze_with_expert_params_for_state_action_representation_only(with_expert)

    assert first == second


# --------------------------------------------------------------------------- outer half


def test_policy_level_keeps_only_the_state_action_projections():
    model = _FakePolicy()

    _configure(model)

    kept = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert kept == [
        "action_in_proj.bias",
        "action_in_proj.weight",
        "action_out_proj.bias",
        "action_out_proj.weight",
        "state_proj.bias",
        "state_proj.weight",
        "with_expert.da_head.bias",
        "with_expert.da_head.weight",
        "with_expert.discrete_action_embedding.weight",
    ]


def test_time_mlps_and_modality_embeddings_are_frozen():
    """The time MLPs are hidden-size-only and condition the (frozen) expert; the
    modality embeddings are not a state/action representation. Both stay frozen."""
    model = _FakePolicy()

    _configure(model)

    assert not any(p.requires_grad for p in model.time_mlp_in.parameters())
    assert not any(p.requires_grad for p in model.time_mlp_out.parameters())
    assert not any(p.requires_grad for p in model.modality_embedding.parameters())


def test_motion_module_is_frozen_unlike_the_vision_only_helper():
    """Regression guard: ``freeze_policy_level_params_for_vision_only`` carves the
    RLDX ``motion_module`` out by name so it keeps training. Inheriting that clause
    here would silently train a freshly-initialized temporal block in a run whose
    whole premise is that only the dataset-facing adapters move."""
    model = _FakePolicy()

    _configure(model)

    assert not any(p.requires_grad for p in model.video_encoder.motion_module.parameters())


def test_policy_without_state_proj_is_fine():
    """pi06 has no ``state_proj`` at all, and pi05 only builds one for
    ``state_type='continuous'``. A missing attribute is not an error."""
    model = _FakePolicy(state_proj=False)

    _configure(model)

    kept = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert "action_in_proj.weight" in kept
    assert not any(n.startswith("state_proj") for n in kept)


def test_inner_params_are_not_re_enabled_by_the_outer_helper():
    """The outer helper must skip everything the inner one already configured,
    rather than re-deriving it — otherwise the two halves can disagree."""
    model = _FakePolicy()
    freeze_with_expert_params_for_state_action_representation_only(model.with_expert)

    freeze_policy_level_params_for_state_action_representation_only(model, model.with_expert)

    assert all(p.requires_grad for p in model.with_expert.discrete_action_embedding.parameters())
    assert not any(p.requires_grad for p in model.with_expert.language_model.parameters())


def test_normalization_buffers_are_never_unfrozen():
    """``Normalize``/``Unnormalize`` register per-dataset statistics as
    ``nn.Parameter(requires_grad=False)``, so they show up in ``named_parameters()``
    under names containing 'state'/'actions'. Selecting the trainable set by NAME
    would turn normalization statistics into learned parameters — and the optimizer
    factory selects on ``requires_grad`` alone, so nothing downstream would catch it."""

    class _PolicyWithNormBuffers(_FakePolicy):
        def __init__(self):
            super().__init__()
            self.normalize_inputs = nn.Module()
            self.normalize_inputs.buffer_state = nn.Parameter(torch.zeros(4), requires_grad=False)
            self.normalize_targets = nn.Module()
            self.normalize_targets.buffer_actions = nn.Parameter(torch.zeros(4), requires_grad=False)

    model = _PolicyWithNormBuffers()

    _configure(model)

    assert not model.normalize_inputs.buffer_state.requires_grad
    assert not model.normalize_targets.buffer_actions.requires_grad


def test_backward_reaches_only_the_representation_params():
    model = _FakePolicy()
    _configure(model)

    x = torch.randn(2, 4)
    y = model.action_out_proj(model.with_expert.language_model(model.action_in_proj(x)))
    y.sum().backward()

    assert model.action_in_proj.weight.grad is not None
    assert model.action_out_proj.weight.grad is not None
    assert model.with_expert.language_model.weight.grad is None
    assert model.time_mlp_in.weight.grad is None


# --------------------------------------------------------------------------- train() pinning


def test_train_mode_pins_the_frozen_trunk_to_eval():
    with_expert = _FakeWithExpert()
    freeze_with_expert_params_for_state_action_representation_only(with_expert)

    with_expert.train(True)
    set_with_expert_train_mode_for_state_action_representation_only(with_expert, True)

    assert not with_expert.vision_tower.training
    assert not with_expert.language_model.training
    assert not with_expert.expert.training
    # The wrapper's own dropout fires inside the decoder loop, i.e. inside the frozen
    # trunk: it must be pinned to eval too, so `config.dropout` is a no-op under the flag.
    assert not with_expert.dropout.training
    # ... while the trained representation follows the requested mode.
    assert with_expert.discrete_action_embedding.training
    assert with_expert.da_head.training


def test_train_mode_false_leaves_everything_in_eval():
    with_expert = _FakeWithExpert()

    # Mirror the real call sequence: the wrapper's `train()` runs `super().train(mode)`
    # (which sets the wrapper's own flag) before delegating to the helper.
    with_expert.train(False)
    set_with_expert_train_mode_for_state_action_representation_only(with_expert, False)

    assert not any(m.training for m in with_expert.modules())


# --------------------------------------------------------------------------- config validation


@dataclass
class _FakeConfig:
    train_state_action_representation_only: bool = True
    train_expert_only: bool = False
    train_vision_encoder_only: bool = False
    knowledge_insulation: bool = False


def test_validator_is_a_noop_when_the_flag_is_off():
    cfg = _FakeConfig(train_state_action_representation_only=False, train_expert_only=True)

    validate_state_action_representation_only_config(cfg, policy_name="fake", has_discrete_actions=True)


@pytest.mark.parametrize("conflicting", ["train_expert_only", "train_vision_encoder_only"])
def test_validator_rejects_the_other_train_only_modes(conflicting):
    cfg = _FakeConfig(**{conflicting: True})

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_state_action_representation_only_config(cfg, policy_name="fake", has_discrete_actions=True)


def test_validator_warns_when_there_is_no_discrete_action_pathway(caplog):
    cfg = _FakeConfig()

    with caplog.at_level(logging.WARNING):
        validate_state_action_representation_only_config(cfg, policy_name="pi0", has_discrete_actions=False)

    assert "no discrete-action" in caplog.text
    assert "pi0" in caplog.text


def test_validator_does_not_warn_when_the_discrete_pathway_exists(caplog):
    cfg = _FakeConfig()

    with caplog.at_level(logging.WARNING):
        validate_state_action_representation_only_config(cfg, policy_name="pi05", has_discrete_actions=True)

    assert "no discrete-action" not in caplog.text


def test_validator_warns_under_knowledge_insulation(caplog):
    """KI splits the trainable set across two losses; it is a warning, not an error,
    so a CE-only or MSE-only run stays possible on purpose."""
    cfg = _FakeConfig(knowledge_insulation=True)

    with caplog.at_level(logging.WARNING):
        validate_state_action_representation_only_config(cfg, policy_name="pi05", has_discrete_actions=True)

    assert "knowledge_insulation" in caplog.text
    assert "CE loss" in caplog.text


def test_validator_does_not_warn_without_knowledge_insulation(caplog):
    cfg = _FakeConfig(knowledge_insulation=False)

    with caplog.at_level(logging.WARNING):
        validate_state_action_representation_only_config(cfg, policy_name="pi05", has_discrete_actions=True)

    assert "knowledge_insulation" not in caplog.text
