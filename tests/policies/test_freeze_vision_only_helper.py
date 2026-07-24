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

"""Unit tests for ``freeze_policy_level_params_for_vision_only``.

The helper backs the ``train_vision_encoder_only`` flag: it freezes every
policy-level (outer) parameter that is NOT part of the vision/video encoder,
while leaving the inner ``*WithExpertModel`` params untouched (their own
``set_requires_grad`` already configured the vision pathway) and preserving any
``motion_module`` (the pi05_mem RLDX video encoder's own temporal block, which
*is* part of the video encoder).
"""

from __future__ import annotations

import torch
from torch import nn

from opentau.policies.utils import freeze_policy_level_params_for_vision_only


class _FakeWithExpert(nn.Module):
    """Stands in for a ``*WithExpertModel``: a trainable vision tower + a frozen LLM."""

    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Linear(4, 4)
        self.multi_modal_projector = nn.Linear(4, 4)
        self.language_model = nn.Linear(4, 4)


class _FakeVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_module = nn.Linear(4, 4)


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.with_expert = _FakeWithExpert()
        self.video_encoder = _FakeVideoEncoder()
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)
        self.state_proj = nn.Linear(4, 4)


def _simulate_inner_set_requires_grad(with_expert: _FakeWithExpert) -> None:
    """Mimic the inner set_requires_grad: freeze the LLM, keep the vision pathway."""
    for p in with_expert.language_model.parameters():
        p.requires_grad = False


def test_freezes_outer_projections_only():
    model = _FakePolicy()
    _simulate_inner_set_requires_grad(model.with_expert)

    freeze_policy_level_params_for_vision_only(model, model.with_expert)

    # The outer projections are frozen ...
    assert not any(p.requires_grad for p in model.action_in_proj.parameters())
    assert not any(p.requires_grad for p in model.action_out_proj.parameters())
    assert not any(p.requires_grad for p in model.state_proj.parameters())


def test_preserves_inner_with_expert_requires_grad():
    model = _FakePolicy()
    _simulate_inner_set_requires_grad(model.with_expert)

    freeze_policy_level_params_for_vision_only(model, model.with_expert)

    # The vision pathway (already trainable) is untouched ...
    assert all(p.requires_grad for p in model.with_expert.vision_tower.parameters())
    assert all(p.requires_grad for p in model.with_expert.multi_modal_projector.parameters())
    # ... and the frozen LLM stays frozen (not accidentally re-enabled).
    assert not any(p.requires_grad for p in model.with_expert.language_model.parameters())


def test_motion_module_stays_trainable():
    """The RLDX video encoder's ``motion_module`` is part of the video encoder and
    must remain trainable even though it lives OUTSIDE the with-expert model."""
    model = _FakePolicy()
    _simulate_inner_set_requires_grad(model.with_expert)

    freeze_policy_level_params_for_vision_only(model, model.with_expert)

    assert all(p.requires_grad for p in model.video_encoder.motion_module.parameters())


def test_no_video_encoder_is_fine():
    """A policy without a video encoder / motion_module (e.g. pi0) still freezes
    exactly its outer projections."""

    class _PolicyNoVideo(nn.Module):
        def __init__(self):
            super().__init__()
            self.with_expert = _FakeWithExpert()
            self.action_in_proj = nn.Linear(4, 4)

    model = _PolicyNoVideo()
    _simulate_inner_set_requires_grad(model.with_expert)

    freeze_policy_level_params_for_vision_only(model, model.with_expert)

    assert all(p.requires_grad for p in model.with_expert.vision_tower.parameters())
    assert not any(p.requires_grad for p in model.action_in_proj.parameters())


def test_only_vision_pathway_trainable_end_to_end():
    """The invariant the flag promises: after the helper runs, every trainable
    param is either in the vision pathway or the motion_module."""
    model = _FakePolicy()
    _simulate_inner_set_requires_grad(model.with_expert)

    freeze_policy_level_params_for_vision_only(model, model.with_expert)

    allowed = {id(p) for p in model.with_expert.vision_tower.parameters()}
    allowed |= {id(p) for p in model.with_expert.multi_modal_projector.parameters()}
    allowed |= {id(p) for p in model.video_encoder.motion_module.parameters()}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert id(p) in allowed, f"{name} unexpectedly trainable"


def test_backward_updates_only_vision_pathway():
    """A gradient step touches only the still-trainable params."""
    model = _FakePolicy()
    _simulate_inner_set_requires_grad(model.with_expert)
    freeze_policy_level_params_for_vision_only(model, model.with_expert)

    x = torch.randn(2, 4)
    # Route x through the vision tower then the (frozen) action proj so the graph
    # spans both trainable and frozen params.
    y = model.action_in_proj(model.with_expert.vision_tower(x))
    y.sum().backward()

    assert model.with_expert.vision_tower.weight.grad is not None
    assert model.action_in_proj.weight.grad is None
    assert model.with_expert.language_model.weight.grad is None
