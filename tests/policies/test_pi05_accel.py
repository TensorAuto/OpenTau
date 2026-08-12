#!/usr/bin/env python

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

"""CPU coverage for the ``accel`` uncertainty proxy wired into pi05's Euler loop.

``tests/policies/test_accel.py`` pins the estimator itself. These tests pin the *wiring*:
that the meter sees the sampler's real per-step velocities in the right order, that the
frozen real-time-chunking rows are excluded, that the whole thing is inert when disabled,
and that the policy-level ``last_accel`` follows the re-plan cadence rather than the env
step cadence.

Built on the same lightweight ``object.__new__(PI05FlowMatching)`` shell the execution-
horizon tests use, so no PaliGemma weights are needed.
"""

import math

import pytest
import torch
from torch import nn

from opentau.policies.accel import AccelMeter
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import PI05FlowMatching, PI05Policy

MAX_STATE_DIM = 32
MAX_ACTION_DIM = 4
HIDDEN = 8


class _StubBackbone(nn.Module):
    """Minimal stand-in for ``PaliGemmaWithExpertModel`` — only the prefix pass is used."""

    def forward(self, *, inputs_embeds, **kwargs):
        batch = inputs_embeds[0].shape[0]
        return (torch.zeros(batch, 1, HIDDEN), None), []


def _config(chunk_size=4, n_action_steps=4, max_delay=0, num_steps=4):
    return PI05Config(
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        max_delay=max_delay,
        num_steps=num_steps,
        max_state_dim=MAX_STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
    )


def _flow_matching(cfg, velocities):
    """A ``PI05FlowMatching`` shell whose ``denoise_step`` replays a scripted velocity list.

    Scripting the velocities is what lets these tests assert an exact ``accel`` — the real
    denoise_step would need the full backbone and would make the expected value unknowable.
    """
    fm = object.__new__(PI05FlowMatching)
    nn.Module.__init__(fm)
    fm.config = cfg
    fm.paligemma_with_expert = _StubBackbone()

    def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
        batch = lang_tokens.shape[0]
        return (
            torch.zeros(batch, 1, HIDDEN),
            torch.ones(batch, 1, dtype=torch.bool),
            torch.zeros(batch, 1, dtype=torch.bool),
            None,
        )

    calls = {"n": 0}

    def denoise_step(prefix_pad_masks, past_key_values, x_t, time):
        v = velocities[calls["n"]]
        calls["n"] += 1
        return v.expand_as(x_t).clone() if v.shape != x_t.shape else v.clone()

    fm.embed_prefix = embed_prefix
    fm.denoise_step = denoise_step
    fm._denoise_calls = calls
    return fm


def _sample(fm, cfg, *, batch=1, accel=None, delay=0):
    return PI05FlowMatching.sample_actions(
        fm,
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros(batch, 1, dtype=torch.long),
        lang_masks=torch.ones(batch, 1, dtype=torch.bool),
        action_prefix=torch.zeros(batch, cfg.chunk_size, cfg.max_action_dim),
        delay=torch.tensor(delay),
        noise=torch.zeros(batch, cfg.chunk_size, cfg.max_action_dim),
        state=None,
        accel=accel,
    )


def _velocity(value, cfg, batch=1):
    return torch.full((batch, cfg.chunk_size, cfg.max_action_dim), float(value))


# --------------------------------------------------------------------------------------
# The loop feeds the meter.
# --------------------------------------------------------------------------------------


def test_constant_velocity_field_scores_zero_through_the_real_loop():
    """End-to-end through ``PI05FlowMatching.sample_actions``: a field that returns the
    same velocity at every Euler step is the maximally-certain case and must score 0."""
    cfg = _config(num_steps=4)
    fm = _flow_matching(cfg, [_velocity(1.0, cfg)] * 4)
    meter = AccelMeter(prefix=3, batch_size=1, device=torch.device("cpu"))

    _sample(fm, cfg, accel=meter)

    assert fm._denoise_calls["n"] == cfg.num_steps
    assert meter.steps == 3
    assert meter.value().item() == 0.0


def test_meter_receives_the_velocities_in_loop_order():
    """The prefix must be the FIRST Euler steps. Scripted so only the last transition bends:
    a prefix of 2 sees none of it, the full schedule does."""
    cfg = _config(num_steps=4)
    velocities = [_velocity(1.0, cfg), _velocity(1.0, cfg), _velocity(1.0, cfg), _velocity(9.0, cfg)]

    short = AccelMeter(prefix=2, batch_size=1, device=torch.device("cpu"))
    _sample(_flow_matching(cfg, velocities), cfg, accel=short)
    assert short.value().item() == 0.0

    full = AccelMeter(prefix=4, batch_size=1, device=torch.device("cpu"))
    _sample(_flow_matching(cfg, velocities), cfg, accel=full)
    assert full.value().item() > 0.0


def test_accel_matches_a_hand_computed_value_through_the_loop():
    """Velocities 1 -> 2 -> 1 over a (4 rows x 4 dims) chunk of constant entries.

    Each velocity has norm ``v * 4`` (sqrt(16) elements), each difference has norm ``1 * 4``,
    so accel_3 = 3 * (4 + 4) / (4 + 8 + 4) = 24 / 16 = 1.5.
    """
    cfg = _config(num_steps=3)
    fm = _flow_matching(cfg, [_velocity(1.0, cfg), _velocity(2.0, cfg), _velocity(1.0, cfg)])
    meter = AccelMeter(prefix=3, batch_size=1, device=torch.device("cpu"))

    _sample(fm, cfg, accel=meter)

    assert meter.value().item() == pytest.approx(1.5, rel=1e-6)


def test_disabled_accel_leaves_the_sampled_actions_bit_identical():
    """The feature must be inert when off — same trajectory, same output, no side effects."""
    cfg = _config(num_steps=4)
    velocities = [_velocity(v, cfg) for v in (1.0, 2.0, 1.5, 0.5)]

    without = _sample(_flow_matching(cfg, velocities), cfg, accel=None)
    with_meter = _sample(
        _flow_matching(cfg, velocities),
        cfg,
        accel=AccelMeter(prefix=3, batch_size=1, device=torch.device("cpu")),
    )
    assert torch.equal(without, with_meter)


def test_per_sample_scores_survive_the_loop():
    """Batched multi-env rollouts: sample 0 travels straight, sample 1 bends."""
    cfg = _config(num_steps=3)
    batch = 2

    def split(a, b):
        v = torch.empty(batch, cfg.chunk_size, cfg.max_action_dim)
        v[0] = a
        v[1] = b
        return v

    velocities = [split(1.0, 1.0), split(1.0, 5.0), split(1.0, 1.0)]
    fm = _flow_matching(cfg, velocities)
    meter = AccelMeter(prefix=3, batch_size=batch, device=torch.device("cpu"))

    _sample(fm, cfg, batch=batch, accel=meter)

    value = meter.value()
    assert value.shape == (batch,)
    assert value[0].item() == 0.0
    assert value[1].item() > 0.0


# --------------------------------------------------------------------------------------
# Real-time chunking: the frozen prefix rows must not be scored.
# --------------------------------------------------------------------------------------


def test_frozen_rtc_prefix_rows_are_excluded():
    """With ``delay=2``, rows 0-1 are overwritten with the committed actions before every
    ``denoise_step`` and their conditioning time is pinned to 0, so their velocities describe
    nothing. Here they carry all the bend; the executed rows are constant, so the score must
    be exactly 0. Without the row mask this reports a large spurious value.
    """
    cfg = _config(chunk_size=4, n_action_steps=4, max_delay=2, num_steps=3)

    def velocity(frozen_value):
        v = torch.ones(1, cfg.chunk_size, cfg.max_action_dim)
        v[:, :2] = frozen_value
        return v

    fm = _flow_matching(cfg, [velocity(1.0), velocity(50.0), velocity(-50.0)])
    meter = AccelMeter(prefix=3, batch_size=1, device=torch.device("cpu"))

    _sample(fm, cfg, accel=meter, delay=2)

    assert meter.value().item() == 0.0


def test_rows_outside_the_executed_window_are_excluded():
    """``select_action`` applies ``actions[delay : delay + n_action_steps]`` and re-plans, so
    later rows never reach the robot. Here only row 0 is executed and it is constant, while
    the unexecuted tail carries the bend."""
    cfg = _config(chunk_size=4, n_action_steps=1, max_delay=0, num_steps=3)

    def velocity(tail_value):
        v = torch.ones(1, cfg.chunk_size, cfg.max_action_dim)
        v[:, 1:] = tail_value
        return v

    fm = _flow_matching(cfg, [velocity(1.0), velocity(50.0), velocity(-50.0)])
    meter = AccelMeter(prefix=3, batch_size=1, device=torch.device("cpu"))

    _sample(fm, cfg, accel=meter)

    assert meter.value().item() == 0.0


# --------------------------------------------------------------------------------------
# Policy-level `last_accel` contract.
# --------------------------------------------------------------------------------------


def _bare_policy(cfg):
    policy = object.__new__(PI05Policy)
    policy.config = cfg
    policy.eval = lambda: None  # bypass nn.Module.eval (no __init__ was run)
    policy.accel_prefix = None
    PI05Policy.reset(policy)
    return policy


def test_last_accel_follows_the_replan_cadence_not_the_step_cadence():
    """One ``sample_actions`` feeds ``n_action_steps`` env steps, so ``last_accel`` must be
    populated only on the step that actually re-planned. A consumer that read it every step
    would silently record the same score ``n_action_steps`` times and inflate its stream.
    """
    chunk, n_steps = 4, 3
    cfg = _config(chunk_size=chunk, n_action_steps=n_steps, max_delay=0)
    policy = _bare_policy(cfg)

    calls = {"n": 0}

    def fake_sample_actions(batch, noise=None, action_prefix=None, delay=None):
        calls["n"] += 1
        policy.last_accel = [float(calls["n"])]
        return torch.zeros(1, chunk, MAX_ACTION_DIM)

    policy.sample_actions = fake_sample_actions
    batch = {"state": torch.zeros(1, MAX_STATE_DIM)}

    observed = []
    for _ in range(2 * n_steps):
        PI05Policy.select_action(policy, batch)
        observed.append(policy.last_accel)

    assert calls["n"] == 2
    # Populated on the re-planning step, None while the queue drains.
    assert observed == [[1.0], None, None, [2.0], None, None]


def test_reset_clears_the_accel_state():
    """Per-episode state must not leak across ``env.reset()`` — the eval loop calls
    ``policy.reset()`` once per rollout batch."""
    policy = _bare_policy(_config())
    policy.last_accel = [0.5]
    policy.last_accel_provenance = object()

    PI05Policy.reset(policy)

    assert policy.last_accel is None
    assert policy.last_accel_provenance is None


def test_accel_is_disabled_by_default():
    """A cost-free score is still a score nothing should read implicitly — its scale is only
    meaningful against a calibration."""
    policy = _bare_policy(_config())
    assert policy.accel_prefix is None
    assert policy.last_accel is None


def test_default_prefix_for_the_shipped_schedule():
    """Every shipped pi05 config uses ``num_steps=10``; the paper's online detector uses the
    second-to-last prefix."""
    from opentau.policies.accel import default_prefix

    assert PI05Config().num_steps == 10
    assert default_prefix(PI05Config().num_steps) == 9


def test_hand_computed_value_is_scale_free():
    """``accel`` is a ratio, so uniformly rescaling every velocity must not change it. This
    is what makes a single threshold portable across chunks whose velocity magnitudes differ.
    """
    cfg = _config(num_steps=3)
    scores = []
    for scale in (1.0, 1000.0):
        fm = _flow_matching(
            cfg, [_velocity(1.0 * scale, cfg), _velocity(2.0 * scale, cfg), _velocity(1.0 * scale, cfg)]
        )
        meter = AccelMeter(prefix=3, batch_size=1, device=torch.device("cpu"))
        _sample(fm, cfg, accel=meter)
        scores.append(meter.value().item())
    assert scores[0] == pytest.approx(scores[1], rel=1e-5)
    assert not math.isnan(scores[0])
