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

"""The hold/warmup/constant/cooldown schedule, and its non-interference with everything else.

The curve is transcribed from Black Forest Labs' published FLUX 3 Action DROID recipe:

    trunk: zero LR through 1,000; warmup through 3,000; constant to 25,000;
           linear cooldown 25,000 -> 30,000
    heads: warmup over 1,000; constant to 25,000; same cooldown

A schedule is a deterministic function, so it is asserted at exact indices rather than
sampled -- this is one of the few things in this port that can be pinned precisely without
running training.
"""

import pytest
import torch

from opentau.optim.schedulers import (
    ConstantSchedulerConfig,
    CosineDecayWithWarmupSchedulerConfig,
    HoldWarmupConstantCooldownSchedulerConfig,
    LRSchedulerConfig,
)


def _optimizer(n_groups: int = 2, lr: float = 1.0):
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr} for _ in range(n_groups)]
    return torch.optim.SGD(groups, lr=lr)


# --------------------------------------------------------------------------- the curve
@pytest.mark.parametrize(
    ("step", "trunk", "head"),
    [
        (0, 0.0, 0.0),  # trunk held at zero; heads start warming immediately
        (999, 0.0, 0.999),
        (1000, 0.0, 1.0),  # hold ends; head warmup complete
        (2000, 0.5, 1.0),  # trunk halfway through its 1000->3000 warmup
        (3000, 1.0, 1.0),
        (25000, 1.0, 1.0),  # constant right up to the cooldown
        (27500, 0.5, 0.5),  # halfway down the linear cooldown
        (30000, 0.0, 0.0),
    ],
)
def test_curve_matches_the_published_recipe(step, trunk, head):
    cfg = HoldWarmupConstantCooldownSchedulerConfig()
    assert cfg._trunk_factor(step) == pytest.approx(trunk, abs=1e-6)
    assert cfg._head_factor(step) == pytest.approx(head, abs=1e-6)


def test_the_two_curves_genuinely_differ_early_and_agree_late():
    """The whole point of per-group lambdas: heads train while the trunk is frozen.

    A single shared schedule would either freeze the freshly initialized heads for 1,000
    updates or start the pretrained trunk at full LR -- the recipe deliberately does
    neither.
    """
    cfg = HoldWarmupConstantCooldownSchedulerConfig()
    assert cfg._trunk_factor(500) == 0.0 and cfg._head_factor(500) > 0.0
    for late in (5_000, 20_000, 27_500):
        assert cfg._trunk_factor(late) == pytest.approx(cfg._head_factor(late))


def test_build_assigns_the_head_curve_only_to_the_named_groups():
    cfg = HoldWarmupConstantCooldownSchedulerConfig()
    opt = _optimizer(n_groups=2)
    sched = cfg.build(opt, num_training_steps=30_000)
    opt.step()  # optimizer first, or torch warns about schedule ordering
    sched.step()  # advance to update 1 so the two curves are distinguishable
    trunk_lr, head_lr = (g["lr"] for g in opt.param_groups)
    assert trunk_lr == 0.0, "group 0 is the trunk and is held at zero"
    assert head_lr > 0.0, "group 1 is the heads and warms immediately"


def test_a_head_index_beyond_the_optimizer_raises():
    """The grouping is the policy's, not the scheduler's, so a mismatch must be loud."""
    cfg = HoldWarmupConstantCooldownSchedulerConfig(head_param_group_indices=(5,))
    with pytest.raises(ValueError, match="out of range"):
        cfg.build(_optimizer(n_groups=2), num_training_steps=30_000)


@pytest.mark.parametrize("bad", [(-1,), (-2, 0), (0, -1)])
def test_a_negative_head_index_is_rejected(bad):
    """``(-1,)`` is a natural spelling for "the last group" and would fail silently.

    The lookup iterates ``range(len(param_groups))``, so a negative entry matches nothing
    and hands *every* group the trunk curve -- freezing the freshly initialized heads for
    the first 1,000 updates with no error and plausible-looking logs.
    """
    cfg = HoldWarmupConstantCooldownSchedulerConfig(head_param_group_indices=bad)
    with pytest.raises(ValueError, match="out of range"):
        cfg.build(_optimizer(n_groups=2), num_training_steps=30_000)


def test_a_horizon_longer_than_the_schedule_warns(caplog):
    """Past ``total_steps`` every group sits at ``final_lr_scale`` -- 0.0 by default.

    That is silently frozen training, so a run longer than the schedule must say so.
    """
    cfg = HoldWarmupConstantCooldownSchedulerConfig()
    with caplog.at_level("WARNING"):
        cfg.build(_optimizer(n_groups=2), num_training_steps=50_000)
    assert "schedule ends at" in caplog.text
    assert "final_lr_scale=0.0" in caplog.text


def test_a_matching_horizon_does_not_warn(caplog):
    cfg = HoldWarmupConstantCooldownSchedulerConfig()
    with caplog.at_level("WARNING"):
        cfg.build(_optimizer(n_groups=2), num_training_steps=30_000)
    assert not caplog.records


def test_single_group_optimizer_gets_the_trunk_curve():
    cfg = HoldWarmupConstantCooldownSchedulerConfig(head_param_group_indices=())
    opt = _optimizer(n_groups=1)
    sched = cfg.build(opt, num_training_steps=30_000)
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hold_steps": 3_000, "num_warmup_steps": 1_000},  # hold past warmup
        {"cooldown_start": 40_000},  # cooldown after the end
        {"final_lr_scale": 1.5},  # scale out of range
        {"head_warmup_steps": 0},  # zero-length warmup
        # head warmup running into the cooldown makes the curve non-monotonic: it is
        # still ramping while the cooldown scales down, peaking below 1 then cliff-dropping
        {"head_warmup_steps": 27_000},
    ],
)
def test_incoherent_schedules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HoldWarmupConstantCooldownSchedulerConfig(**kwargs)


# --------------------------------------------------------------------------- non-interference
def test_it_is_registered_without_disturbing_the_existing_schedulers():
    """Purely additive: this must not change what any existing name resolves to."""
    assert (
        LRSchedulerConfig.get_choice_class("hold_warmup_constant_cooldown")
        is HoldWarmupConstantCooldownSchedulerConfig
    )
    assert LRSchedulerConfig.get_choice_class("constant") is ConstantSchedulerConfig
    assert (
        LRSchedulerConfig.get_choice_class("cosine_decay_with_warmup") is CosineDecayWithWarmupSchedulerConfig
    )


def test_existing_schedulers_still_produce_a_single_shared_lambda():
    """A policy that does not opt in must see byte-identical behaviour.

    The new schedule is the only one that hands ``LambdaLR`` a *list*; pinning that the
    others still pass one callable is what makes "additive" checkable rather than asserted.
    """
    opt = _optimizer(n_groups=2)
    sched = ConstantSchedulerConfig(num_warmup_steps=0).build(opt, 1_000)
    assert len({id(f) for f in sched.lr_lambdas}) == 1

    opt2 = _optimizer(n_groups=2)
    cosine = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=100, peak_lr=1.0, decay_lr=0.1
    )
    assert len({id(f) for f in cosine.build(opt2, 100).lr_lambdas}) == 1


def test_it_is_reachable_through_the_optimizer_factory():
    """The reason this scheduler was withdrawn once: its two curves need two param groups.

    ``make_optimizer_and_scheduler`` previously could not deliver them — the preset path
    returned param-group dicts that the ``requires_grad`` filter choked on, and the
    non-preset path built from flat ``policy.parameters()``. Both are fixed, so this walks
    the real factory rather than hand-building an optimizer, which is what every other test
    in this file does.
    """
    import torch

    from opentau.optim.factory import _trainable_params

    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 5.0},
    ]
    filtered = _trainable_params(groups)
    assert len(filtered) == 2, "the factory must preserve param groups, not flatten them"
    opt = torch.optim.SGD(filtered, lr=1.0)
    sched = HoldWarmupConstantCooldownSchedulerConfig().build(opt, num_training_steps=30_000)
    opt.step()
    sched.step()
    trunk, head = (g["lr"] for g in opt.param_groups)
    assert trunk == 0.0, "trunk must be held at zero through the hold window"
    assert head > 0.0, "heads must warm immediately"


def test_no_policy_selects_it_by_default():
    """Opt-in only: every policy's own preset must be unchanged by this PR."""
    import opentau
    from opentau.policies.factory import make_policy_config

    checked = 0
    for name in opentau.available_policies:
        try:
            preset = make_policy_config(name).get_scheduler_preset()
        except Exception:  # noqa: BLE001 - a policy needing ctor args is not this test's subject
            continue
        checked += 1
        assert not isinstance(preset, HoldWarmupConstantCooldownSchedulerConfig), (
            f"{name} would silently change its learning-rate schedule"
        )
    # Without this the `except: continue` makes the whole pin vacuous -- if every policy
    # started raising, the loop would inspect nothing and still pass.
    assert checked >= 5, f"only {checked} policy presets were actually inspected"
