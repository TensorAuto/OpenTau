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

"""Param groups reaching the optimizer.

``get_optim_params()`` is documented as returning "parameters to optimize", and most
policies return a flat list. A policy that wants per-group hyperparameters returns param-
group dicts instead — which the factory's ``requires_grad`` filter used to choke on with
``AttributeError: 'dict' object has no attribute 'requires_grad'``. The effect was silent
and total: such a policy could not be trained at all, and the per-group learning rates it
asked for never applied.
"""

import pytest
import torch

from opentau.optim.factory import _trainable_params


def _param(trainable: bool = True):
    p = torch.nn.Parameter(torch.zeros(1))
    p.requires_grad_(trainable)
    return p


# --------------------------------------------------------------------------- flat (unchanged)
def test_a_flat_list_is_still_filtered_by_requires_grad():
    """Every policy but one returns this shape; its behaviour must not move.

    Asserted by object identity and order against the expression this replaced, not just
    by count — eleven of the twelve registered policies take this path, and a subtle
    reordering would silently change which parameters land in which optimizer state.
    """
    params = [_param(True), _param(False), _param(True), _param(True)]
    expected = [p for p in params if p.requires_grad]  # the pre-existing filter, verbatim
    kept = _trainable_params(params)
    assert [id(p) for p in kept] == [id(p) for p in expected]


def test_only_the_policy_that_needs_groups_takes_the_new_branch():
    """The group branch must be unreachable for every policy that returns flat params.

    This is the guarantee that matters for the other eleven policies: the new code path is
    entered solely on ``isinstance(first, dict)``, so a flat return value cannot reach it.
    """
    flat = [_param(), _param()]
    assert all(isinstance(p, torch.nn.Parameter) for p in _trainable_params(flat))

    groups = [{"params": [_param()], "lr": 1e-4}]
    assert all(isinstance(g, dict) for g in _trainable_params(groups))


def test_every_registered_policy_still_gets_its_own_presets():
    """A sweep, so a change here cannot quietly re-point another policy's training.

    Both edits in this change sit on the shared training path, so the check is that every
    policy still resolves the optimizer and scheduler it resolved before.
    """
    import opentau
    from opentau.policies.factory import make_policy_config

    checked = 0
    for name in opentau.available_policies:
        try:
            cfg = make_policy_config(name)
            optimizer, scheduler = cfg.get_optimizer_preset(), cfg.get_scheduler_preset()
        except Exception:  # noqa: BLE001 - a policy needing ctor args is not this test's subject
            continue
        checked += 1
        assert optimizer is not None, f"{name} lost its optimizer preset"
        assert scheduler is not None, f"{name} lost its scheduler preset"
    assert checked >= 10, f"only {checked} policies were actually inspected"


def test_a_generator_of_parameters_is_accepted():
    """``policy.parameters()`` is a generator, and it was consumed lazily before."""
    kept = _trainable_params(p for p in [_param(True), _param(False)])
    assert len(kept) == 1


def test_an_empty_iterable_is_harmless():
    assert _trainable_params([]) == []


# --------------------------------------------------------------------------- param groups
def test_param_group_dicts_survive_instead_of_raising():
    """The regression this fixes: filtering used to be applied to the dicts themselves."""
    groups = [
        {"params": [_param(), _param()], "lr": 1e-4},
        {"params": [_param()], "lr": 5e-4},
    ]
    kept = _trainable_params(groups)
    assert [g["lr"] for g in kept] == [1e-4, 5e-4], "per-group hyperparameters must survive"
    assert [len(g["params"]) for g in kept] == [2, 1]


def test_frozen_parameters_are_removed_from_within_a_group():
    """The filter has to reach inside, not just pass the group through."""
    groups = [{"params": [_param(True), _param(False), _param(True)], "lr": 1e-4}]
    (kept,) = _trainable_params(groups)
    assert len(kept["params"]) == 2


def test_a_fully_frozen_group_is_kept_so_indices_stay_stable():
    """Dropping it would silently renumber the groups after it.

    Anything addressing a group by position — a per-group learning-rate schedule, say —
    would then be remapped onto a different group, changing which parameters it governs
    with no error.
    """
    groups = [
        {"params": [_param(False)], "lr": 1e-4},
        {"params": [_param(True)], "lr": 5e-4},
    ]
    kept = _trainable_params(groups)
    assert len(kept) == 2, "group count must not depend on what happens to be frozen"
    assert kept[0]["params"] == []
    assert kept[1]["lr"] == 5e-4


def test_extra_group_keys_are_carried_through():
    """A group may carry any optimizer kwarg; none of them are ours to drop."""
    groups = [{"params": [_param()], "lr": 1e-4, "weight_decay": 0.05, "betas": (0.9, 0.99)}]
    (kept,) = _trainable_params(groups)
    assert kept["weight_decay"] == 0.05
    assert kept["betas"] == (0.9, 0.99)


def test_the_result_is_accepted_by_a_real_optimizer():
    """The point of the exercise: torch must take what we hand it."""
    groups = [
        {"params": [_param()], "lr": 1e-4},
        {"params": [_param()], "lr": 5e-4},
    ]
    opt = torch.optim.AdamW(_trainable_params(groups))
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 5e-4]


# --------------------------------------------------------------------------- the policy that needs it
def test_flux3_action_gets_its_five_times_head_learning_rate():
    """End-to-end on the policy this was blocking, through the real factory.

    Black Forest Labs' recipe runs the freshly initialized embodiment heads at 5x the
    trunk's learning rate. Before this fix the optimizer could not be built at all, so that
    ratio was unreachable rather than merely wrong.
    """
    import torch.nn as nn

    from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
    from opentau.configs.train import TrainPipelineConfig
    from opentau.configs.types import FeatureType, PolicyFeature
    from opentau.datasets.transforms import ImageTransformsConfig
    from opentau.optim.factory import make_optimizer_and_scheduler
    from opentau.policies.flux3_action.configuration_flux3_action import Flux3ActionConfig
    from opentau.policies.flux3_action.modeling_flux3_action import Flux3ActionPolicy
    from opentau.policies.flux3_action.models.text_encoder import MockTextEncoder

    class _StubVAE(nn.Module):
        pass

    cfg = Flux3ActionConfig(
        video_vae_id="stub",
        text_encoder_id="stub",
        torch_dtype="float32",
        dit_config={
            "hidden_size": 256,
            "num_heads": 4,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": [16, 16, 16, 16],
            "context_in_dim": 512,
            "vec_in_dim": 64,
        },
    )
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    policy = Flux3ActionPolicy(cfg, video_vae=_StubVAE(), text_encoder=MockTextEncoder())

    ds = DatasetConfig(
        repo_id="m",
        root="/tmp/m",
        image_transforms=ImageTransformsConfig(enable=False),
        episodes=[0],
        video_backend=None,
    )
    train = TrainPipelineConfig(
        dataset_mixture=DatasetMixtureConfig(datasets=[ds], weights=[1.0], action_freq=15.0),
        policy=cfg,
        batch_size=2,
        action_chunk=32,
        use_policy_training_preset=True,
        steps=100,
    )
    train.validate()
    optimizer, _ = make_optimizer_and_scheduler(train, policy)

    assert len(optimizer.param_groups) == 2, "trunk and heads must remain separate groups"
    lrs = [g["lr"] for g in optimizer.param_groups]
    assert max(lrs) / min(lrs) == pytest.approx(5.0), "the recipe's 5x head LR must survive"
    # and the recipe's other optimizer settings must reach it too
    assert tuple(optimizer.param_groups[0]["betas"]) == (0.9, 0.99)
    assert optimizer.param_groups[0]["weight_decay"] == 0.05


# --------------------------------------------------------------------------- malformed input
def test_mixing_bare_parameters_and_groups_is_refused_clearly():
    """torch forbids the mix; the diagnostic should say so rather than leak an AttributeError.

    Sniffing only the first entry meant a flat-first list containing a group later produced
    the original ``'dict' object has no attribute 'requires_grad'`` — the very error this
    change exists to remove, from a different cause.
    """
    with pytest.raises(TypeError, match="mixed bare parameters"):
        _trainable_params([_param(), {"params": [_param()], "lr": 1e-4}])

    with pytest.raises(TypeError, match="every entry to be a group"):
        _trainable_params([{"params": [_param()], "lr": 1e-4}, _param()])


def test_a_group_without_params_is_refused_clearly():
    with pytest.raises(KeyError, match="no 'params' key"):
        _trainable_params([{"lr": 1e-4}])


def test_group_params_may_be_any_iterable():
    """A policy may build groups lazily; consuming them once here is fine."""
    (kept,) = _trainable_params([{"params": (p for p in [_param(), _param()]), "lr": 1e-4}])
    assert len(kept["params"]) == 2
