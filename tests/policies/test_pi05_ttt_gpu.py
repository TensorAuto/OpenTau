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

"""GPU tests for the pi05_ttt policy end to end.

Constructing a ``PI05TTTPolicy`` builds a full PaliGemma tower and fetches the
PaliGemma and FAST tokenizers, so these cannot run in the CPU suite. The layer
math, config validation and sequence-folding helpers are covered on CPU in
``test_pi05_ttt.py``; what is left for here is what only a real model shows.

Two things shape how these are written, both learned by running them:

* **Everything is bfloat16.** ``_preferred_dtype()`` in the dual-tower forward
  pins activations to bf16, so a float32 policy hard-errors on the first
  ``q_proj``. It also matters for memory: the policy is 3.45B parameters, which
  is ~13.8 GiB in float32 and leaves no room for a second one.
* **The no-op test does not build a second policy to compare against.** On the
  single-timestep path ``PI05TTTPolicy`` delegates to ``PI05Policy.forward``,
  which never passes ``ttt_state``, so TTT provably cannot run there and
  comparing the two policies would pin nothing. The property worth pinning
  lives on the *sequence* path: with the gate shut the model must be blind to
  its own history, and with the gate open it must not be.
"""

from __future__ import annotations

import pytest
import torch

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTPolicy
from tests.utils import require_vram_gib

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

# Measured peak on an RTX 3090 (24 GiB card, which reports 23.57 GiB): 6.64 GiB
# for a bf16 policy plus a 4-timestep sequence forward at chunk_size=50. The
# floor sits well above that to cover the CUDA context, allocator fragmentation
# and the backward pass, while staying below what a 24 GiB card reports — an
# earlier 24.0 floor silently skipped every test on exactly the card they were
# written for.
_MIN_VRAM_GIB = 16.0


def _config(**overrides) -> PI05TTTConfig:
    """Builds a small pi05_ttt config for tests.

    Args:
        **overrides: Field overrides.

    Returns:
        A config with a short chunk and few registers, to keep runtime down.
    """
    defaults = {
        "chunk_size": 4,
        "n_action_steps": 4,
        "n_register_tokens": 2,
        "ttt_num_heads": 8,
        "sequence_length": 4,
        "tbptt_segment_length": 2,
        "num_steps": 2,
        "train_ttt_only": True,
        # The sequence path raises on predict_response=True rather than
        # silently dropping the response cross-entropy term.
        "predict_response": False,
        # Normally filled in by `make_policy(ds_meta=...)` from the dataset.
        # Hardcoded here so the tests need no Hub dataset; the names and shapes
        # are what `lerobot/droid_100` actually produces through
        # `make_dataset_mixture`, so a batch built to this schema is the same
        # shape the training path sees.
        "input_features": {
            "state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        "output_features": {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
        },
    }
    defaults.update(overrides)
    return PI05TTTConfig(**defaults)


def _stats(config: PI05TTTConfig) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Builds neutral normalization statistics for the synthetic features.

    Without these the `Normalize` buffers stay at infinity and the first
    forward asserts. Identity-ish values (zero mean, unit std, [0, 1] range)
    keep the normalization a no-op so the tests measure the model, not the
    scaling.

    Args:
        config: Policy configuration, for the padded state/action widths.

    Returns:
        A single-dataset stats list in the shape `Normalize` expects.
    """

    def entry(dim: int) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.zeros(dim),
            "std": torch.ones(dim),
            "min": torch.zeros(dim),
            "max": torch.ones(dim),
            "q01": torch.zeros(dim),
            "q99": torch.ones(dim),
        }

    return [{"state": entry(config.max_state_dim), "actions": entry(config.max_action_dim)}]


def _policy(config: PI05TTTConfig) -> PI05TTTPolicy:
    """Builds a policy on the GPU in the dtype the dual-tower forward requires.

    Args:
        config: Policy configuration.

    Returns:
        A bfloat16 CUDA policy in eval mode.

    Note:
        ``.eval()`` is a separate statement, never chained: this repo's
        ``PaliGemmaWithExpertModel.train`` override returns ``None`` rather than
        ``self``, so ``policy = policy.eval()`` would silently yield ``None``.
    """
    torch.manual_seed(0)
    policy = (
        PI05TTTPolicy(config, per_dataset_stats=_stats(config), dataset_names=["synthetic"])
        .to("cuda")
        .to(torch.bfloat16)
    )
    policy.eval()
    return policy


def _sequence_batch(config: PI05TTTConfig, num_timesteps: int) -> dict:
    """Builds a trajectory batch with a leading timestep axis.

    Args:
        config: Policy configuration.
        num_timesteps: Policy calls per trajectory.

    Returns:
        A batch shaped for the sequence path.
    """
    return {
        "camera0": torch.rand(1, num_timesteps, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
        "state": torch.rand(1, num_timesteps, config.max_state_dim, device="cuda", dtype=torch.bfloat16),
        "actions": torch.rand(
            1,
            num_timesteps,
            config.chunk_size,
            config.max_action_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "prompt": ["assemble the gear"],
    }


def _perturb_first_timestep(batch: dict) -> dict:
    """Returns a copy of ``batch`` with only timestep 0 changed.

    Args:
        batch: A sequence batch.

    Returns:
        A deep-enough copy whose timestep 0 state and actions are resampled.
    """
    altered = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    altered["state"][:, 0] = torch.rand_like(altered["state"][:, 0])
    altered["actions"][:, 0] = torch.rand_like(altered["actions"][:, 0])
    return altered


def _final_timestep_mask(num_timesteps: int) -> torch.Tensor:
    """Builds a loss mask that scores only the last timestep.

    Any dependence on earlier timesteps must then travel through the memory,
    which is exactly what the history tests probe.

    Args:
        num_timesteps: Policy calls per trajectory.

    Returns:
        A ``(1, num_timesteps)`` bool mask, True only at the final position.
    """
    mask = torch.zeros(1, num_timesteps, dtype=torch.bool, device="cuda")
    mask[:, -1] = True
    return mask


def _set_gate(policy: PI05TTTPolicy, value: float) -> None:
    """Sets every tanh gate to a fixed value.

    Args:
        policy: The policy to modify in place.
        value: The value written into every gate's ``alpha``.
    """
    for layer in policy.model.paligemma_with_expert.gemma_expert.model.layers:
        torch.nn.init.constant_(layer.ttt_gate.alpha, value)


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_shut_gate_makes_the_model_blind_to_its_own_history():
    """With the gate at zero the loss must not depend on earlier timesteps.

    TTT is the only path across timesteps, so closing the gate must sever it
    completely — that is what licenses initializing from a pretrained π₀.₅
    checkpoint without damaging it.
    """
    config = _config()
    policy = _policy(config)
    _set_gate(policy, 0.0)

    batch = _sequence_batch(config, num_timesteps=4)
    altered = _perturb_first_timestep(batch)
    mask = _final_timestep_mask(4)

    with torch.no_grad():
        torch.manual_seed(1)
        first = policy.forward({**batch, "loss_mask": mask})["MSE"]
        torch.manual_seed(1)
        second = policy.forward({**altered, "loss_mask": mask})["MSE"]

    torch.testing.assert_close(first, second, atol=1e-3, rtol=1e-2)


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_open_gate_makes_the_model_see_its_own_history():
    """With the gate open, earlier timesteps must change the final loss.

    The mutation-killing counterpart to the test above: without it, a TTT branch
    that had been accidentally disconnected would pass that one perfectly.
    """
    config = _config()
    policy = _policy(config)
    _set_gate(policy, 1.0)

    batch = _sequence_batch(config, num_timesteps=4)
    altered = _perturb_first_timestep(batch)
    mask = _final_timestep_mask(4)

    with torch.no_grad():
        torch.manual_seed(1)
        first = policy.forward({**batch, "loss_mask": mask})["MSE"]
        torch.manual_seed(1)
        second = policy.forward({**altered, "loss_mask": mask})["MSE"]

    assert not torch.allclose(first, second, atol=1e-3), (
        "changing timestep 0 left the final timestep's loss untouched with the gate wide "
        "open — the TTT branch is not carrying history"
    )


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_sequence_forward_produces_gradients_for_every_added_parameter():
    """The sequence path must run and reach W_0, the gates and the registers.

    ``W_0`` is the one most easily left dangling: it receives gradient only
    through the *first* TBPTT segment, so a detach in the wrong place turns it
    into a permanently random initialization that nothing complains about.
    """
    config = _config()
    torch.manual_seed(0)
    policy = (
        PI05TTTPolicy(config, per_dataset_stats=_stats(config), dataset_names=["synthetic"])
        .to("cuda")
        .to(torch.bfloat16)
    )
    batch = _sequence_batch(config, num_timesteps=4)

    losses = policy.forward(batch)
    (losses["MSE"] + losses["CE"]).backward()

    first_layer = policy.model.paligemma_with_expert.gemma_expert.model.layers[0]
    assert first_layer.ttt.w1_init.grad is not None, "W_0 never received gradient"
    assert first_layer.ttt.w1_init.grad.abs().sum() > 0
    assert first_layer.ttt_gate.alpha.grad is not None
    assert first_layer.ttt_gate.alpha.grad.abs().sum() > 0
    assert policy.model.register_tokens.grad is not None
    assert policy.model.register_tokens.grad.abs().sum() > 0
    leaked = [n for n, p in policy.named_parameters() if not p.requires_grad and p.grad is not None]
    assert not leaked, f"frozen parameters received gradients: {leaked[:5]}"


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_context_only_timesteps_contribute_no_target():
    """A masked timestep must drop out of the loss without breaking it.

    The all-context case is the one that bites: it must produce a finite zero
    rather than a NaN from an empty denominator, because raising instead would
    be a data-dependent branch that can deadlock a distributed run.
    """
    config = _config()
    policy = _policy(config)
    batch = _sequence_batch(config, num_timesteps=4)

    with torch.no_grad():
        torch.manual_seed(1)
        full = policy.forward(dict(batch))["MSE"]
        half = torch.ones(1, 4, dtype=torch.bool, device="cuda")
        half[:, :2] = False
        torch.manual_seed(1)
        partial = policy.forward({**batch, "loss_mask": half})["MSE"]
        torch.manual_seed(1)
        none = policy.forward({**batch, "loss_mask": torch.zeros(1, 4, dtype=torch.bool, device="cuda")})[
            "MSE"
        ]

    assert not torch.allclose(full, partial), "loss_mask did not change the loss"
    assert torch.isfinite(none), "an all-context sequence produced a non-finite loss"
    assert none.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_train_ttt_only_freezes_exactly_the_pretrained_weights():
    """``train_ttt_only`` must leave only the newly added parameters trainable.

    This is the paper's pretraining stage. If it leaked the pretrained weights
    into the trainable set, the run would be a full finetune wearing a
    pretraining label.
    """
    config = _config(train_ttt_only=True)
    policy = PI05TTTPolicy(config, per_dataset_stats=_stats(config), dataset_names=["synthetic"])
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert trainable, "train_ttt_only left nothing trainable"
    for name in trainable:
        assert ".ttt." in name or ".ttt_gate." in name or name.endswith("register_tokens"), (
            f"{name} should have been frozen by train_ttt_only"
        )
    assert any(not p.requires_grad for p in policy.parameters()), "nothing was frozen"


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_rollout_memory_advances_once_per_call_and_resets():
    """One ``select_action`` must advance memory by exactly one timestep.

    ``config.num_steps`` denoising steps run inside each call, and only the last
    one's fast-weight update may be adopted. If every denoising step committed,
    inference memory would advance ``num_steps`` times faster than it ever did
    in training.
    """
    config = _config()
    policy = _policy(config)
    observation = {
        "camera0": torch.rand(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
        "state": torch.rand(1, config.max_state_dim, device="cuda", dtype=torch.bfloat16),
        "prompt": ["assemble the gear"],
    }

    assert policy.model._carried_fast_weights == {}
    with torch.no_grad():
        policy.select_action(dict(observation))
    assert policy.model._carried_fast_weights, "memory was not carried out of the call"
    assert policy.model._inference_token_position == config.n_expert_tokens_per_timestep

    policy.reset()
    assert policy.model._carried_fast_weights == {}, "reset() leaked memory across episodes"
    assert policy.model._inference_token_position == 0
