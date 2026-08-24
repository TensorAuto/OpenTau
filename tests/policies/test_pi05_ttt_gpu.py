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
``test_pi05_ttt.py``; what is left for here is the part only a real model can
show: that the gate makes the policy a no-op against stock π₀.₅ at
initialization, that the sequence path runs and produces gradients, that
``train_ttt_only`` freezes what it claims to, and that the rollout memory
advances per policy call and resets on ``reset()``.

The no-op test is the important one. It is what licenses initializing from an
existing π₀.₅ checkpoint at all: if the randomly initialized TTT layers
perturbed the pretrained action expert on step 0, the pretrained skills would be
damaged before training had a chance to decide how much memory to use.
"""

from __future__ import annotations

import pytest
import torch

from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTPolicy
from tests.utils import require_vram_gib

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

# A PaliGemma-3B tower in bfloat16 plus an 18-layer action expert with TTT
# layers and a short trajectory's activations.
_MIN_VRAM_GIB = 24.0


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
        # The sequence path raises on predict_response=True (see modeling docstring).
        "predict_response": False,
    }
    defaults.update(overrides)
    return PI05TTTConfig(**defaults)


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_gate_makes_ttt_a_no_op_at_initialization():
    """At ``ttt_gate_init=0`` the policy must match stock π₀.₅ bit for bit.

    Pinned at exactly zero rather than the default 0.001 so the assertion can
    be exact; ``test_default_gate_is_a_small_perturbation`` covers the shipped
    default. Together they pin that the gate — not a disconnected TTT branch —
    is what preserves the pretrained behaviour.
    """
    torch.manual_seed(0)
    ttt_config = _config(ttt_gate_init=0.0, n_register_tokens=0, train_ttt_only=False)
    ttt_policy = PI05TTTPolicy(ttt_config).to("cuda").eval()

    base_config = PI05Config(
        **{field: getattr(ttt_config, field) for field in ("chunk_size", "n_action_steps", "num_steps")}
    )
    torch.manual_seed(0)
    base_policy = PI05Policy(base_config).to("cuda").eval()
    # Copy the shared weights across so the only difference is the TTT branch.
    missing, unexpected = base_policy.load_state_dict(
        {k: v for k, v in ttt_policy.state_dict().items() if k in base_policy.state_dict()},
        strict=False,
    )
    assert not unexpected, f"unexpected keys when mirroring weights: {unexpected}"

    batch = _single_timestep_batch(ttt_config, device="cuda")
    with torch.no_grad():
        torch.manual_seed(1)
        ttt_losses = ttt_policy.forward(dict(batch))
        torch.manual_seed(1)
        base_losses = base_policy.forward(dict(batch))

    torch.testing.assert_close(ttt_losses["MSE"], base_losses["MSE"], atol=1e-4, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_default_gate_is_a_small_perturbation():
    """The shipped 0.001 gate must move the loss a little, not a lot.

    A gate that had been accidentally disconnected would give an exact match
    here, and a gate initialized too wide would swamp the pretrained policy.
    """
    torch.manual_seed(0)
    config = _config(train_ttt_only=False)
    policy = PI05TTTPolicy(config).to("cuda").eval()
    batch = _single_timestep_batch(config, device="cuda")

    with torch.no_grad():
        torch.manual_seed(1)
        gated = policy.forward(dict(batch))["MSE"]
        for layer in policy.model.paligemma_with_expert.gemma_expert.model.layers:
            torch.nn.init.zeros_(layer.ttt_gate.alpha)
        torch.manual_seed(1)
        ungated = policy.forward(dict(batch))["MSE"]

    assert not torch.allclose(gated, ungated, atol=1e-7), "the tanh gate is not wired in"
    assert (gated - ungated).abs() < 0.5 * ungated.abs(), "0.001 gate perturbed the loss wildly"


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_sequence_forward_produces_gradients_for_ttt_parameters():
    """The sequence path must run and reach every added parameter."""
    torch.manual_seed(0)
    config = _config()
    policy = PI05TTTPolicy(config).to("cuda")
    batch = _sequence_batch(config, batch_size=1, num_timesteps=4, device="cuda")

    losses = policy.forward(batch)
    (losses["MSE"] + losses["CE"]).backward()

    first_layer = policy.model.paligemma_with_expert.gemma_expert.model.layers[0]
    assert first_layer.ttt.w1_init.grad is not None, "W_0 never received gradient"
    assert first_layer.ttt.w1_init.grad.abs().sum() > 0
    assert first_layer.ttt_gate.alpha.grad is not None
    assert policy.model.register_tokens.grad is not None


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_train_ttt_only_freezes_the_pretrained_weights():
    """``train_ttt_only`` must leave exactly the new parameters trainable.

    This is the paper's pretraining stage. If it leaked the pretrained weights
    into the trainable set, the run would be a full finetune wearing a
    pretraining label.
    """
    policy = PI05TTTPolicy(_config(train_ttt_only=True))
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert trainable, "nothing is trainable"
    for name in trainable:
        assert ".ttt." in name or ".ttt_gate." in name or name.endswith("register_tokens"), (
            f"{name} should have been frozen by train_ttt_only"
        )
    # Frozen tensors may still reach the optimizer (every policy in the repo
    # returns `self.parameters()`); AdamW allocates no state for a param whose
    # grad stays None, so what matters is only that requires_grad is right.
    frozen = [name for name, p in policy.named_parameters() if not p.requires_grad]
    assert frozen, "train_ttt_only froze nothing"


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_context_only_timesteps_still_move_the_memory():
    """A masked-out timestep must contribute no target but still update memory.

    This asymmetry is the mechanism behind the paper's in-context video
    imitation and DAgger Distillation. If the mask short-circuited the forward
    instead of only the loss, both capabilities would be unreachable.
    """
    torch.manual_seed(0)
    config = _config()
    policy = PI05TTTPolicy(config).to("cuda")
    batch = _sequence_batch(config, batch_size=1, num_timesteps=4, device="cuda")

    supervise_all = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    supervise_last_two = supervise_all.clone()
    supervise_last_two[:, :2] = False

    torch.manual_seed(1)
    full = policy.forward({**batch, "loss_mask": supervise_all})["MSE"]
    torch.manual_seed(1)
    partial = policy.forward({**batch, "loss_mask": supervise_last_two})["MSE"]

    assert not torch.allclose(full, partial), "loss_mask did not change the loss"
    assert torch.isfinite(partial), "masked loss produced a non-finite value"


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_rollout_memory_advances_per_call_and_resets():
    """One ``select_action`` must advance memory by exactly one timestep.

    ``config.num_steps`` denoising steps happen inside each call; only the last
    one's update may be adopted. If every denoising step committed, inference
    memory would advance ``num_steps`` times faster than training ever did.
    """
    torch.manual_seed(0)
    config = _config()
    policy = PI05TTTPolicy(config).to("cuda").eval()
    observation = _single_timestep_observation(config, device="cuda")

    assert policy.model._carried_fast_weights == {}
    with torch.no_grad():
        policy.select_action(dict(observation))
    assert policy.model._carried_fast_weights, "memory was not carried out of the call"
    assert policy.model._inference_token_position == config.n_expert_tokens_per_timestep

    with torch.no_grad():
        policy.select_action(dict(observation))
    assert policy.model._inference_token_position == 2 * config.n_expert_tokens_per_timestep

    policy.reset()
    assert policy.model._carried_fast_weights == {}, "reset() leaked memory across episodes"
    assert policy.model._inference_token_position == 0


@pytest.mark.gpu
@pytest.mark.slow
@require_vram_gib(_MIN_VRAM_GIB)
def test_best_of_n_with_memory_fails_loudly():
    """Best-of-N plus a populated memory has no defined answer; it must raise."""
    torch.manual_seed(0)
    config = _config()
    policy = PI05TTTPolicy(config).to("cuda").eval()
    observation = _single_timestep_observation(config, device="cuda")

    with torch.no_grad():
        policy.select_action(dict(observation))
    with pytest.raises(NotImplementedError, match="best-of-N"), torch.no_grad():
        policy.model.sample_actions(*_sample_actions_args(policy, observation), n_candidates=4)


# ---------------------------------------------------------------------------
# Batch builders. Kept explicit rather than fixture-based so each test reads
# top to bottom; the shapes are the contract under test.
# ---------------------------------------------------------------------------


def _single_timestep_batch(config: PI05TTTConfig, device: str) -> dict[str, torch.Tensor]:
    """Builds a flat single-timestep training batch.

    Args:
        config: Policy configuration.
        device: Device to allocate on.

    Returns:
        A batch shaped for the non-sequence path.
    """
    return {
        "observation.images.top": torch.rand(1, 3, 224, 224, device=device),
        "observation.state": torch.rand(1, config.max_state_dim, device=device),
        "actions": torch.rand(1, config.chunk_size, config.max_action_dim, device=device),
        "task": ["assemble the gear"],
    }


def _sequence_batch(
    config: PI05TTTConfig, batch_size: int, num_timesteps: int, device: str
) -> dict[str, torch.Tensor]:
    """Builds a trajectory batch with a leading timestep axis.

    Args:
        config: Policy configuration.
        batch_size: Number of trajectories.
        num_timesteps: Policy calls per trajectory.
        device: Device to allocate on.

    Returns:
        A batch shaped for the sequence path.
    """
    return {
        "observation.images.top": torch.rand(batch_size, num_timesteps, 3, 224, 224, device=device),
        "observation.state": torch.rand(batch_size, num_timesteps, config.max_state_dim, device=device),
        "actions": torch.rand(
            batch_size, num_timesteps, config.chunk_size, config.max_action_dim, device=device
        ),
        "task": ["assemble the gear"] * batch_size,
    }


def _single_timestep_observation(config: PI05TTTConfig, device: str) -> dict[str, torch.Tensor]:
    """Builds an inference observation (no action targets).

    Args:
        config: Policy configuration.
        device: Device to allocate on.

    Returns:
        An observation dict suitable for ``select_action``.
    """
    return {
        "observation.images.top": torch.rand(1, 3, 224, 224, device=device),
        "observation.state": torch.rand(1, config.max_state_dim, device=device),
        "task": ["assemble the gear"],
    }


def _sample_actions_args(policy: PI05TTTPolicy, observation: dict[str, torch.Tensor]) -> tuple:
    """Prepares positional arguments for a direct ``sample_actions`` call.

    Args:
        policy: The policy under test.
        observation: Raw observation dict.

    Returns:
        The positional argument tuple ``sample_actions`` expects.
    """
    batch = policy.normalize_inputs(dict(observation), policy._resolve_dataset_index(observation))
    images, img_masks = policy.prepare_images(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    action_prefix = torch.zeros(
        1, policy.config.chunk_size, policy.config.max_action_dim, device=lang_tokens.device
    )
    delay = torch.zeros(1, dtype=torch.long, device=lang_tokens.device)
    return images, img_masks, lang_tokens, lang_masks, action_prefix, delay
