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

"""Contract tests for the opt-in ``knowledge_insulation`` policy-config flag.

Knowledge insulation (π0.5) ``.detach()``-es the VLM prefix KV cache before the
flow-matching action expert reads it, so the action loss does not backpropagate
into the VLM backbone. It is now gated on a per-policy ``knowledge_insulation``
config field (default ``True``, preserving the historical always-on behavior).
These CPU tests guard the config contract across every policy that performs the
detach:

  * the field exists and defaults to ``True``;
  * it can be turned off (``knowledge_insulation=False``);
  * it survives the real ``draccus`` save/load round-trip used by
    ``_save_pretrained`` / ``from_pretrained``;
  * an old ``config.json`` that predates the field decodes back to ``True``
    (backward compatibility for existing checkpoints).

The behavioral proof that the flag actually opens/closes the gradient path lives
behind a GPU forward+backward (the backbone is too heavy for CPU) and is not
exercised here.
"""

import dataclasses

import draccus
import pytest
import torch

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi07.low_level.configuration_pi07_low_level import PI07LowLevelConfig
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)

# Every policy whose training forward performs the knowledge-insulation detach.
KI_CONFIG_CLASSES = [
    PI05Config,
    PI05MemConfig,
    PI06Config,
    PI07LowLevelConfig,
    PI07PaligemmaLowLevelConfig,
]


@pytest.mark.parametrize("config_cls", KI_CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_knowledge_insulation_field_defaults_true(config_cls):
    """The flag must exist as a dataclass field defaulting to True so existing
    checkpoints keep their historical (always-insulated) behavior."""
    fields = {f.name: f for f in dataclasses.fields(config_cls)}
    assert "knowledge_insulation" in fields, f"{config_cls.__name__} is missing knowledge_insulation"
    assert fields["knowledge_insulation"].default is True

    cfg = config_cls()
    assert cfg.knowledge_insulation is True


@pytest.mark.parametrize("config_cls", KI_CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_knowledge_insulation_can_be_disabled(config_cls):
    """The flag must be settable to False (end-to-end action training)."""
    cfg = config_cls(knowledge_insulation=False)
    assert cfg.knowledge_insulation is False


@pytest.mark.parametrize("config_cls", KI_CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_knowledge_insulation_survives_round_trip(config_cls):
    """The value must round-trip through the real draccus encode/decode path
    (the one ``_save_pretrained`` / ``from_pretrained`` use)."""
    cfg = config_cls(knowledge_insulation=False)
    data = draccus.encode(cfg, declared_type=PreTrainedConfig)
    assert data["knowledge_insulation"] is False

    decoded = draccus.decode(PreTrainedConfig, data)
    assert type(decoded) is config_cls
    assert decoded.knowledge_insulation is False


@pytest.mark.parametrize("config_cls", KI_CONFIG_CLASSES, ids=lambda c: c.__name__)
def test_legacy_config_without_field_decodes_true(config_cls):
    """A pre-existing config.json saved before this field was added (i.e. the
    key is absent) must decode back to True — backward compatibility."""
    data = draccus.encode(config_cls(), declared_type=PreTrainedConfig)
    data.pop("knowledge_insulation", None)

    decoded = draccus.decode(PreTrainedConfig, data)
    assert type(decoded) is config_cls
    assert decoded.knowledge_insulation is True


def _trainable_vlm_kv_param(policy):
    """Return a (name, param) for a trainable weight inside the VLM language-model
    stack that produces the prefix KV cache.

    Every PaliGemma language-model layer's ``k_proj`` / ``v_proj`` feeds the KV
    cache that the action expert cross-attends to, so the flow-matching action
    (MSE) loss can only reach these weights *through* that KV cache — which
    knowledge insulation detaches. Only the vision tower is frozen by default, so
    these language-model weights carry ``requires_grad=True``.
    """
    lm = policy.model.paligemma_with_expert.paligemma.language_model
    for name, p in lm.named_parameters():
        if p.requires_grad and "layers." in name and ("k_proj" in name or "v_proj" in name) and p.ndim == 2:
            return name, p
    for name, p in lm.named_parameters():  # fallback: any trainable layer weight
        if p.requires_grad and "layers." in name and p.ndim == 2:
            return name, p
    raise AssertionError("no trainable VLM language-model layer weight found")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("knowledge_insulation", [True, False])
def test_pi05_knowledge_insulation_controls_vlm_gradient(
    pi05_training_config, lerobot_dataset_metadata, knowledge_insulation
):
    """Behavioral proof that the toggle opens/closes the action→VLM gradient path.

    The action (flow-matching ``MSE``) loss is backpropagated *in isolation*
    (the FAST/``CE`` term always trains the VLM, so it would mask the effect).

      * ``knowledge_insulation=True``  → the prefix KV cache is detached before
        the action expert, so a VLM language-model weight that feeds the KV
        receives **no** gradient from the MSE-only backward (``grad is None``).
      * ``knowledge_insulation=False`` → the same weight **does** receive a
        gradient (the action loss flows into the VLM backbone).

    The action-side output projection gets a gradient in *both* cases — a sanity
    check that the MSE backward actually ran.
    """
    from opentau.policies.pi05.modeling_pi05 import PI05Policy

    config = pi05_training_config.policy
    config.knowledge_insulation = knowledge_insulation
    config.train_expert_only = False  # keep the VLM backbone trainable

    policy = PI05Policy(config, per_dataset_stats=[lerobot_dataset_metadata.stats])

    batch_size = 1
    batch = {
        "camera0": torch.randn(batch_size, 3, 224, 224),
        "camera1": torch.randn(batch_size, 3, 224, 224),
        "state": torch.randn(batch_size, config.max_state_dim),
        "actions": torch.randn(batch_size, config.chunk_size, config.max_action_dim),
        "prompt": ["Pick up the red block"],
        "response": ["Pick up the red block"],
        "img_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }

    policy.to(dtype=torch.bfloat16, device="cuda")
    batch_cuda = {
        k: v.to("cuda", non_blocking=True, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch_cuda["action_is_pad"] = batch_cuda["action_is_pad"].to(dtype=torch.bool)

    try:
        losses = policy.forward(batch_cuda)
        assert "MSE" in losses

        vlm_name, vlm_param = _trainable_vlm_kv_param(policy)
        action_param = next(policy.model.action_out_proj.parameters())

        policy.zero_grad(set_to_none=True)
        losses["MSE"].backward()

        # MSE always trains the action expert / output projection.
        assert action_param.grad is not None and torch.count_nonzero(action_param.grad) > 0

        if knowledge_insulation:
            assert vlm_param.grad is None, (
                f"KI on: VLM weight {vlm_name} unexpectedly received an MSE gradient"
            )
        else:
            assert vlm_param.grad is not None and torch.count_nonzero(vlm_param.grad) > 0, (
                f"KI off: VLM weight {vlm_name} should receive an MSE gradient"
            )
    finally:
        # Free ~6 GB of PaliGemma weights so adjacent GPU tests don't OOM.
        del policy
        torch.cuda.empty_cache()
