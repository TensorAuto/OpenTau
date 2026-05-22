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

"""DeepSpeed ZeRO-2 smoke for ``PaliGemmaWithExpertModel`` attention backends.

Builds a small ``PaliGemmaWithExpertModel`` with synthetic inputs, wraps it
in ``accelerate.Accelerator`` (DeepSpeed ZeRO-2 backend), runs a handful
of optimizer steps, and asserts that:

  - the loss is finite on every rank,
  - the loss decreases (well-conditioned for a synthetic target),
  - no NCCL collective is mismatched in size across ranks (would
    surface as a hang/abort).

The point is exercising the same fwd/bwd path on the same DeepSpeed
config the team trains under (see ``configs/examples/accelerate_deepspeed_config.yaml``,
``zero_stage: 2``) — flex_attention's Triton kernel runs *inside* each
rank's autograd graph, so the DeepSpeed contract is purely "gradients
look normal at all-reduce time." If that holds for sdpa it should hold
for flex; this script proves it concretely.

Run on 2x GPUs::

    accelerate launch --config_file configs/examples/accelerate_deepspeed_config.yaml \\
        --num_processes 2 \\
        src/opentau/scripts/smoke_paligemma_attention_deepspeed.py \\
        --attention-implementation flex
"""

from __future__ import annotations

import argparse

import torch
from accelerate import Accelerator
from transformers.models.auto import CONFIG_MAPPING

from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)


def _build_tiny_engine_config(attention_implementation: str) -> PaliGemmaWithExpertConfig:
    """Smallest config that still drives the real ``forward`` path end-to-end."""
    cfg = PaliGemmaWithExpertConfig(
        freeze_vision_encoder=False,
        train_expert_only=False,
        dropout=0.0,
        attention_implementation=attention_implementation,
        discrete_action_vocab_size=64,
    )
    # Tiny PaliGemma — same trick used by tests/policies/test_pi0.py's
    # ``_make_tiny_pi0_engine_config``: build with defaults then overwrite the
    # nested configs to keep the parameter count manageable.
    cfg.paligemma_config = CONFIG_MAPPING["paligemma"](
        bos_token_id=2,
        eos_token_id=1,
        hidden_size=128,
        image_token_index=64,
        pad_token_id=0,
        projection_dim=128,
        text_config={
            "model_type": "gemma",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "head_dim": 256,  # exercise the head_dim=256 path the real model uses
            "vocab_size": 128,
            "max_position_embeddings": 256,
            "torch_dtype": "float32",
            "hidden_activation": "gelu_pytorch_tanh",
            "use_adarms": False,
            "adarms_cond_dim": None,
        },
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_image_tokens": 4,
            "patch_size": 14,
            "projection_dim": 128,
            "projector_hidden_act": "gelu_fast",
            "vision_use_head": False,
        },
    )
    cfg.gemma_expert_config = CONFIG_MAPPING["gemma"](
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=256,
        vocab_size=128,
        torch_dtype="float32",
        hidden_activation="gelu_pytorch_tanh",
        use_adarms=False,
        adarms_cond_dim=None,
    )
    return cfg


def _make_synthetic_batch(b: int, s_prefix: int, s_action: int, hidden: int, device, dtype):
    torch.manual_seed(0)
    prefix = torch.randn(b, s_prefix, hidden, device=device, dtype=dtype)
    action = torch.randn(b, s_action, hidden, device=device, dtype=dtype)

    total = s_prefix + s_action
    pad_masks = torch.ones(b, total, dtype=torch.bool, device=device)
    att_masks = torch.zeros(b, total, dtype=torch.int32, device=device)
    # block-causal: prefix attends bidirectionally to itself; action causal w.r.t. self + prefix.
    att_masks[:, s_prefix] = 1
    for i in range(1, s_action):
        att_masks[:, s_prefix + i] = 1

    cumsum = torch.cumsum(att_masks, dim=1)
    attention_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    attention_mask = attention_mask & (pad_masks[:, None, :] & pad_masks[:, :, None])

    position_ids = torch.arange(total, device=device).unsqueeze(0).expand(b, -1)
    target = torch.randn(b, s_action, hidden, device=device, dtype=dtype)
    return [prefix, action], attention_mask, position_ids, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-implementation", default="flex", choices=["eager", "sdpa", "flex"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    # flex_attention's Inductor lowering on Ampere needs Q_LEN to be a
    # multiple of 64 (see ``_flex_can_use``); defaults here pick shapes
    # that exercise the real fused kernel rather than the SDPA fallback.
    parser.add_argument("--s-prefix", type=int, default=128)
    parser.add_argument("--s-action", type=int, default=128)
    args = parser.parse_args()

    accelerator = Accelerator()
    # Under DeepSpeed, accelerate.prepare() needs to know the micro-batch size
    # from either (a) a prepared dataloader or (b) the deepspeed_config dict.
    # We're using synthetic in-script batches, so populate the dict here.
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
        # The example accelerate_deepspeed_config.yaml uses ``mixed_precision: no``
        # because the team's real training script ships its own master-weight /
        # bf16 plumbing (see scripts/train.py). For this self-contained smoke we
        # just enable DeepSpeed's native bf16 mode so the bf16-cast paligemma
        # weights round-trip through deepspeed.initialize unchanged.
        ds_cfg["bf16"] = {"enabled": True}
    device = accelerator.device

    cfg = _build_tiny_engine_config(args.attention_implementation)
    model = PaliGemmaWithExpertModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)

    hidden = cfg.paligemma_config.text_config.hidden_size
    losses = []
    for step in range(args.steps):
        inputs_embeds, attention_mask, position_ids, target = _make_synthetic_batch(
            args.batch_size, args.s_prefix, args.s_action, hidden, device, torch.bfloat16
        )

        out_embeds, _ = model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            n_cross_att_tokens=None,
            use_cache=False,
            fill_kv_cache=False,
        )
        action_out = out_embeds[1]  # action-expert output
        loss = torch.nn.functional.mse_loss(action_out.float(), target.float())
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        gathered = accelerator.gather(loss.detach().unsqueeze(0)).cpu().tolist()
        if accelerator.is_main_process:
            print(f"[step {step}] loss per rank: {gathered}")
        losses.append(loss.item())
        assert torch.isfinite(loss), f"rank {accelerator.process_index} step {step} loss not finite"

    if accelerator.is_main_process:
        # On well-conditioned synthetic data with a learnable target, the loss
        # should drop by step `steps-1` — same monotonic-improvement check the
        # team uses for engine-level smokes.
        assert losses[-1] < losses[0], f"loss did not improve: first={losses[0]} last={losses[-1]}"
        print(f"OK: {args.attention_implementation} loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
