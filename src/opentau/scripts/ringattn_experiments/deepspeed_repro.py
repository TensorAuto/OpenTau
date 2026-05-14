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

"""Local DeepSpeed ZeRO-2 reproducer for the step-2 NaN.

Mirrors the bf16-ZeRO-2 production setup as closely as 2× RTX 3090s can:
- DeepSpeed engine wraps a tiny Gemma 3 with expert
- AdamW optimizer
- 2 ranks, single ring spanning WORLD
- 5 training steps on synthetic prefix inputs
- Reports loss + grad-norm + any NaN-bearing parameter after each step

Run with::

    accelerate launch --config_file /tmp/zero2-2gpu-bf16.yaml \\
        -m opentau.scripts.ringattn_experiments.deepspeed_repro
"""

from __future__ import annotations

import logging

import torch
from accelerate import Accelerator

from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertModel
from opentau.scripts.ringattn_experiments.common import (
    attention_implementation,
    build_tiny_gemma3_config,
    synthesise_prefix_inputs,
)


def _check_finite(model, label):
    """Print any non-finite parameters / gradients on this rank."""
    rank = torch.distributed.get_rank()
    bad = []
    for name, p in model.named_parameters():
        nf_p = (~torch.isfinite(p)).sum().item()
        nf_g = (~torch.isfinite(p.grad)).sum().item() if p.grad is not None else 0
        if nf_p > 0 or nf_g > 0:
            bad.append((name, nf_p, nf_g, p.numel()))
    if bad:
        print(f"[rank {rank}] {label}: {len(bad)} bad params:")
        for name, nfp, nfg, total in bad[:10]:
            print(f"    {name}: param non-finite={nfp}/{total} grad non-finite={nfg}")
    else:
        print(f"[rank {rank}] {label}: all params/grads finite")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    accelerator = Accelerator()
    rank = accelerator.process_index
    device = accelerator.device

    torch.manual_seed(0 + rank * 0)  # same seed across ranks => same data

    cfg = build_tiny_gemma3_config(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=1024,
    )
    cfg.attention_implementation = "ring"
    model = Gemma3WithExpertModel(config=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Tell DeepSpeed the micro-batch size since we don't pass a dataloader.
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    seq_len = 1024
    hidden = 512
    embs, attn_mask, position_ids = synthesise_prefix_inputs(1, seq_len, hidden, device=device, seed=42)
    # Same data on every rank — what ring SP expects.
    torch.distributed.broadcast(embs, src=0)
    torch.distributed.broadcast(attn_mask, src=0)
    torch.distributed.broadcast(position_ids, src=0)

    # pi07-like discrete-action head + CE loss path. Project the last
    # `da_slice_len` positions of the model output to a small vocab and CE-
    # supervise against random labels. This exercises the SAME gradient
    # pattern as production: only a slice of the seq has non-zero grad_O.
    vocab_size = 64
    da_head = torch.nn.Linear(hidden, vocab_size, bias=False, dtype=torch.bfloat16, device=device)
    torch.distributed.broadcast(da_head.weight.data, src=0)
    da_slice_len = 32
    labels = torch.randint(0, vocab_size, (1, da_slice_len), device=device, dtype=torch.long)
    torch.distributed.broadcast(labels, src=0)

    for step in range(5):
        embs_step = embs.detach().clone().requires_grad_(True)
        inner = accelerator.unwrap_model(model)
        with attention_implementation(inner, "ring"):
            (out, _), _ = model.forward(
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs_step, None],
                n_cross_att_tokens=seq_len,
                use_cache=False,
                fill_kv_cache=True,
            )
        assert out is not None
        # Slice the last positions and compute CE; only this slice has
        # non-zero gradient back to the model output, mirroring pi07.
        slice_out = out[:, -da_slice_len:]
        logits = da_head(slice_out).float()
        logits = logits.reshape(-1, vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, labels.reshape(-1))
        accelerator.backward(loss)
        if accelerator.is_main_process:
            n_nf = (~torch.isfinite(out)).sum().item()
            print(f"=== step {step} ===")
            print(f"  forward.out non-finite: {n_nf}/{out.numel()}")
            print(f"  loss: {loss.item():.6g}")
        _check_finite(model, f"step {step} after backward")
        optimizer.step()
        optimizer.zero_grad()
        _check_finite(model, f"step {step} after optimizer.step")

    accelerator.end_training()


if __name__ == "__main__":
    main()
