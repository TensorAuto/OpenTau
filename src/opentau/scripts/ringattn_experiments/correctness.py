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

"""Numerical-equivalence check: ring vs SDPA on the tiny pi07 prefix forward.

Run on N GPUs with::

    torchrun --nproc_per_node=2 -m opentau.scripts.ringattn_experiments.correctness

What's checked
--------------
The script runs three forwards on the same seeded inputs (same model
parameters, broadcast from rank 0):

1. ``eager`` — fp32 score path, bf16 outputs. The most numerically stable
   reference but uses a math chain different from ring's online softmax.
2. ``sdpa`` — PyTorch's fused kernel; what the model uses by default in
   production.
3. ``ring`` — our paper-style ring attention (this PR).

The interesting comparison is **ring vs SDPA**: ring is the new code
path, SDPA is what it replaces. The eager vs SDPA diff is reported as a
baseline so a reader can tell whether ring's discrepancy is "this is how
much bf16 reassociation noise the model already eats today" or "this is
new noise introduced by ring".

Bit-equivalence of the *attention kernel* alone is proven separately by
``unit_test.py`` — there the max abs diff between eager and ring is at
the 1e-7 level when computed in identical fp32 chains, which is all we
need to be sure the algebra is right.
"""

from __future__ import annotations

import torch

from opentau.scripts.ringattn_experiments.common import (
    attention_implementation,
    build_model,
    build_tiny_gemma3_config,
    cleanup_distributed,
    init_distributed,
    synthesise_prefix_inputs,
)


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def main() -> None:
    rank, world_size, device = init_distributed()
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 256
    hidden_size = 256

    cfg = build_tiny_gemma3_config(hidden_size=hidden_size)
    model = build_model(cfg, device=device).eval()
    # Broadcast parameters and buffers from rank 0 so every rank starts from
    # an identical model (HF init can be non-deterministic across ranks).
    for p in model.parameters():
        torch.distributed.broadcast(p.data, src=0)
    for b in model.buffers():
        torch.distributed.broadcast(b.data, src=0)

    embs, attn_mask, position_ids = synthesise_prefix_inputs(
        batch_size, seq_len, hidden_size, device=device, seed=42
    )

    def run(impl: str) -> torch.Tensor:
        with attention_implementation(model, impl):
            (out, _), _ = model.forward(
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                n_cross_att_tokens=seq_len,
                use_cache=False,
                fill_kv_cache=True,
            )
        assert out is not None
        return out

    with torch.no_grad():
        eager_out = run("eager")
        sdpa_out = run("sdpa")
        ring_out = run("ring")

    eager_norm = eager_out.float().norm().item()
    sdpa_norm = sdpa_out.float().norm().item()
    ring_norm = ring_out.float().norm().item()
    eager_vs_sdpa = _diff(eager_out, sdpa_out)
    eager_vs_ring = _diff(eager_out, ring_out)
    sdpa_vs_ring = _diff(sdpa_out, ring_out)

    if rank == 0:
        print(f"world_size = {world_size}")
        print(f"output shape = {tuple(eager_out.shape)}")
        print(f"||eager|| = {eager_norm:.4f}   ||sdpa|| = {sdpa_norm:.4f}   ||ring|| = {ring_norm:.4f}")
        print(f"max |eager - sdpa|  = {eager_vs_sdpa[0]:.3e}   mean = {eager_vs_sdpa[1]:.3e}")
        print(f"max |eager - ring|  = {eager_vs_ring[0]:.3e}   mean = {eager_vs_ring[1]:.3e}")
        print(f"max |sdpa  - ring|  = {sdpa_vs_ring[0]:.3e}   mean = {sdpa_vs_ring[1]:.3e}")
        # Ring should be ~ as close to SDPA as SDPA is to eager. We give a
        # 1.5x tolerance so a slightly different reassociation order doesn't
        # trip the check.
        ok = sdpa_vs_ring[0] <= 1.5 * eager_vs_sdpa[0]
        print("PASS" if ok else "FAIL")

    cleanup_distributed()


if __name__ == "__main__":
    main()
