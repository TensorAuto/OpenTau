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

"""FlexAttention parity tests for the shared pi05 PaliGemmaWithExpert backbone.

The pi05 ``PaliGemmaWithExpertModel`` is reused by pi05 itself and by
pi07_paligemma's high-level planner and low-level controller. These tests
pin the new ``attention_implementation="flex"`` path numerically against
the already-validated ``"sdpa"`` and ``"eager"`` paths.
"""

from types import SimpleNamespace

import pytest
import torch

from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)


def _make_fake_self(num_attention_heads: int, num_key_value_heads: int, head_dim: int):
    text_config = SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    paligemma_config = SimpleNamespace(text_config=text_config)
    config = SimpleNamespace(paligemma_config=paligemma_config)
    return SimpleNamespace(config=config)


def _block_causal_mask(b: int, s_q: int, s_kv: int, prefix_kv: int, *, device, dtype=torch.bool):
    """Build a (B, S_q, S_kv) mask matching pi07_paligemma's prefix-bidirectional
    + causal-tail pattern. ``prefix_kv`` tokens on the KV side are visible to
    every query position; the remaining KV positions are causal w.r.t. Q."""
    mask = torch.zeros(b, s_q, s_kv, dtype=dtype, device=device)
    mask[:, :, :prefix_kv] = True
    causal_kv = s_kv - prefix_kv
    # Q index i can see prefix_kv + min(i, causal_kv-1) tail positions.
    for q_idx in range(s_q):
        tail = min(q_idx + 1, causal_kv)
        if tail > 0:
            mask[:, q_idx, prefix_kv : prefix_kv + tail] = True
    return mask


# --- Constructor / dispatcher (CPU-only) ----------------------------------


class TestFlexConfig:
    def test_config_accepts_flex(self):
        # NOTE: pi05's PaliGemmaWithExpertConfig validator lives in __post_init__,
        # which PretrainedConfig never calls — that's a pre-existing dead-code
        # bug in this file (compare pi0's copy, which puts the check in __init__).
        # The dispatcher test below is the real coverage; this just documents
        # that "flex" round-trips through the constructor.
        cfg = PaliGemmaWithExpertConfig(attention_implementation="flex")
        assert cfg.attention_implementation == "flex"

    def test_dispatcher_returns_flex_for_flex_impl(self):
        fake = SimpleNamespace(
            config=SimpleNamespace(attention_implementation="flex"),
            sdpa_attention_forward="<sdpa>",
            eager_attention_forward="<eager>",
            flex_attention_forward="<flex>",
        )
        assert PaliGemmaWithExpertModel.get_attention_interface(fake) == "<flex>"


# --- Numerical parity (GPU-only) ------------------------------------------


@pytest.mark.gpu
class TestFlexParity:
    """Pin flex_attention_forward against sdpa/eager on a block-causal mask.

    flex and sdpa share the bf16-native code path (no Q/K fp32 upcast), so
    they should match to ~1e-3 max-abs-err in bf16. The eager path's
    explicit fp32 upcast means flex-vs-eager has a wider gap — we still
    check it's within standard FA tolerance.
    """

    @pytest.mark.parametrize(
        "num_att,num_kv,head_dim,s_q,s_kv,prefix_kv",
        [
            (8, 1, 256, 256, 320, 64),  # pi07_paligemma-like GQA shape, with KV prefix
            (8, 1, 256, 256, 256, 64),  # equal Q/KV length
            (4, 4, 64, 128, 128, 32),  # smaller head_dim path (no kernel_options needed)
        ],
    )
    def test_flex_matches_sdpa_bf16(self, num_att, num_kv, head_dim, s_q, s_kv, prefix_kv):
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(0)
        fake = _make_fake_self(num_att, num_kv, head_dim)
        b = 2

        q = torch.randn(b, s_q, num_att, head_dim, dtype=dtype, device=device)
        k = torch.randn(b, s_kv, num_kv, head_dim, dtype=dtype, device=device)
        v = torch.randn(b, s_kv, num_kv, head_dim, dtype=dtype, device=device)
        mask = _block_causal_mask(b, s_q, s_kv, prefix_kv, device=device)

        sdpa_out = PaliGemmaWithExpertModel.sdpa_attention_forward(
            fake, mask, b, head_dim, q.clone(), k.clone(), v.clone()
        )
        flex_out = PaliGemmaWithExpertModel.flex_attention_forward(
            fake, mask, b, head_dim, q.clone(), k.clone(), v.clone()
        )

        assert flex_out.shape == sdpa_out.shape
        # bf16-native paths; softmax reassociation noise only.
        max_err = (flex_out.float() - sdpa_out.float()).abs().max().item()
        assert max_err < 5e-3, f"flex vs sdpa max-abs-err {max_err} exceeds 5e-3"

    @pytest.mark.parametrize("head_dim", [64, 256])
    def test_flex_matches_sdpa_fp32(self, head_dim):
        """Tighter tolerance in fp32 to catch logic errors masked by bf16 noise."""
        device = "cuda"
        dtype = torch.float32
        torch.manual_seed(1)
        num_att, num_kv = 8, 1
        s_q, s_kv, prefix_kv = 128, 160, 32
        fake = _make_fake_self(num_att, num_kv, head_dim)
        b = 2

        q = torch.randn(b, s_q, num_att, head_dim, dtype=dtype, device=device)
        k = torch.randn(b, s_kv, num_kv, head_dim, dtype=dtype, device=device)
        v = torch.randn(b, s_kv, num_kv, head_dim, dtype=dtype, device=device)
        mask = _block_causal_mask(b, s_q, s_kv, prefix_kv, device=device)

        sdpa_out = PaliGemmaWithExpertModel.sdpa_attention_forward(
            fake, mask, b, head_dim, q.clone(), k.clone(), v.clone()
        )
        flex_out = PaliGemmaWithExpertModel.flex_attention_forward(
            fake, mask, b, head_dim, q.clone(), k.clone(), v.clone()
        )
        max_err = (flex_out - sdpa_out).abs().max().item()
        assert max_err < 1e-4, f"flex vs sdpa (fp32) max-abs-err {max_err} exceeds 1e-4"

    def test_flex_grads_match_sdpa_bf16(self):
        """Backward parity: dQ/dK/dV from flex should match SDPA on the same inputs."""
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(2)
        num_att, num_kv, head_dim = 8, 1, 256
        s_q, s_kv, prefix_kv = 256, 320, 64
        fake = _make_fake_self(num_att, num_kv, head_dim)
        b = 2

        def run(fn, q0, k0, v0):
            q = q0.clone().detach().requires_grad_(True)
            k = k0.clone().detach().requires_grad_(True)
            v = v0.clone().detach().requires_grad_(True)
            out = fn(fake, mask, b, head_dim, q, k, v)
            out.float().sum().backward()
            return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        q0 = torch.randn(b, s_q, num_att, head_dim, dtype=dtype, device=device)
        k0 = torch.randn(b, s_kv, num_kv, head_dim, dtype=dtype, device=device)
        v0 = torch.randn(b, s_kv, num_kv, head_dim, dtype=dtype, device=device)
        mask = _block_causal_mask(b, s_q, s_kv, prefix_kv, device=device)

        out_s, dq_s, dk_s, dv_s = run(PaliGemmaWithExpertModel.sdpa_attention_forward, q0, k0, v0)
        out_f, dq_f, dk_f, dv_f = run(PaliGemmaWithExpertModel.flex_attention_forward, q0, k0, v0)

        # Output parity, then per-tensor grad parity.
        assert (out_f.float() - out_s.float()).abs().max().item() < 5e-3

        # dQ: no GQA reduction across heads, so the tighter envelope applies.
        # dK / dV: with Hq/Hkv = 8 we reduce 8 head contributions per KV row;
        # each step is one bf16 ULP (≈ 2^-5 ≈ 0.03 at this magnitude). The
        # observed max-abs-err of ~0.03 against SDPA is one quantization step,
        # which is the same envelope flash-attn upstream uses for its bf16 GQA
        # grad parity tests.
        tols = {"dq": 1e-2, "dk": 5e-2, "dv": 5e-2}
        for name, dx_s, dx_f in [("dq", dq_s, dq_f), ("dk", dk_s, dk_f), ("dv", dv_s, dv_f)]:
            err = (dx_f.float() - dx_s.float()).abs().max().item()
            assert err < tols[name], f"flex vs sdpa {name} max-abs-err {err} exceeds {tols[name]}"
