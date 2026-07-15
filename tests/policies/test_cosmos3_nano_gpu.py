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

"""GPU smoke tests for the cosmos3_nano policy.

cosmos3_nano shares all modeling code with cosmos3 (``test_cosmos3_gpu.py`` covers the
architecture on CUDA, including single-layer conditioning); these run the inner
``Cosmos3FlowMatching`` built from a ``Cosmos3NanoConfig`` on CUDA in bf16 with a tiny
random ``qwen3_vl`` config (no 8B download), locking the nano config plumbing on GPU:
bf16 forward + backward with the frozen-backbone / trainable-expert split, and
bit-identical loss determinism. The full 8B (``TensorAuto/cosmos3-reason-8b``) load +
memory-fit is exercised separately as a manual smoke on the target hardware.
"""

import pytest
import torch
from transformers import Qwen3VLConfig

from opentau.policies.cosmos3.modeling_cosmos3 import Cosmos3FlowMatching
from opentau.policies.cosmos3_nano.configuration_cosmos3_nano import Cosmos3NanoConfig

CHUNK = 4
MAX_ADIM, MAX_SDIM = 8, 8


def _tiny_qwen3vl_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config={
            "model_type": "qwen3_vl_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 200,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
        },
        vision_config={
            "model_type": "qwen3_vl",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "depth": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [1, 2],
            "num_position_embeddings": 256,
            "in_channels": 3,
        },
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        tie_word_embeddings=False,
    )


def _tiny_config(**overrides) -> Cosmos3NanoConfig:
    kwargs = {
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "max_action_dim": MAX_ADIM,
        "max_state_dim": MAX_SDIM,
        "proj_width": 16,
        "num_steps": 3,
        "attention_implementation": "eager",
        "load_pretrained_backbone": False,
        "dropout": 0.0,
        "expert_hidden_size": 32,
        "expert_intermediate_size": 64,
        "expert_num_hidden_layers": 2,
        "expert_num_attention_heads": 4,
        "expert_num_key_value_heads": 2,
        "expert_head_dim": 16,
        "expert_adarms_cond_dim": 16,
    }
    kwargs.update(overrides)
    return Cosmos3NanoConfig(**kwargs)


def _build(device, **overrides):
    torch.manual_seed(0)
    model = Cosmos3FlowMatching(_tiny_config(**overrides), qwen3vl_config=_tiny_qwen3vl_config())
    return model.to(device=device, dtype=torch.bfloat16)


def _batch(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(20, 200, (2, 6), generator=g).to(device)
    attention_mask = torch.ones(2, 6, dtype=torch.long, device=device)
    state = torch.randn(2, MAX_SDIM, generator=g).to(device)
    actions = torch.randn(2, CHUNK, MAX_ADIM, generator=g).to(device)
    noise = torch.randn(2, CHUNK, MAX_ADIM, generator=g).to(device)
    time = torch.rand(2, generator=g).to(device)
    return input_ids, attention_mask, state, actions, noise, time


@pytest.mark.gpu
def test_cosmos3_nano_gpu_forward_backward():
    """bf16 forward + backward on CUDA via Cosmos3NanoConfig: finite loss, expert grads
    flow, backbone frozen."""
    device = "cuda"
    model = _build(device)
    iid, am, st, act, noise, time = _batch(device)
    out = model(
        input_ids=iid,
        attention_mask=am,
        pixel_values=None,
        image_grid_thw=None,
        state=st,
        actions=act,
        noise=noise,
        time=time,
    )
    assert torch.isfinite(out["MSE"])
    out["MSE"].backward()
    expert = model.qwen3vl_with_expert.expert
    backbone = model.qwen3vl_with_expert.backbone
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in expert.parameters())
    assert all(p.grad is None for p in backbone.parameters())


@pytest.mark.gpu
def test_cosmos3_nano_gpu_determinism():
    """Two eval forwards with identical inputs give a bit-identical loss on GPU."""
    device = "cuda"
    losses = []
    for _ in range(2):
        model = _build(device).eval()
        iid, am, st, act, noise, time = _batch(device, seed=3)
        with torch.no_grad():
            out = model(
                input_ids=iid,
                attention_mask=am,
                pixel_values=None,
                image_grid_thw=None,
                state=st,
                actions=act,
                noise=noise,
                time=time,
            )
        losses.append(out["MSE"].item())
    assert losses[0] == losses[1], f"non-deterministic loss on GPU: {losses}"
