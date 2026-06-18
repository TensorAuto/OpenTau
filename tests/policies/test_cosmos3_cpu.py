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

"""CPU smoke tests for the cosmos3 policy.

These exercise the inner ``Cosmos3FlowMatching`` (frozen tiny Qwen3-VL prefix + trainable
expert + flow matching) on a deliberately tiny random ``qwen3_vl`` config, so they run on
CPU in the gating ``-m "not gpu and not network"`` suite without downloading the 32B
backbone. The factory registration is covered by ``test_policies.py``.
"""

import pytest
import torch
from transformers import Qwen3VLConfig

from opentau.policies.cosmos3.configuration_cosmos3 import Cosmos3Config
from opentau.policies.cosmos3.modeling_cosmos3 import Cosmos3FlowMatching
from opentau.policies.utils import PerSampleLoss

# Small vision/text special-token ids that fit inside the tiny vocab below.
IMAGE_TOKEN_ID = 10
VIDEO_TOKEN_ID = 11
VISION_START_ID = 12
VISION_END_ID = 13
VOCAB = 200

CHUNK = 4
ACTION_DIM = 6
STATE_DIM = 5
MAX_ACTION_DIM = 8
MAX_STATE_DIM = 8


def _tiny_qwen3vl_config() -> Qwen3VLConfig:
    """A minimally-sized Qwen3-VL config (head_dim/kv-heads/layers must match the expert)."""
    return Qwen3VLConfig(
        text_config={
            "model_type": "qwen3_vl_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": VOCAB,
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
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=False,
    )


def _tiny_cosmos3_config(**overrides) -> Cosmos3Config:
    kwargs = {
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "max_action_dim": MAX_ACTION_DIM,
        "max_state_dim": MAX_STATE_DIM,
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
    return Cosmos3Config(**kwargs)


def _build_model() -> Cosmos3FlowMatching:
    torch.manual_seed(0)
    return Cosmos3FlowMatching(_tiny_cosmos3_config(), qwen3vl_config=_tiny_qwen3vl_config())


def _text_batch(bsize: int = 2, n_text: int = 6):
    input_ids = torch.randint(20, VOCAB, (bsize, n_text))
    attention_mask = torch.ones(bsize, n_text, dtype=torch.long)
    state = torch.randn(bsize, STATE_DIM)
    state = torch.nn.functional.pad(state, (0, MAX_STATE_DIM - STATE_DIM))
    actions = torch.randn(bsize, CHUNK, MAX_ACTION_DIM)
    return input_ids, attention_mask, state, actions


def _image_inputs(bsize: int = 2):
    """One tiny image per sample: grid [1, 4, 4] -> 4 merged image tokens, 16 patches."""
    grid_t, grid_h, grid_w = 1, 4, 4
    merged = (grid_h // 2) * (grid_w // 2)  # spatial_merge_size = 2 -> 2x2 = 4 tokens/image
    n_patches = grid_t * grid_h * grid_w
    patch_dim = 3 * 2 * 16 * 16  # in_channels * temporal_patch_size * patch_size**2
    # Per-sample sequence: <vision_start> <image>*merged <vision_end> + text
    img_block = [VISION_START_ID] + [IMAGE_TOKEN_ID] * merged + [VISION_END_ID]
    text_tail = [25, 30, 35]
    seq = img_block + text_tail
    input_ids = torch.tensor([seq] * bsize, dtype=torch.long)
    attention_mask = torch.ones(bsize, len(seq), dtype=torch.long)
    pixel_values = torch.randn(bsize * n_patches, patch_dim)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * bsize, dtype=torch.long)
    return input_ids, attention_mask, pixel_values, image_grid_thw


def test_cosmos3_inner_forward_text_only():
    model = _build_model()
    input_ids, attention_mask, state, actions = _text_batch()
    noise = torch.randn_like(actions)
    time = torch.rand(actions.shape[0])
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        state=state,
        actions=actions,
        noise=noise,
        time=time,
    )
    assert set(out) == {"MSE", "CE"}
    assert out["MSE"].ndim == 0 and torch.isfinite(out["MSE"])
    assert torch.equal(out["CE"], torch.zeros((), dtype=out["CE"].dtype))


def test_cosmos3_inner_forward_with_image():
    model = _build_model()
    input_ids, attention_mask, pixel_values, image_grid_thw = _image_inputs()
    bsize = input_ids.shape[0]
    state = torch.randn(bsize, MAX_STATE_DIM)
    actions = torch.randn(bsize, CHUNK, MAX_ACTION_DIM)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        state=state,
        actions=actions,
    )
    assert torch.isfinite(out["MSE"])


def test_cosmos3_backbone_is_frozen_expert_trains():
    model = _build_model()
    backbone_grad = [p.requires_grad for p in model.qwen3vl_with_expert.backbone.parameters()]
    expert_grad = [p.requires_grad for p in model.qwen3vl_with_expert.expert.parameters()]
    assert not any(backbone_grad), "backbone must be fully frozen (train_expert_only)"
    assert all(expert_grad), "expert must be trainable"
    # A backward pass populates expert grads but no backbone grads.
    input_ids, attention_mask, state, actions = _text_batch()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        state=state,
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.rand(actions.shape[0]),
    )
    out["MSE"].backward()
    assert any(p.grad is not None for p in model.qwen3vl_with_expert.expert.parameters())
    assert all(p.grad is None for p in model.qwen3vl_with_expert.backbone.parameters())


def test_cosmos3_per_sample_loss_shapes():
    model = _build_model()
    input_ids, attention_mask, state, actions = _text_batch(bsize=3)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        state=state,
        actions=actions,
        noise=torch.randn_like(actions),
        time=torch.rand(3),
        return_per_sample=True,
    )
    assert isinstance(out["MSE_per_sample"], PerSampleLoss)
    assert isinstance(out["CE_per_sample"], PerSampleLoss)
    assert out["MSE_per_sample"].sum.shape == (3,)
    assert torch.equal(out["CE_per_sample"].count, torch.zeros(3))


def test_cosmos3_forward_is_deterministic():
    """Two forwards with identical inputs/seed produce bit-identical loss (CLAUDE.md rule 3)."""
    losses = []
    for _ in range(2):
        model = _build_model()  # re-seeded identically inside _build_model
        input_ids, attention_mask, state, actions = _text_batch()
        # Re-seed the input draw so both runs see identical tensors.
        torch.manual_seed(1234)
        noise = torch.randn_like(actions)
        time = torch.rand(actions.shape[0])
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=None,
            state=state,
            actions=actions,
            noise=noise,
            time=time,
        )
        losses.append(out["MSE"].item())
    assert losses[0] == losses[1], f"non-deterministic loss: {losses}"


def test_cosmos3_sample_actions_shape():
    model = _build_model()
    input_ids, attention_mask, state, _ = _text_batch()
    bsize = input_ids.shape[0]
    action_prefix = torch.zeros(bsize, CHUNK, MAX_ACTION_DIM)
    delay = torch.tensor(0, dtype=torch.long)
    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        state=state,
        action_prefix=action_prefix,
        delay=delay,
    )
    assert actions.shape == (bsize, CHUNK, MAX_ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_cosmos3_config_validation():
    with pytest.raises(ValueError):
        Cosmos3Config(attention_implementation="flash_cuda")
    with pytest.raises(ValueError):
        Cosmos3Config(n_action_steps=100, chunk_size=50)
    with pytest.raises(ValueError):
        # query heads not a multiple of kv heads
        Cosmos3Config(expert_num_attention_heads=15, expert_num_key_value_heads=8)


def test_cosmos3_expert_under_1b_params():
    """The default (real) expert + projections must be < 1B trainable params."""
    cfg = Cosmos3Config()  # real defaults; head geometry from Qwen3-VL-32B
    # Count expert params analytically without building the 32B backbone.
    h = cfg.expert_hidden_size
    inter = cfg.expert_intermediate_size
    hq, hkv, hd = cfg.expert_num_attention_heads, cfg.expert_num_key_value_heads, cfg.expert_head_dim
    c = cfg.expert_adarms_cond_dim
    per_layer = (
        h * (hq * hd)  # q_proj
        + 2 * h * (hkv * hd)  # k_proj + v_proj
        + (hq * hd) * h  # o_proj
        + 2 * hd  # q_norm + k_norm
        + 3 * h * inter  # gated MLP
        + 2 * (c * 3 * h + 3 * h)  # two AdaRMS dense (weight + bias)
    )
    total = per_layer * cfg.expert_num_hidden_layers + (c * 3 * h + 3 * h)  # + final norm
    assert total < 1_000_000_000, f"expert has {total:,} params (>= 1B)"
