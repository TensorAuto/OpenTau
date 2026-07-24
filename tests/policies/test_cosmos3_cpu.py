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
from opentau.policies.cosmos3.modeling_cosmos3 import Cosmos3FlowMatching, Cosmos3Policy
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


def _tiny_qwen3vl_config(num_hidden_layers: int = 2) -> Qwen3VLConfig:
    """A minimally-sized Qwen3-VL config (head_dim/kv-heads/layers must match the expert)."""
    return Qwen3VLConfig(
        text_config={
            "model_type": "qwen3_vl_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": num_hidden_layers,
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


def _build_model(qwen3vl_num_layers: int = 2, **overrides) -> Cosmos3FlowMatching:
    torch.manual_seed(0)
    return Cosmos3FlowMatching(
        _tiny_cosmos3_config(**overrides), qwen3vl_config=_tiny_qwen3vl_config(qwen3vl_num_layers)
    )


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


# ---- single-layer conditioning (condition_on_layer) ----


def test_cosmos3_single_layer_forward_and_grads():
    """condition_on_layer=k: forward is finite, the backbone is truncated to k+1 layers,
    and only the expert receives gradients (the truncated backbone stays frozen)."""
    model = _build_model(condition_on_layer=0)
    we = model.qwen3vl_with_expert
    assert we.condition_on_layer == 0
    assert we.num_layers == 1
    assert len(we.backbone.model.language_model.layers) == 1  # deeper layers never allocated

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
    assert torch.isfinite(out["MSE"])
    out["MSE"].backward()
    assert any(p.grad is not None for p in we.expert.parameters())
    assert all(p.grad is None for p in we.backbone.parameters())


def test_cosmos3_single_layer_allows_shallower_expert():
    """In single-layer mode the expert depth is free (need not equal the backbone depth)."""
    model = _build_model(qwen3vl_num_layers=2, condition_on_layer=1, expert_num_hidden_layers=5)
    assert len(model.qwen3vl_with_expert.expert.layers) == 5
    assert model.qwen3vl_with_expert.num_layers == 2  # condition_on_layer=1 -> first 2 layers
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
    assert torch.isfinite(out["MSE"])


def test_cosmos3_single_layer_negative_index():
    """Negative condition_on_layer indexes from the end (Python-style)."""
    model = _build_model(qwen3vl_num_layers=2, condition_on_layer=-1, expert_num_hidden_layers=3)
    assert model.qwen3vl_with_expert.condition_on_layer == 1  # -1 -> last of 2 layers
    assert model.qwen3vl_with_expert.num_layers == 2


def test_cosmos3_single_layer_out_of_range_raises():
    with pytest.raises(ValueError):
        _build_model(qwen3vl_num_layers=2, condition_on_layer=2)  # valid indices are 0, 1
    with pytest.raises(ValueError):
        _build_model(qwen3vl_num_layers=2, condition_on_layer=-3)


def test_cosmos3_default_mode_requires_matching_depth():
    """With condition_on_layer=None the expert depth must still equal the backbone depth."""
    with pytest.raises(ValueError):
        _build_model(qwen3vl_num_layers=2, expert_num_hidden_layers=3)  # 3 != 2


def test_cosmos3_single_layer_truncation_kv_matches_full_backbone():
    """Truncating the backbone to k+1 layers yields the *same* KV at the selected layer as
    the full backbone: deepstack vision features are injected only into the earliest layers,
    so layer k's output (and its cached KV) depends only on layers 0..k."""
    k = 2  # backbone depth 4 -> truncate to 3 layers (layer 3 dropped)
    full = _build_model(qwen3vl_num_layers=4, expert_num_hidden_layers=4)  # condition_on_layer=None
    trunc = _build_model(qwen3vl_num_layers=4, condition_on_layer=k, expert_num_hidden_layers=4)
    # Pin the shared backbone weights so the comparison isolates truncation, not init RNG.
    trunc.qwen3vl_with_expert.backbone.load_state_dict(
        full.qwen3vl_with_expert.backbone.state_dict(), strict=False
    )

    input_ids, attention_mask, pixel_values, image_grid_thw = _image_inputs()

    def cached(model):
        pos, _ = model.qwen3vl_with_expert.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )
        return model.qwen3vl_with_expert.run_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    full_kv = cached(full)
    trunc_kv = cached(trunc)
    assert len(full_kv) == 4 and len(trunc_kv) == k + 1
    fk, fv = full_kv[k]
    tk, tv = trunc_kv[k]  # the selected layer is the last truncated entry
    assert torch.equal(fk, tk)
    assert torch.equal(fv, tv)


def test_cosmos3_single_layer_deterministic():
    """Two seeded single-layer forwards produce bit-identical loss (CLAUDE.md rule 3)."""
    losses = []
    for _ in range(2):
        model = _build_model(qwen3vl_num_layers=2, condition_on_layer=0, expert_num_hidden_layers=3)
        input_ids, attention_mask, state, actions = _text_batch()
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


def test_cosmos3_single_layer_sample_actions_shape():
    model = _build_model(qwen3vl_num_layers=2, condition_on_layer=0, expert_num_hidden_layers=3)
    input_ids, attention_mask, state, _ = _text_batch()
    bsize = input_ids.shape[0]
    actions = model.sample_actions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        state=state,
        action_prefix=torch.zeros(bsize, CHUNK, MAX_ACTION_DIM),
        delay=torch.tensor(0, dtype=torch.long),
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


def test_first_tensor_skips_non_tensor_prompt():
    """``_first_tensor`` must skip the non-tensor ``prompt`` entry (the wrapper relies on it
    to infer device / batch size without ``AttributeError`` on the string list)."""
    from opentau.policies.cosmos3.modeling_cosmos3 import _first_tensor

    batch = {"prompt": ["pick up the block", "place it"], "actions": torch.zeros(2, CHUNK, MAX_ACTION_DIM)}
    ref = _first_tensor(batch)
    assert isinstance(ref, torch.Tensor)
    assert ref.shape[0] == 2


class _FakeTokenizer:
    """Records encode/decode calls; one whitespace-separated word == one token id."""

    def __init__(self):
        self.encode_calls: list[tuple[str, list[int]]] = []
        self.decode_calls: list[list[int]] = []

    def encode(self, text, add_special_tokens=False):
        ids = list(range(len(text.split())))
        self.encode_calls.append((text, ids))
        return ids

    def decode(self, ids):
        ids = list(ids)
        self.decode_calls.append(ids)
        return " ".join(f"t{i}" for i in ids)


class _FakeProcessor:
    """Minimal stand-in for the Qwen3-VL processor that records what it is handed."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.seen_texts = None
        self.seen_images = "UNSET"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Flatten just the text content so the test can read back the (truncated) prompt.
        return " ".join(c["text"] for m in messages for c in m["content"] if c["type"] == "text")

    def __call__(self, text, images, return_tensors, padding):
        self.seen_texts = text
        self.seen_images = images
        bsize, seqlen = len(text), 6
        return {
            "input_ids": torch.zeros(bsize, seqlen, dtype=torch.long),
            "attention_mask": torch.ones(bsize, seqlen, dtype=torch.long),
        }


def test_cosmos3_prompt_max_length_truncation():
    """``prepare_multimodal_inputs`` must truncate the language prompt to
    ``prompt_max_length`` tokens (text only -- image placeholders are never clipped).

    This locks the wrapper-level truncation on the CPU gate; the real Qwen3-VL processor
    is mocked so no 32B backbone / network is needed.
    """
    cfg = _tiny_cosmos3_config(prompt_max_length=4)
    policy = Cosmos3Policy(cfg, qwen3vl_config=_tiny_qwen3vl_config())
    policy.processor = _FakeProcessor()

    batch = {
        "prompt": ["a b c d e f g h", "x y"],  # 8 tokens -> truncate to 4; 2 tokens -> keep
        "state": torch.zeros(2, MAX_STATE_DIM),
    }
    mm = policy.prepare_multimodal_inputs(batch)

    tok = policy.processor.tokenizer
    # The long prompt was sliced to prompt_max_length ids; the short one passed through.
    assert tok.decode_calls == [[0, 1, 2, 3], [0, 1]]
    # The (decoded) text reaching the processor reflects the per-sample truncation.
    assert [len(t.split()) for t in policy.processor.seen_texts] == [4, 2]
    # No image features -> the image argument is None, not an empty list.
    assert policy.processor.seen_images is None
    assert mm["input_ids"].shape[0] == 2


def test_cosmos3_partial_unfreeze_backbone_receives_grad():
    """With train_expert_only=False the cached KV keeps its graph, so the unfrozen text
    backbone receives gradients (otherwise unfreezing would be a silent no-op)."""
    torch.manual_seed(0)
    cfg = _tiny_cosmos3_config(train_expert_only=False, freeze_vision_encoder=True)
    model = Cosmos3FlowMatching(cfg, qwen3vl_config=_tiny_qwen3vl_config())
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
    text_backbone = model.qwen3vl_with_expert.backbone.model.language_model
    assert any(p.grad is not None for p in text_backbone.parameters()), (
        "unfrozen text backbone must receive gradients via the (non-detached) cached KV"
    )


def test_cosmos3_train_vision_encoder_only_config_validation():
    # cosmos3 defaults train_expert_only=True, so the bare combo is mutually exclusive.
    with pytest.raises(ValueError, match="mutually exclusive"):
        _tiny_cosmos3_config(train_vision_encoder_only=True)
    # With the expert-only regime off, the vision encoder must be unfrozen.
    with pytest.raises(ValueError, match="freeze_vision_encoder"):
        _tiny_cosmos3_config(train_expert_only=False, train_vision_encoder_only=True)


def test_cosmos3_train_vision_encoder_only_freezes_all_but_visual():
    model = _build_model(
        train_expert_only=False,
        freeze_vision_encoder=False,
        train_vision_encoder_only=True,
    )
    we = model.qwen3vl_with_expert

    # The Qwen3-VL vision tower (its merger / deepstack projector lives inside it) trains ...
    assert all(p.requires_grad for p in we.backbone.model.visual.parameters())
    # ... the LLM backbone and the action expert are frozen ...
    assert not any(p.requires_grad for p in we.backbone.model.language_model.parameters())
    assert not any(p.requires_grad for p in we.expert.parameters())
    # ... and every policy-level projection is frozen.
    for proj in (
        model.action_in_proj,
        model.action_out_proj,
        model.time_mlp_in,
        model.time_mlp_out,
        model.adarms_proj,
        model.state_proj,
    ):
        assert not any(p.requires_grad for p in proj.parameters())

    # Nothing trainable lives outside the visual tower.
    visual_ids = {id(p) for p in we.backbone.model.visual.parameters()}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert id(p) in visual_ids, f"{name} unexpectedly trainable"


def test_cosmos3_train_vision_encoder_only_grad_flows_to_visual():
    """The no_grad ctx in run_prefix must stay off (train_expert_only=False), so the
    vision tower's loss backpropagates into ``backbone.model.visual``."""
    model = _build_model(
        train_expert_only=False,
        freeze_vision_encoder=False,
        train_vision_encoder_only=True,
    )
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
        noise=torch.randn_like(actions),
        time=torch.rand(bsize),
    )
    out["MSE"].backward()

    we = model.qwen3vl_with_expert
    assert any(p.grad is not None for p in we.backbone.model.visual.parameters()), (
        "vision tower must receive gradients under train_vision_encoder_only"
    )
    assert all(p.grad is None for p in we.expert.parameters())
    assert all(p.grad is None for p in we.backbone.model.language_model.parameters())
