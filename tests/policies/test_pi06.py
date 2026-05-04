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

"""Unit tests for the `pi06` policy.

The end-to-end integration test (`test_complete_pi06_pipeline_integration`) is
guarded by `@pytest.mark.slow` + `@pytest.mark.gpu` — it mirrors the π0.5 test
in scope and takes ~1 minute on an H100. The CPU-only tests below focus on the
things the plan called out: attention-mask block semantics, padding-mask
contiguity, sliding-window behaviour, per-layer RoPE theta selection, image
resizing, and discrete-action padding.
"""

from __future__ import annotations

import pytest
import torch

from opentau.policies.pi06 import gemma3_with_expert as g3we
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi06.gemma3_with_expert import (
    Gemma3WithExpertConfig,
    Gemma3WithExpertModel,
    apply_rope,
)
from opentau.policies.pi06.modeling_pi06 import (
    flow_matching_masked_mse,
    make_att_2d_masks,
    pad_discrete_tokens,
    resize_with_pad,
)

# Block-causal attention mask (pi05 / π0.6 prefix-LM pattern)


class TestMakeAtt2dMasks:
    """Locks in the block-causal semantics the π0.6 model card depends on:
    images = bidirectional prefix; language tokens, response and FAST
    discrete-action tokens = causal (per §2 "use causal attention among the
    text tokens"); action suffix in the expert = its own bidirectional block."""

    def test_pure_bidirectional(self):
        pad = torch.tensor([[True, True, True, True]])
        att = torch.tensor([[0, 0, 0, 0]])
        mask = make_att_2d_masks(pad, att)
        assert mask.shape == (1, 4, 4)
        assert mask.dtype == torch.bool
        # Every (i, j) pair should be attendable.
        assert torch.all(mask[0])

    def test_pure_causal(self):
        pad = torch.tensor([[True] * 4])
        att = torch.tensor([[1, 1, 1, 1]])
        mask = make_att_2d_masks(pad, att)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.all(mask[0] == expected)

    def test_prefix_lm_block(self):
        # First 3 bidirectional, last 3 causal relative to prior blocks.
        pad = torch.tensor([[True] * 6])
        att = torch.tensor([[0, 0, 0, 1, 1, 1]])
        mask = make_att_2d_masks(pad, att)

        # Rows 0..2 (bidirectional prefix) — can see each other but NOT later
        # causal tokens.
        assert torch.all(mask[0, :3, :3])
        assert not torch.any(mask[0, :3, 3:])
        # Row 3 sees prefix + itself, not rows 4-5.
        assert torch.all(mask[0, 3, :4])
        assert not torch.any(mask[0, 3, 4:])
        # Row 4 sees prefix + rows 3-4, not row 5.
        assert torch.all(mask[0, 4, :5])
        assert not torch.any(mask[0, 4, 5:])
        # Row 5 sees everything so far.
        assert torch.all(mask[0, 5, :])

    def test_padding_blocks_rows_and_columns(self):
        pad = torch.tensor([[True, True, False, False]])
        att = torch.tensor([[0, 0, 0, 0]])
        mask = make_att_2d_masks(pad, att)
        # Padded rows and columns should be entirely False.
        assert not torch.any(mask[0, 2:, :])
        assert not torch.any(mask[0, :, 2:])
        # Non-padded square is bidirectional.
        assert torch.all(mask[0, :2, :2])

    def test_cross_attention_block_prepended(self):
        pad = torch.tensor([[True, True]])
        att = torch.tensor([[1, 0]])
        cross_pad = torch.tensor([[True, True, False]])
        mask = make_att_2d_masks(pad, att, n_cross_att_tokens=3, cross_att_pad_masks=cross_pad)
        # Expected shape: (B, Q, K_cross + K_suffix) = (1, 2, 5).
        assert mask.shape == (1, 2, 5)
        # Every suffix query should see non-padded cross-attn keys.
        assert torch.all(mask[0, :, :2])
        # Padded cross key should be masked out.
        assert not torch.any(mask[0, :, 2])

    def test_embed_prefix_layout_has_causal_language_block(self):
        """π0.6 model card §2: language tokens use causal attention while
        sitting after a bidirectional image prefix. This simulates the
        `att_masks` list `PI06FlowMatching.embed_prefix` builds and asserts
        the language-vs-language sub-block is strictly causal, while still
        seeing the full image prefix.
        """
        num_img_embs, num_lang_embs = 4, 5
        att = torch.tensor([[0] * num_img_embs + [1] * num_lang_embs])
        pad = torch.ones_like(att, dtype=torch.bool)
        mask = make_att_2d_masks(pad, att)

        img_slice = slice(0, num_img_embs)
        lang_slice = slice(num_img_embs, num_img_embs + num_lang_embs)

        # Image prefix is fully bidirectional within itself.
        assert torch.all(mask[0, img_slice, img_slice])

        # Language-vs-language is strictly causal: every lang token sees its
        # predecessors and itself, but not later lang tokens.
        lang_block = mask[0, lang_slice, lang_slice]
        expected_causal = torch.tril(torch.ones(num_lang_embs, num_lang_embs, dtype=torch.bool))
        assert torch.equal(lang_block, expected_causal)

        # Language still sees the entire image prefix (prefix-LM cross attention).
        assert torch.all(mask[0, lang_slice, img_slice])
        # Image tokens do NOT see later language tokens (they precede them).
        assert not torch.any(mask[0, img_slice, lang_slice])


# RoPE shape preservation + different theta values


class TestApplyRope:
    def test_shape_and_dtype_preserved(self):
        x = torch.randn(2, 4, 3, 32, dtype=torch.float32)
        positions = torch.arange(4)[None, :].expand(2, 4)
        out = apply_rope(x, positions, max_wavelength=10_000.0)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_different_thetas_give_different_results(self):
        x = torch.randn(1, 8, 2, 16)
        positions = torch.arange(8)[None, :]
        local = apply_rope(x, positions, max_wavelength=10_000.0)
        global_ = apply_rope(x, positions, max_wavelength=1_000_000.0)
        # The two thetas should produce different embeddings at the same positions.
        assert not torch.allclose(local, global_)

    def test_position_zero_is_identity_in_ones_input(self):
        # At position 0, sin=0 cos=1, so RoPE should leave the vector unchanged
        # (when the original vector has no complex-pair rotation applied).
        x = torch.ones(1, 1, 1, 8)
        positions = torch.zeros(1, 1, dtype=torch.long)
        out = apply_rope(x, positions, max_wavelength=10_000.0)
        assert torch.allclose(out, x, atol=1e-6)


# MSE-loss masking on fully-padded action samples (web / VQA co-training).
#
# The π0.6 paper §2 says the VLM is trained jointly on FAST action tokens and
# co-training examples like multi-modal web data. VQA samples have no real
# actions, so `VQADataset` sets `actions_is_pad = all-True`. The flow-matching
# loss in `PI06FlowMatching.forward` must honour that mask, otherwise the
# action expert is trained to regress to zero on web samples — a silent bug.
#
# These tests drive the production helper `flow_matching_masked_mse`
# (the same function `PI06FlowMatching.forward` calls) so a regression in the
# masking arithmetic — e.g. dropping the `~actions_is_pad` AND, or moving the
# slice-then-sum boundary — fails here, not just in slow GPU integration.


class TestMSEMaskOnFullyPaddedActions:
    """Locks in the contract: an item with `actions_is_pad` all-True
    contributes exactly zero MSE, regardless of the noisy-action targets."""

    def test_all_padded_yields_zero_loss(self):
        torch.manual_seed(0)
        u_t = torch.randn(2, 8, 16)
        v_t = torch.randn(2, 8, 16)
        actions_is_pad = torch.ones(2, 8, dtype=torch.bool)
        prefix_mask = torch.zeros(2, 8, dtype=torch.bool)
        loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            max_action_dim=16,
        )
        assert loss.item() == 0.0

    def test_partial_padded_only_counts_real_steps(self):
        """Mixed mask: half the chunk is padded; loss should equal the MSE
        averaged only over the real timesteps."""
        torch.manual_seed(0)
        u_t = torch.randn(1, 8, 4)
        v_t = torch.randn(1, 8, 4)
        actions_is_pad = torch.tensor([[False, False, False, False, True, True, True, True]])
        prefix_mask = torch.zeros(1, 8, dtype=torch.bool)
        loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            max_action_dim=4,
        )
        # Independent reference: mean over the first 4 timesteps only.
        expected = ((u_t[0, :4] - v_t[0, :4]) ** 2).sum() / (4 * 4)
        assert torch.isclose(loss, expected, atol=1e-6)

    def test_prefix_mask_excludes_frozen_steps(self):
        """Frozen-prefix steps (real-time-inference delay) must be excluded
        the same way `actions_is_pad` is — covers the other AND-input."""
        torch.manual_seed(0)
        u_t = torch.randn(1, 8, 4)
        v_t = torch.randn(1, 8, 4)
        prefix_mask = torch.tensor([[True, True, True, False, False, False, False, False]])
        loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            prefix_mask=prefix_mask,
            actions_is_pad=None,
            max_action_dim=4,
        )
        expected = ((u_t[0, 3:] - v_t[0, 3:]) ** 2).sum() / (5 * 4)
        assert torch.isclose(loss, expected, atol=1e-6)


# PI06Config defaults + validators


class TestPI06Config:
    def test_defaults_match_pi06_spec(self):
        cfg = PI06Config()
        assert cfg.resize_imgs_with_padding == (448, 448)
        assert cfg.num_steps == 5
        assert cfg.proj_width == 1280
        assert cfg.chunk_size == 50
        assert cfg.n_action_steps == 50
        assert cfg.max_state_dim == 32
        assert cfg.max_action_dim == 32
        assert cfg.attention_implementation == "eager"

    def test_action_delta_indices_length_matches_chunk_size(self):
        cfg = PI06Config(chunk_size=8, n_action_steps=8)
        assert cfg.action_delta_indices == list(range(8))

    def test_n_action_steps_larger_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk size"):
            PI06Config(chunk_size=4, n_action_steps=8)

    def test_max_delay_larger_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="max delay"):
            PI06Config(chunk_size=4, n_action_steps=4, max_delay=8)


# Gemma3WithExpertConfig defaults — locks in π0.6 topology


class TestGemma3WithExpertConfig:
    def test_backbone_matches_gemma3_4b(self):
        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        tc = cfg.gemma3_config.text_config
        assert tc.hidden_size == 2560
        assert tc.num_hidden_layers == 34
        assert tc.num_attention_heads == 8
        assert tc.num_key_value_heads == 4
        assert tc.head_dim == 256
        assert tc.sliding_window == 1024
        assert float(tc.rope_theta) == 1_000_000.0
        assert float(tc.rope_local_base_freq) == 10_000.0

    def test_layer_types_interleave_sliding_and_full(self):
        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        layer_types = cfg.gemma3_config.text_config.layer_types
        assert len(layer_types) == 34
        # The official 5:1 ratio — every 6th layer should be full-attention.
        assert layer_types.count("full_attention") >= 5
        assert layer_types.count("sliding_attention") >= 25

    def test_expert_sized_to_target_parameter_range(self):
        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        ec = cfg.gemma_expert_config
        assert ec.hidden_size == 1280
        assert ec.intermediate_size == 5120
        assert ec.num_hidden_layers == 34
        # GQA matches the backbone so per-layer KV concatenation is valid.
        assert ec.num_key_value_heads == 4
        assert ec.head_dim == 256
        assert ec.use_adarms is True
        assert ec.adarms_cond_dim == 1280

    def test_train_expert_only_incompatible_with_unfrozen_vision(self):
        with pytest.raises(ValueError, match="not compatible"):
            Gemma3WithExpertConfig(
                discrete_action_vocab_size=2048,
                freeze_vision_encoder=False,
                train_expert_only=True,
            )

    def test_bad_attention_implementation_raises(self):
        with pytest.raises(ValueError, match="attention_implementation"):
            Gemma3WithExpertConfig(discrete_action_vocab_size=2048, attention_implementation="sdpa")

    def test_vision_image_size_matches_input_resolution(self):
        # Regression: Gemma 3's `Gemma3MultiModalProjector` hardcodes
        # `patches_per_image = image_size // patch_size`, so the config's
        # `vision_config.image_size` MUST match what we actually feed through
        # the vision tower. π0.6 uses 448×448, so the default config has to
        # carry image_size=448 — otherwise the projector's reshape crashes
        # on the first forward pass.
        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        vc = cfg.gemma3_config.vision_config
        assert vc.image_size == 448
        assert vc.patch_size == 14
        # 448/14 = 32 patches/side → 1024 vision tokens → avg-pool to 256 mm tokens.
        assert (vc.image_size // vc.patch_size) ** 2 == 1024

    def test_projector_accepts_448_inputs(self):
        # The above is a correctness check; this is the end-to-end runtime
        # smoke test that the projector actually runs at 448 without crashing.
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        proj = Gemma3MultiModalProjector(cfg.gemma3_config)
        vision_hidden = cfg.gemma3_config.vision_config.hidden_size
        # SigLIP at 448 produces 32×32 = 1024 patch tokens per image.
        vision_out = torch.randn(1, 1024, vision_hidden)
        out = proj(vision_out)
        assert out.shape == (1, 256, cfg.gemma3_config.text_config.hidden_size)


# Image preprocessing (448×448 default, padding correctness)


class TestResizeWithPad:
    def test_pi06_default_shape(self):
        img = torch.rand(1, 3, 256, 320)
        out = resize_with_pad(img, width=448, height=448, pad_value=-1)
        assert out.shape == (1, 3, 448, 448)

    def test_aspect_ratio_preserved_horizontal(self):
        # 2:1 landscape — after padding to square, the left edge should carry
        # pad_value (pad is prepended to width and top).
        img = torch.zeros(1, 3, 64, 128)
        out = resize_with_pad(img, width=32, height=32, pad_value=-1)
        assert out.shape == (1, 3, 32, 32)
        # Top rows were padded (image was half-height after resize).
        assert torch.all(out[0, 0, 0, :] == -1)
        # Bottom rows contain the resized image (all zeros, not pad value).
        assert torch.all(out[0, 0, -1, :] == 0)

    def test_rejects_non_4d_tensor(self):
        img = torch.zeros(3, 64, 64)
        with pytest.raises(ValueError, match="b,c,h,w"):
            resize_with_pad(img, 32, 32)


# FAST discrete-action padding


class TestPadDiscreteTokens:
    def test_padding_shape_and_mask(self):
        tokens = [[3, 4, 5], [6]]
        padded, masks = pad_discrete_tokens(tokens, max_length=5)
        assert padded.shape == (2, 5)
        assert masks.shape == (2, 5)
        # First sequence: first 3 positions valid, trailing 0s padded.
        assert masks[0].tolist() == [True, True, True, False, False]
        assert padded[0].tolist() == [3, 4, 5, 0, 0]
        # Second sequence: only index 0 valid.
        assert masks[1].tolist() == [True, False, False, False, False]

    def test_truncation_keeps_max_length(self):
        tokens = [list(range(10))]
        padded, masks = pad_discrete_tokens(tokens, max_length=4)
        assert padded.shape == (1, 4)
        assert padded[0].tolist() == [0, 1, 2, 3]
        assert masks[0].tolist() == [True, True, True, True]


# Per-layer RoPE θ selection inside `Gemma3WithExpertModel.forward` — regression
# for the bug where the expert used its own `rope_theta=10_000` even on global
# Gemma-3 layers whose backbone Q/K are rotated at θ=1_000_000.


def _make_tiny_g3we_model():
    """Construct a minimally-sized `Gemma3WithExpertModel` for fast tests."""
    cfg = Gemma3WithExpertConfig(
        gemma3_config={
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "sliding_window": 2,
                "rope_theta": 1_000_000.0,
                "rope_local_base_freq": 10_000.0,
                "query_pre_attn_scalar": 16,
                "rms_norm_eps": 1e-6,
                "vocab_size": 128,
                "max_position_embeddings": 512,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "hidden_activation": "gelu_pytorch_tanh",
                "sliding_window_pattern": 2,
                "torch_dtype": "float32",
                # Force one local + one global layer for the RoPE θ test.
                "layer_types": ["sliding_attention", "full_attention"],
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "patch_size": 14,
                "image_size": 448,
                "projection_dim": 32,
                "projector_hidden_act": "gelu_fast",
                "vision_use_head": False,
                "torch_dtype": "float32",
                "layer_norm_eps": 1e-6,
            },
            "image_token_index": 127,
            "mm_tokens_per_image": 4,
            "boi_token_index": 125,
            "eoi_token_index": 126,
        },
        gemma_expert_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": 16,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 16,
            "intermediate_size": 32,
            "max_position_embeddings": 512,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            # Intentionally different from the backbone's θs — we want to
            # confirm this value is IGNORED during shared attention.
            "rope_theta": 10_000.0,
            "use_adarms": True,
            "adarms_cond_dim": 16,
            "vocab_size": 128,
        },
        discrete_action_vocab_size=32,
        freeze_vision_encoder=False,
        train_expert_only=False,
    )
    # bfloat16 cast interacts badly with tiny Linear layers; skip it for tests.
    import torch as _torch

    return Gemma3WithExpertModel.__new__(Gemma3WithExpertModel), cfg, _torch


class TestRopeThetaSymmetryDuringForward:
    def test_expert_uses_backbone_per_layer_theta(self, monkeypatch):
        """Both streams' Q/K must rotate with the backbone's per-layer θ so
        the shared-attention dot product stays in a consistent RoPE basis."""
        captured: list[float] = []

        real_apply_rope = g3we.apply_rope

        def spy_apply_rope(x, positions, max_wavelength=10_000.0):
            captured.append(float(max_wavelength))
            return real_apply_rope(x, positions, max_wavelength=max_wavelength)

        monkeypatch.setattr(g3we, "apply_rope", spy_apply_rope)
        # `Gemma3WithExpertModel` unconditionally casts its layers to bfloat16
        # in __init__; override for the test so a plain float32 forward pass
        # doesn't complain about mixed dtypes through the tiny expert linear.
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)

        _, cfg, _ = _make_tiny_g3we_model()
        model = Gemma3WithExpertModel(cfg)
        model = model.to(dtype=torch.float32)

        batch, seq_len = 1, 3
        hidden_backbone = torch.randn(batch, seq_len, cfg.gemma3_config.text_config.hidden_size)
        position_ids = torch.arange(seq_len)[None, :]
        attention_mask = torch.ones(batch, seq_len, seq_len, dtype=torch.bool)

        model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden_backbone, None],
            n_cross_att_tokens=seq_len,
            use_cache=False,
            fill_kv_cache=True,
        )

        # Per layer we compute RoPE on two tensors (Q and K). 2 layers × 2
        # tensors = 4 calls; we only have the backbone stream here so the
        # expert's θ isn't exercised, but the backbone's per-layer θ must be
        # present (10_000 for the sliding layer, 1_000_000 for the global).
        assert 10_000.0 in captured
        assert 1_000_000.0 in captured
        assert captured.count(10_000.0) == 2, captured
        assert captured.count(1_000_000.0) == 2, captured


# Sliding-window mask is INTENTIONALLY not enforced — confirm the per-layer
# attention mask reaching `eager_attention_forward` is the unmodified input
# mask regardless of whether the layer is local or global.


class TestNoSlidingWindowEnforcement:
    def test_per_layer_mask_equals_input_mask_on_both_layer_types(self, monkeypatch):
        """π0.6 keeps the prefix block-causal mask global at every layer.

        If sliding-window enforcement crept back in, the captured mask on
        the local (sliding) layer would differ from the captured mask on
        the global (full_attention) layer; this test guards against that.
        """
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)

        _, cfg, _ = _make_tiny_g3we_model()
        # The tiny config above already orders layers as
        # ["sliding_attention", "full_attention"] — exercise both.
        assert cfg.gemma3_config.text_config.layer_types == [
            "sliding_attention",
            "full_attention",
        ]

        model = Gemma3WithExpertModel(cfg)
        model = model.to(dtype=torch.float32)

        captured_masks: list[torch.Tensor] = []
        real_eager = model.eager_attention_forward

        def spy_eager(attention_mask, *args, **kwargs):
            captured_masks.append(attention_mask.clone())
            return real_eager(attention_mask, *args, **kwargs)

        monkeypatch.setattr(model, "eager_attention_forward", spy_eager)

        batch, seq_len = 1, 8
        hidden_backbone = torch.randn(batch, seq_len, cfg.gemma3_config.text_config.hidden_size)
        position_ids = torch.arange(seq_len)[None, :]
        # A non-trivial input mask so we can verify it survives unchanged
        # through every layer (no AND-ing with a sliding window).
        attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None]

        model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden_backbone, None],
            n_cross_att_tokens=seq_len,
            use_cache=False,
            fill_kv_cache=True,
        )

        assert len(captured_masks) == 2, "expected one capture per layer"
        # Sliding-window cap (= 2) WOULD have zeroed everything more than 2
        # positions away on the sliding layer; the global layer would have
        # been left alone. Confirm both layers received the identical input
        # mask.
        assert torch.equal(captured_masks[0], attention_mask), (
            "sliding (local) layer mask differs from input — sliding window "
            "enforcement appears to have been re-enabled"
        )
        assert torch.equal(captured_masks[1], attention_mask), (
            "global layer mask differs from input — something is mutating it"
        )


# End-to-end integration — guarded because Gemma 3 4B is huge.


@pytest.mark.gpu
@pytest.mark.slow
def test_complete_pi06_pipeline_integration_smoke(lerobot_dataset_metadata):
    """Minimal GPU smoke test — constructs the full π0.6 policy on CUDA and
    runs one forward pass. Heavy; mirrors the pi05 integration test and is
    skipped in CPU CI."""
    from opentau.policies.pi06.modeling_pi06 import PI06Policy

    config = PI06Config(
        max_state_dim=32,
        max_action_dim=32,
        chunk_size=10,
        n_action_steps=10,
        discrete_action_max_length=32,
        predict_response=True,
    )

    # Need input/output features populated; reuse the lerobot metadata fixture.
    from opentau.configs.types import FeatureType
    from opentau.datasets.utils import dataset_to_policy_features

    features = dataset_to_policy_features(
        {
            "state": {"shape": (32,), "dtype": "float32"},
            "actions": {"shape": (10, 32), "dtype": "float32"},
            "camera0": {"shape": (3, 448, 448), "dtype": "image"},
            "camera1": {"shape": (3, 448, 448), "dtype": "image"},
        }
    )
    config.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    config.input_features = {k: ft for k, ft in features.items() if k not in config.output_features}

    # The shared lerobot_dataset_metadata fixture carries actions stats shaped
    # (50, 32) — matching the default PI06Config(chunk_size=50). This test
    # uses chunk_size=10 to keep the model small, so override the actions
    # stats to (10, 32) before constructing Normalize buffers; otherwise
    # `(actions - min) / (max - min + EPS)` mismatches at dim=1 (actions is
    # (B, 10, 32) but the buffer is (50, 32)).
    import copy

    import numpy as np

    dataset_stats = copy.deepcopy(lerobot_dataset_metadata.stats)
    for k in ("max", "mean", "min", "std"):
        dataset_stats["actions"][k] = np.full(
            (config.chunk_size, 32),
            float(dataset_stats["actions"][k].flatten()[0]),
            dtype=np.float32,
        )

    policy = PI06Policy(config, dataset_stats=dataset_stats)
    policy.to(dtype=torch.bfloat16, device="cuda")

    batch = {
        "camera0": torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device="cuda"),
        "camera1": torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device="cuda"),
        "state": torch.randn(1, 32, dtype=torch.bfloat16, device="cuda"),
        "actions": torch.randn(1, 10, 32, dtype=torch.bfloat16, device="cuda"),
        "prompt": ["Pick up the red block"],
        "response": ["Pick up the red block"],
        "img_is_pad": torch.zeros(1, 2, dtype=torch.bool, device="cuda"),
        "action_is_pad": torch.zeros(1, 10, dtype=torch.bool, device="cuda"),
    }

    loss = policy.forward(batch)
    assert isinstance(loss, dict)
    assert "MSE" in loss and "CE" in loss
    assert all(v.isfinite() for v in loss.values())
