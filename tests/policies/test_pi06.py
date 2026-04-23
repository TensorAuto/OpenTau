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

from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.policies.pi06.gemma3_with_expert import (
    Gemma3WithExpertConfig,
    _build_sliding_window_mask,
    apply_rope,
)
from opentau.policies.pi06.modeling_pi06 import (
    make_att_2d_masks,
    pad_discrete_tokens,
    resize_with_pad,
)

# Block-causal attention mask (pi05 / π0.6 prefix-LM pattern)


class TestMakeAtt2dMasks:
    """Locks in the block-causal semantics the π0.6 model card depends on:
    image + text = one bidirectional block, response / discrete-action =
    subsequent causal blocks, action suffix = its own bidirectional block."""

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


# Sliding-window mask (Gemma 3 local attention)


class TestSlidingWindowMask:
    def test_window_smaller_than_sequence(self):
        # With a sliding window of 3, only positions within |i-j|<3 should be True.
        mask = _build_sliding_window_mask(seq_len=6, window=3, device=torch.device("cpu"))
        assert mask.shape == (6, 6)
        for i in range(6):
            for j in range(6):
                expected = abs(i - j) < 3
                assert mask[i, j].item() == expected, f"mismatch at ({i}, {j})"

    def test_window_covers_full_sequence(self):
        mask = _build_sliding_window_mask(seq_len=4, window=100, device=torch.device("cpu"))
        assert torch.all(mask)

    def test_window_of_one_is_identity(self):
        mask = _build_sliding_window_mask(seq_len=4, window=1, device=torch.device("cpu"))
        assert torch.all(mask == torch.eye(4, dtype=torch.bool))


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

    policy = PI06Policy(config, dataset_stats=lerobot_dataset_metadata.stats)
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
