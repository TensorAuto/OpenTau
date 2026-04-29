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

"""CPU-only regression tests for the ``pi07`` policy backbone.

Covers the architectural invariants ported from PR #178 (`pi06`):

  * Gemma 3's ``Gemma3TextScaledWordEmbedding`` already scales by
    ``sqrt(hidden_size)`` — the planner code must NOT apply a second manual
    ``* math.sqrt(hidden_size)`` to text embeddings.  Otherwise text tokens
    are scaled to ~51× the image-token magnitude.
  * The vision-config ``image_size`` MUST match the planner's input
    resolution; ``Gemma3MultiModalProjector`` hardcodes
    ``patches_per_image = image_size // patch_size``.
  * Per-layer RoPE ``θ`` is applied uniformly to backbone *and* expert at the
    same layer — the shared-attention dot product would otherwise mix RoPE
    bases.
  * Gemma 3's 1024-token sliding-window pattern is **not** enforced; every
    layer receives the unmodified block-causal prefix-LM mask.
"""

from __future__ import annotations

import pytest
import torch

from opentau.policies.pi07 import gemma3_with_expert as g3we
from opentau.policies.pi07.gemma3_with_expert import (
    Gemma3WithExpertConfig,
    Gemma3WithExpertModel,
)

# Shared tiny config helper


def _make_tiny_g3we_cfg() -> Gemma3WithExpertConfig:
    """Construct a minimally-sized ``Gemma3WithExpertConfig`` for fast tests.

    The text config is sized for two layers — one ``sliding_attention`` and
    one ``full_attention`` — so the per-layer RoPE-θ and no-sliding-window
    invariants can be exercised in a single forward pass.  Vision config
    matches the production 448 / 14 / 32-patch grid so the projector reshape
    runs without crashing.
    """
    return Gemma3WithExpertConfig(
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
            # Intentionally different from the backbone θ so the test can
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


# Vision config invariants


class TestVisionConfig:
    def test_vision_image_size_matches_input_resolution(self):
        """Regression: ``Gemma3MultiModalProjector`` hardcodes
        ``patches_per_image = image_size // patch_size``, so the default
        config's ``vision_config.image_size`` MUST equal what the planners
        actually feed (448).  Otherwise the projector reshape crashes on
        the first forward."""
        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        vc = cfg.gemma3_config.vision_config
        assert vc.image_size == 448
        assert vc.patch_size == 14
        # 448 / 14 = 32 patches/side → 1024 vision tokens → avg-pool to 256
        # multimodal tokens.
        assert (vc.image_size // vc.patch_size) ** 2 == 1024

    def test_projector_accepts_448_inputs(self):
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048)
        proj = Gemma3MultiModalProjector(cfg.gemma3_config)
        vision_hidden = cfg.gemma3_config.vision_config.hidden_size
        # SigLIP at 448 produces 32×32 = 1024 patch tokens per image.
        vision_out = torch.randn(1, 1024, vision_hidden)
        out = proj(vision_out)
        assert out.shape == (1, 256, cfg.gemma3_config.text_config.hidden_size)


# Per-layer RoPE-θ symmetry across backbone / expert


class TestRopeThetaSymmetryDuringForward:
    def test_expert_uses_backbone_per_layer_theta(self, monkeypatch):
        """Both streams' Q/K must rotate with the backbone's per-layer θ so
        the shared-attention dot product stays in a consistent RoPE basis.

        Sliding (local) layers use ``rope_local_base_freq=10_000``; global
        layers use ``rope_theta=1_000_000``.  Each layer calls ``apply_rope``
        once for Q and once for K — a 2-layer config therefore generates 4
        captures, two per θ.
        """
        captured: list[float] = []
        real_apply_rope = g3we.apply_rope

        def spy_apply_rope(x, positions, max_wavelength=10_000.0):
            captured.append(float(max_wavelength))
            return real_apply_rope(x, positions, max_wavelength=max_wavelength)

        monkeypatch.setattr(g3we, "apply_rope", spy_apply_rope)
        # Skip the unconditional bf16 cast in __init__ so a plain float32
        # forward through tiny linears doesn't complain about mixed dtypes.
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)

        cfg = _make_tiny_g3we_cfg()
        model = Gemma3WithExpertModel(cfg).to(dtype=torch.float32)

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

        # 2 layers × 2 tensors (Q, K) per layer = 4 captures.  Sliding layer
        # (idx 0) uses 10_000; global layer (idx 1) uses 1_000_000.
        assert 10_000.0 in captured
        assert 1_000_000.0 in captured
        assert captured.count(10_000.0) == 2, captured
        assert captured.count(1_000_000.0) == 2, captured


# Sliding-window mask is intentionally NOT enforced


class TestNoSlidingWindowEnforcement:
    def test_per_layer_mask_equals_input_mask_on_both_layer_types(self, monkeypatch):
        """π0.7 keeps the prefix block-causal mask global at every layer.

        Gemma 3 pretraining uses a 1024-token sliding window on local layers,
        but π0.7 needs bidirectional attention across **all** image tokens.
        If sliding-window enforcement crept back in, the captured mask on the
        sliding (local) layer would differ from the one on the global layer.
        """
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)

        cfg = _make_tiny_g3we_cfg()
        assert cfg.gemma3_config.text_config.layer_types == [
            "sliding_attention",
            "full_attention",
        ]
        model = Gemma3WithExpertModel(cfg).to(dtype=torch.float32)

        captured_masks: list[torch.Tensor] = []
        real_eager = model.eager_attention_forward

        def spy_eager(attention_mask, *args, **kwargs):
            captured_masks.append(attention_mask.clone())
            return real_eager(attention_mask, *args, **kwargs)

        monkeypatch.setattr(model, "eager_attention_forward", spy_eager)

        batch, seq_len = 1, 8
        hidden_backbone = torch.randn(batch, seq_len, cfg.gemma3_config.text_config.hidden_size)
        position_ids = torch.arange(seq_len)[None, :]
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
        # been left alone.  Confirm both layers received the identical input
        # mask.
        assert torch.equal(captured_masks[0], attention_mask), (
            "sliding (local) layer mask differs from input — sliding window "
            "enforcement appears to have been re-enabled"
        )
        assert torch.equal(captured_masks[1], attention_mask), (
            "global layer mask differs from input — something is mutating it"
        )


# Embedding-magnitude invariant — guards against the double-scale bug


class TestEmbeddingMagnitudeInvariant:
    """Gemma 3's ``embed_tokens`` is a ``Gemma3TextScaledWordEmbedding`` that
    already multiplies by ``sqrt(hidden_size)`` internally.  Apply a manual
    ``* math.sqrt(hidden_size)`` on top of ``embed_language_tokens`` and
    text tokens end up at ~51× image-token magnitude, corrupting the
    bidirectional prefix attention and the FAST/response cross-entropy heads.

    These tests pin the invariant at the embedding level (not inside the
    planner) so a regression is flagged regardless of which planner gets the
    scaling wrong.
    """

    def test_embed_language_tokens_already_scaled(self, monkeypatch):
        """``embed_language_tokens`` returns the scaled embedding directly."""
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)

        cfg = _make_tiny_g3we_cfg()
        model = Gemma3WithExpertModel(cfg).to(dtype=torch.float32)

        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        embedded = model.embed_language_tokens(tokens)
        # The embedding table's underlying weights have stdev ≈ initializer
        # range; after the internal sqrt(hidden_size) scaling the embedded
        # tokens have stdev ≈ initializer_range × sqrt(hidden).  Assert the
        # scaling was applied (embedded > raw weights * 1.0) by sampling the
        # raw weight matrix and comparing magnitudes.
        lm = getattr(model.gemma3, "language_model", model.gemma3.model.language_model)
        raw_weights = lm.embed_tokens.weight
        embedded_std = embedded.detach().std().item()
        raw_std = raw_weights.detach().std().item()
        # Scaled embedding should be markedly larger than the raw row.
        assert embedded_std > raw_std, (
            f"Gemma3TextScaledWordEmbedding does not appear to have applied its "
            f"sqrt(hidden) scaling: embedded_std={embedded_std} <= raw_std={raw_std}"
        )

    def test_embedded_token_stdev_matches_internal_scaling(self, monkeypatch):
        """``embed_language_tokens`` output stdev must be ≈ ``raw_weight_stdev
        × sqrt(hidden)`` — and not an additional factor of ``sqrt(hidden)``
        on top.  An untrained Gemma3WithExpertModel does not give us a well-
        calibrated image-token magnitude to compare against (random SigLIP +
        random projector → tiny stdev), so we verify the Gemma 3 internal
        scaling alone fires, and rely on the source-level lint
        (``TestNoManualSqrtScalingInPlannerSource``) plus the GPU integration
        tests to catch a manual second scale further up the call stack.
        """
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)

        cfg = _make_tiny_g3we_cfg()
        model = Gemma3WithExpertModel(cfg).to(dtype=torch.float32)

        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        embedded = model.embed_language_tokens(tokens)

        lm = getattr(model.gemma3, "language_model", model.gemma3.model.language_model)
        raw_weight_std = lm.embed_tokens.weight.detach().std().item()
        embedded_std = embedded.detach().std().item()
        hidden = cfg.gemma3_config.text_config.hidden_size
        expected_factor = hidden**0.5

        # Allow ±0.5× slack because the sample stdev over 5 token rows is noisy.
        actual_factor = embedded_std / max(raw_weight_std, 1e-8)
        assert 0.5 * expected_factor < actual_factor < 1.5 * expected_factor, (
            f"embedded_std/raw_std = {actual_factor:.2f} but expected ≈ "
            f"sqrt(hidden) = {expected_factor:.2f}.  A second manual "
            f"`* sqrt(hidden)` would push this ratio to ~{expected_factor**2:.0f}."
        )


# Spec sanity for the embed_prefix call sites — guards against accidental
# reintroduction of the manual scaling.


class TestNoManualSqrtScalingInPlannerSource:
    """Source-level invariant: planner files must not call
    ``math.sqrt(*_dim)`` on a tensor whose immediate origin is
    ``embed_language_tokens``.  This is a fast file-level lint so a
    regression is caught even if no test exercises that prefix slot.
    """

    @pytest.mark.parametrize(
        "module_path",
        [
            "src/opentau/policies/pi07/low_level_planner/modeling_pi07_low_level.py",
            "src/opentau/policies/pi07/high_level_planner/modeling_pi07_high_level.py",
        ],
    )
    def test_no_sqrt_emb_dim_left(self, module_path):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        text = (repo_root / module_path).read_text()
        # ``math.sqrt(*_dim)`` is only ever applied to a language-embedding
        # tensor in this codebase; if it appears in a pi07 planner file, the
        # double-scale fix from #178 has been undone.
        assert "math.sqrt(" not in text, (
            f"{module_path} contains a residual math.sqrt(...) call; "
            "Gemma 3's embed_tokens already scales by sqrt(hidden_size). "
            "Reintroducing the manual scaling double-scales text embeddings."
        )
