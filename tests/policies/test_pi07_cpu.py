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


# `prepare_metadata` segment-string construction
#
# The two planners share an identical contract for assembling the metadata
# prompt: each per-sample segment is appended only when the field is *not*
# padded, and a sample with zero active segments emits an empty string (not
# the literal "Metadata: ").  These tests pin that contract on the CPU side
# without instantiating the full Gemma 3 backbone — we bind the unbound
# method to a SimpleNamespace and stub the tokenizer to capture the exact
# strings the loop produced.


def _make_tokenizer_capture():
    """Return a (tokenizer_stub, captured_list) pair.

    Use the captured list to assert on the metadata strings the planner
    constructed, independent of the real Gemma 3 tokenizer.
    """
    import types

    captured: list[str] = []

    def stub_call(metadata, **kwargs):
        captured.extend(metadata)
        batch_size = len(metadata)
        max_length = kwargs.get("max_length", 4)
        return {
            "input_ids": torch.zeros((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
        }

    tokenizer = types.SimpleNamespace()
    tokenizer.__call__ = stub_call
    return tokenizer, captured


def _make_fake_planner(metadata_max_length: int = 4):
    """Construct a minimal ``self`` stand-in for the planners' ``prepare_metadata``.

    The unbound method only reads ``self.language_tokenizer`` and
    ``self.config.metadata_max_length``, so a plain SimpleNamespace is
    sufficient — no Gemma 3 backbone, no HuggingFace download.
    """
    import types

    tokenizer, captured = _make_tokenizer_capture()
    fake = types.SimpleNamespace(
        language_tokenizer=tokenizer,
        config=types.SimpleNamespace(metadata_max_length=metadata_max_length),
    )
    return fake, captured


class TestPrepareMetadataSegments:
    """Verify the per-sample metadata string construction in both planners.

    Covers:
      * ``robot_type`` / ``control_mode`` (PR-introduced) emit ``"Robot: ..."``
        and ``"Control: ..."`` segments only when non-empty.
      * Missing batch keys default to "fully padded" → empty string.
      * The high- and low-level planners agree on the all-empty case.
    """

    @staticmethod
    def _planner_methods():
        from opentau.policies.pi07.high_level_planner.modeling_pi07_high_level import (
            PI07HighLevelPlannerPolicy,
        )
        from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
            PI07LowLevelPlannerPolicy,
        )

        return {
            "low": PI07LowLevelPlannerPolicy.prepare_metadata,
            "high": PI07HighLevelPlannerPolicy.prepare_metadata,
        }

    @pytest.mark.parametrize("planner", ["low", "high"])
    def test_robot_and_control_present(self, planner):
        """Both robot_type and control_mode populated → both segments emitted."""
        method = self._planner_methods()[planner]
        fake, captured = _make_fake_planner()

        batch_size = 2
        batch = {
            "state": torch.zeros(batch_size, 1),
            "robot_type": ["franka", "ur5"],
            "control_mode": ["joint", "ee"],
        }

        method(fake, batch)

        assert len(captured) == batch_size
        for line, robot, ctrl in zip(captured, ["franka", "ur5"], ["joint", "ee"], strict=True):
            assert line.startswith("Metadata: ")
            assert f"Robot: {robot}, " in line
            assert f"Control: {ctrl}, " in line

    @pytest.mark.parametrize("planner", ["low", "high"])
    def test_robot_and_control_absent_emits_empty_string(self, planner):
        """No metadata keys present → all samples emit ``""`` (no fabricated values)."""
        method = self._planner_methods()[planner]
        fake, captured = _make_fake_planner()

        batch_size = 3
        batch = {"state": torch.zeros(batch_size, 1)}

        method(fake, batch)

        assert captured == ["", "", ""]

    @pytest.mark.parametrize("planner", ["low", "high"])
    def test_one_present_one_empty(self, planner):
        """Mixed: only ``robot_type`` populated → only ``Robot:`` segment emitted."""
        method = self._planner_methods()[planner]
        fake, captured = _make_fake_planner()

        batch_size = 2
        batch = {
            "state": torch.zeros(batch_size, 1),
            "robot_type": ["franka", "ur5"],
            "control_mode": ["", ""],
        }

        method(fake, batch)

        assert len(captured) == batch_size
        for line, robot in zip(captured, ["franka", "ur5"], strict=True):
            assert line.startswith("Metadata: ")
            assert f"Robot: {robot}, " in line
            assert "Control:" not in line

    @pytest.mark.parametrize("planner", ["low", "high"])
    def test_per_sample_empty_emits_empty_string(self, planner):
        """Within a batch, samples whose every field is empty/padded emit ``""``,
        even when sibling samples have populated fields.  Pins the
        ``if segments else ""`` guard at both planner sites so the two
        planners agree on the all-empty contract.
        """
        method = self._planner_methods()[planner]
        fake, captured = _make_fake_planner()

        batch_size = 2
        batch = {
            "state": torch.zeros(batch_size, 1),
            "robot_type": ["franka", ""],
            "control_mode": ["joint", ""],
        }

        method(fake, batch)

        assert len(captured) == batch_size
        assert captured[0].startswith("Metadata: ")
        assert captured[1] == ""

    def test_low_level_missing_speed_pad_does_not_fabricate_value(self):
        """Regression for the zeros→ones default fix: with ``speed`` and
        ``speed_is_pad`` both absent from the batch, the low-level planner
        must NOT emit ``"Speed: 0.0"`` — that would surface a value the
        caller never provided.
        """
        from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
            PI07LowLevelPlannerPolicy,
        )

        method = PI07LowLevelPlannerPolicy.prepare_metadata
        fake, captured = _make_fake_planner()

        batch_size = 2
        batch = {"state": torch.zeros(batch_size, 1)}

        method(fake, batch)

        for line in captured:
            assert "Speed:" not in line
            assert "Quality:" not in line
            assert "Mistake:" not in line


# `embed_prefix` conditional-block guards
#
# The low-level planner skips entire prefix blocks when their availability
# masks are all-False (response_masks, subgoal_img_masks, metadata_masks).
# Pinning that on the CPU side means a regression that re-introduces a
# spurious causal boundary (att_masks=[1,...]) on a fully-padded slot is
# caught without firing up the Gemma 3 backbone.


def _make_fake_flow_matching(*, hidden: int = 4, n_video_tokens: int = 3):
    """Construct a minimal stand-in for ``PI07LowLevelPlannerFlowMatching``
    so ``embed_prefix`` runs end-to-end without instantiating the real
    Gemma 3 backbone.

    All language tokens, image tokens, and video tokens project to deterministic
    zero tensors with the correct shape — the test only asserts on the
    structure of ``embs`` / ``pad_masks`` / ``att_masks`` (lengths, sample-mask
    rows), not on numeric values.
    """
    import types

    class _FakeGemma3WithExpert:
        def embed_language_tokens(self, tokens):
            return torch.zeros((*tokens.shape, hidden), dtype=torch.float32)

        def embed_image(self, image):
            # Each image becomes 2 tokens — small enough to keep prefix lengths readable.
            return torch.zeros(image.shape[0], 2, hidden, dtype=torch.float32)

        def embed_discrete_actions(self, da):
            return torch.zeros((*da.shape, hidden), dtype=torch.float32)

    class _FakeTokenizer:
        # Every indicator phrase encodes to exactly 2 tokens. embed_prefix
        # uses tokenizer.encode() to get the literal token ids, then later
        # passes them back through embed_language_tokens for the embedding.
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    def _state_proj(state):
        # state: (B, T, D) → (B, T, hidden)
        return torch.zeros(state.shape[0], state.shape[1], hidden, dtype=torch.float32)

    def _embed_video(video):
        # video: (B, T, C, H, W) → (B, n_video_tokens, hidden)
        return torch.zeros(video.shape[0], n_video_tokens, hidden, dtype=torch.float32)

    fake = types.SimpleNamespace(
        gemma3_with_expert=_FakeGemma3WithExpert(),
        language_tokenizer=_FakeTokenizer(),
        state_proj=_state_proj,
        embed_video=_embed_video,
        config=types.SimpleNamespace(discrete_action_max_length=2),
    )
    return fake


def _embed_prefix_method():
    from opentau.policies.pi07.low_level_planner.modeling_pi07_low_level import (
        PI07LowLevelPlannerFlowMatching,
    )

    return PI07LowLevelPlannerFlowMatching.embed_prefix


def _build_default_inputs(*, batch_size: int = 2, prompt_len: int = 3, t_state: int = 1):
    """Build the minimal kwargs every embed_prefix invocation needs."""
    return {
        "videos": [torch.zeros(batch_size, t_state, 3, 4, 4)],
        "vid_masks": [torch.ones(batch_size, dtype=torch.bool)],
        "lang_tokens": torch.zeros(batch_size, prompt_len, dtype=torch.long),
        "lang_masks": torch.ones(batch_size, prompt_len, dtype=torch.bool),
        "state": torch.zeros(batch_size, t_state, 7),
        "response_tokens": None,
        "response_masks": None,
        "metadata_tokens": None,
        "metadata_masks": None,
    }


class TestEmbedPrefixConditionalGuards:
    """Pin per-block toggles in ``embed_prefix``.

    The three conditional emit-or-skip toggles (response, subgoal, metadata)
    are normally exercised only by the GPU integration tests. These CPU tests
    fake the Gemma 3 backbone so a regression that emits a spurious header
    with all-False masks (or fails to emit a real header on a mixed batch)
    is caught directly.
    """

    def test_all_optional_blocks_absent_skips_emission(self):
        """All-False response_masks + no subgoal_images + all-False metadata_masks →
        the prefix collapses to ``videos + lang + State: + state + ":\\n"``.

        With ``has_any_optional == False`` the state-end separator collapses
        from ``", "`` to ``":\\n"`` (same fake-tokenizer length: 2 tokens) and
        the trailing ``";\\n "`` prefix-end is omitted entirely so it cannot
        dangle as a spurious separator before ``"Action: "``.
        """
        method = _embed_prefix_method()
        fake = _make_fake_flow_matching()
        bsize = 2
        kwargs = _build_default_inputs(batch_size=bsize)

        # Empty response/metadata still passed in: tokens with all-False masks.
        kwargs["response_tokens"] = torch.zeros(bsize, 5, dtype=torch.long)
        kwargs["response_masks"] = torch.zeros(bsize, 5, dtype=torch.bool)
        kwargs["metadata_tokens"] = torch.zeros(bsize, 4, dtype=torch.long)
        kwargs["metadata_masks"] = torch.zeros(bsize, 4, dtype=torch.bool)
        # No subgoals at all.
        kwargs["subgoal_images"] = []
        kwargs["subgoal_img_masks"] = []

        embs, pad_masks, att_masks = method(fake, **kwargs)

        # Expected layout (each video produces 3 fake video tokens, each
        # indicator encodes to 2 tokens, hidden=4):
        #   videos:         3
        #   lang:           3 (prompt_len)
        #   "State: ":      2
        #   state(T=1):     1
        #   ":\n":          2  (state-end; collapsed from ", " because no optionals)
        # Total = 11  (";\n " prefix-end is omitted with no optional content)
        assert embs.shape == (bsize, 11, 4)
        assert pad_masks.shape == (bsize, 11)
        assert att_masks.shape == (bsize, 11)
        # No causal boundaries (1's) anywhere in the att_masks — every block
        # should remain bidirectional with all-skip optional blocks.
        assert int(att_masks[0].sum().item()) == 0, (
            "att_masks contained a causal boundary — a guarded block was emitted "
            "anyway: this would corrupt the cumsum for every subsequent token."
        )

    def test_mixed_subgoal_availability_zeros_pad_only_samples(self):
        """In a mixed batch (some samples have subgoals, others don't), the
        ``Subgoal: `` header and trailing comma must follow the per-sample
        availability mask — pad-only samples get False on those header tokens.
        """
        method = _embed_prefix_method()
        fake = _make_fake_flow_matching()
        bsize = 2
        kwargs = _build_default_inputs(batch_size=bsize)

        # Sample 0 has a subgoal, sample 1 does not.
        per_sample_mask = torch.tensor([True, False])
        subgoal_img = torch.zeros(bsize, 3, 4, 4)
        kwargs["subgoal_images"] = [subgoal_img]
        kwargs["subgoal_img_masks"] = [per_sample_mask]
        kwargs["metadata_tokens"] = torch.zeros(bsize, 4, dtype=torch.long)
        kwargs["metadata_masks"] = torch.zeros(bsize, 4, dtype=torch.bool)

        _, pad_masks, _ = method(fake, **kwargs)

        # The Subgoal: header is 2 tokens (fake tokenizer); the image is 2 tokens
        # (fake embed_image); the trailing ", " is 2 tokens. The mask values for
        # those 6 tokens must mirror per_sample_mask on every emitted slot.
        # Easiest assertion: at least one row has pad=False everywhere in the
        # subgoal block (sample 1), and at least one row has pad=True (sample 0).
        # We don't pin exact slice boundaries here — instead we check that the
        # union of per_sample_mask=False rows are all-zero on the subgoal slice.
        # Total length sanity:
        #   videos(3) + lang(3) + "State: "(2) + state(1) + ", "(2)
        #     + "Subgoal: "(2) + subgoal_img(2) + ", "(2)
        #     + ";\n "(2) = 19
        assert pad_masks.shape == (bsize, 19)
        # The subgoal block is the 6 tokens at indices 11..16 (after the
        # prefix-LM transition); the sample with no subgoal must be False
        # across all 6, and the sample with a subgoal must be True across all 6.
        subgoal_block = pad_masks[:, 11:17]
        assert subgoal_block[0].all().item() is True, (
            "sample 0 (has subgoal) should have all-True pad mask on the subgoal block"
        )
        assert (~subgoal_block[1]).all().item() is True, (
            "sample 1 (no subgoal) should have all-False pad mask on the subgoal block — "
            "otherwise the prefix-LM block opening on a pad-only sample injects a "
            "spurious unmasked indicator token"
        )

    def test_response_mask_any_true_emits_block(self):
        """When at least one sample has a real response, the response block
        IS emitted and contributes a causal boundary token.
        """
        method = _embed_prefix_method()
        fake = _make_fake_flow_matching()
        bsize = 2
        kwargs = _build_default_inputs(batch_size=bsize)
        kwargs["response_tokens"] = torch.zeros(bsize, 5, dtype=torch.long)
        # Sample 0 has real response; sample 1 is all padded.
        kwargs["response_masks"] = torch.tensor(
            [[True, True, True, False, False], [False, False, False, False, False]]
        )
        kwargs["subgoal_images"] = []
        kwargs["subgoal_img_masks"] = []
        kwargs["metadata_tokens"] = torch.zeros(bsize, 4, dtype=torch.long)
        kwargs["metadata_masks"] = torch.zeros(bsize, 4, dtype=torch.bool)

        _, _, att_masks = method(fake, **kwargs)

        # Without the response block, att_masks would be all-zeros (see
        # test_all_optional_blocks_absent_skips_emission). With the response
        # block, att_masks gets exactly one ``1`` (the prefix-LM block opening
        # at the start of the 5-token response span).
        per_sample_sum = int(att_masks[0].sum().item())
        assert per_sample_sum == 1, (
            f"expected exactly one causal boundary from the response block opening, got {per_sample_sum}"
        )

    def test_discrete_actions_indicator_uses_per_token_causal_blocks(self):
        """The ``"Action: "`` indicator must use ``[1]*N`` (one causal block per
        token), not ``[1] + [0]*(N-1)`` (single bidirectional block).

        The buggy ``[1] + [0]*(N-1)`` pattern collapses the indicator into one
        bidirectional block, shifting the cumsum at the indicator -> first-discrete
        boundary by N-1 and breaking the discrete-action CE loss. This test pins
        the tail of ``att_masks`` to all 1's after the indicator, so a regression
        to the old pattern fails immediately.
        """
        method = _embed_prefix_method()
        fake = _make_fake_flow_matching()
        bsize = 2
        kwargs = _build_default_inputs(batch_size=bsize)
        # No optional middle blocks — keeps the prefix layout deterministic so
        # the indicator + discrete-action span sits exactly at the tail.
        kwargs["subgoal_images"] = []
        kwargs["subgoal_img_masks"] = []

        num_action_tokens = 3
        kwargs["discrete_actions"] = torch.zeros(bsize, num_action_tokens, dtype=torch.long)
        kwargs["discrete_action_masks"] = torch.ones(bsize, num_action_tokens, dtype=torch.bool)

        _, pad_masks, att_masks = method(fake, **kwargs)

        # Expected layout (no optional blocks; fake tokenizer encodes every
        # indicator phrase to 2 tokens; ``discrete_action_emb`` matches its
        # input shape):
        #   videos(3) + lang(3) + "State: "(2) + state(1) + ":\n"(2)
        #     + "Action: "(2) + discrete_actions(3) = 16
        num_indicator_tokens = 2
        base_prefix_len = 11
        total_len = base_prefix_len + num_indicator_tokens + num_action_tokens
        assert att_masks.shape == (bsize, total_len)
        assert pad_masks.shape == (bsize, total_len)

        # The first ``base_prefix_len`` positions are bidirectional (no causal
        # boundaries) — same invariant as test_all_optional_blocks_absent.
        assert int(att_masks[0, :base_prefix_len].sum().item()) == 0

        # Tail invariant: every position from the indicator onward must be 1
        # (per-token causal blocks for the indicator + one causal step per
        # discrete action). A regression to ``[1] + [0]*(N-1)`` on the
        # indicator would put zeros at indices ``base_prefix_len + 1 ..
        # base_prefix_len + num_indicator_tokens - 1`` and the assertion below
        # would fail.
        tail = att_masks[0, base_prefix_len:]
        assert int(tail.sum().item()) == num_indicator_tokens + num_action_tokens, (
            f"expected all-ones tail of length {num_indicator_tokens + num_action_tokens} "
            f"(indicator + discrete actions), got {tail.tolist()} — a zero in the indicator "
            "span signals a regression to the old [1]+[0]*(N-1) pattern, which shifts the "
            "cumsum at the indicator -> first-discrete boundary and breaks the CE loss."
        )


# Engine-level config validation: SDPA accepted, fa2 falls back to eager with
# a warning (was: NotImplementedError), garbage values raise.


class TestGemma3WithExpertConfig:
    def test_bad_attention_implementation_raises(self):
        with pytest.raises(ValueError, match="attention_implementation"):
            Gemma3WithExpertConfig(discrete_action_vocab_size=2048, attention_implementation="garbage")

    def test_sdpa_attention_implementation_accepted(self):
        cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048, attention_implementation="sdpa")
        assert cfg.attention_implementation == "sdpa"

    def test_fa2_falls_back_to_eager_with_warning(self, caplog):
        with caplog.at_level("WARNING"):
            cfg = Gemma3WithExpertConfig(discrete_action_vocab_size=2048, attention_implementation="fa2")
        # The config field itself is preserved so callers can introspect what
        # they asked for; the dispatcher (see TestPi07AttentionDispatcher) is
        # what falls back to eager at runtime.
        assert cfg.attention_implementation == "fa2"
        assert any("falling back to 'eager'" in record.message for record in caplog.records)


# Attention dispatcher: routes attention_implementation to the right forward.


class TestPi07AttentionDispatcher:
    def test_dispatcher_returns_sdpa_for_sdpa_impl(self, monkeypatch):
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)
        cfg = _make_tiny_g3we_cfg()
        cfg.attention_implementation = "sdpa"
        model = Gemma3WithExpertModel(cfg)
        assert model.get_attention_interface() == model.sdpa_attention_forward

    def test_dispatcher_returns_eager_for_eager_impl(self, monkeypatch):
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)
        cfg = _make_tiny_g3we_cfg()
        cfg.attention_implementation = "eager"
        model = Gemma3WithExpertModel(cfg)
        assert model.get_attention_interface() == model.eager_attention_forward

    def test_dispatcher_falls_back_to_eager_for_fa2(self, monkeypatch):
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)
        cfg = _make_tiny_g3we_cfg()
        cfg.attention_implementation = "fa2"
        model = Gemma3WithExpertModel(cfg)
        # Old behaviour raised NotImplementedError; new behaviour falls back
        # to eager (warning was already emitted at config construction time).
        assert model.get_attention_interface() == model.eager_attention_forward


# SDPA vs eager equivalence — drives BOTH streams (backbone + expert) so the
# Q/K/V concat path along the seq axis is covered. Pi07 is the first policy
# in this family with that coverage requirement (pi06 tested backbone-only).


class TestPi07SdpaEquivalence:
    def test_eager_vs_sdpa_outputs_close(self, monkeypatch):
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)
        cfg = _make_tiny_g3we_cfg()
        cfg.attention_implementation = "eager"

        torch.manual_seed(0)
        model_eager = Gemma3WithExpertModel(cfg).to(dtype=torch.float32)

        cfg_sdpa = _make_tiny_g3we_cfg()
        cfg_sdpa.attention_implementation = "sdpa"
        torch.manual_seed(0)
        model_sdpa = Gemma3WithExpertModel(cfg_sdpa).to(dtype=torch.float32)
        # Mirror weights so the only delta is the attention math.
        model_sdpa.load_state_dict(model_eager.state_dict(), strict=True)

        model_eager.eval()
        model_sdpa.eval()

        batch, seq_len = 1, 4
        backbone_h = cfg.gemma3_config.text_config.hidden_size
        expert_h = cfg.gemma_expert_config.hidden_size
        hidden_backbone = torch.randn(batch, seq_len, backbone_h)
        hidden_expert = torch.randn(batch, seq_len, expert_h)
        position_ids = torch.arange(seq_len)[None, :]
        # Both streams concat along the seq axis → (B, 2*seq_len, 2*seq_len).
        attention_mask = torch.tril(torch.ones(2 * seq_len, 2 * seq_len, dtype=torch.bool))[None]
        # Expert AdaRMS layers are constructed with cond_dim=adarms_cond_dim
        # and have no `weight` parameter — passing cond=None into the cond=None
        # branch would crash. Provide real conditioning matching cond_dim, and
        # let it broadcast over the per-stream sequence dim.
        adarms_cond_value = torch.randn(batch, cfg.gemma_expert_config.adarms_cond_dim)
        adarms_cond = [None, adarms_cond_value]

        out_eager, _ = model_eager(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden_backbone, hidden_expert],
            n_cross_att_tokens=2 * seq_len,
            use_cache=False,
            fill_kv_cache=True,
            adarms_cond=adarms_cond,
        )
        out_sdpa, _ = model_sdpa(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden_backbone, hidden_expert],
            n_cross_att_tokens=2 * seq_len,
            use_cache=False,
            fill_kv_cache=True,
            adarms_cond=adarms_cond,
        )

        assert out_eager[0] is not None and out_sdpa[0] is not None
        assert out_eager[1] is not None and out_sdpa[1] is not None
        # Eager upcasts QK to fp32; SDPA accumulates the softmax in fp32 too.
        # Numerical match is tight but not bit-identical due to reassociation.
        assert torch.allclose(out_eager[0], out_sdpa[0], atol=1e-4, rtol=1e-4)
        assert torch.allclose(out_eager[1], out_sdpa[1], atol=1e-4, rtol=1e-4)


# Gradient-checkpointing forward equivalence — pins that ``_run_layer``
# extraction is bit-identical to the original inlined loop body.


class TestPi07GradCkptEquivalence:
    def test_grad_ckpt_forward_matches_no_ckpt(self, monkeypatch):
        """``gradient_checkpointing=True`` must not change the forward output.

        Drives BOTH streams (backbone + expert) so the per-layer Q/K/V
        concat path (which is what ``_run_layer`` wraps) is the actual
        codepath under test. Train mode + ``dropout=0`` is required: the
        checkpoint wrap is gated on ``self.training``, and dropout would
        introduce stochasticity that masks any extraction bug.
        """
        monkeypatch.setattr(g3we, "_preferred_dtype", lambda: torch.float32)
        cfg = _make_tiny_g3we_cfg()
        cfg.dropout = 0.0

        torch.manual_seed(0)
        model_no_ckpt = Gemma3WithExpertModel(cfg).to(dtype=torch.float32)
        model_no_ckpt.train()

        cfg_ckpt = _make_tiny_g3we_cfg()
        cfg_ckpt.dropout = 0.0
        cfg_ckpt.gradient_checkpointing = True
        torch.manual_seed(0)
        model_ckpt = Gemma3WithExpertModel(cfg_ckpt).to(dtype=torch.float32)
        model_ckpt.load_state_dict(model_no_ckpt.state_dict(), strict=True)
        model_ckpt.train()

        batch, seq_len = 1, 4
        backbone_h = cfg.gemma3_config.text_config.hidden_size
        expert_h = cfg.gemma_expert_config.hidden_size
        hidden_backbone = torch.randn(batch, seq_len, backbone_h)
        hidden_expert = torch.randn(batch, seq_len, expert_h)
        position_ids = torch.arange(seq_len)[None, :]
        attention_mask = torch.tril(torch.ones(2 * seq_len, 2 * seq_len, dtype=torch.bool))[None]
        # Expert AdaRMS layers require non-None cond — see TestPi07SdpaEquivalence
        # for the same construction.
        adarms_cond_value = torch.randn(batch, cfg.gemma_expert_config.adarms_cond_dim)
        adarms_cond = [None, adarms_cond_value]

        out_a, _ = model_no_ckpt(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden_backbone, hidden_expert],
            n_cross_att_tokens=2 * seq_len,
            use_cache=False,
            fill_kv_cache=True,
            adarms_cond=adarms_cond,
        )
        out_b, _ = model_ckpt(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden_backbone, hidden_expert],
            n_cross_att_tokens=2 * seq_len,
            use_cache=False,
            fill_kv_cache=True,
            adarms_cond=adarms_cond,
        )

        assert out_a[0] is not None and out_b[0] is not None
        assert out_a[1] is not None and out_b[1] is not None
        # ckpt only re-runs forward in backward; forward numerics are
        # identical up to floating-point reassociation in autograd.
        assert torch.allclose(out_a[0], out_b[0], atol=1e-6, rtol=1e-6)
        assert torch.allclose(out_a[1], out_b[1], atol=1e-6, rtol=1e-6)


# Policy-level → vlm_config plumbing in __post_init__ — pins that
# --policy.attention_implementation and --policy.gradient_checkpointing
# CLI overrides reach the engine for both planners.


class TestPi07ConfigPlumbing:
    def test_post_init_plumbs_grad_ckpt_into_vlm_config(self):
        from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
            PI07HighLevelPlannerConfig,
        )
        from opentau.policies.pi07.low_level_planner.configuration_pi07_low_level import (
            PI07LowLevelPlannerConfig,
        )

        hl = PI07HighLevelPlannerConfig(gradient_checkpointing=True)
        assert hl.vlm_config.gradient_checkpointing is True

        ll = PI07LowLevelPlannerConfig(gradient_checkpointing=True)
        assert ll.vlm_config.gradient_checkpointing is True

    def test_post_init_plumbs_attention_impl_into_vlm_config(self):
        from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
            PI07HighLevelPlannerConfig,
        )
        from opentau.policies.pi07.low_level_planner.configuration_pi07_low_level import (
            PI07LowLevelPlannerConfig,
        )

        hl = PI07HighLevelPlannerConfig(attention_implementation="sdpa")
        assert hl.vlm_config.attention_implementation == "sdpa"

        ll = PI07LowLevelPlannerConfig(attention_implementation="sdpa")
        assert ll.vlm_config.attention_implementation == "sdpa"

    def test_post_init_preserves_explicit_vlm_config_when_policy_default(self):
        """Direct overrides via --policy.vlm_config.* must survive when the
        policy-level field is at its default ('eager' / False)."""
        from opentau.policies.pi07.high_level_planner.configuration_pi07_high_level import (
            PI07HighLevelPlannerConfig,
        )

        cfg = PI07HighLevelPlannerConfig(
            vlm_config=Gemma3WithExpertConfig(
                attention_implementation="sdpa",
                gradient_checkpointing=True,
                discrete_action_vocab_size=2048,
            )
        )
        # Policy-level fields stay at their defaults; vlm_config keeps its
        # explicit values (the __post_init__ plumbing is gated on the
        # policy-level field being NON-default).
        assert cfg.attention_implementation == "eager"
        assert cfg.gradient_checkpointing is False
        assert cfg.vlm_config.attention_implementation == "sdpa"
        assert cfg.vlm_config.gradient_checkpointing is True
