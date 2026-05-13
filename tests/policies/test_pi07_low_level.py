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

import pytest
import torch

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.pi07.gemma3_with_expert import Gemma3WithExpertConfig
from opentau.policies.pi07.low_level.configuration_pi07_low_level import (
    PI07LowLevelConfig,
)
from opentau.policies.pi07.low_level.modeling_pi07_low_level import (
    PI07LowLevelFlowMatching,
    PI07LowLevelPolicy,
    make_att_2d_masks,
)

# Tiny VLM config so that the full forward pass fits within 24 GB GPU memory.
# Backbone: 2 text layers × 512 hidden, 2 KV heads × 128 head_dim (=256 total).
# Expert:   2 layers × 256 hidden, matching head_dim/KV heads.
# Vision:   2 SigLIP layers, 448 image_size (matching production resolution).
_TINY_TEXT_HIDDEN = 512
_TINY_EXPERT_HIDDEN = 256
_TINY_HEAD_DIM = 128
_TINY_NUM_LAYERS = 2

_TINY_VLM_CONFIG = Gemma3WithExpertConfig(
    gemma3_config={
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": _TINY_TEXT_HIDDEN,
            "intermediate_size": _TINY_TEXT_HIDDEN * 4,
            "num_hidden_layers": _TINY_NUM_LAYERS,
            "num_attention_heads": _TINY_TEXT_HIDDEN // _TINY_HEAD_DIM,
            "num_key_value_heads": 2,
            "head_dim": _TINY_HEAD_DIM,
            "query_pre_attn_scalar": _TINY_HEAD_DIM,
            "sliding_window": 1024,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262_208,
            "max_position_embeddings": 8192,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window_pattern": 6,
            "torch_dtype": "float32",
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "patch_size": 14,
            "image_size": 448,
            "projection_dim": _TINY_TEXT_HIDDEN,
            "projector_hidden_act": "gelu_fast",
            "vision_use_head": False,
            "torch_dtype": "float32",
            "layer_norm_eps": 1e-6,
        },
        "image_token_index": 262144,
        "mm_tokens_per_image": 256,
        "boi_token_index": 255999,
        "eoi_token_index": 256000,
        "initializer_range": 0.02,
    },
    gemma_expert_config={
        "model_type": "gemma",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "head_dim": _TINY_HEAD_DIM,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": _TINY_EXPERT_HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": _TINY_EXPERT_HIDDEN * 4,
        "max_position_embeddings": 8192,
        "num_attention_heads": _TINY_TEXT_HIDDEN // _TINY_HEAD_DIM,
        "num_hidden_layers": _TINY_NUM_LAYERS,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "torch_dtype": "float32",
        "use_adarms": True,
        "adarms_cond_dim": _TINY_EXPERT_HIDDEN,
        "use_cache": True,
        "vocab_size": 262_208,
    },
    freeze_vision_encoder=True,
    train_expert_only=False,
    attention_implementation="eager",
    load_pretrained_gemma3=False,
    dropout=0.1,
)

# Config defaults used across the test.
NUM_CAMERAS = 2
# SpaceTimeSiglip pipes per-frame patches through the Gemma 3 multimodal
# projector, which always reduces to ``mm_tokens_per_image=256`` tokens per
# camera regardless of the underlying 32x32 patch grid (1024 patches at 448x448).
SPACETIME_SIGLIP_TOKENS_PER_CAMERA = 256
NUM_SUBGOAL_CAMERAS = 1
SIGLIP_TOKENS_PER_SUBGOAL = 256
IMAGE_SIZE = 448
PROMPT_MAX_LENGTH = 256
RESPONSE_MAX_LENGTH = 52
METADATA_MAX_LENGTH = 52
DISCRETE_ACTION_MAX_LENGTH = 32
# chunk_size=50 must match the ``lerobot_dataset_metadata`` fixture's action
# stats shape (50, 32); the normalizer buffers are built from those stats.
# n_obs_steps is reduced from the production default (8) to 2 so that the
# full-resolution 448×448 video tensors fit within 24 GB GPU memory.  T=2 is
# the minimum that exercises SpaceTime temporal attention (T=1 short-circuits
# to spatial-only).
CHUNK_SIZE = 50
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32

# For training the state is provided as (B, n_obs_steps, D) so T = n_obs_steps.
N_OBS_STEPS = 2

VIDEO_TOKENS = NUM_CAMERAS * SPACETIME_SIGLIP_TOKENS_PER_CAMERA  # 512
LANG_START = VIDEO_TOKENS  # 512
SUBGOAL_TOKENS = NUM_SUBGOAL_CAMERAS * SIGLIP_TOKENS_PER_SUBGOAL  # 256

# For inference: no discrete actions. SpaceTimeSiglip requires the full
# temporal window, so state keeps all T == N_OBS_STEPS timesteps.
INFER_STATE_TOKENS = N_OBS_STEPS


class TestPI07LowLevelIntegration:
    """Integration tests for the PI07 low-level component pipeline."""

    @staticmethod
    def _make_config() -> PI07LowLevelConfig:
        config = PI07LowLevelConfig(
            n_obs_steps=N_OBS_STEPS,
            chunk_size=CHUNK_SIZE,
            n_action_steps=CHUNK_SIZE,
            max_state_dim=MAX_STATE_DIM,
            max_action_dim=MAX_ACTION_DIM,
            prompt_max_length=PROMPT_MAX_LENGTH,
            response_max_length=RESPONSE_MAX_LENGTH,
            metadata_max_length=METADATA_MAX_LENGTH,
            discrete_action_max_length=DISCRETE_ACTION_MAX_LENGTH,
            proj_width=_TINY_EXPERT_HIDDEN,
            vlm_config=_TINY_VLM_CONFIG,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MIN_MAX,
                "ACTION": NormalizationMode.MEAN_STD,
            },
        )
        config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            "camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(MAX_STATE_DIM,)),
        }
        config.output_features = {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(CHUNK_SIZE, MAX_ACTION_DIM)),
        }
        return config

    @staticmethod
    def _indicator_lens(tokenizer):
        """Fixed strings inserted by ``embed_prefix`` (matches modeling layout).

        Resolved against the live Gemma 3 tokenizer, which may segment these
        spans differently than PaliGemma's tokenizer.
        """
        return {
            "state_lead": len(tokenizer.encode("State: ", add_special_tokens=False)),
            "comma": len(tokenizer.encode(", ", add_special_tokens=False)),
            "subgoal_lead": len(tokenizer.encode("Subgoal: ", add_special_tokens=False)),
            "prefix_end": len(tokenizer.encode(";\n ", add_special_tokens=False)),
            "action_lead": len(tokenizer.encode("Action: ", add_special_tokens=False)),
        }

    @classmethod
    def _train_prefix_total(cls, tokenizer) -> int:
        # Layout (per π0.7 paper Fig. 19, image goals after the text prompt):
        # video | lang | "State: " | state(T) | ", " | response | metadata |
        # ";\n " | "Subgoal: " | subgoal_imgs | ", " | "Action: " | discrete
        m = cls._indicator_lens(tokenizer)
        p = 0
        p += VIDEO_TOKENS
        p += PROMPT_MAX_LENGTH
        p += m["state_lead"] + N_OBS_STEPS + m["comma"]
        p += RESPONSE_MAX_LENGTH
        p += METADATA_MAX_LENGTH
        p += m["prefix_end"]
        p += m["subgoal_lead"] + SUBGOAL_TOKENS + m["comma"]
        p += m["action_lead"] + DISCRETE_ACTION_MAX_LENGTH
        return p

    @classmethod
    def _infer_prefix_total(cls, tokenizer) -> int:
        # Same Fig. 19 layout as training, minus the discrete-action block.
        m = cls._indicator_lens(tokenizer)
        p = 0
        p += VIDEO_TOKENS
        p += PROMPT_MAX_LENGTH
        p += m["state_lead"] + INFER_STATE_TOKENS + m["comma"]
        p += RESPONSE_MAX_LENGTH
        p += METADATA_MAX_LENGTH
        p += m["prefix_end"]
        p += m["subgoal_lead"] + SUBGOAL_TOKENS + m["comma"]
        return p

    @staticmethod
    def _check_ones_before_zeros(mask_slice):
        """Check that in a 1-D boolean mask all Trues precede all Falses."""
        mask = mask_slice.cpu().numpy()
        first_zero = None
        for idx, val in enumerate(mask):
            if val == 0:
                first_zero = idx
                break
        if first_zero is not None:
            assert all(v == 0 for v in mask[first_zero:]), f"Zeros not contiguous: {mask}"
            assert all(v == 1 for v in mask[:first_zero]), f"Ones not contiguous: {mask}"
        else:
            assert all(v == 1 for v in mask), f"Expected all ones: {mask}"

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _verify_pad_masks(self, prefix_pad_masks, suffix_pad_masks, tokenizer, inference_mode=False):
        assert prefix_pad_masks.shape[0] == 1
        total = self._infer_prefix_total(tokenizer) if inference_mode else self._train_prefix_total(tokenizer)
        assert prefix_pad_masks.shape[1] == total
        assert prefix_pad_masks.dtype == torch.bool
        assert suffix_pad_masks.shape == (1, CHUNK_SIZE)
        assert suffix_pad_masks.dtype == torch.bool

        m = self._indicator_lens(tokenizer)

        lang_slice = slice(LANG_START, LANG_START + PROMPT_MAX_LENGTH)
        state_t = INFER_STATE_TOKENS if inference_mode else N_OBS_STEPS

        # Layout (post-Fig. 19): video | lang | "State: " | state | ", " |
        # response | metadata | ";\n " | "Subgoal: " | subgoal_imgs | ", " |
        # "Action: " | discrete
        resp_lo = LANG_START + PROMPT_MAX_LENGTH + m["state_lead"] + state_t + m["comma"]
        resp_slice = slice(resp_lo, resp_lo + RESPONSE_MAX_LENGTH)

        meta_lo = resp_lo + RESPONSE_MAX_LENGTH
        meta_slice = slice(meta_lo, meta_lo + METADATA_MAX_LENGTH)

        sg_lo = meta_lo + METADATA_MAX_LENGTH + m["prefix_end"] + m["subgoal_lead"]
        sg_slice = slice(sg_lo, sg_lo + SUBGOAL_TOKENS)

        for i in range(prefix_pad_masks.shape[0]):
            assert torch.all(prefix_pad_masks[i, :VIDEO_TOKENS] == 1)
            self._check_ones_before_zeros(prefix_pad_masks[i, lang_slice])
            self._check_ones_before_zeros(prefix_pad_masks[i, resp_slice])
            self._check_ones_before_zeros(prefix_pad_masks[i, meta_slice])
            assert torch.all(prefix_pad_masks[i, sg_slice] == 1)

            if not inference_mode:
                da_lo = sg_lo + SUBGOAL_TOKENS + m["comma"] + m["action_lead"]
                da_slice = slice(da_lo, da_lo + DISCRETE_ACTION_MAX_LENGTH)
                self._check_ones_before_zeros(prefix_pad_masks[i, da_slice])

            self._check_ones_before_zeros(suffix_pad_masks[i])

    def _verify_position_ids(
        self,
        prefix_position_ids,
        suffix_position_ids,
        prefix_pad_masks,
        suffix_pad_masks,
        tokenizer,
        inference_mode=False,
    ):
        expected_prefix = torch.cumsum(prefix_pad_masks, dim=1) - 1
        assert torch.equal(prefix_position_ids, expected_prefix)

        if inference_mode:
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        else:
            # Training: model's prefix_offsets exclude both the "Action: " indicator
            # and the discrete-action span from cross-attention (matches pi05's
            # discrete_action_indicator_max_length logic) so the action expert sees
            # the same prefix length at train and inference.
            action_lead_len = self._indicator_lens(tokenizer)["action_lead"]
            prefix_offsets = torch.sum(
                prefix_pad_masks[:, : -(action_lead_len + DISCRETE_ACTION_MAX_LENGTH)], dim=-1
            )[:, None]

        expected_suffix = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        assert torch.equal(suffix_position_ids, expected_suffix)

    def _verify_vlm_attention_mask(
        self, vlm_attention_mask, prefix_pad_masks, prefix_att_masks, inference_mode=False
    ):
        del inference_mode  # same rule as training
        expected = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        assert torch.equal(vlm_attention_mask, expected), (
            f"VLM attention mask mismatch vs make_att_2d_masks.\n"
            f"Diff indices: {(vlm_attention_mask != expected).nonzero(as_tuple=False)[:20]}"
        )

    def _verify_action_expert_attention_mask(
        self,
        action_expert_attention_mask,
        prefix_pad_masks,
        suffix_pad_masks,
        suffix_att_masks,
        tokenizer,
        inference_mode=False,
    ):
        if inference_mode:
            num_cross = prefix_pad_masks.shape[1]
        else:
            # Training: cross-attention excludes both the "Action: " indicator and
            # the discrete-action span (mirrors the prefix_offsets logic above).
            action_lead_len = self._indicator_lens(tokenizer)["action_lead"]
            num_cross = prefix_pad_masks.shape[1] - action_lead_len - DISCRETE_ACTION_MAX_LENGTH

        expected = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross],
        )
        assert torch.equal(action_expert_attention_mask, expected), (
            f"Action expert attention mask mismatch vs make_att_2d_masks.\n"
            f"Diff indices: {(action_expert_attention_mask != expected).nonzero(as_tuple=False)[:20]}"
        )

    # ------------------------------------------------------------------
    # Main integration test
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="Requires too much memory, does not fit on RTX 3090 24GB")
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_complete_pi07_low_level_pipeline(self, lerobot_dataset_metadata):
        """Test the PI07 low-level component pipeline: forward (training) and select_action (inference)."""

        config = self._make_config()
        policy = PI07LowLevelPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)
        tokenizer = policy.model.language_tokenizer

        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, N_OBS_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE),
            "camera1": torch.randn(batch_size, N_OBS_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE),
            "subgoal0": torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
            "state": torch.randn(batch_size, N_OBS_STEPS, MAX_STATE_DIM),
            "actions": torch.randn(batch_size, CHUNK_SIZE, MAX_ACTION_DIM),
            "prompt": ["Pick up the red block"],
            "response": ["Grasp the red block"],
            "speed": torch.tensor([50]),
            "quality": torch.tensor([3]),
            "mistake": torch.tensor([0]),
            "speed_is_pad": torch.tensor([False]),
            "quality_is_pad": torch.tensor([False]),
            "mistake_is_pad": torch.tensor([False]),
            "subgoal_is_pad": torch.tensor([False]),
            "action_is_pad": torch.cat(
                [
                    torch.zeros(batch_size, CHUNK_SIZE // 2, dtype=torch.bool),
                    torch.ones(batch_size, CHUNK_SIZE - CHUNK_SIZE // 2, dtype=torch.bool),
                ],
                dim=1,
            ),
        }

        policy.to(dtype=torch.bfloat16, device="cuda")
        batch_cuda = {
            key: value.to("cuda", non_blocking=True, dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        batch_cuda["action_is_pad"] = batch_cuda["action_is_pad"].to(dtype=torch.bool)

        # ── Monkey-patch to capture intermediate tensors ──────────────
        captured = {}
        original_gemma3_forward = policy.model.gemma3_with_expert.forward
        original_embed_prefix = policy.model.embed_prefix
        original_embed_suffix = policy.model.embed_suffix

        def capture_forward(*args, **kwargs):
            if kwargs["inputs_embeds"][0] is not None:
                captured["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["vlm_position_ids"] = kwargs["position_ids"].clone()
            else:
                captured["action_expert_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["action_expert_position_ids"] = kwargs["position_ids"].clone()
            return original_gemma3_forward(*args, **kwargs)

        def capture_embed_prefix(*args, **kwargs):
            # Truncate discrete actions to inject padding (same workaround as PI05 test).
            half = DISCRETE_ACTION_MAX_LENGTH // 2
            discrete_actions = kwargs["discrete_actions"]
            discrete_action_masks = kwargs["discrete_action_masks"]
            kwargs["discrete_actions"] = torch.cat(
                (
                    discrete_actions[:, :half],
                    torch.zeros((1, half), dtype=discrete_actions.dtype, device=discrete_actions.device),
                ),
                dim=-1,
            )
            kwargs["discrete_action_masks"] = torch.cat(
                (
                    discrete_action_masks[:, :half],
                    torch.zeros((1, half), dtype=torch.bool, device=discrete_action_masks.device),
                ),
                dim=-1,
            )
            result = original_embed_prefix(*args, **kwargs)
            captured["prefix_pad_masks"] = result[1].clone()
            captured["prefix_att_masks"] = result[2].clone()
            return result

        def capture_embed_suffix(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            captured["suffix_pad_masks"] = result[1].clone()
            captured["suffix_att_masks"] = result[2].clone()
            return result

        policy.model.gemma3_with_expert.forward = capture_forward
        policy.model.embed_prefix = capture_embed_prefix
        policy.model.embed_suffix = capture_embed_suffix

        # ── Training forward pass ────────────────────────────────────
        loss = policy.forward(batch_cuda)

        # Restore originals.
        policy.model.gemma3_with_expert.forward = original_gemma3_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.embed_suffix = original_embed_suffix

        # Verify normalize / unnormalize round-trip.
        action_output = {"actions": batch["actions"].to("cuda")}
        assert torch.allclose(
            action_output["actions"],
            policy.unnormalize_outputs(policy.normalize_targets(action_output))["actions"],
            atol=1e-6,
        )

        # Verify all expected captures are present.
        for var in [
            "prefix_pad_masks",
            "prefix_att_masks",
            "suffix_pad_masks",
            "suffix_att_masks",
            "vlm_2d_attention_mask",
            "vlm_position_ids",
            "action_expert_2d_attention_mask",
            "action_expert_position_ids",
        ]:
            assert var in captured, f"{var} was not captured"

        assert captured["vlm_2d_attention_mask"].dtype == torch.bool
        assert captured["action_expert_2d_attention_mask"].dtype == torch.bool
        assert captured["prefix_pad_masks"].dtype == torch.bool
        assert captured["suffix_pad_masks"].dtype == torch.bool

        self._verify_pad_masks(captured["prefix_pad_masks"], captured["suffix_pad_masks"], tokenizer)
        self._verify_position_ids(
            captured["vlm_position_ids"],
            captured["action_expert_position_ids"],
            captured["prefix_pad_masks"],
            captured["suffix_pad_masks"],
            tokenizer,
        )
        self._verify_vlm_attention_mask(
            captured["vlm_2d_attention_mask"],
            captured["prefix_pad_masks"],
            captured["prefix_att_masks"],
        )
        self._verify_action_expert_attention_mask(
            captured["action_expert_2d_attention_mask"],
            captured["prefix_pad_masks"],
            captured["suffix_pad_masks"],
            captured["suffix_att_masks"],
            tokenizer,
        )

        assert isinstance(loss, dict)
        assert "MSE" in loss
        assert "CE" in loss
        assert all(v.isfinite() for v in loss.values())

        # Reset and check queue cleared.
        policy.reset()
        assert len(policy._action_queue) == 0

        # Optimizer params are non-empty.
        assert len(list(policy.get_optim_params())) > 0

        # ── Inference via select_action ──────────────────────────────
        captured_infer = {}

        def capture_forward_infer(*args, **kwargs):
            if kwargs["inputs_embeds"][0] is not None and kwargs.get("past_key_values") is None:
                captured_infer["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured_infer["vlm_position_ids"] = kwargs["position_ids"].clone()
            else:
                captured_infer["action_expert_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured_infer["action_expert_position_ids"] = kwargs["position_ids"].clone()
            return original_gemma3_forward(*args, **kwargs)

        def capture_embed_prefix_infer(*args, **kwargs):
            result = original_embed_prefix(*args, **kwargs)
            captured_infer["prefix_pad_masks"] = result[1].clone()
            captured_infer["prefix_att_masks"] = result[2].clone()
            return result

        def capture_embed_suffix_infer(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            captured_infer["suffix_pad_masks"] = result[1].clone()
            captured_infer["suffix_att_masks"] = result[2].clone()
            return result

        policy.model.gemma3_with_expert.forward = capture_forward_infer
        policy.model.embed_prefix = capture_embed_prefix_infer
        policy.model.embed_suffix = capture_embed_suffix_infer

        # Inference batch: SpaceTimeSiglip requires T == n_obs_steps frames, so
        # videos stay 5-D (B, T, C, H, W) and state stays 3-D (B, T, D).
        infer_batch = {
            "camera0": batch_cuda["camera0"],  # (B, T, C, H, W)
            "camera1": batch_cuda["camera1"],
            "subgoal0": batch_cuda["subgoal0"],  # already (B, C, H, W)
            "state": batch_cuda["state"],  # (B, T, D)
            "prompt": ["Pick up the red block"],
            "response": ["Grasp the red block"],
            "speed": torch.tensor([50], device="cuda"),
            "quality": torch.tensor([3], device="cuda"),
            "mistake": torch.tensor([0], device="cuda"),
            "speed_is_pad": torch.tensor([False], device="cuda"),
            "quality_is_pad": torch.tensor([False], device="cuda"),
            "mistake_is_pad": torch.tensor([False], device="cuda"),
            "subgoal_is_pad": torch.tensor([False], device="cuda"),
        }
        action = policy.select_action(infer_batch)

        # Restore originals.
        policy.model.gemma3_with_expert.forward = original_gemma3_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.embed_suffix = original_embed_suffix

        for var in [
            "prefix_pad_masks",
            "prefix_att_masks",
            "suffix_pad_masks",
            "suffix_att_masks",
            "vlm_2d_attention_mask",
            "vlm_position_ids",
            "action_expert_2d_attention_mask",
            "action_expert_position_ids",
        ]:
            assert var in captured_infer, f"{var} was not captured for select_action"

        assert captured_infer["vlm_2d_attention_mask"].dtype == torch.bool
        assert captured_infer["action_expert_2d_attention_mask"].dtype == torch.bool
        assert captured_infer["prefix_pad_masks"].dtype == torch.bool
        assert captured_infer["suffix_pad_masks"].dtype == torch.bool

        self._verify_pad_masks(
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            tokenizer,
            inference_mode=True,
        )
        self._verify_position_ids(
            captured_infer["vlm_position_ids"],
            captured_infer["action_expert_position_ids"],
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            tokenizer,
            inference_mode=True,
        )
        self._verify_vlm_attention_mask(
            captured_infer["vlm_2d_attention_mask"],
            captured_infer["prefix_pad_masks"],
            captured_infer["prefix_att_masks"],
            inference_mode=True,
        )
        self._verify_action_expert_attention_mask(
            captured_infer["action_expert_2d_attention_mask"],
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            captured_infer["suffix_att_masks"],
            tokenizer,
            inference_mode=True,
        )

        assert action.shape == (1, MAX_ACTION_DIM)

    @pytest.mark.skip(reason="Requires too much memory, does not fit on RTX 3090 24GB")
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_no_optionals_path_on_real_gemma3(self, lerobot_dataset_metadata):
        """Lock in the no-optionals prefix layout end-to-end on the real Gemma 3 backbone.

        Constructs a batch with empty response (full ``response_drop_prob=1.0``
        equivalent), all metadata fields padded (full metadata dropout), and
        ``subgoal_is_pad=True`` (full ``subgoal_drop_prob=1.0`` equivalent) so
        that ``has_any_optional`` is False and ``embed_prefix`` should:

          * collapse the state-end separator from ``", "`` to ``":\\n"``,
          * skip the metadata block entirely,
          * skip the trailing ``";\\n "`` prefix-end.

        Without this guard a regression that flips ``has_any_optional``
        semantics would still pass the existing ``_train_prefix_total`` /
        ``_infer_prefix_total`` shape checks (which hard-code the
        all-optionals layout) and the structural CPU tests in
        ``test_pi07_cpu.py``, but produce miscalibrated cross-attention
        offsets on real Gemma 3 weights.
        """
        config = self._make_config()
        policy = PI07LowLevelPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)
        tokenizer = policy.model.language_tokenizer

        batch_size = 1
        # Zero-action batch so the discrete-action CE path is still exercised
        # while the optional middle blocks are entirely absent.
        batch = {
            "camera0": torch.randn(batch_size, N_OBS_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE),
            "camera1": torch.randn(batch_size, N_OBS_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE),
            "subgoal0": torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
            "state": torch.randn(batch_size, N_OBS_STEPS, MAX_STATE_DIM),
            "actions": torch.randn(batch_size, CHUNK_SIZE, MAX_ACTION_DIM),
            "prompt": ["Pick up the red block"],
            # Empty response → all-padding response_masks → has_response == False.
            "response": [""],
            # All metadata fields padded → empty metadata string → has_metadata == False.
            "speed": torch.tensor([0]),
            "quality": torch.tensor([0]),
            "mistake": torch.tensor([0]),
            "speed_is_pad": torch.tensor([True]),
            "quality_is_pad": torch.tensor([True]),
            "mistake_is_pad": torch.tensor([True]),
            # Subgoal padded → all-False subgoal_img_masks → has_subgoal == False.
            "subgoal_is_pad": torch.tensor([True]),
            "action_is_pad": torch.zeros(batch_size, CHUNK_SIZE, dtype=torch.bool),
        }

        policy.to(dtype=torch.bfloat16, device="cuda")
        batch_cuda = {
            key: value.to("cuda", non_blocking=True, dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        batch_cuda["action_is_pad"] = batch_cuda["action_is_pad"].to(dtype=torch.bool)
        batch_cuda["speed_is_pad"] = batch_cuda["speed_is_pad"].to(dtype=torch.bool)
        batch_cuda["quality_is_pad"] = batch_cuda["quality_is_pad"].to(dtype=torch.bool)
        batch_cuda["mistake_is_pad"] = batch_cuda["mistake_is_pad"].to(dtype=torch.bool)
        batch_cuda["subgoal_is_pad"] = batch_cuda["subgoal_is_pad"].to(dtype=torch.bool)

        captured = {}
        original_embed_prefix = policy.model.embed_prefix

        def capture_embed_prefix(*args, **kwargs):
            result = original_embed_prefix(*args, **kwargs)
            captured["prefix_pad_masks"] = result[1].clone()
            captured["prefix_att_masks"] = result[2].clone()
            return result

        policy.model.embed_prefix = capture_embed_prefix
        try:
            loss = policy.forward(batch_cuda)
        finally:
            policy.model.embed_prefix = original_embed_prefix

        # Loss is finite — the integrated path on real Gemma 3 weights survived
        # the no-optionals layout (cross-attention offsets and position IDs both
        # consistent with the collapsed prefix).
        assert isinstance(loss, dict)
        assert all(v.isfinite() for v in loss.values())

        # Pin the collapsed prefix length: VIDEO + LANG + state-block (state_lead
        # + N_OBS_STEPS + state-end-no-opt) + Action: + DISCRETE — no response,
        # no subgoal, no metadata, and no ";\n " prefix-end.
        m = self._indicator_lens(tokenizer)
        state_end_no_opt_len = len(tokenizer.encode(":\n", add_special_tokens=False))
        expected_prefix_len = (
            VIDEO_TOKENS
            + PROMPT_MAX_LENGTH
            + m["state_lead"]
            + N_OBS_STEPS
            + state_end_no_opt_len
            + m["action_lead"]
            + DISCRETE_ACTION_MAX_LENGTH
        )
        assert captured["prefix_pad_masks"].shape == (batch_size, expected_prefix_len), (
            f"no-optionals prefix length mismatch: got {captured['prefix_pad_masks'].shape[1]}, "
            f"expected {expected_prefix_len}. A regression that re-emits the metadata block, "
            f"the trailing ';\\n ' prefix-end, or fails to collapse the state-end separator "
            f"would change this length."
        )

        # Per π0.7 paper §VI.B (post-fix): the only bidirectional block at the
        # head of the prefix is the video span; every text/state span after it
        # opens its own causal block (text: ``[1] * N``; state: ``[1] + [0]*(T-1)``).
        # So the first causal boundary should be at the start of the language
        # span (right after the video block). Mirrors the CPU test invariant in
        # ``test_pi07_cpu.py::test_all_optional_blocks_absent_skips_emission``
        # but on the real Gemma 3 tokenizer.
        prefix_att_masks = captured["prefix_att_masks"][0].cpu()
        first_one = int((prefix_att_masks == 1).nonzero(as_tuple=False).flatten()[0].item())
        expected_first_one = VIDEO_TOKENS  # = LANG_START
        assert first_one == expected_first_one, (
            f"first causal boundary at position {first_one}, expected {expected_first_one} "
            f"(start of language span). A later ``1`` signals a regression to the pre-fix "
            f"pattern that lumped lang/State:/state-end into the bidirectional video block."
        )


class TestPI07LowLevelRegression:
    """GPU regression tests pinning the low-level component signature/dtype fixes.

    Covers the changes made to ``embed_prefix``, ``embed_suffix``,
    ``prepare_metadata``, and the metadata-zip ``strict=True`` switch.
    """

    @staticmethod
    def _make_policy(lerobot_dataset_metadata) -> PI07LowLevelPolicy:
        config = TestPI07LowLevelIntegration._make_config()
        policy = PI07LowLevelPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)
        policy.to(dtype=torch.bfloat16, device="cuda")
        return policy

    @staticmethod
    def _make_metadata_batch(batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "state": torch.randn(batch_size, N_OBS_STEPS, MAX_STATE_DIM, device="cuda", dtype=torch.bfloat16),
            "speed": torch.tensor([50] * batch_size, device="cuda"),
            "quality": torch.tensor([3] * batch_size, device="cuda"),
            "mistake": torch.tensor([0] * batch_size, device="cuda"),
            "speed_is_pad": torch.tensor([False] * batch_size, device="cuda"),
            "quality_is_pad": torch.tensor([False] * batch_size, device="cuda"),
            "mistake_is_pad": torch.tensor([False] * batch_size, device="cuda"),
        }

    @pytest.mark.skip(reason="Requires too much memory, does not fit on RTX 3090 24GB")
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_prepare_metadata_always_returns_tensors(self, lerobot_dataset_metadata):
        """prepare_metadata returns (Tensor, Tensor) — never (None, None) — with the documented shapes."""
        policy = self._make_policy(lerobot_dataset_metadata)
        batch = self._make_metadata_batch(batch_size=2)

        tokens, masks = policy.prepare_metadata(batch)

        assert isinstance(tokens, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert tokens.shape == (2, METADATA_MAX_LENGTH)
        assert masks.shape == (2, METADATA_MAX_LENGTH)
        assert masks.dtype == torch.bool

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_prepare_metadata_zip_strict_catches_mismatch(self, lerobot_dataset_metadata):
        """The ``strict=True`` zip in prepare_metadata raises on length mismatch."""
        policy = self._make_policy(lerobot_dataset_metadata)
        batch = self._make_metadata_batch(batch_size=2)
        # Truncate quality to length 1 to break the zip.
        batch["quality"] = batch["quality"][:1]
        batch["quality_is_pad"] = batch["quality_is_pad"][:1]

        with pytest.raises(ValueError):
            policy.prepare_metadata(batch)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_embed_suffix_returns_bool_att_masks(self, lerobot_dataset_metadata):
        """The suffix att_masks must be bool, not embs.dtype (was a copy-paste bug)."""
        policy = self._make_policy(lerobot_dataset_metadata)

        bsize = 1
        noisy_actions = torch.randn(bsize, CHUNK_SIZE, MAX_ACTION_DIM, device="cuda", dtype=torch.bfloat16)
        timestep = torch.zeros(bsize, CHUNK_SIZE, device="cuda", dtype=torch.bfloat16)

        _, _, att_masks, _ = policy.model.embed_suffix(noisy_actions, timestep)

        assert att_masks.dtype == torch.bool, f"Expected torch.bool, got {att_masks.dtype}"

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_embed_prefix_metadata_response_are_required(self):
        """response_tokens / response_masks / metadata_tokens / metadata_masks are positional, no defaults."""
        import inspect

        params = inspect.signature(PI07LowLevelFlowMatching.embed_prefix).parameters
        for name in ("response_tokens", "response_masks", "metadata_tokens", "metadata_masks"):
            assert params[name].default is inspect.Parameter.empty, (
                f"{name} should be a required parameter (no default), got default={params[name].default}"
            )


class TestGemma3WithExpertFSDPWrap:
    """Smoke-level GPU coverage that the InterleavedDecoderLayer-based model
    can be wrapped under FSDP and run one forward+backward without raising.

    The CPU invariants in ``test_pi07_cpu.py::TestInterleavedDecoderLayer``
    pin param uniqueness / state_dict prefix / dispatch lookup, but a wrap-
    target regression (e.g. someone re-introduces the language_model.layers
    nesting or shares a Parameter between two modules) would only fail at
    actual FSDP construction. Until commit aaefa6e + this PR that failure
    was only caught by the 8×A100 throughput matrix; this test makes it
    catchable on any single-GPU box.
    """

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_fsdp_wrap_forward_backward(self):
        """Wrap the model with FSDP (single-rank process group on the local
        GPU) and exercise one forward + backward. Catches param double-
        registration, shared-parameter handles, and unwrappable submodules."""
        import functools
        import os

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from opentau.policies.pi07.gemma3_with_expert import (
            Gemma3WithExpertModel,
            InterleavedDecoderLayer,
        )

        if not torch.cuda.is_available():
            pytest.skip("FSDP wrap test requires CUDA")

        # Single-rank process group on the local GPU; the test owns init /
        # teardown so it doesn't pollute other GPU tests in the same pytest
        # session. world_size=1 keeps the all-gather / reduce-scatter hooks
        # active (their no-op fast paths still walk the FlatParameter
        # machinery) while not requiring multi-GPU infrastructure.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        already_initialized = torch.distributed.is_initialized()
        if not already_initialized:
            # Set the CUDA device and drain stale state from earlier GPU tests
            # in the same pytest session before init_process_group, since NCCL
            # binds its communicator to whatever device is current at init time
            # (issue #283 hint about stale GPU context). When a prior test has
            # already initialized the PG, NCCL is bound to whichever device that
            # test picked and re-pinning here can't unbind it — so this drain is
            # only meaningful on the path where we own the init.
            torch.cuda.set_device(0)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # device_id pins the new PG to the local GPU explicitly, so NCCL
            # forms its communicator eagerly on the correct device instead of
            # the lazy default-device guess (PyTorch ≥ 2.0).
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=1,
                rank=0,
                device_id=torch.device("cuda", 0),
            )
        try:
            cfg = Gemma3WithExpertConfig(
                gemma3_config=_TINY_VLM_CONFIG.gemma3_config.to_dict(),
                gemma_expert_config=_TINY_VLM_CONFIG.gemma_expert_config.to_dict(),
                # Skip the in-init bf16 cast so FSDP MixedPrecision can manage
                # the fp32-master / bf16-compute split itself (the production
                # FSDP regime — see profile_step.py:215-218 and train.py).
                disable_internal_bf16_cast=True,
                freeze_vision_encoder=True,
                train_expert_only=False,
                attention_implementation="eager",
                # Production wrappers (PI07LowLevelPolicy) inject this from the
                # FAST tokenizer; here we set it explicitly so the bare
                # Gemma3WithExpertModel can build its discrete-action head.
                discrete_action_vocab_size=32,
            )

            model = Gemma3WithExpertModel(cfg).to(device="cuda", dtype=torch.float32)

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={InterleavedDecoderLayer},
            )
            # Match the production FSDP regime (accelerate's `mixed_precision: bf16`)
            # — without it, the model's fp32 weights collide with `_preferred_dtype()`
            # casting activations to bf16 inside `gemma3_with_expert.py:forward`.
            #
            # ``reduce_dtype=fp32`` (rather than bf16) for the single-rank smoke test:
            # NCCL 2.27.5 hits an ``ncclUnhandledCudaError`` on bf16 ``all_reduce`` over
            # a single-rank PG inside FSDP's ``_reduce_grad_no_shard`` fast path
            # (issue #283). fp32 reduce is also a valid production setting (accelerate
            # ``fsdp_reduce_dtype: fp32``) and the test's purpose is FSDP-wrap topology,
            # not the bf16 reduce path itself — multi-rank bf16-reduce coverage lives
            # in the regression matrix.
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
            wrapped = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                use_orig_params=True,
                device_id=torch.cuda.current_device(),
            )

            batch, seq_len = 1, 4
            hidden_size = cfg.gemma3_config.text_config.hidden_size
            inputs = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.float32)
            position_ids = torch.arange(seq_len, device="cuda")[None, :]
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"))[None]

            outs, _ = wrapped(
                attention_mask=attn_mask,
                position_ids=position_ids,
                inputs_embeds=[inputs, None],
            )
            assert outs[0] is not None
            # One backward to exercise the FSDP all-gather / reduce-scatter
            # hooks on every interleaved layer — the path that would deadlock
            # if the wrap target was misaligned.
            outs[0].sum().backward()

            # Confirm the wrap actually saw the InterleavedDecoderLayer (vs.
            # falling back to a single root-level wrap that hides regressions).
            wrapped_modules = [m for m in wrapped.modules() if isinstance(m, FSDP)]
            assert any(isinstance(m.module, InterleavedDecoderLayer) for m in wrapped_modules), (
                "transformer_auto_wrap_policy did not pick up InterleavedDecoderLayer"
            )
        finally:
            if not already_initialized and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
