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
from opentau.policies.pi07_paligemma.high_level_planner.configuration_pi07_high_level import (
    PI07HighLevelPlannerConfig,
)
from opentau.policies.pi07_paligemma.high_level_planner.modeling_pi07_high_level import (
    PI07HighLevelPlannerPolicy,
    make_att_2d_masks,
)

# Config defaults used across the test.
NUM_CAMERAS = 2
SIGLIP_TOKENS_PER_CAMERA = 256
PROMPT_MAX_LENGTH = 256
METADATA_MAX_LENGTH = 52
MEMORY_MAX_LENGTH = 52
RESPONSE_MAX_LENGTH = 52
MAX_STATE_DIM = 32

# Token offsets (fixed by config and image tokenization).
IMAGE_TOKENS = NUM_CAMERAS * SIGLIP_TOKENS_PER_CAMERA  # 512
LANG_START = IMAGE_TOKENS  # 512
METADATA_START = LANG_START + PROMPT_MAX_LENGTH  # 768
METADATA_END = METADATA_START + METADATA_MAX_LENGTH  # 820


class TestPI07HighLevelPlannerIntegration:
    """Integration tests for the PI07 high-level planner pipeline."""

    @staticmethod
    def _make_config() -> PI07HighLevelPlannerConfig:
        config = PI07HighLevelPlannerConfig(
            n_obs_steps=1,
            max_state_dim=MAX_STATE_DIM,
            prompt_max_length=PROMPT_MAX_LENGTH,
            metadata_max_length=METADATA_MAX_LENGTH,
            memory_max_length=MEMORY_MAX_LENGTH,
            response_max_length=RESPONSE_MAX_LENGTH,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
            },
        )
        config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(MAX_STATE_DIM,)),
        }
        config.output_features = {}
        return config

    @staticmethod
    def _indicator_lens(tokenizer):
        """Lengths of fixed text spans inserted by ``embed_prefix`` / inference."""
        return {
            "prefix_end": len(tokenizer.encode(";\n ", add_special_tokens=False)),
            "memory_lead": len(tokenizer.encode("Updated Memory: ", add_special_tokens=False)),
            "subtask_lead": len(tokenizer.encode("Subtask: ", add_special_tokens=False)),
        }

    @classmethod
    def _train_prefix_total(cls, tokenizer) -> int:
        meta = cls._indicator_lens(tokenizer)
        mem_tokens_start = METADATA_END + meta["prefix_end"] + meta["memory_lead"]
        return mem_tokens_start + MEMORY_MAX_LENGTH + meta["subtask_lead"] + RESPONSE_MAX_LENGTH

    @classmethod
    def _infer_embed_prefix_total(cls, tokenizer) -> int:
        meta = cls._indicator_lens(tokenizer)
        return METADATA_END + meta["prefix_end"] + meta["memory_lead"]

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

    def _verify_pad_masks(self, prefix_pad_masks, tokenizer):
        meta = self._indicator_lens(tokenizer)
        total = self._train_prefix_total(tokenizer)
        assert prefix_pad_masks.shape == (1, total)
        assert prefix_pad_masks.dtype == torch.bool

        mem_tokens_start = METADATA_END + meta["prefix_end"] + meta["memory_lead"]
        resp_tokens_start = mem_tokens_start + MEMORY_MAX_LENGTH + meta["subtask_lead"]

        for i in range(1):
            assert torch.all(prefix_pad_masks[i, :IMAGE_TOKENS] == 1)
            self._check_ones_before_zeros(prefix_pad_masks[i, LANG_START:METADATA_START])
            self._check_ones_before_zeros(prefix_pad_masks[i, METADATA_START:METADATA_END])
            self._check_ones_before_zeros(
                prefix_pad_masks[i, mem_tokens_start : mem_tokens_start + MEMORY_MAX_LENGTH]
            )
            self._check_ones_before_zeros(
                prefix_pad_masks[i, resp_tokens_start : resp_tokens_start + RESPONSE_MAX_LENGTH]
            )

    def _verify_position_ids(self, prefix_position_ids, prefix_pad_masks):
        expected = torch.cumsum(prefix_pad_masks, dim=1) - 1
        assert torch.equal(prefix_position_ids, expected)

    def _verify_vlm_attention_mask(self, vlm_attention_mask, prefix_pad_masks, prefix_att_masks):
        expected = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        assert torch.equal(vlm_attention_mask, expected), (
            f"VLM attention mask mismatch vs make_att_2d_masks.\n"
            f"Diff indices: {(vlm_attention_mask != expected).nonzero(as_tuple=False)[:20]}"
        )

    # ------------------------------------------------------------------
    # Main integration test
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="run on local machine")
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_complete_pi07_high_level_planner_pipeline(self, lerobot_dataset_metadata):
        """Test the PI07 high-level planner: forward (training) and sample_actions (inference)."""

        config = self._make_config()
        policy = PI07HighLevelPlannerPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)
        tokenizer = policy.model.language_tokenizer

        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, 3, 224, 224),
            "camera1": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, MAX_STATE_DIM),
            "prompt": ["Pick up the red block"],
            "past_memory": ["Robot is near the table"],
            "speed": torch.tensor([50]),
            "quality": torch.tensor([3]),
            "mistake": torch.tensor([0]),
            "speed_is_pad": torch.tensor([False]),
            "quality_is_pad": torch.tensor([False]),
            "mistake_is_pad": torch.tensor([False]),
            "next_memory": ["Robot is grasping the red block"],
            "response": ["Grasp the red block"],
        }

        policy.to(dtype=torch.bfloat16, device="cuda")
        batch_cuda = {
            key: value.to("cuda", non_blocking=True, dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

        # ── Monkey-patch to capture intermediate tensors ──────────────
        captured = {}
        original_paligemma_forward = policy.model.paligemma_with_expert.forward
        original_embed_prefix = policy.model.embed_prefix

        def capture_forward(*args, **kwargs):
            if kwargs.get("past_key_values") is None:
                captured["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["vlm_position_ids"] = kwargs["position_ids"].clone()
            return original_paligemma_forward(*args, **kwargs)

        def capture_embed_prefix(*args, **kwargs):
            result = original_embed_prefix(*args, **kwargs)
            captured["prefix_pad_masks"] = result[1].clone()
            captured["prefix_att_masks"] = result[2].clone()
            return result

        policy.model.paligemma_with_expert.forward = capture_forward
        policy.model.embed_prefix = capture_embed_prefix

        # ── Training forward pass ────────────────────────────────────
        loss = policy.forward(batch_cuda)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix

        # Verify captures.
        for var in ["prefix_pad_masks", "prefix_att_masks", "vlm_2d_attention_mask", "vlm_position_ids"]:
            assert var in captured, f"{var} was not captured"

        assert captured["vlm_2d_attention_mask"].dtype == torch.bool
        assert captured["prefix_pad_masks"].dtype == torch.bool

        self._verify_pad_masks(captured["prefix_pad_masks"], tokenizer)
        self._verify_position_ids(captured["vlm_position_ids"], captured["prefix_pad_masks"])
        self._verify_vlm_attention_mask(
            captured["vlm_2d_attention_mask"],
            captured["prefix_pad_masks"],
            captured["prefix_att_masks"],
        )

        assert isinstance(loss, dict)
        assert "MSE" in loss
        assert "CE" in loss
        assert loss["MSE"].item() == 0.0
        assert loss["CE"].isfinite()

        # Optimizer params are non-empty.
        assert len(list(policy.get_optim_params())) > 0

        # ── Inference via sample_actions ──────────────────────────────
        captured_infer = {}
        original_infer_autoregressive = policy.model.infer_autoregressive
        step_counter = [0]

        def capture_infer_autoregressive(*args, **kwargs):
            prev_prefix_pad_masks = kwargs.get("prefix_pad_masks")
            result = original_infer_autoregressive(*args, **kwargs)
            (
                _prefix_out,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                prefix_offsets,
                tokens,
                _past_kv,
            ) = result

            # Verify prefix grows by exactly one token each step.
            assert prefix_pad_masks.shape[1] == prev_prefix_pad_masks.shape[1] + 1, (
                f"Step {step_counter[0]}: prefix should grow by 1"
            )
            assert prefix_embs.shape[1] == prefix_pad_masks.shape[1]

            assert prefix_att_masks.shape[1] == prefix_pad_masks.shape[1]

            captured_infer["last_prefix_pad_masks"] = prefix_pad_masks.clone()
            captured_infer["last_prefix_offsets"] = prefix_offsets.clone()
            step_counter[0] += 1
            return result

        def capture_forward_infer(*args, **kwargs):
            if kwargs.get("past_key_values") is None:
                captured_infer["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured_infer["vlm_position_ids"] = kwargs["position_ids"].clone()
            return original_paligemma_forward(*args, **kwargs)

        def capture_embed_prefix_infer(*args, **kwargs):
            result = original_embed_prefix(*args, **kwargs)
            captured_infer["prefix_pad_masks"] = result[1].clone()
            captured_infer["prefix_att_masks"] = result[2].clone()
            return result

        policy.model.paligemma_with_expert.forward = capture_forward_infer
        policy.model.embed_prefix = capture_embed_prefix_infer
        policy.model.infer_autoregressive = capture_infer_autoregressive

        infer_batch = {
            "camera0": batch_cuda["camera0"],
            "camera1": batch_cuda["camera1"],
            "state": batch_cuda["state"],
            "prompt": ["Pick up the red block"],
            "past_memory": ["Robot is near the table"],
            "speed": torch.tensor([50], device="cuda"),
            "quality": torch.tensor([3], device="cuda"),
            "mistake": torch.tensor([0], device="cuda"),
            "speed_is_pad": torch.tensor([False], device="cuda"),
            "quality_is_pad": torch.tensor([False], device="cuda"),
            "mistake_is_pad": torch.tensor([False], device="cuda"),
        }
        memory_tokens, response_tokens = policy.sample_actions(infer_batch)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.infer_autoregressive = original_infer_autoregressive

        # Verify total number of autoregressive steps (memory + response only).
        assert step_counter[0] == MEMORY_MAX_LENGTH + RESPONSE_MAX_LENGTH

        # Verify inference captures.
        assert "vlm_2d_attention_mask" in captured_infer
        assert "prefix_pad_masks" in captured_infer
        assert "prefix_att_masks" in captured_infer
        assert "last_prefix_pad_masks" in captured_infer
        assert "last_prefix_offsets" in captured_infer

        infer_base = self._infer_embed_prefix_total(tokenizer)
        assert captured_infer["vlm_2d_attention_mask"].shape == (1, infer_base, infer_base)
        assert captured_infer["vlm_2d_attention_mask"].dtype == torch.bool

        init_expected = make_att_2d_masks(
            captured_infer["prefix_pad_masks"],
            captured_infer["prefix_att_masks"],
        )
        assert torch.equal(captured_infer["vlm_2d_attention_mask"], init_expected), (
            f"Inference VLM 2D mask vs make_att_2d_masks.\n"
            f"Diff: {(captured_infer['vlm_2d_attention_mask'] != init_expected).nonzero(as_tuple=False)[:20]}"
        )

        subtask_lead = self._indicator_lens(tokenizer)["subtask_lead"]
        final_prefix_len = infer_base + MEMORY_MAX_LENGTH + subtask_lead + RESPONSE_MAX_LENGTH
        assert captured_infer["last_prefix_pad_masks"].shape == (1, final_prefix_len)

        # Output shapes.
        assert memory_tokens.shape == (1, MEMORY_MAX_LENGTH)
        assert response_tokens.shape == (1, RESPONSE_MAX_LENGTH)
