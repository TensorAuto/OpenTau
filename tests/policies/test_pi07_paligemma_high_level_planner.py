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
)

# Config defaults used across the test.
NUM_CAMERAS = 2
SIGLIP_TOKENS_PER_CAMERA = 256
PROMPT_MAX_LENGTH = 256
MEMORY_MAX_LENGTH = 52
RESPONSE_MAX_LENGTH = 52
MAX_STATE_DIM = 32

# Token offsets for training prefix:
#   images(512) | language(256) | memory(52) | response(52) = 872
IMAGE_TOKENS = NUM_CAMERAS * SIGLIP_TOKENS_PER_CAMERA  # 512
LANG_START = IMAGE_TOKENS  # 512
MEMORY_START = LANG_START + PROMPT_MAX_LENGTH  # 768
RESPONSE_START = MEMORY_START + MEMORY_MAX_LENGTH  # 820
TRAIN_PREFIX_LEN = RESPONSE_START + RESPONSE_MAX_LENGTH  # 872

# For inference: no memory or response tokens initially.
INFER_PREFIX_LEN = IMAGE_TOKENS + PROMPT_MAX_LENGTH  # 768


class TestPI07HighLevelPlannerIntegration:
    """Integration tests for the PI07 high-level planner pipeline."""

    @staticmethod
    def _make_config() -> PI07HighLevelPlannerConfig:
        config = PI07HighLevelPlannerConfig(
            n_obs_steps=1,
            max_state_dim=MAX_STATE_DIM,
            prompt_max_length=PROMPT_MAX_LENGTH,
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

    def _verify_pad_masks(self, prefix_pad_masks):
        assert prefix_pad_masks.shape == (1, TRAIN_PREFIX_LEN)
        assert prefix_pad_masks.dtype == torch.bool

        for i in range(1):
            # Image tokens should never be padded.
            assert torch.all(prefix_pad_masks[i, :IMAGE_TOKENS] == 1)
            # Language, memory, response tokens can be padded at the end.
            self._check_ones_before_zeros(prefix_pad_masks[i, LANG_START:MEMORY_START])
            self._check_ones_before_zeros(prefix_pad_masks[i, MEMORY_START:RESPONSE_START])
            self._check_ones_before_zeros(prefix_pad_masks[i, RESPONSE_START:TRAIN_PREFIX_LEN])

    def _verify_position_ids(self, prefix_position_ids, prefix_pad_masks):
        assert prefix_position_ids.shape == (1, TRAIN_PREFIX_LEN)
        assert prefix_position_ids.dtype == torch.long

        for i in range(1):
            pids = prefix_position_ids[i]
            pads = prefix_pad_masks[i]
            for j in range(1, len(pids)):
                if pads[j] == 1:
                    assert pids[j] == pids[j - 1] + 1, (
                        f"Position ID should increment at {j}: {pids[j - 1]} -> {pids[j]}"
                    )
                else:
                    assert pids[j] == pids[j - 1], (
                        f"Position ID should stay same at padded {j}: {pids[j - 1]} -> {pids[j]}"
                    )

    def _verify_vlm_attention_mask(self, vlm_attention_mask, prefix_pad_masks):
        """Verify the VLM attention mask for training.

        Expected pattern:
            - images + language (0..767): bidirectional (cumsum=0).
            - memory (768..819): causal (cumsum=1..52).
            - response (820..871): causal (cumsum=53..104).
        Bidirectional tokens cannot attend to memory/response tokens.
        Memory/response tokens can see all preceding non-padded tokens.
        """
        assert vlm_attention_mask.shape == (1, TRAIN_PREFIX_LEN, TRAIN_PREFIX_LEN)
        assert vlm_attention_mask.dtype == torch.bool

        for i in range(1):
            mask = vlm_attention_mask[i].cpu()
            pads = prefix_pad_masks[i].cpu()

            n_lang = pads[LANG_START:MEMORY_START].sum().item()
            n_mem = pads[MEMORY_START:RESPONSE_START].sum().item()
            n_resp = pads[RESPONSE_START:TRAIN_PREFIX_LEN].sum().item()

            expected = torch.ones(TRAIN_PREFIX_LEN, TRAIN_PREFIX_LEN, dtype=torch.bool)

            # Zero out rows/cols for padded language tokens.
            lang_pad_start = LANG_START + n_lang
            expected[lang_pad_start:MEMORY_START, :] = 0
            expected[:, lang_pad_start:MEMORY_START] = 0

            # Zero out rows/cols for padded memory tokens.
            mem_pad_start = MEMORY_START + n_mem
            expected[mem_pad_start:RESPONSE_START, :] = 0
            expected[:, mem_pad_start:RESPONSE_START] = 0

            # Zero out rows/cols for padded response tokens.
            resp_pad_start = RESPONSE_START + n_resp
            expected[resp_pad_start:TRAIN_PREFIX_LEN, :] = 0
            expected[:, resp_pad_start:TRAIN_PREFIX_LEN] = 0

            # Bidirectional block (images + language) cannot attend to memory or response.
            expected[:MEMORY_START, MEMORY_START:] = 0

            # Memory tokens: causal among themselves, can attend to images+language.
            mem_causal = torch.tril(torch.ones(n_mem, n_mem, dtype=torch.bool))
            expected[MEMORY_START : MEMORY_START + n_mem, MEMORY_START : MEMORY_START + n_mem] = mem_causal
            # Memory cannot attend to response tokens.
            expected[MEMORY_START:RESPONSE_START, RESPONSE_START:] = 0

            # Response tokens: causal among themselves, can attend to images+language+memory.
            resp_causal = torch.tril(torch.ones(n_resp, n_resp, dtype=torch.bool))
            expected[RESPONSE_START : RESPONSE_START + n_resp, RESPONSE_START : RESPONSE_START + n_resp] = (
                resp_causal
            )

            assert torch.all(mask == expected), (
                f"VLM attention mask mismatch.\n"
                f"Diff indices: {(mask != expected).nonzero(as_tuple=False)[:20]}"
            )

    # ------------------------------------------------------------------
    # Main integration test
    # ------------------------------------------------------------------

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_complete_pi07_high_level_planner_pipeline(self, lerobot_dataset_metadata):
        """Test the PI07 high-level planner: forward (training) and sample_actions (inference)."""

        config = self._make_config()
        policy = PI07HighLevelPlannerPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)

        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, 3, 224, 224),
            "camera1": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, MAX_STATE_DIM),
            "prompt": ["Pick up the red block"],
            "past_memory": ["Robot is near the table"],
            "memory": ["Robot is grasping the red block"],
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
            return result

        policy.model.paligemma_with_expert.forward = capture_forward
        policy.model.embed_prefix = capture_embed_prefix

        # ── Training forward pass ────────────────────────────────────
        loss = policy.forward(batch_cuda)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix

        # Verify captures.
        for var in ["prefix_pad_masks", "vlm_2d_attention_mask", "vlm_position_ids"]:
            assert var in captured, f"{var} was not captured"

        assert captured["vlm_2d_attention_mask"].dtype == torch.bool
        assert captured["prefix_pad_masks"].dtype == torch.bool

        self._verify_pad_masks(captured["prefix_pad_masks"])
        self._verify_position_ids(captured["vlm_position_ids"], captured["prefix_pad_masks"])
        self._verify_vlm_attention_mask(captured["vlm_2d_attention_mask"], captured["prefix_pad_masks"])

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

            # Each new autoregressive token gets att_mask=1 (causal).
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
        }
        memory_tokens, response_tokens = policy.sample_actions(infer_batch)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.infer_autoregressive = original_infer_autoregressive

        # Verify total number of autoregressive steps.
        assert step_counter[0] == MEMORY_MAX_LENGTH + RESPONSE_MAX_LENGTH

        # Verify inference captures.
        assert "vlm_2d_attention_mask" in captured_infer
        assert "last_prefix_pad_masks" in captured_infer
        assert "last_prefix_offsets" in captured_infer

        # Initial VLM attention mask should be for inference prefix (images + language only).
        assert captured_infer["vlm_2d_attention_mask"].shape == (1, INFER_PREFIX_LEN, INFER_PREFIX_LEN)
        assert captured_infer["vlm_2d_attention_mask"].dtype == torch.bool

        # Initial prefix (images + language) should be fully bidirectional.
        init_pad = captured_infer["prefix_pad_masks"][0].cpu()
        init_mask = captured_infer["vlm_2d_attention_mask"][0].cpu()
        expected_init = init_pad[:, None] & init_pad[None, :]
        assert torch.all(init_mask == expected_init), "Inference prefix should be fully bidirectional"

        # After autoregressive generation, prefix should have grown by memory + response tokens.
        final_prefix_len = INFER_PREFIX_LEN + MEMORY_MAX_LENGTH + RESPONSE_MAX_LENGTH
        assert captured_infer["last_prefix_pad_masks"].shape == (1, final_prefix_len)

        # Output shapes.
        assert memory_tokens.shape == (1, MEMORY_MAX_LENGTH)
        assert response_tokens.shape == (1, RESPONSE_MAX_LENGTH)
