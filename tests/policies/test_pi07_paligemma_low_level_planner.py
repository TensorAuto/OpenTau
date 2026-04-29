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
from opentau.policies.pi07_paligemma.low_level_planner.configuration_pi07_low_level import (
    PI07lowlevelPlannerConfig,
)
from opentau.policies.pi07_paligemma.low_level_planner.modeling_pi07_low_level import (
    PI07LowLevelPlannerPolicy,
    make_att_2d_masks,
)

# Config defaults used across the test.
NUM_CAMERAS = 2
VJEPA2_TOKENS_PER_CAMERA = 256
NUM_SUBGOAL_CAMERAS = 1
SIGLIP_TOKENS_PER_SUBGOAL = 256
PROMPT_MAX_LENGTH = 256
RESPONSE_MAX_LENGTH = 52
METADATA_MAX_LENGTH = 52
DISCRETE_ACTION_MAX_LENGTH = 32
CHUNK_SIZE = 50
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32

# For training the state is provided as (B, n_obs_steps, D) so T = n_obs_steps.
N_OBS_STEPS = 8

VIDEO_TOKENS = NUM_CAMERAS * VJEPA2_TOKENS_PER_CAMERA  # 512
LANG_START = VIDEO_TOKENS  # 512
SUBGOAL_TOKENS = NUM_SUBGOAL_CAMERAS * SIGLIP_TOKENS_PER_SUBGOAL  # 256

# For inference: no discrete actions; state is 1 timestep for embed_prefix.
INFER_STATE_TOKENS = 1


class TestPI07LowLevelPlannerIntegration:
    """Integration tests for the PI07 low-level planner pipeline."""

    @staticmethod
    def _make_config() -> PI07lowlevelPlannerConfig:
        config = PI07lowlevelPlannerConfig(
            n_obs_steps=N_OBS_STEPS,
            chunk_size=CHUNK_SIZE,
            n_action_steps=CHUNK_SIZE,
            max_state_dim=MAX_STATE_DIM,
            max_action_dim=MAX_ACTION_DIM,
            prompt_max_length=PROMPT_MAX_LENGTH,
            response_max_length=RESPONSE_MAX_LENGTH,
            metadata_max_length=METADATA_MAX_LENGTH,
            discrete_action_max_length=DISCRETE_ACTION_MAX_LENGTH,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MIN_MAX,
                "ACTION": NormalizationMode.MEAN_STD,
            },
        )
        config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(MAX_STATE_DIM,)),
        }
        config.output_features = {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(CHUNK_SIZE, MAX_ACTION_DIM)),
        }
        return config

    @staticmethod
    def _indicator_lens(tokenizer):
        """Fixed strings inserted by ``embed_prefix`` (matches modeling layout)."""
        return {
            "state_lead": len(tokenizer.encode("State: ", add_special_tokens=False)),
            "comma": len(tokenizer.encode(", ", add_special_tokens=False)),
            "subgoal_lead": len(tokenizer.encode("Subgoal: ", add_special_tokens=False)),
            "prefix_end": len(tokenizer.encode(";\n ", add_special_tokens=False)),
            "action_lead": len(tokenizer.encode("Action: ", add_special_tokens=False)),
        }

    @classmethod
    def _train_prefix_total(cls, tokenizer) -> int:
        m = cls._indicator_lens(tokenizer)
        p = 0
        p += VIDEO_TOKENS
        p += PROMPT_MAX_LENGTH
        p += m["state_lead"] + N_OBS_STEPS + m["comma"]
        p += RESPONSE_MAX_LENGTH
        p += m["subgoal_lead"] + SUBGOAL_TOKENS + m["comma"]
        p += METADATA_MAX_LENGTH
        p += m["prefix_end"]
        p += m["action_lead"] + DISCRETE_ACTION_MAX_LENGTH
        return p

    @classmethod
    def _infer_prefix_total(cls, tokenizer) -> int:
        m = cls._indicator_lens(tokenizer)
        p = 0
        p += VIDEO_TOKENS
        p += PROMPT_MAX_LENGTH
        p += m["state_lead"] + INFER_STATE_TOKENS + m["comma"]
        p += RESPONSE_MAX_LENGTH
        p += m["subgoal_lead"] + SUBGOAL_TOKENS + m["comma"]
        p += METADATA_MAX_LENGTH
        p += m["prefix_end"]
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

        resp_lo = LANG_START + PROMPT_MAX_LENGTH + m["state_lead"] + state_t + m["comma"]
        resp_slice = slice(resp_lo, resp_lo + RESPONSE_MAX_LENGTH)

        sg_lo = resp_lo + RESPONSE_MAX_LENGTH + m["subgoal_lead"]
        sg_slice = slice(sg_lo, sg_lo + SUBGOAL_TOKENS)

        meta_lo = sg_lo + SUBGOAL_TOKENS + m["comma"]
        meta_slice = slice(meta_lo, meta_lo + METADATA_MAX_LENGTH)

        for i in range(prefix_pad_masks.shape[0]):
            assert torch.all(prefix_pad_masks[i, :VIDEO_TOKENS] == 1)
            self._check_ones_before_zeros(prefix_pad_masks[i, lang_slice])
            self._check_ones_before_zeros(prefix_pad_masks[i, resp_slice])
            assert torch.all(prefix_pad_masks[i, sg_slice] == 1)
            self._check_ones_before_zeros(prefix_pad_masks[i, meta_slice])

            if not inference_mode:
                da_lo = meta_lo + METADATA_MAX_LENGTH + m["prefix_end"] + m["action_lead"]
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
            prefix_offsets = torch.sum(prefix_pad_masks[:, :-DISCRETE_ACTION_MAX_LENGTH], dim=-1)[:, None]

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
        inference_mode=False,
    ):
        if inference_mode:
            num_cross = prefix_pad_masks.shape[1]
        else:
            num_cross = prefix_pad_masks.shape[1] - DISCRETE_ACTION_MAX_LENGTH

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

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_complete_pi07_low_level_pipeline(self, lerobot_dataset_metadata):
        """Test the PI07 low-level planner pipeline: forward (training) and select_action (inference)."""

        config = self._make_config()
        policy = PI07LowLevelPlannerPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)
        tokenizer = policy.model.language_tokenizer

        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, N_OBS_STEPS, 3, 224, 224),
            "camera1": torch.randn(batch_size, N_OBS_STEPS, 3, 224, 224),
            "subgoal0": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, N_OBS_STEPS, MAX_STATE_DIM),
            "actions": torch.randn(batch_size, CHUNK_SIZE, MAX_ACTION_DIM),
            "prompt": ["Pick up the red block"],
            "response": ["Grasp the red block"],
            "speed": torch.tensor([500]),
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
        original_paligemma_forward = policy.model.paligemma_with_expert.forward
        original_embed_prefix = policy.model.embed_prefix
        original_embed_suffix = policy.model.embed_suffix

        def capture_forward(*args, **kwargs):
            if kwargs["inputs_embeds"][0] is not None:
                captured["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["vlm_position_ids"] = kwargs["position_ids"].clone()
            else:
                captured["action_expert_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["action_expert_position_ids"] = kwargs["position_ids"].clone()
            return original_paligemma_forward(*args, **kwargs)

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

        policy.model.paligemma_with_expert.forward = capture_forward
        policy.model.embed_prefix = capture_embed_prefix
        policy.model.embed_suffix = capture_embed_suffix

        # ── Training forward pass ────────────────────────────────────
        loss = policy.forward(batch_cuda)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
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
            return original_paligemma_forward(*args, **kwargs)

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

        policy.model.paligemma_with_expert.forward = capture_forward_infer
        policy.model.embed_prefix = capture_embed_prefix_infer
        policy.model.embed_suffix = capture_embed_suffix_infer

        # Inference batch: state is 2-D (B, D), images are 4-D (B, C, H, W).
        infer_batch = {
            "camera0": batch_cuda["camera0"][:, 0],  # (B, C, H, W)
            "camera1": batch_cuda["camera1"][:, 0],
            "subgoal0": batch_cuda["subgoal0"],  # already (B, C, H, W)
            "state": batch_cuda["state"][:, 0],  # (B, D)
            "prompt": ["Pick up the red block"],
            "response": ["Grasp the red block"],
            "speed": torch.tensor([500], device="cuda"),
            "quality": torch.tensor([3], device="cuda"),
            "mistake": torch.tensor([0], device="cuda"),
            "speed_is_pad": torch.tensor([False], device="cuda"),
            "quality_is_pad": torch.tensor([False], device="cuda"),
            "mistake_is_pad": torch.tensor([False], device="cuda"),
            "subgoal_is_pad": torch.tensor([False], device="cuda"),
        }
        action = policy.select_action(infer_batch)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
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
            inference_mode=True,
        )

        assert action.shape == (1, MAX_ACTION_DIM)


class TestPI07LowLevelPlannerRegression:
    """GPU regression tests pinning the low-level planner signature/dtype fixes.

    Covers the changes made to ``embed_prefix``, ``embed_suffix``,
    ``prepare_metadata``, and the metadata-zip ``strict=True`` switch.
    """

    @staticmethod
    def _make_policy(lerobot_dataset_metadata) -> PI07LowLevelPlannerPolicy:
        config = TestPI07LowLevelPlannerIntegration._make_config()
        policy = PI07LowLevelPlannerPolicy(config, dataset_stats=lerobot_dataset_metadata.stats)
        policy.to(dtype=torch.bfloat16, device="cuda")
        return policy

    @staticmethod
    def _make_metadata_batch(batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "state": torch.randn(batch_size, N_OBS_STEPS, MAX_STATE_DIM, device="cuda", dtype=torch.bfloat16),
            "speed": torch.tensor([500] * batch_size, device="cuda"),
            "quality": torch.tensor([3] * batch_size, device="cuda"),
            "mistake": torch.tensor([0] * batch_size, device="cuda"),
            "speed_is_pad": torch.tensor([False] * batch_size, device="cuda"),
            "quality_is_pad": torch.tensor([False] * batch_size, device="cuda"),
            "mistake_is_pad": torch.tensor([False] * batch_size, device="cuda"),
        }

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

        from opentau.policies.pi07_paligemma.low_level_planner.modeling_pi07_low_level import (
            PI07LowLevelPlannerFlowMatching,
        )

        params = inspect.signature(PI07LowLevelPlannerFlowMatching.embed_prefix).parameters
        for name in ("response_tokens", "response_masks", "metadata_tokens", "metadata_masks"):
            assert params[name].default is inspect.Parameter.empty, (
                f"{name} should be a required parameter (no default), got default={params[name].default}"
            )
