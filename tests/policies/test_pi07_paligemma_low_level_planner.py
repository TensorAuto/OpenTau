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

# Token offsets for training prefix:
#   video(512) | lang(256) | response(52) | state(8) | subgoal(256) | metadata(52) | discrete_actions(32)
VIDEO_TOKENS = NUM_CAMERAS * VJEPA2_TOKENS_PER_CAMERA  # 512
LANG_START = VIDEO_TOKENS  # 512
RESPONSE_START = LANG_START + PROMPT_MAX_LENGTH  # 768
STATE_START = RESPONSE_START + RESPONSE_MAX_LENGTH  # 820
SUBGOAL_START = STATE_START + N_OBS_STEPS  # 828
SUBGOAL_TOKENS = NUM_SUBGOAL_CAMERAS * SIGLIP_TOKENS_PER_SUBGOAL  # 256
METADATA_START = SUBGOAL_START + SUBGOAL_TOKENS  # 1084
DISCRETE_ACTION_START = METADATA_START + METADATA_MAX_LENGTH  # 1136
TRAIN_PREFIX_LEN = DISCRETE_ACTION_START + DISCRETE_ACTION_MAX_LENGTH  # 1168

# For inference: no discrete actions, state is 1 token (2D -> unsqueezed).
INFER_STATE_TOKENS = 1
INFER_SUBGOAL_START = VIDEO_TOKENS + PROMPT_MAX_LENGTH + RESPONSE_MAX_LENGTH + INFER_STATE_TOKENS  # 821
INFER_METADATA_START = INFER_SUBGOAL_START + SUBGOAL_TOKENS  # 1077
INFER_PREFIX_LEN = INFER_METADATA_START + METADATA_MAX_LENGTH  # 1129

# Cross-attention tokens = prefix minus discrete actions.
CROSS_ATT_TOKENS_TRAIN = TRAIN_PREFIX_LEN - DISCRETE_ACTION_MAX_LENGTH  # 1136
CROSS_ATT_TOKENS_INFER = INFER_PREFIX_LEN  # 1129


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

    def _verify_pad_masks(self, prefix_pad_masks, suffix_pad_masks, inference_mode=False):
        assert prefix_pad_masks.shape[0] == 1
        expected_prefix_len = INFER_PREFIX_LEN if inference_mode else TRAIN_PREFIX_LEN
        assert prefix_pad_masks.shape[1] == expected_prefix_len
        assert prefix_pad_masks.dtype == torch.bool
        assert suffix_pad_masks.shape == (1, CHUNK_SIZE)
        assert suffix_pad_masks.dtype == torch.bool

        for i in range(prefix_pad_masks.shape[0]):
            # Video tokens should never be padded.
            assert torch.all(prefix_pad_masks[i, :VIDEO_TOKENS] == 1)
            # Language tokens can be padded at the end.
            self._check_ones_before_zeros(prefix_pad_masks[i, LANG_START:RESPONSE_START])
            # Response tokens can be padded at the end.
            self._check_ones_before_zeros(
                prefix_pad_masks[i, RESPONSE_START : RESPONSE_START + RESPONSE_MAX_LENGTH]
            )
            if not inference_mode:
                # Subgoal image tokens should never be padded.
                assert torch.all(prefix_pad_masks[i, SUBGOAL_START : SUBGOAL_START + SUBGOAL_TOKENS] == 1)
                # Metadata tokens can be padded at the end.
                self._check_ones_before_zeros(prefix_pad_masks[i, METADATA_START:DISCRETE_ACTION_START])
                # Discrete action tokens can be padded at the end.
                self._check_ones_before_zeros(prefix_pad_masks[i, DISCRETE_ACTION_START:TRAIN_PREFIX_LEN])
            else:
                # Subgoal image tokens should never be padded (inference).
                assert torch.all(
                    prefix_pad_masks[i, INFER_SUBGOAL_START : INFER_SUBGOAL_START + SUBGOAL_TOKENS] == 1
                )
                # Metadata tokens can be padded at the end (inference).
                self._check_ones_before_zeros(prefix_pad_masks[i, INFER_METADATA_START:INFER_PREFIX_LEN])
            # Suffix (action chunk) can be padded.
            self._check_ones_before_zeros(suffix_pad_masks[i])

    def _verify_position_ids(
        self,
        prefix_position_ids,
        suffix_position_ids,
        prefix_pad_masks,
        suffix_pad_masks,
        inference_mode=False,
    ):
        expected_prefix_len = INFER_PREFIX_LEN if inference_mode else TRAIN_PREFIX_LEN
        assert prefix_position_ids.shape == (1, expected_prefix_len)
        assert prefix_position_ids.dtype == torch.long
        assert suffix_position_ids.shape == (1, CHUNK_SIZE)
        assert suffix_position_ids.dtype == torch.long

        def _check_ids(position_ids, pad_masks):
            for j in range(1, len(position_ids)):
                if pad_masks[j] == 1:
                    assert position_ids[j] == position_ids[j - 1] + 1, (
                        f"Position ID should increment at {j}: {position_ids[j - 1]} -> {position_ids[j]}"
                    )
                else:
                    assert position_ids[j] == position_ids[j - 1], (
                        f"Position ID should stay same at padded {j}: {position_ids[j - 1]} -> {position_ids[j]}"
                    )

        for i in range(1):
            _check_ids(prefix_position_ids[i], prefix_pad_masks[i])
            _check_ids(suffix_position_ids[i], suffix_pad_masks[i])

            if not inference_mode:
                # Suffix starts after the cross-attention portion of the prefix.
                cross_att_end = DISCRETE_ACTION_START  # = prefix minus discrete actions
                assert suffix_position_ids[i, 0] == prefix_position_ids[i, cross_att_end]

    def _verify_vlm_attention_mask(self, vlm_attention_mask, prefix_pad_masks, inference_mode=False):
        expected_len = INFER_PREFIX_LEN if inference_mode else TRAIN_PREFIX_LEN
        assert vlm_attention_mask.shape == (1, expected_len, expected_len)
        assert vlm_attention_mask.dtype == torch.bool

        for i in range(1):
            mask = vlm_attention_mask[i].cpu()
            pads = prefix_pad_masks[i].cpu()

            if not inference_mode:
                # Build the expected mask.
                # Blocks (cumsum progression):
                #   video + lang + response + state (bidir, cumsum=0)
                #   | subgoal images (bidir block, [1,0,...,0], cumsum=1)
                #   | metadata (bidir block, [1,0,...,0], cumsum=2)
                #   | discrete_actions (causal, [1] each, cumsum=3,4,...)
                expected = torch.ones(expected_len, expected_len, dtype=torch.bool)

                # Determine non-padded counts in each paddable section.
                n_lang = pads[LANG_START:RESPONSE_START].sum().item()
                n_resp = pads[RESPONSE_START:STATE_START].sum().item()
                n_meta = pads[METADATA_START:DISCRETE_ACTION_START].sum().item()
                n_disc = pads[DISCRETE_ACTION_START:TRAIN_PREFIX_LEN].sum().item()

                # Zero out rows/cols for padded language tokens.
                lang_pad_start = LANG_START + n_lang
                expected[lang_pad_start:RESPONSE_START, :] = 0
                expected[:, lang_pad_start:RESPONSE_START] = 0

                # Zero out rows/cols for padded response tokens.
                resp_pad_start = RESPONSE_START + n_resp
                expected[resp_pad_start:STATE_START, :] = 0
                expected[:, resp_pad_start:STATE_START] = 0

                # Zero out rows/cols for padded metadata tokens.
                meta_pad_start = METADATA_START + n_meta
                expected[meta_pad_start:DISCRETE_ACTION_START, :] = 0
                expected[:, meta_pad_start:DISCRETE_ACTION_START] = 0

                # Zero out rows/cols for padded discrete action tokens.
                disc_pad_start = DISCRETE_ACTION_START + n_disc
                expected[disc_pad_start:TRAIN_PREFIX_LEN, :] = 0
                expected[:, disc_pad_start:TRAIN_PREFIX_LEN] = 0

                # Core bidir (video+lang+resp+state) cannot attend to subgoal, metadata, or discrete.
                expected[:SUBGOAL_START, SUBGOAL_START:] = 0

                # Subgoal tokens: bidir among themselves, can see core bidir.
                # Cannot attend to metadata or discrete actions.
                expected[SUBGOAL_START:METADATA_START, METADATA_START:] = 0

                # Metadata tokens: bidir among themselves, can see core bidir + subgoal.
                # Cannot attend to discrete action tokens.
                expected[METADATA_START:DISCRETE_ACTION_START, DISCRETE_ACTION_START:] = 0

                # Discrete action tokens: causal among themselves, can attend to
                # everything before (core bidir + subgoal + metadata).
                da_start = DISCRETE_ACTION_START
                causal = torch.tril(torch.ones(n_disc, n_disc, dtype=torch.bool))
                expected[da_start : da_start + n_disc, da_start : da_start + n_disc] = causal

                assert torch.all(mask == expected), (
                    f"VLM attention mask mismatch at batch {i}.\n"
                    f"Diff indices: {(mask != expected).nonzero(as_tuple=False)[:20]}"
                )
            else:
                # In inference, no discrete actions.
                # Blocks (cumsum progression):
                #   video + lang + response + state (bidir, cumsum=0)
                #   | subgoal images (bidir block, cumsum=1)
                #   | metadata (bidir block, cumsum=2)
                n_lang = pads[LANG_START : LANG_START + PROMPT_MAX_LENGTH].sum().item()
                n_resp = (
                    pads[
                        LANG_START + PROMPT_MAX_LENGTH : LANG_START + PROMPT_MAX_LENGTH + RESPONSE_MAX_LENGTH
                    ]
                    .sum()
                    .item()
                )
                n_meta = pads[INFER_METADATA_START:INFER_PREFIX_LEN].sum().item()

                expected = torch.ones(expected_len, expected_len, dtype=torch.bool)
                lang_pad_start = LANG_START + n_lang
                resp_start_infer = LANG_START + PROMPT_MAX_LENGTH
                expected[lang_pad_start:resp_start_infer, :] = 0
                expected[:, lang_pad_start:resp_start_infer] = 0

                resp_pad_start = resp_start_infer + n_resp
                state_start_infer = resp_start_infer + RESPONSE_MAX_LENGTH
                expected[resp_pad_start:state_start_infer, :] = 0
                expected[:, resp_pad_start:state_start_infer] = 0

                # Zero out rows/cols for padded metadata tokens.
                meta_pad_start_infer = INFER_METADATA_START + n_meta
                expected[meta_pad_start_infer:INFER_PREFIX_LEN, :] = 0
                expected[:, meta_pad_start_infer:INFER_PREFIX_LEN] = 0

                # Core bidir cannot attend to subgoal or metadata.
                expected[:INFER_SUBGOAL_START, INFER_SUBGOAL_START:] = 0

                # Subgoal tokens: bidir among themselves, can see core bidir.
                # Cannot attend to metadata.
                expected[INFER_SUBGOAL_START:INFER_METADATA_START, INFER_METADATA_START:] = 0

                assert torch.all(mask == expected), (
                    f"VLM attention mask mismatch (inference) at batch {i}.\n"
                    f"Diff indices: {(mask != expected).nonzero(as_tuple=False)[:20]}"
                )

    def _verify_action_expert_attention_mask(
        self, action_expert_attention_mask, prefix_pad_masks, suffix_pad_masks, inference_mode=False
    ):
        cross_att = CROSS_ATT_TOKENS_INFER if inference_mode else CROSS_ATT_TOKENS_TRAIN
        total_cols = cross_att + CHUNK_SIZE
        assert action_expert_attention_mask.shape == (1, CHUNK_SIZE, total_cols)
        assert action_expert_attention_mask.dtype == torch.bool

        for i in range(1):
            mask = action_expert_attention_mask[i].cpu()
            n_action = suffix_pad_masks[i].sum().item()

            expected = torch.ones(CHUNK_SIZE, total_cols, dtype=torch.bool)

            prefix_pads = prefix_pad_masks[i, :cross_att].cpu()
            # Zero out columns where prefix tokens are padded.
            for col in range(cross_att):
                if not prefix_pads[col]:
                    expected[:, col] = 0

            # Zero out rows for padded action tokens.
            expected[n_action:, :] = 0
            # Zero out columns for padded action tokens in the suffix portion.
            expected[:, cross_att + n_action :] = 0

            assert torch.all(mask == expected), (
                f"Action expert attention mask mismatch at batch {i}.\n"
                f"Diff indices: {(mask != expected).nonzero(as_tuple=False)[:20]}"
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
            return result

        def capture_embed_suffix(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            captured["suffix_pad_masks"] = result[1].clone()
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
            "suffix_pad_masks",
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

        self._verify_pad_masks(captured["prefix_pad_masks"], captured["suffix_pad_masks"])
        self._verify_position_ids(
            captured["vlm_position_ids"],
            captured["action_expert_position_ids"],
            captured["prefix_pad_masks"],
            captured["suffix_pad_masks"],
        )
        self._verify_vlm_attention_mask(captured["vlm_2d_attention_mask"], captured["prefix_pad_masks"])
        self._verify_action_expert_attention_mask(
            captured["action_expert_2d_attention_mask"],
            captured["prefix_pad_masks"],
            captured["suffix_pad_masks"],
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
            return result

        def capture_embed_suffix_infer(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            captured_infer["suffix_pad_masks"] = result[1].clone()
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
            "suffix_pad_masks",
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
            captured_infer["prefix_pad_masks"], captured_infer["suffix_pad_masks"], inference_mode=True
        )
        self._verify_position_ids(
            captured_infer["vlm_position_ids"],
            captured_infer["action_expert_position_ids"],
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            inference_mode=True,
        )
        self._verify_vlm_attention_mask(
            captured_infer["vlm_2d_attention_mask"],
            captured_infer["prefix_pad_masks"],
            inference_mode=True,
        )
        self._verify_action_expert_attention_mask(
            captured_infer["action_expert_2d_attention_mask"],
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
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
