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
from einops import rearrange

from opentau.policies.pi05.modeling_pi05 import (
    PI05Policy,
)


class TestPI05Integration:
    """Integration tests for the complete PI05 pipeline."""

    def _check_autoregressive_prefix_pads(self, prefix_pad_masks, response_tokens, prev_prefix_pad_masks):
        assert prefix_pad_masks.shape[0] == 1
        assert prefix_pad_masks.shape[1] == prev_prefix_pad_masks.shape[1] + 1
        assert prefix_pad_masks.dtype == torch.bool

        has_eos = (response_tokens == 1).any(dim=1, keepdim=True)
        has_pad = (response_tokens == 0).any(dim=1, keepdim=True)

        response_pad_masks = ~(has_eos | has_pad)
        assert torch.all(torch.cat((prev_prefix_pad_masks, response_pad_masks), dim=1) == prefix_pad_masks)

    def _check_autoregressive_prefix_position_ids(self, prefix_offsets, response_tokens, prev_prefix_offsets):
        assert prefix_offsets.shape[0] == 1
        assert prefix_offsets.shape[1] == 1
        assert prefix_offsets.dtype == torch.long

        response_token_position_id = (~(response_tokens[:, -1] == 0)).long()
        assert torch.all(prefix_offsets == prev_prefix_offsets + response_token_position_id)

    def _check_autoregressive_prefix_attention_mask(
        self, prefix_att_masks, response_tokens, prev_prefix_att_masks
    ):
        assert prefix_att_masks.shape[0] == 1
        assert prefix_att_masks.shape[1] == prev_prefix_att_masks.shape[1] + 1
        assert prefix_att_masks.dtype == torch.bfloat16

        response_token_attention_mask = rearrange((~(response_tokens[:, -1] == 0)).bfloat16(), "s -> 1 s")
        assert torch.all(
            torch.cat((prev_prefix_att_masks, response_token_attention_mask), dim=1) == prefix_att_masks
        )

    def _verify_pad_masks(self, prefix_pad_masks, suffix_pad_masks, inference_mode=False):
        """Verify the pad masks are correct. This assumes all images are not padded. Language embeddings and action chunks can be padded.

        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        suffix_pad_masks: tensor with shape (batch_size, seq_len)
        inference_mode: boolean indicating if the pad masks were created using the forward method (training) or select_action method (inference)
        """
        assert prefix_pad_masks.shape[0] == 1
        assert prefix_pad_masks.shape[1] == 852 if not inference_mode else 768
        assert prefix_pad_masks.dtype == torch.bool
        assert suffix_pad_masks.shape[0] == 1
        assert suffix_pad_masks.shape[1] == 50
        assert suffix_pad_masks.dtype == torch.bool

        def _check_ones_before_zeros(mask_slice):
            """Check that in a 1D mask, all ones come before all zeros."""
            mask = mask_slice.cpu().numpy()
            first_zero_idx = None
            for idx, val in enumerate(mask):
                if val == 0:
                    first_zero_idx = idx
                    break
            if first_zero_idx is not None:
                # All elements after first_zero_idx must be zero
                assert all(v == 0 for v in mask[first_zero_idx:]), f"Zeros not contiguous at end: {mask}"
                # All elements before first_zero_idx must be one
                assert all(v == 1 for v in mask[:first_zero_idx]), f"Ones not contiguous at start: {mask}"
            else:
                # All ones
                assert all(v == 1 for v in mask), f"Expected all ones in {mask}"

        batch_size = prefix_pad_masks.shape[0]
        for i in range(batch_size):
            assert torch.all(prefix_pad_masks[i, :512] == 1)  # image tokens should not be padded
            _check_ones_before_zeros(prefix_pad_masks[i, 512:768])  # prompt tokens
            if not inference_mode:
                _check_ones_before_zeros(prefix_pad_masks[i, 820:852])  # discrete action tokens
                _check_ones_before_zeros(prefix_pad_masks[i, 768:820])  # response tokens

            _check_ones_before_zeros(suffix_pad_masks[i, 0:50])  # action chunks

    def _verify_position_ids(
        self,
        prefix_position_ids,
        suffix_position_ids,
        prefix_pad_masks,
        suffix_pad_masks,
        last_prefix_offsets=None,
        inference_mode=False,
    ):
        """Verify the position ids are correct. They should increment by 1 for each non-padded token and stay the same for padded tokens.

        prefix_position_ids: tensor with shape (batch_size, seq_len)
        suffix_position_ids: tensor with shape (batch_size, seq_len)
        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        suffix_pad_masks: tensor with shape (batch_size, seq_len)
        inference_mode: boolean indicating if the position ids were created using the forward method (training) or select_action method (inference)
        """
        assert prefix_position_ids.shape[0] == 1
        assert prefix_position_ids.shape[1] == 852 if not inference_mode else 768
        assert prefix_position_ids.dtype == torch.long
        assert suffix_position_ids.shape[0] == 1
        assert suffix_position_ids.shape[1] == 50
        assert suffix_position_ids.dtype == torch.long

        def _check_position_ids_with_padding(position_ids, pad_masks):
            """Check that position IDs increment correctly for non-padded tokens and stay the same for padded tokens."""
            # Check that position IDs follow the rule: increment for non-padded tokens, stay same for padded tokens
            for i in range(1, len(position_ids)):
                if pad_masks[i] == 1:  # non-padded token
                    # Should increment from previous position
                    assert position_ids[i] == position_ids[i - 1] + 1, (
                        f"Position ID should increment at index {i}: {position_ids[i - 1]} -> {position_ids[i]}"
                    )
                else:  # padded token
                    # Should stay the same as previous position
                    assert position_ids[i] == position_ids[i - 1], (
                        f"Position ID should stay same at padded index {i}: {position_ids[i - 1]} -> {position_ids[i]}"
                    )

        batch_size = prefix_position_ids.shape[0]
        for i in range(batch_size):
            # Check entire prefix position IDs array
            _check_position_ids_with_padding(prefix_position_ids[i], prefix_pad_masks[i])

            # Check entire suffix position IDs array
            _check_position_ids_with_padding(suffix_position_ids[i], suffix_pad_masks[i])

            # check that the prefix offset is correct
            # the suffix position ids should start after the prefix position ids (minus the response tokens)
            if not inference_mode:
                assert suffix_position_ids[i, 0] == prefix_position_ids[i, 820]
            else:
                assert suffix_position_ids[i, 0] == last_prefix_offsets.item() + 1

    def _verify_vlm_attention_mask(self, vlm_attention_mask, prefix_pad_masks, inference_mode=False):
        """Verify the VLM attention mask is correct.

        vlm_attention_mask: tensor with shape (batch_size, seq_len, seq_len)
        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        inference_mode: boolean indicating if the attention mask were created using the forward method (training) or select_action method (inference)
        """
        assert vlm_attention_mask.shape[0] == 1
        assert vlm_attention_mask.shape[1] == 852 if not inference_mode else 768
        assert vlm_attention_mask.shape[2] == 852 if not inference_mode else 768
        assert vlm_attention_mask.dtype == torch.bool

        batch_size = vlm_attention_mask.shape[0]
        for i in range(batch_size):
            # construct correct attention mask
            # see diagram here: https://drive.google.com/file/d/1x3pM8SoIf9rqAG4-rZxmVvrqkorqhU-s/view?usp=sharing
            correct_vlm_attention_mask = torch.ones(852, 852, dtype=torch.bool)

            # pad tokens should not be attended to or attend to any other tokens
            num_non_padded_prompt_tokens = prefix_pad_masks[i, 512:768].sum()
            num_non_padded_response_tokens = prefix_pad_masks[i, 768:820].sum()
            num_non_padded_discrete_action_tokens = prefix_pad_masks[i, 820:852].sum()
            prompt_start_idx, response_start_idx, discrete_action_start_idx = 512, 768, 820

            # set the masks for pad tokens in prompt to 0
            correct_vlm_attention_mask[
                prompt_start_idx + num_non_padded_prompt_tokens : response_start_idx, :
            ] = 0
            correct_vlm_attention_mask[
                response_start_idx + num_non_padded_response_tokens : discrete_action_start_idx, :
            ] = 0
            correct_vlm_attention_mask[
                discrete_action_start_idx + num_non_padded_discrete_action_tokens : 852, :
            ] = 0

            correct_vlm_attention_mask[
                :, prompt_start_idx + num_non_padded_prompt_tokens : response_start_idx
            ] = 0
            correct_vlm_attention_mask[
                :, response_start_idx + num_non_padded_response_tokens : discrete_action_start_idx
            ] = 0
            correct_vlm_attention_mask[
                :, discrete_action_start_idx + num_non_padded_discrete_action_tokens : 852
            ] = 0

            # nothing should attend to discrete action tokens (other than discrete action tokens)
            correct_vlm_attention_mask[
                :,
                response_start_idx:,
            ] = 0

            correct_vlm_attention_mask[
                discrete_action_start_idx : discrete_action_start_idx + num_non_padded_discrete_action_tokens,
                response_start_idx : response_start_idx + num_non_padded_response_tokens,
            ] = 1

            # discrete action tokens should have a causal attention mask when attending to other discrete action tokens
            # Create causal mask: each token can attend to itself and all previous tokens
            discrete_action_causal_mask = torch.tril(
                torch.ones(
                    num_non_padded_discrete_action_tokens,
                    num_non_padded_discrete_action_tokens,
                    dtype=torch.bool,
                )
            )
            correct_vlm_attention_mask[
                discrete_action_start_idx : discrete_action_start_idx + num_non_padded_discrete_action_tokens,
                discrete_action_start_idx : discrete_action_start_idx + num_non_padded_discrete_action_tokens,
            ] = discrete_action_causal_mask

            response_causal_mask = torch.tril(
                torch.ones(
                    num_non_padded_response_tokens,
                    num_non_padded_response_tokens,
                    dtype=torch.bool,
                )
            )
            correct_vlm_attention_mask[
                response_start_idx : response_start_idx + num_non_padded_response_tokens,
                response_start_idx : response_start_idx + num_non_padded_response_tokens,
            ] = response_causal_mask

            # discrete action tokens are not used in inference
            if inference_mode:
                correct_vlm_attention_mask = correct_vlm_attention_mask[:768, :768]

            assert torch.all(vlm_attention_mask[i].cpu() == correct_vlm_attention_mask.cpu())

    def _verify_action_expert_attention_mask(
        self, action_expert_attention_mask, prefix_pad_masks, suffix_pad_masks
    ):
        """Verify the action expert attention mask is correct.

        action_expert_attention_mask: tensor with shape (batch_size, seq_len, seq_len)
        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        suffix_pad_masks: tensor with shape (batch_size, seq_len)
        """
        assert action_expert_attention_mask.shape[0] == 1
        assert action_expert_attention_mask.shape[1] == 50
        assert action_expert_attention_mask.shape[2] == 870
        assert action_expert_attention_mask.dtype == torch.bool

        batch_size = action_expert_attention_mask.shape[0]
        for i in range(batch_size):
            # construct correct attention mask
            # see diagram here: https://drive.google.com/file/d/19oKbjVdPBQzXF_Wt6PhfIRY3RaAxUGnd/view?usp=sharing
            correct_action_expert_attention_mask = torch.ones(50, 870, dtype=torch.bool)

            # pad tokens should not be attended to or attend to any other tokens
            num_non_padded_action_tokens = suffix_pad_masks[i, 0:50].sum()
            num_non_padded_prompt_tokens = prefix_pad_masks[i, 512:768].sum()
            num_non_padded_response_tokens = prefix_pad_masks[i, 768:820].sum()
            action_start_idx, prompt_start_idx, response_start_idx = 820, 512, 768
            # set attention mask for prompt pad tokens to 0
            correct_action_expert_attention_mask[
                :, prompt_start_idx + num_non_padded_prompt_tokens : response_start_idx
            ] = 0

            correct_action_expert_attention_mask[
                :, response_start_idx + num_non_padded_response_tokens : action_start_idx
            ] = 0

            # set attention mask for action pad tokens to 0
            correct_action_expert_attention_mask[num_non_padded_action_tokens:, :] = 0
            correct_action_expert_attention_mask[:, action_start_idx + num_non_padded_action_tokens :] = 0

            assert torch.all(
                action_expert_attention_mask[i].cpu() == correct_action_expert_attention_mask.cpu()
            )

    @pytest.mark.gpu
    @pytest.mark.slow  # ~1 mins
    def test_complete_pi05_pipeline_integration(self, pi05_training_config, lerobot_dataset_metadata):
        """Test the complete PI05 pipeline from data loading to model execution."""

        # Initialize policy with unified training mode
        config = pi05_training_config.policy
        policy = PI05Policy(config, dataset_stats=lerobot_dataset_metadata.stats)

        # Test data preparation pipeline
        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, 3, 224, 224),
            "camera1": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, config.max_state_dim),
            "actions": torch.randn(batch_size, config.chunk_size, config.max_action_dim),
            "prompt": ["Pick up the red block"],
            "response": ["Pick up the red block"],
            "img_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
            "action_is_pad": torch.cat(
                [
                    torch.zeros(batch_size, config.chunk_size // 2, dtype=torch.bool),
                    torch.ones(batch_size, config.chunk_size - config.chunk_size // 2, dtype=torch.bool),
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

        # Capture intermediate variables for inspection by monkey-patching the paligemma_with_expert forward method
        captured_variables = {}

        def capture_variables_forward(*args, **kwargs):
            # Extract the variables we want to capture from the kwargs
            if kwargs["inputs_embeds"][0] is not None:
                vlm_attention_mask = kwargs.get("attention_mask")
                vlm_position_ids = kwargs.get("position_ids")
                action_expert_attention_mask = None
                action_expert_position_ids = None
            else:
                vlm_attention_mask = None
                vlm_position_ids = None
                action_expert_attention_mask = kwargs.get("attention_mask")
                action_expert_position_ids = kwargs.get("position_ids")

            # Capture the attention masks and position IDs
            if vlm_attention_mask is not None:
                captured_variables["vlm_2d_attention_mask"] = vlm_attention_mask.clone()
            if action_expert_attention_mask is not None:
                captured_variables["action_expert_2d_attention_mask"] = action_expert_attention_mask.clone()
            if vlm_position_ids is not None:
                captured_variables["vlm_position_ids"] = vlm_position_ids.clone()
            if action_expert_position_ids is not None:
                captured_variables["action_expert_position_ids"] = action_expert_position_ids.clone()

            # Call the original forward method
            return original_paligemma_forward(*args, **kwargs)

        # Store original paligemma forward method and replace it
        original_paligemma_forward = policy.model.paligemma_with_expert.forward
        policy.model.paligemma_with_expert.forward = capture_variables_forward

        # Also capture prefix_pad_masks and suffix_pad_masks by monkey-patching the embed methods
        original_embed_prefix = policy.model.embed_prefix
        original_embed_suffix = policy.model.embed_suffix
        original_infer_response = policy.model.infer_response

        def capture_embed_prefix(*args, **kwargs):
            # workaround to have paddings in discrete actions, otherwise the tokenizer creates 1500 non-padded tokens, which can't be handled in given gpu memory
            args1 = list(args)
            args1[-2] = torch.concat(
                (args1[-2][:, :16], torch.zeros((1, 16), device=args1[-2].device)), dim=-1
            )
            args1[-1] = torch.concat(
                (args1[-1][:, :16], torch.zeros((1, 16), dtype=torch.bool, device=args1[-1].device)), dim=-1
            )
            args1 = tuple(args1)
            result = original_embed_prefix(*args1, **kwargs)
            prefix_embs, prefix_pad_masks, prefix_att_masks = result
            captured_variables["prefix_pad_masks"] = prefix_pad_masks.clone()
            return result

        def capture_embed_suffix(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = result
            captured_variables["suffix_pad_masks"] = suffix_pad_masks.clone()
            return result

        policy.model.embed_prefix = capture_embed_prefix
        policy.model.embed_suffix = capture_embed_suffix

        # Test forward pass
        loss = policy.forward(batch_cuda)

        # Restore original methods
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.embed_suffix = original_embed_suffix

        # check normalize and unnormalize by applying it on actions
        normalize_actions = policy.normalize_targets
        unnormalize_actions = policy.unnormalize_outputs
        action_output = {"actions": batch["actions"].to("cuda")}
        assert torch.allclose(
            action_output["actions"],
            unnormalize_actions(normalize_actions(action_output))["actions"],
            atol=1e-6,
        )

        # Inspect all captured variables
        expected_vars = [
            "prefix_pad_masks",
            "suffix_pad_masks",
            "vlm_position_ids",
            "action_expert_position_ids",
            "vlm_2d_attention_mask",
            "action_expert_2d_attention_mask",
        ]

        for var_name in expected_vars:
            assert var_name in captured_variables, f"{var_name} was not captured"

        # Basic assertions about the variables
        assert captured_variables["vlm_2d_attention_mask"].dtype == torch.bool, (
            "VLM attention mask should be boolean"
        )
        assert captured_variables["action_expert_2d_attention_mask"].dtype == torch.bool, (
            "Action expert attention mask should be boolean"
        )
        assert captured_variables["prefix_pad_masks"].dtype == torch.bool, (
            "Prefix pad masks should be boolean"
        )
        assert captured_variables["suffix_pad_masks"].dtype == torch.bool, (
            "Suffix pad masks should be boolean"
        )

        # self._verify_pad_masks(captured_variables["prefix_pad_masks"], captured_variables["suffix_pad_masks"])
        # self._verify_position_ids(
        #     captured_variables["vlm_position_ids"],
        #     captured_variables["action_expert_position_ids"],
        #     captured_variables["prefix_pad_masks"],
        #     captured_variables["suffix_pad_masks"],
        # )
        # self._verify_vlm_attention_mask(
        #     captured_variables["vlm_2d_attention_mask"], captured_variables["prefix_pad_masks"]
        # )
        # self._verify_action_expert_attention_mask(
        #     captured_variables["action_expert_2d_attention_mask"],
        #     captured_variables["prefix_pad_masks"],
        #     captured_variables["suffix_pad_masks"],
        # )

        assert isinstance(loss, dict)
        assert "MSE" in loss
        assert "CE" in loss
        assert all(v.isfinite() for v in loss.values())

        # Test reset functionality
        policy.reset()
        assert len(policy._action_queue) == 0

        # Test optimization parameters
        optim_params = policy.get_optim_params()
        assert len(list(optim_params)) > 0

        # --------------------------------- Run the same test but for select_action --------------------------------------
        captured_variables_select_action = {}

        def capture_infer_response(*args, **kwargs):
            prev_prefix_pad_masks = args[2]
            prev_prefix_offsets = args[5]
            prev_prefix_att_masks = args[3]
            result = original_infer_response(*args, **kwargs)
            _, _, prefix_pad_masks, prefix_att_masks, prefix_offsets, response_tokens, _ = result
            self._check_autoregressive_prefix_pads(prefix_pad_masks, response_tokens, prev_prefix_pad_masks)
            self._check_autoregressive_prefix_position_ids(
                prefix_offsets, response_tokens, prev_prefix_offsets
            )
            self._check_autoregressive_prefix_attention_mask(
                prefix_att_masks, response_tokens, prev_prefix_att_masks
            )
            captured_variables_select_action["last_prefix_pad_masks"] = prefix_pad_masks.clone()
            captured_variables_select_action["last_prefix_att_masks"] = prefix_att_masks.clone()
            captured_variables_select_action["last_prefix_offsets"] = prefix_offsets.clone()
            return result

        def capture_variables_forward_select_action(*args, **kwargs):
            # Extract the variables we want to capture from the kwargs
            if kwargs["inputs_embeds"][0] is not None and kwargs.get("past_key_values") is None:
                vlm_attention_mask = kwargs.get("attention_mask")
                vlm_position_ids = kwargs.get("position_ids")
                action_expert_attention_mask = None
                action_expert_position_ids = None
            else:
                vlm_attention_mask = None
                vlm_position_ids = None
                action_expert_attention_mask = kwargs.get("attention_mask")
                action_expert_position_ids = kwargs.get("position_ids")

            # Capture the attention masks and position IDs
            if vlm_attention_mask is not None:
                captured_variables_select_action["vlm_2d_attention_mask"] = vlm_attention_mask.clone()
            if action_expert_attention_mask is not None:
                captured_variables_select_action["action_expert_2d_attention_mask"] = (
                    action_expert_attention_mask.clone()
                )
            if vlm_position_ids is not None:
                captured_variables_select_action["vlm_position_ids"] = vlm_position_ids.clone()
            if action_expert_position_ids is not None:
                captured_variables_select_action["action_expert_position_ids"] = (
                    action_expert_position_ids.clone()
                )

            # Call the original forward method
            return original_paligemma_forward(*args, **kwargs)

        # Store original paligemma forward method and replace it for select_action
        original_paligemma_forward_select_action = policy.model.paligemma_with_expert.forward
        policy.model.paligemma_with_expert.forward = capture_variables_forward_select_action
        policy.model.infer_response = capture_infer_response

        # Also capture prefix_pad_masks and suffix_pad_masks by monkey-patching the embed methods for select_action
        original_embed_prefix_select_action = policy.model.embed_prefix
        original_embed_suffix_select_action = policy.model.embed_suffix

        def capture_embed_prefix_select_action(*args, **kwargs):
            result = original_embed_prefix_select_action(*args, **kwargs)
            prefix_embs, prefix_pad_masks, prefix_att_masks = result
            captured_variables_select_action["prefix_pad_masks"] = prefix_pad_masks.clone()
            return result

        def capture_embed_suffix_select_action(*args, **kwargs):
            result = original_embed_suffix_select_action(*args, **kwargs)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = result
            captured_variables_select_action["suffix_pad_masks"] = suffix_pad_masks.clone()
            return result

        policy.model.embed_prefix = capture_embed_prefix_select_action
        policy.model.embed_suffix = capture_embed_suffix_select_action

        # Call select_action with variable capturing
        action = policy.select_action(batch_cuda)

        # Restore original methods
        policy.model.paligemma_with_expert.forward = original_paligemma_forward_select_action
        policy.model.embed_prefix = original_embed_prefix_select_action
        policy.model.embed_suffix = original_embed_suffix_select_action

        # Verify that the same variables were captured for select_action
        expected_vars_select_action = [
            "prefix_pad_masks",
            "suffix_pad_masks",
            "vlm_position_ids",
            "action_expert_position_ids",
            "vlm_2d_attention_mask",
            "action_expert_2d_attention_mask",
            "last_prefix_pad_masks",
            "last_prefix_att_masks",
            "last_prefix_offsets",
        ]

        for var_name in expected_vars_select_action:
            assert var_name in captured_variables_select_action, (
                f"{var_name} was not captured for select_action"
            )

        # Basic assertions about the variables captured for select_action
        assert captured_variables_select_action["vlm_2d_attention_mask"].dtype == torch.bool, (
            "VLM attention mask should be boolean for select_action"
        )
        assert captured_variables_select_action["action_expert_2d_attention_mask"].dtype == torch.bool, (
            "Action expert attention mask should be boolean for select_action"
        )
        assert captured_variables_select_action["prefix_pad_masks"].dtype == torch.bool, (
            "Prefix pad masks should be boolean for select_action"
        )
        assert captured_variables_select_action["suffix_pad_masks"].dtype == torch.bool, (
            "Suffix pad masks should be boolean for select_action"
        )

        # Verify the captured variables for select_action using the same verification methods
        self._verify_pad_masks(
            captured_variables_select_action["prefix_pad_masks"],
            captured_variables_select_action["suffix_pad_masks"],
            inference_mode=True,
        )
        self._verify_position_ids(
            captured_variables_select_action["vlm_position_ids"],
            captured_variables_select_action["action_expert_position_ids"],
            captured_variables_select_action["prefix_pad_masks"],
            captured_variables_select_action["suffix_pad_masks"],
            captured_variables_select_action["last_prefix_offsets"],
            inference_mode=True,
        )
        self._verify_vlm_attention_mask(
            captured_variables_select_action["vlm_2d_attention_mask"],
            captured_variables_select_action["prefix_pad_masks"],
            inference_mode=True,
        )
        self._verify_action_expert_attention_mask(
            captured_variables_select_action["action_expert_2d_attention_mask"],
            captured_variables_select_action["last_prefix_pad_masks"],
            captured_variables_select_action["suffix_pad_masks"],
        )

        assert action.shape == (1, policy.config.max_action_dim)
