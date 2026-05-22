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

from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import (
    PI05Policy,
)

PALIGEMMA_TOKENIZER_EOS_IDS = 1
PALIGEMMA_TOKENIZER_PAD_IDS = 0


class TestPI05Integration:
    """Integration tests for the complete PI05 pipeline."""

    def _check_autoregressive_prefix_pads(
        self, prefix_pad_masks, response_tokens, prev_prefix_pad_masks, prefix_embs
    ):
        assert prefix_pad_masks.shape[0] == 1
        assert prefix_pad_masks.shape[1] == prev_prefix_pad_masks.shape[1] + 1
        assert prefix_pad_masks.dtype == torch.bool
        assert prefix_embs.shape[1] == prefix_pad_masks.shape[1]

        has_eos = (response_tokens == PALIGEMMA_TOKENIZER_EOS_IDS).any(dim=1, keepdim=True)
        has_pad = (response_tokens == PALIGEMMA_TOKENIZER_PAD_IDS).any(dim=1, keepdim=True)

        response_pad_masks = ~(has_eos | has_pad)
        assert torch.all(torch.cat((prev_prefix_pad_masks, response_pad_masks), dim=1) == prefix_pad_masks)

    def _check_autoregressive_prefix_position_ids(self, prefix_offsets, response_tokens, prev_prefix_offsets):
        assert prefix_offsets.shape[0] == 1
        assert prefix_offsets.shape[1] == 1
        assert prefix_offsets.dtype == torch.long

        response_token_position_id_increment = (~(response_tokens[:, -1] == 0)).long()
        assert torch.all(prefix_offsets == prev_prefix_offsets + response_token_position_id_increment)

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

        Token layout (training, 858 total):
            Images(512) | Prompt(256) | ResponseInd(3) | Response(52) | ActionInd(3) | DiscreteAction(32)
        Token layout (inference, 771 total):
            Images(512) | Prompt(256) | ResponseInd(3)
        """
        total_train, total_infer = 858, 771
        expected_len = total_infer if inference_mode else total_train
        assert prefix_pad_masks.shape == (1, expected_len)
        assert prefix_pad_masks.dtype == torch.bool
        assert suffix_pad_masks.shape == (1, 50)
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
                assert all(v == 0 for v in mask[first_zero_idx:]), f"Zeros not contiguous at end: {mask}"
                assert all(v == 1 for v in mask[:first_zero_idx]), f"Ones not contiguous at start: {mask}"
            else:
                assert all(v == 1 for v in mask), f"Expected all ones in {mask}"

        batch_size = prefix_pad_masks.shape[0]
        for i in range(batch_size):
            assert torch.all(prefix_pad_masks[i, :512] == 1)  # image tokens should not be padded
            _check_ones_before_zeros(prefix_pad_masks[i, 512:768])  # prompt tokens
            assert torch.all(prefix_pad_masks[i, 768:771] == 1)  # response indicator (always present)
            if not inference_mode:
                _check_ones_before_zeros(prefix_pad_masks[i, 771:823])  # response tokens
                assert torch.all(prefix_pad_masks[i, 823:826] == 1)  # action indicator (always present)
                _check_ones_before_zeros(prefix_pad_masks[i, 826:858])  # discrete action tokens

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
        total_train, total_infer = 858, 771
        expected_len = total_infer if inference_mode else total_train
        assert prefix_position_ids.shape == (1, expected_len)
        assert prefix_position_ids.dtype == torch.long
        assert suffix_position_ids.shape == (1, 50)
        assert suffix_position_ids.dtype == torch.long

        def _check_position_ids_with_padding(position_ids, pad_masks):
            """Check that position IDs increment correctly for non-padded tokens and stay the same for padded tokens."""
            for i in range(1, len(position_ids)):
                if pad_masks[i] == 1:  # non-padded token
                    assert position_ids[i] == position_ids[i - 1] + 1, (
                        f"Position ID should increment at index {i}: {position_ids[i - 1]} -> {position_ids[i]}"
                    )
                else:  # padded token
                    assert position_ids[i] == position_ids[i - 1], (
                        f"Position ID should stay same at padded index {i}: {position_ids[i - 1]} -> {position_ids[i]}"
                    )

        batch_size = prefix_position_ids.shape[0]
        for i in range(batch_size):
            _check_position_ids_with_padding(prefix_position_ids[i], prefix_pad_masks[i])
            _check_position_ids_with_padding(suffix_position_ids[i], suffix_pad_masks[i])

            if not inference_mode:
                # Action expert positions start after cross-att tokens (excluding action indicator + discrete actions)
                assert suffix_position_ids[i, 0] == prefix_position_ids[i, 823]
            else:
                assert suffix_position_ids[i, 0] == last_prefix_offsets.item() + 1

    def _verify_vlm_attention_mask(self, vlm_attention_mask, prefix_pad_masks, inference_mode=False):
        """Verify the VLM attention mask follows a prefix-LM pattern.

        Token layout (training, 858):
            [Images(512)][Prompt(256)] bidirectional | [RespInd(3)][Response(52)][ActInd(3)][Discrete(32)] causal
        Token layout (inference, 771):
            [Images(512)][Prompt(256)] bidirectional | [RespInd(3)] causal

        Bidirectional tokens have full mutual attention. Causal tokens attend to
        all bidirectional tokens plus causal tokens up to and including themselves.
        Padded tokens neither attend nor are attended to.
        """
        total_train, total_infer = 858, 771
        prompt_start = 512
        causal_start = 768
        response_start = 771
        action_ind_start = 823
        discrete_action_start = 826

        total = total_infer if inference_mode else total_train
        assert vlm_attention_mask.shape == (1, total, total)
        assert vlm_attention_mask.dtype == torch.bool

        for i in range(vlm_attention_mask.shape[0]):
            correct = torch.zeros(total, total, dtype=torch.bool)

            # Bidirectional block: full mutual attention (images + prompt)
            correct[:causal_start, :causal_start] = 1

            # Causal tokens attend to all bidirectional tokens
            correct[causal_start:, :causal_start] = 1

            # Causal self-attention (lower triangular)
            causal_len = total - causal_start
            correct[causal_start:, causal_start:] = torch.tril(
                torch.ones(causal_len, causal_len, dtype=torch.bool)
            )

            # Zero out padding: prompt section
            num_np_prompt = prefix_pad_masks[i, prompt_start:causal_start].sum().item()
            prompt_pad_start = prompt_start + num_np_prompt
            correct[prompt_pad_start:causal_start, :] = 0
            correct[:, prompt_pad_start:causal_start] = 0

            if not inference_mode:
                # Response indicator (768-770): never padded — no masking needed
                # Response section padding
                num_np_resp = prefix_pad_masks[i, response_start:action_ind_start].sum().item()
                resp_pad_start = response_start + num_np_resp
                correct[resp_pad_start:action_ind_start, :] = 0
                correct[:, resp_pad_start:action_ind_start] = 0

                # Action indicator (823-825): never padded — no masking needed
                # Discrete action section padding
                num_np_da = prefix_pad_masks[i, discrete_action_start:total_train].sum().item()
                da_pad_start = discrete_action_start + num_np_da
                correct[da_pad_start:total_train, :] = 0
                correct[:, da_pad_start:total_train] = 0

            assert torch.all(vlm_attention_mask[i].cpu() == correct.cpu())

    def _verify_action_expert_attention_mask(
        self, action_expert_attention_mask, prefix_pad_masks, suffix_pad_masks
    ):
        """Verify the action expert attention mask is correct.

        The action expert cross-attends to 823 prefix tokens (everything except action
        indicator + discrete actions) and has 50 action tokens, giving shape (50, 873).

        Cross-att layout (823 tokens):
            Images(512) | Prompt(256) | ResponseInd(3) | Response(52)
        """
        n_cross = 823
        n_suffix = 50
        total_attn = n_cross + n_suffix  # 873
        prompt_start = 512
        causal_start = 768
        response_start = 771

        assert action_expert_attention_mask.shape == (1, n_suffix, total_attn)
        assert action_expert_attention_mask.dtype == torch.bool

        for i in range(action_expert_attention_mask.shape[0]):
            correct = torch.ones(n_suffix, total_attn, dtype=torch.bool)

            num_np_action = suffix_pad_masks[i, :n_suffix].sum().item()
            num_np_prompt = prefix_pad_masks[i, prompt_start:causal_start].sum().item()
            num_np_resp = prefix_pad_masks[i, response_start:n_cross].sum().item()

            # Zero out padded prompt columns (response indicator 768-770 is never padded)
            correct[:, prompt_start + num_np_prompt : causal_start] = 0

            # Zero out padded response columns
            correct[:, response_start + num_np_resp : n_cross] = 0

            # Zero out padded action rows and columns
            correct[num_np_action:, :] = 0
            correct[:, n_cross + num_np_action :] = 0

            assert torch.all(action_expert_attention_mask[i].cpu() == correct.cpu())

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

        self._verify_pad_masks(captured_variables["prefix_pad_masks"], captured_variables["suffix_pad_masks"])
        self._verify_position_ids(
            captured_variables["vlm_position_ids"],
            captured_variables["action_expert_position_ids"],
            captured_variables["prefix_pad_masks"],
            captured_variables["suffix_pad_masks"],
        )
        self._verify_vlm_attention_mask(
            captured_variables["vlm_2d_attention_mask"], captured_variables["prefix_pad_masks"]
        )
        self._verify_action_expert_attention_mask(
            captured_variables["action_expert_2d_attention_mask"],
            captured_variables["prefix_pad_masks"],
            captured_variables["suffix_pad_masks"],
        )

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
        original_infer_response = policy.model.infer_response

        def capture_infer_response(*args, **kwargs):
            prev_prefix_pad_masks = args[2]
            prev_prefix_offsets = args[5]
            prev_prefix_att_masks = args[3]
            result = original_infer_response(*args, **kwargs)
            _, prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_offsets, response_tokens, _ = result
            self._check_autoregressive_prefix_pads(
                prefix_pad_masks, response_tokens, prev_prefix_pad_masks, prefix_embs
            )
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


# Loc-token regression — confirm the `<locNNNN>` strings flow through pi05's
# tokenizer + embedding + response_ce_loss path without shape errors and
# produce a finite loss. Guarded by GPU because instantiating the full
# PaliGemma backbone is heavy.


@pytest.mark.gpu
@pytest.mark.slow
def test_pi05_loc_tokens_in_response_produce_finite_loss(pi05_training_config, lerobot_dataset_metadata):
    """π0.5: a response containing `<loc0042>` should encode each loc string
    as a single token (not BPE-fragment), get embedded by the existing
    PaliGemma `embed_language_tokens`, and produce a finite `response_ce_loss`
    on a one-batch forward pass. Regression for the `ensure_loc_tokens`
    promotion wired in `PI05Policy.__init__` / `PI05FlowMatching.__init__`."""

    config = pi05_training_config.policy
    policy = PI05Policy(config, dataset_stats=lerobot_dataset_metadata.stats)

    # PI05Policy and PI05FlowMatching share a single tokenizer instance so
    # token IDs cannot drift between the two layers.
    assert policy.language_tokenizer is policy.model.language_tokenizer
    # The promotion makes <locNNNN> a single-token match on the (shared)
    # tokenizer.
    tok = policy.language_tokenizer
    assert len(tok.encode("<loc0042>", add_special_tokens=False)) == 1
    assert len(tok.encode("<loc1023>", add_special_tokens=False)) == 1

    batch_size = 1
    batch = {
        "camera0": torch.randn(batch_size, 3, 224, 224),
        "camera1": torch.randn(batch_size, 3, 224, 224),
        "state": torch.randn(batch_size, config.max_state_dim),
        "actions": torch.randn(batch_size, config.chunk_size, config.max_action_dim),
        "prompt": ["detect red block"],
        "response": ["<loc0234><loc0567><loc0890><loc1023> red block"],
        "img_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }

    policy.to(dtype=torch.bfloat16, device="cuda")
    batch_cuda = {
        k: v.to("cuda", non_blocking=True, dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch_cuda["action_is_pad"] = batch_cuda["action_is_pad"].to(dtype=torch.bool)

    try:
        loss = policy.forward(batch_cuda)
        assert isinstance(loss, dict)
        assert "MSE" in loss and "CE" in loss
        assert all(v.isfinite() for v in loss.values()), f"Non-finite loss with loc tokens: {loss}"
    finally:
        # Free ~6 GB of PaliGemma weights so adjacent GPU tests in the same
        # process don't OOM on a single-GPU dev box.
        del policy
        torch.cuda.empty_cache()


class TestPI05ExecutionHorizon:
    """Regression coverage for ``n_action_steps`` as a short execution horizon.

    ``chunk_size`` is the trained prediction horizon (always decoded);
    ``n_action_steps`` (<= chunk_size) is how many actions are executed before
    re-querying. ``n_action_steps < chunk_size`` used to broadcast-crash the
    denoise/MSE paths and trip the queue assert; these CPU tests guard that path.
    """

    MAX_STATE_DIM = 32
    MAX_ACTION_DIM = 32

    @classmethod
    def _config(cls, chunk_size=10, n_action_steps=3, max_delay=0):
        return PI05Config(
            n_obs_steps=1,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_delay=max_delay,
            max_state_dim=cls.MAX_STATE_DIM,
            max_action_dim=cls.MAX_ACTION_DIM,
        )

    def test_guard_rejects_short_horizon_with_delay(self):
        # A shortened execution horizon is not compatible with the real-time
        # delay prefix path; the config must reject it loudly.
        with pytest.raises(ValueError, match="max_delay"):
            self._config(chunk_size=10, n_action_steps=3, max_delay=2)

    def test_guard_allows_short_horizon_without_delay(self):
        cfg = self._config(chunk_size=10, n_action_steps=3, max_delay=0)
        assert (cfg.chunk_size, cfg.n_action_steps, cfg.max_delay) == (10, 3, 0)

    def test_guard_allows_full_horizon_with_delay(self):
        # n_action_steps == chunk_size keeps the real-time-delay path available.
        cfg = self._config(chunk_size=10, n_action_steps=10, max_delay=2)
        assert cfg.max_delay == 2

    def test_select_action_executes_first_n_then_requeries(self):
        """Model decodes the full ``chunk_size`` chunk, but ``select_action``
        executes only the first ``n_action_steps`` before re-querying.
        """
        chunk_size, n_steps, bsz = 10, 3, 2
        cfg = self._config(chunk_size=chunk_size, n_action_steps=n_steps, max_delay=0)

        policy = object.__new__(PI05Policy)
        policy.config = cfg
        policy.eval = lambda: None  # bypass nn.Module.eval (no __init__ was run)
        PI05Policy.reset(policy)

        calls = {"n": 0}

        def fake_sample_actions(batch, noise=None, action_prefix=None, delay=None):
            # Return a full chunk_size chunk; element value encodes
            # (call_index * 1000 + timestep) so we can assert which timesteps run.
            calls["n"] += 1
            ts = torch.arange(chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
            return (calls["n"] * 1000 + ts).expand(bsz, chunk_size, self.MAX_ACTION_DIM).clone()

        policy.sample_actions = fake_sample_actions
        batch = {"state": torch.zeros(bsz, self.MAX_STATE_DIM)}

        # First n_steps actions all come from a single decode (call 1), in order.
        acts = [PI05Policy.select_action(policy, batch) for _ in range(n_steps)]
        assert calls["n"] == 1
        assert [tuple(a.shape) for a in acts] == [(bsz, self.MAX_ACTION_DIM)] * n_steps
        assert [a[0, 0].item() for a in acts] == [1000.0, 1001.0, 1002.0]

        # Queue drained after n_action_steps -> next call re-queries (call 2).
        a_next = PI05Policy.select_action(policy, batch)
        assert calls["n"] == 2
        assert a_next[0, 0].item() == 2000.0

    def test_embed_suffix_attn_mask_spans_full_chunk(self, monkeypatch):
        """Real-path guard the mocked select_action test above misses.

        ``embed_suffix`` must build the action attention mask over the full
        ``chunk_size`` (= the noise/x_t length), not ``n_action_steps``. Before the
        fix it used ``n_action_steps``, so for ``n_action_steps < chunk_size`` the
        returned ``att_masks`` (len ``n_action_steps``) and ``pad_masks`` (len
        ``chunk_size``) mismatched and ``make_att_2d_masks`` crashed -- in both the
        training forward and the inference denoise step, upstream of the Euler step.
        """
        import torch.nn as nn

        from opentau.policies.pi05.modeling_pi05 import PI05FlowMatching

        mod = "opentau.policies.pi05.modeling_pi05"
        monkeypatch.setattr(f"{mod}._preferred_dtype", lambda: torch.float32)
        monkeypatch.setattr(
            f"{mod}.create_sinusoidal_pos_embedding",
            lambda timestep, dim, **kw: torch.zeros(timestep.shape[0], dim),
        )

        cfg = self._config(chunk_size=10, n_action_steps=3, max_delay=0)
        fm = object.__new__(PI05FlowMatching)
        nn.Module.__init__(fm)  # set up _modules so we can attach the stub layers
        fm.config = cfg
        fm.action_in_proj = nn.Identity()
        fm.time_mlp_in = nn.Identity()
        fm.time_mlp_out = nn.Identity()

        out = PI05FlowMatching.embed_suffix(
            fm, torch.zeros(1, cfg.chunk_size, cfg.max_action_dim), torch.zeros(1)
        )
        pad_masks, att_masks = out[1], out[2]
        assert pad_masks.shape[1] == att_masks.shape[1] == cfg.chunk_size
