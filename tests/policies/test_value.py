import pytest
import torch

from opentau.policies.value.modeling_value import ValueFunction


class TestValueFunctionIntegration:
    """Integration tests for the complete Value Function pipeline."""

    def _verify_pad_masks(self, pad_masks):
        """Verify the pad masks are correct. This assumes all images are not padded. Language embeddings can be padded.

        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        """
        assert pad_masks.shape[0] == 1
        assert pad_masks.shape[1] == 566
        assert pad_masks.dtype == torch.bool

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

        batch_size = pad_masks.shape[0]
        for i in range(batch_size):
            assert torch.all(pad_masks[i, :512] == 1)  # image tokens should not be padded
            _check_ones_before_zeros(pad_masks[i, 512:564])  # prompt tokens
            assert torch.all(pad_masks[i, 565:] == 1)  # state and cls token should not be padded

    def _verify_position_ids(self, position_ids, pad_masks):
        """Verify the position ids are correct. They should increment by 1 for each non-padded token and stay the same for padded tokens.

        position_ids: tensor with shape (batch_size, seq_len)
        pad_masks: tensor with shape (batch_size, seq_len)
        """
        assert position_ids.shape[0] == 1
        assert position_ids.shape[1] == 566
        assert position_ids.dtype == torch.long

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

        batch_size = position_ids.shape[0]
        for i in range(batch_size):
            # Check entire prefix position IDs array
            _check_position_ids_with_padding(position_ids[i], pad_masks[i])

    def _verify_vlm_attention_mask(self, vlm_attention_mask, pad_masks):
        """Verify the VLM attention mask is correct.

        vlm_attention_mask: tensor with shape (batch_size, seq_len, seq_len)
        pad_masks: tensor with shape (batch_size, seq_len)
        """
        assert vlm_attention_mask.shape[0] == 1
        assert vlm_attention_mask.shape[1] == 566
        assert vlm_attention_mask.shape[2] == 566
        assert vlm_attention_mask.dtype == torch.bool

        batch_size = vlm_attention_mask.shape[0]
        for i in range(batch_size):
            # construct correct attention mask
            # see diagram here: https://drive.google.com/file/d/12lhS72bnQrKyL4iCfEj6SDRSPi1NABBj/view?usp=sharing
            correct_vlm_attention_mask = torch.ones(566, 566, dtype=torch.bool)

            # pad tokens should not be attended to or attend to any other tokens
            prompt_start_idx, state_start_idx = 512, 564
            num_non_padded_prompt_tokens = pad_masks[i, prompt_start_idx:state_start_idx].sum()
            correct_vlm_attention_mask[
                prompt_start_idx + num_non_padded_prompt_tokens : state_start_idx, :
            ] = 0
            correct_vlm_attention_mask[
                :, prompt_start_idx + num_non_padded_prompt_tokens : state_start_idx
            ] = 0

            assert torch.all(vlm_attention_mask[i].cpu() == correct_vlm_attention_mask.cpu())

    @pytest.mark.gpu
    @pytest.mark.slow  # ~1 mins
    def test_complete_value_function_pipeline_integration(
        self, value_function_training_config, lerobot_dataset_metadata
    ):
        """Test the complete Value Function pipeline from data loading to model execution."""

        # Initialize value function with unified training mode
        config = value_function_training_config.policy
        value_function = ValueFunction(config, dataset_stats=lerobot_dataset_metadata.stats)
        value_function.to(torch.bfloat16)

        # Test data preparation pipeline
        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, 3, 224, 224),
            "camera1": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, config.max_state_dim),
            "actions": torch.randn(batch_size, config.chunk_size, config.max_action_dim),
            "prompt": ["Pick up the red block"],
            "response": ["I will pick up the red block"],
            "loss_type": ["MSE"],
            "img_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
            "action_is_pad": torch.cat(
                [
                    torch.zeros(batch_size, config.chunk_size // 2, dtype=torch.bool),
                    torch.ones(batch_size, config.chunk_size - config.chunk_size // 2, dtype=torch.bool),
                ],
                dim=1,
            ),
            "return_bin_idx": torch.randint(0, 201, (batch_size,)),
            "return_continuous": torch.randn(
                batch_size,
            ),
        }

        value_function.to("cuda")
        batch_cuda = {
            key: value.to("cuda", non_blocking=True, dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

        # Capture intermediate variables for inspection by monkey-patching the siglip_gemma_value forward method
        captured_variables = {}

        def capture_variables_forward(*args, **kwargs):
            # Extract the variables we want to capture from the kwargs
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")

            # Capture the attention masks and position IDs
            if attention_mask is not None:
                captured_variables["attention_mask"] = attention_mask.clone()
            if position_ids is not None:
                captured_variables["position_ids"] = position_ids.clone()

            # Call the original forward method
            return original_siglip_gemma_value_forward(*args, **kwargs)

        # Store original siglip_gemma_value forward method and replace it
        original_siglip_gemma_value_forward = value_function.model.siglip_gemma_value.forward
        value_function.model.siglip_gemma_value.forward = capture_variables_forward

        # Also capture pad_masks by monkey-patching the embed_sequence method
        original_embed_sequence = value_function.model.embed_sequence

        def capture_embed_sequence(*args, **kwargs):
            result = original_embed_sequence(*args, **kwargs)
            embs, pad_masks, att_masks = result
            captured_variables["pad_masks"] = pad_masks.clone()
            return result

        value_function.model.embed_sequence = capture_embed_sequence

        # Test forward pass
        loss = value_function.forward(batch_cuda)

        # Restore original methods
        value_function.model.siglip_gemma_value.forward = original_siglip_gemma_value_forward
        value_function.model.embed_sequence = original_embed_sequence

        # Inspect all captured variables
        expected_vars = [
            "pad_masks",
            "position_ids",
            "attention_mask",
        ]

        for var_name in expected_vars:
            assert var_name in captured_variables, f"{var_name} was not captured"

        # Basic assertions about the variables
        assert captured_variables["attention_mask"].dtype == torch.bool, (
            "VLM attention mask should be boolean"
        )
        assert captured_variables["pad_masks"].dtype == torch.bool, "Pad masks should be boolean"

        self._verify_pad_masks(captured_variables["pad_masks"])
        self._verify_position_ids(
            captured_variables["position_ids"],
            captured_variables["pad_masks"],
        )
        self._verify_vlm_attention_mask(captured_variables["attention_mask"], captured_variables["pad_masks"])

        assert isinstance(loss, dict)
        assert "MSE" in loss
        assert "CE" in loss
        assert all(v.isfinite() for v in loss.values())

        # Test optimization parameters
        optim_params = value_function.get_optim_params()
        assert len(list(optim_params)) > 0

        # --------------------------------- Run the same test but for predict_value --------------------------------------
        captured_variables = {}

        def capture_variables_forward(*args, **kwargs):
            # Extract the variables we want to capture from the kwargs
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")

            # Capture the attention masks and position IDs
            if attention_mask is not None:
                captured_variables["attention_mask"] = attention_mask.clone()
            if position_ids is not None:
                captured_variables["position_ids"] = position_ids.clone()

            # Call the original forward method
            return original_siglip_gemma_value_forward(*args, **kwargs)

        # Store original siglip_gemma_value forward method and replace it
        original_siglip_gemma_value_forward = value_function.model.siglip_gemma_value.forward
        value_function.model.siglip_gemma_value.forward = capture_variables_forward

        # Also capture pad_masks by monkey-patching the embed_sequence method
        original_embed_sequence = value_function.model.embed_sequence

        def capture_embed_sequence(*args, **kwargs):
            result = original_embed_sequence(*args, **kwargs)
            embs, pad_masks, att_masks = result
            captured_variables["pad_masks"] = pad_masks.clone()
            return result

        value_function.model.embed_sequence = capture_embed_sequence

        # Test forward pass
        value = value_function.predict_value(batch_cuda)

        # Restore original methods
        value_function.model.siglip_gemma_value.forward = original_siglip_gemma_value_forward
        value_function.model.embed_sequence = original_embed_sequence

        # Inspect all captured variables
        expected_vars = [
            "pad_masks",
            "position_ids",
            "attention_mask",
        ]

        for var_name in expected_vars:
            assert var_name in captured_variables, f"{var_name} was not captured"

        # Basic assertions about the variables
        assert captured_variables["attention_mask"].dtype == torch.bool, (
            "VLM attention mask should be boolean"
        )
        assert captured_variables["pad_masks"].dtype == torch.bool, "Pad masks should be boolean"

        self._verify_pad_masks(captured_variables["pad_masks"])
        self._verify_position_ids(
            captured_variables["position_ids"],
            captured_variables["pad_masks"],
        )
        self._verify_vlm_attention_mask(captured_variables["attention_mask"], captured_variables["pad_masks"])

        assert isinstance(value, torch.Tensor)
        assert value.shape == (1,)
        assert value.isfinite()
