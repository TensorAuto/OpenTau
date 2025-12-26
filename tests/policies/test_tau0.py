#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lerobot.common.policies.tau0.configuration_tau0 import TAU0Config
from lerobot.common.policies.tau0.modeling_tau0 import (
    TAU0FlowMatching,
    TAU0Policy,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    sample_beta,
)
from lerobot.common.policies.tau0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
    apply_rope,
)
from lerobot.common.utils.fake_tensor import FakeTensorContext, run_with_fake_tensor


class TestTAU0Policy:
    """Test the TAU0Policy class."""

    @run_with_fake_tensor
    def test_tau0_policy_initialization(self, tau0_training_config):
        """Test TAU0Policy initialization with different execution targets."""
        config = tau0_training_config.policy
        # Test unified training mode (execution_target=None)
        policy = TAU0Policy(config)
        assert policy.execution_target is None
        assert hasattr(policy, "language_tokenizer")
        assert hasattr(policy, "model")

        # Test robot-only mode
        policy_robot = TAU0Policy(config, execution_target="robot")
        assert policy_robot.execution_target == "robot"
        assert not hasattr(policy_robot, "language_tokenizer")
        assert hasattr(policy_robot, "model")

        # Test cloud-only mode
        policy_cloud = TAU0Policy(config, execution_target="cloud")
        assert policy_cloud.execution_target == "cloud"
        assert hasattr(policy_cloud, "language_tokenizer")
        assert hasattr(policy_cloud, "model")

    @patch("lerobot.common.policies.tau0.modeling_tau0.TAU0FlowMatching")
    @patch("lerobot.common.policies.tau0.modeling_tau0.AutoTokenizer")
    @run_with_fake_tensor
    def test_tau0_policy_set_execution_target(
        self, mock_autotokenizer, mock_tau0flowmatching, tau0_training_config
    ):
        """Test setting execution target after initialization."""
        mock_autotokenizer.from_pretrained.return_value = None  # avoid API call to HF
        mock_model = mock_tau0flowmatching.return_value
        mock_model.set_execution_target.return_value = None  # avoid building the model

        config = tau0_training_config.policy

        # Test setting to robot mode
        policy = TAU0Policy(config)
        policy.set_execution_target("robot")
        assert policy.execution_target == "robot"
        assert not hasattr(policy, "language_tokenizer")

        # Test setting to cloud mode
        policy = TAU0Policy(config)
        policy.set_execution_target("cloud")
        assert policy.execution_target == "cloud"
        assert hasattr(policy, "language_tokenizer")

        # Test invalid execution target
        with pytest.raises(KeyError):
            policy.set_execution_target("invalid")

    @pytest.mark.gpu
    @pytest.mark.slow  # 7 sec
    @run_with_fake_tensor
    def test_tau0_policy_reset(self, fake_tensor_tau0):
        """Test policy reset functionality."""
        # Add some actions to the queue
        fake_tensor_tau0._action_queue.extend([torch.randn(1, 6) for _ in range(3)])
        assert len(fake_tensor_tau0._action_queue) == 3

        # Reset should clear the queue
        fake_tensor_tau0.reset()
        assert len(fake_tensor_tau0._action_queue) == 0

    @pytest.mark.gpu
    @pytest.mark.slow  # 5 sec
    def test_tau0_policy_get_optim_params(self, fake_tensor_tau0):
        """Test getting optimization parameters."""
        params = fake_tensor_tau0.get_optim_params()
        # Should contain model parameters
        assert len(list(params)) > 0

    @pytest.mark.gpu
    @pytest.mark.slow  # 5 sec
    def test_tau0_policy_prepare_images(self, fake_tensor_tau0):
        """Test image preparation methods."""
        batch_size = 2
        batch = {
            "camera0": torch.randn(batch_size, 3, 224, 224),
            "camera1": torch.randn(batch_size, 3, 224, 224),
            "local_camera0": torch.randn(batch_size, 3, 224, 224),
            "img_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
            "local_img_is_pad": torch.zeros(batch_size, 1, dtype=torch.bool),
        }

        # Test cloud VLM image preparation
        images, img_masks = fake_tensor_tau0.prepare_cloud_vlm_images(batch)
        assert len(images) == 2
        assert images[0].shape == (batch_size, 3, 224, 224)
        assert img_masks.shape == (2, batch_size)

        # Test action expert image preparation
        images, img_masks = fake_tensor_tau0.prepare_action_expert_images(batch)
        assert len(images) == 1
        assert images[0].shape == (batch_size, 3, 224, 224)
        assert img_masks.shape == (1, batch_size)

    @pytest.mark.gpu
    @pytest.mark.slow  # 4 sec
    def test_tau0_policy_prepare_language(self, fake_tensor_tau0):
        """Test language preparation methods."""
        batch_size = 2
        batch = {
            "state": torch.randn((1,), device=fake_tensor_tau0.config.device),
            "prompt": ["Pick up the red block", "Place the cube on the cylinder"],
            "response": ["I will pick up the red block", "I will place the cube"],
            "observation.robot": torch.randn(batch_size, 6),
        }

        # Test prompt preparation
        tokens, masks = fake_tensor_tau0.prepare_prompt(batch)
        assert tokens.shape[0] == batch_size
        assert tokens.shape[1] == fake_tensor_tau0.config.tokenizer_max_length
        assert masks.shape[0] == batch_size
        assert masks.shape[1] == fake_tensor_tau0.config.tokenizer_max_length

        # Test response preparation
        tokens, masks = fake_tensor_tau0.prepare_response(batch)
        assert tokens.shape[0] == batch_size
        assert tokens.shape[1] == fake_tensor_tau0.config.response_max_tokens
        assert masks.shape[0] == batch_size
        assert masks.shape[1] == fake_tensor_tau0.config.response_max_tokens

    @pytest.mark.skip(reason="moved most of this test to the TestTAU0Integration class")
    @pytest.mark.gpu
    @pytest.mark.slow  # 1 mins
    def test_tau0_policy_select_action(self, tau0, tau0_training_config):
        """Test action selection."""
        from lerobot.common.utils.utils import create_dummy_observation

        observation = create_dummy_observation(tau0_training_config, "cuda")
        observation["camera1"] = observation["camera0"]  # HACK: tau0_training_config expects 2 cameras

        # Test action selection
        with torch.inference_mode():
            action = tau0.select_action(observation)
            assert action.shape == (1, 32)

    def test_transform_state_dict_keys(self):
        """Test _transform_state_dict_keys method with various key transformations."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        # Test case 1: Basic key transformations
        state_dict = {
            "model.paligemma_with_expert.paligemma.language_model.lm_head.weight": torch.randn(2, 2),
            "model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight": torch.randn(
                2, 2
            ),
            "model.paligemma_with_expert.paligemma.vision_tower.conv1.weight": torch.randn(2, 2),
            "model.paligemma_with_expert.paligemma.multi_modal_projector.weight": torch.randn(2, 2),
            "model.paligemma_with_expert.gemma_expert.weight": torch.randn(2, 2),  # Should remain unchanged
        }

        transformed = TAU0Policy._transform_state_dict_keys(state_dict)

        # Check transformations
        assert "model.paligemma_with_expert.paligemma.lm_head.weight" in transformed
        assert "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight" in transformed
        assert "model.paligemma_with_expert.paligemma.model.vision_tower.conv1.weight" in transformed
        assert "model.paligemma_with_expert.paligemma.model.multi_modal_projector.weight" in transformed
        assert "model.paligemma_with_expert.gemma_expert.weight" in transformed  # Unchanged

        # Check that values are preserved
        assert torch.equal(
            transformed["model.paligemma_with_expert.paligemma.lm_head.weight"],
            state_dict["model.paligemma_with_expert.paligemma.language_model.lm_head.weight"],
        )
        assert torch.equal(
            transformed["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
            state_dict["model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight"],
        )

    def test_transform_state_dict_keys_tied_weights_lm_head_only(self):
        """Test tied weights handling when only lm_head.weight is present."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        state_dict = {
            "model.paligemma_with_expert.paligemma.lm_head.weight": torch.randn(2, 2),
        }

        transformed = TAU0Policy._transform_state_dict_keys(state_dict)

        # Check that embed_tokens.weight is now tied to lm_head.weight
        expected_embed_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        assert expected_embed_key in transformed
        assert torch.equal(
            transformed[expected_embed_key],
            state_dict["model.paligemma_with_expert.paligemma.lm_head.weight"],
        )

    def test_transform_state_dict_keys_tied_weights_embed_tokens_only(self):
        """Test tied weights handling when only embed_tokens.weight is present."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        state_dict = {
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": torch.randn(
                2, 2
            ),
        }

        transformed = TAU0Policy._transform_state_dict_keys(state_dict)

        # Check that lm_head.weight is now tied to embed_tokens.weight
        expected_lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
        assert expected_lm_head_key in transformed
        assert torch.equal(
            transformed[expected_lm_head_key],
            state_dict["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
        )

    def test_transform_state_dict_keys_tied_weights_both_present(self):
        """Test tied weights handling when both lm_head.weight and embed_tokens.weight are present."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        state_dict = {
            "model.paligemma_with_expert.paligemma.lm_head.weight": torch.randn(100, 200),
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": torch.randn(
                100, 200
            ),
        }

        transformed = TAU0Policy._transform_state_dict_keys(state_dict)

        # Both should remain unchanged since they're both present
        assert torch.equal(
            transformed["model.paligemma_with_expert.paligemma.lm_head.weight"],
            state_dict["model.paligemma_with_expert.paligemma.lm_head.weight"],
        )
        assert torch.equal(
            transformed["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
            state_dict["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"],
        )

    def test_transform_state_dict_keys_no_tied_weights(self):
        """Test when neither lm_head.weight nor embed_tokens.weight are present."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        state_dict = {
            "model.paligemma_with_expert.paligemma.vision_tower.conv1.weight": torch.randn(2, 2),
            "model.paligemma_with_expert.gemma_expert.weight": torch.randn(2, 2),
        }

        transformed = TAU0Policy._transform_state_dict_keys(state_dict)

        # Should only contain the transformed keys, no tied weights
        assert len(transformed) == 2
        assert "model.paligemma_with_expert.paligemma.model.vision_tower.conv1.weight" in transformed
        assert "model.paligemma_with_expert.gemma_expert.weight" in transformed

    def test_transform_state_dict_keys_empty_dict(self):
        """Test with empty state dict."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        state_dict = {}
        transformed = TAU0Policy._transform_state_dict_keys(state_dict)
        assert transformed == {}

    def test_transform_state_dict_keys_no_transformations(self):
        """Test with keys that don't need transformation."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        state_dict = {
            "model.paligemma_with_expert.gemma_expert.weight": torch.randn(2),
            "model.paligemma_with_expert.onboard_vision_encoder.weight": torch.randn(2),
        }

        transformed = TAU0Policy._transform_state_dict_keys(state_dict)

        # Should be identical since no transformations needed
        assert transformed == state_dict

    @pytest.mark.slow  # 3s
    @patch("safetensors.torch.load_file")
    @patch("lerobot.common.policies.tau0.modeling_tau0.log_model_loading_keys")
    @run_with_fake_tensor
    def test_load_as_safetensor_success(self, mock_log_keys, mock_load_file, tau0_training_config):
        """Test successful loading of safetensor file."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        # Mock the state dict that would be loaded from file
        mock_state_dict = {
            "model.paligemma_with_expert.paligemma.language_model.lm_head.weight": torch.randn(257152, 3),
            "model.paligemma_with_expert.paligemma.vision_tower.conv1.weight": torch.randn(2),
        }
        mock_load_file.return_value = mock_state_dict

        # Create a mock model
        model = TAU0Policy(tau0_training_config.policy)

        # Mock the load_state_dict method to return a success message
        mock_msg = type("MockMessage", (), {"missing_keys": [], "unexpected_keys": []})()
        model.load_state_dict = lambda state_dict, strict: mock_msg

        # Test the method
        result = TAU0Policy._load_as_safetensor(model, "test_model.safetensors", "cpu", True)

        # Verify calls
        mock_load_file.assert_called_once_with("test_model.safetensors", device="cpu")
        mock_log_keys.assert_called_once_with([], [])
        assert result == model

    @pytest.mark.slow  # 3s
    @patch("safetensors.torch.load_file")
    @patch("lerobot.common.policies.tau0.modeling_tau0.log_model_loading_keys")
    def test_load_as_safetensor_with_missing_keys(self, mock_log_keys, mock_load_file, tau0_training_config):
        """Test loading with missing keys."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        mock_state_dict = {
            "model.paligemma_with_expert.paligemma.language_model.lm_head.weight": torch.randn(257152, 200),
        }
        mock_load_file.return_value = mock_state_dict

        with FakeTensorContext():
            model = TAU0Policy(tau0_training_config.policy)

            # Mock the load_state_dict method to return missing keys
            mock_msg = type(
                "MockMessage",
                (),
                {"missing_keys": ["missing.weight"], "unexpected_keys": ["unexpected.weight"]},
            )()
            model.load_state_dict = lambda state_dict, strict: mock_msg

            # Test the method
            result = TAU0Policy._load_as_safetensor(model, "test_model.safetensors", "cpu", True)

            # Verify logging was called with the missing/unexpected keys
            mock_log_keys.assert_called_once_with(["missing.weight"], ["unexpected.weight"])
            assert result == model

    @pytest.mark.slow  # 3s
    @patch("safetensors.torch.load_file")
    def test_load_as_safetensor_key_transformation(self, mock_load_file, tau0_training_config):
        """Test that key transformations are applied during loading."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        # Mock state dict with keys that need transformation
        mock_state_dict = {
            "model.paligemma_with_expert.paligemma.language_model.lm_head.weight": torch.randn(257152, 2),
            "model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight": torch.randn(3),
        }
        mock_load_file.return_value = mock_state_dict

        with FakeTensorContext():
            model = TAU0Policy(tau0_training_config.policy)

            # Capture the state dict passed to load_state_dict
            captured_state_dict = None

            def mock_load_state_dict(state_dict, strict):
                nonlocal captured_state_dict
                captured_state_dict = state_dict
                return type("MockMessage", (), {"missing_keys": [], "unexpected_keys": []})()

            model.load_state_dict = mock_load_state_dict

            # Test the method
            TAU0Policy._load_as_safetensor(model, "test_model.safetensors", "cpu", True)

            # Verify that the state dict was transformed
            assert captured_state_dict is not None
            assert "model.paligemma_with_expert.paligemma.lm_head.weight" in captured_state_dict
            assert (
                "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                in captured_state_dict
            )
            # Check that tied weights were handled
            assert captured_state_dict["model.paligemma_with_expert.paligemma.lm_head.weight"].shape == (
                257152,
                2048,
            )
            assert captured_state_dict[
                "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            ].shape == (3,)

    @pytest.mark.slow  # 3s
    @patch("safetensors.torch.load_file")
    def test_load_as_safetensor_different_map_location(self, mock_load_file, tau0_training_config):
        """Test loading with different map_location."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        mock_state_dict = {"test.weight": torch.randn(10, 10)}
        mock_load_file.return_value = mock_state_dict

        with FakeTensorContext():
            model = TAU0Policy(tau0_training_config.policy)
            model.load_state_dict = lambda state_dict, strict: type(
                "MockMessage", (), {"missing_keys": [], "unexpected_keys": []}
            )()

            # Test with different map_location
            TAU0Policy._load_as_safetensor(model, "test_model.safetensors", "cuda", True)
            mock_load_file.assert_called_with("test_model.safetensors", device="cuda")

    @pytest.mark.slow  # 3s
    @patch("safetensors.torch.load_file")
    def test_load_as_safetensor_strict_false(self, mock_load_file, tau0_training_config):
        """Test loading with strict=False."""
        from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy

        mock_state_dict = {"test.weight": torch.randn(10, 10)}
        mock_load_file.return_value = mock_state_dict

        with FakeTensorContext():
            model = TAU0Policy(tau0_training_config.policy)

            # Capture the strict parameter
            captured_strict = None

            def mock_load_state_dict(state_dict, strict):
                nonlocal captured_strict
                captured_strict = strict
                return type("MockMessage", (), {"missing_keys": [], "unexpected_keys": []})()

            model.load_state_dict = mock_load_state_dict

            # Test with strict=False
            TAU0Policy._load_as_safetensor(model, "test_model.safetensors", "cpu", False)
            assert captured_strict is False


class TestTAU0FlowMatching:
    """Test the TAU0FlowMatching class."""

    @pytest.mark.gpu
    @pytest.mark.slow  # 4 sec
    def test_tau0_flow_matching_initialization(self, tau0_training_config):
        """Test TAU0FlowMatching initialization."""
        with FakeTensorContext():
            config = tau0_training_config.policy
            model = TAU0FlowMatching(config)
            assert isinstance(model, nn.Module)
            assert hasattr(model, "paligemma_with_expert")
            assert hasattr(model, "state_proj")
            assert hasattr(model, "action_in_proj")
            assert hasattr(model, "action_out_proj")

    @pytest.mark.gpu
    @pytest.mark.slow  # 2 sec
    def test_tau0_flow_matching_set_execution_target(self, tau0_training_config):
        """Test setting execution target."""
        with FakeTensorContext():
            config = tau0_training_config.policy
            model = TAU0FlowMatching(config)
            model.set_execution_target("robot")
            model.set_execution_target("cloud")
            model.set_execution_target(None)

    @pytest.mark.gpu
    @pytest.mark.slow  # 3.5 sec
    def test_tau0_flow_matching_sample_noise(self, tau0_training_config):
        """Test noise sampling."""
        with FakeTensorContext():
            config = tau0_training_config.policy
            model = TAU0FlowMatching(config)
            shape = (2, 3, 4)
            device = torch.device("cpu")
            noise = model.sample_noise(shape, device)
            assert noise.shape == shape
            assert noise.device == device

    @pytest.mark.gpu
    @pytest.mark.parametrize(argnames="device_str", argvalues=["cpu", "cuda:0"])
    def test_tau0_flow_matching_sample_time(self, device_str, tau0_training_config):
        """Test time sampling."""
        dummy_self = ...
        bsize = 4
        device = torch.device(device_str)
        time = TAU0FlowMatching.sample_time(dummy_self, bsize, device)

        assert time.shape == (bsize,)
        assert time.device == device

    @pytest.mark.gpu
    @pytest.mark.slow  # 2 mins
    def test_tau0_flow_matching_embed_prefix(self, tau0_training_config):
        """Test prefix embedding to ensure correct order and masks."""
        config = tau0_training_config.policy
        model = TAU0FlowMatching(config)

        batch_size = 2
        # Create test data with known dimensions
        images = [torch.randn(batch_size, 3, 224, 224)]
        img_masks = torch.ones(1, batch_size, dtype=torch.bool)
        prompt_tokens = torch.randint(0, 1000, (batch_size, config.tokenizer_max_length))
        prompt_masks = torch.zeros(batch_size, config.tokenizer_max_length, dtype=torch.bool)
        response_tokens = torch.randint(0, 1000, (batch_size, config.response_max_tokens))
        response_masks = torch.ones(batch_size, config.response_max_tokens, dtype=torch.bool)

        # Mock the embedding methods to return predictable outputs
        with (
            patch.object(model.paligemma_with_expert, "embed_image") as mock_embed_image,
            patch.object(model.paligemma_with_expert, "embed_language_tokens") as mock_embed_lang,
        ):
            # create mock emb returns
            num_image_tokens = (
                model.paligemma_with_expert.config.paligemma_config.vision_config.num_image_tokens
            )
            model_hidden_size = model.paligemma_with_expert.config.paligemma_config.hidden_size
            mock_img_emb = torch.ones(batch_size, num_image_tokens, model_hidden_size) * 1
            mock_prompt_emb = torch.ones(batch_size, config.tokenizer_max_length, model_hidden_size) * 2
            mock_response_emb = torch.ones(batch_size, config.response_max_tokens, model_hidden_size) * 3

            # mock embed image
            mock_embed_image.return_value = mock_img_emb

            # Mock language embeddings
            mock_embed_lang.side_effect = [mock_prompt_emb, mock_response_emb]

            embs, pad_masks, att_masks = model.embed_prefix(
                images, img_masks, prompt_tokens, prompt_masks, response_tokens, response_masks
            )

            # Verify basic shapes
            assert embs.shape[0] == batch_size
            assert pad_masks.shape[0] == batch_size
            assert att_masks.shape[0] == batch_size

            # Verify total sequence length: image_tokens + prompt_tokens + response_tokens
            expected_seq_len = num_image_tokens + config.tokenizer_max_length + config.response_max_tokens
            assert embs.shape[1] == expected_seq_len
            assert pad_masks.shape[1] == expected_seq_len
            assert att_masks.shape[1] == expected_seq_len

            # verify that the embs are in the correct order
            assert torch.equal(embs[:, :num_image_tokens], mock_img_emb)
            assert torch.equal(
                embs[:, num_image_tokens : num_image_tokens + config.tokenizer_max_length],
                mock_prompt_emb
                * torch.tensor(
                    model_hidden_size**0.5,
                    dtype=mock_prompt_emb.dtype,
                    device=mock_prompt_emb.device,
                ),
            )
            assert torch.equal(
                embs[:, num_image_tokens + config.tokenizer_max_length :],
                mock_response_emb
                * torch.tensor(
                    model_hidden_size**0.5,
                    dtype=mock_response_emb.dtype,
                    device=mock_response_emb.device,
                ),
            )

            # Verify that the mock methods were called in the correct order
            mock_embed_image.assert_called_once()
            assert mock_embed_lang.call_count == 2  # Once for prompt, once for response

            # Check the order of language embedding calls
            lang_calls = mock_embed_lang.call_args_list
            assert len(lang_calls) == 2
            # First call should be for prompt tokens
            assert torch.equal(lang_calls[0][0][0], prompt_tokens)
            # Second call should be for response tokens
            assert torch.equal(lang_calls[1][0][0], response_tokens)

            # Verify pad_masks structure
            # Image masks should be expanded to match image token count
            # Prompt and response masks should be as provided
            assert torch.all(pad_masks[:, :num_image_tokens] == 1)  # All image tokens should be valid
            assert torch.all(
                pad_masks[:, num_image_tokens : num_image_tokens + config.tokenizer_max_length] == 0
            )  # All prompt tokens should be valid
            assert torch.all(
                pad_masks[:, num_image_tokens + config.tokenizer_max_length :] == 1
            )  # All response tokens should be valid

            # Verify attention masks structure
            # Image tokens should have att_mask = 0 (can attend to each other)
            # Prompt tokens should have att_mask = 0 (can attend to images and each other)
            # Response tokens should have att_mask = 1 (causal attention)
            expected_att_masks = (
                [0] * num_image_tokens + [0] * config.tokenizer_max_length + [1] * config.response_max_tokens
            )
            expected_att_masks = torch.tensor(expected_att_masks, dtype=torch.bool)

            # Check that the attention mask pattern is correct for each batch element
            for i in range(batch_size):
                assert torch.all(att_masks[i] == expected_att_masks)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.gpu
    @pytest.mark.slow  # 2 mins
    def test_tau0_flow_matching_embed_suffix(self, tau0_training_config):
        """Test suffix embedding to ensure correct order and masks."""
        config: TAU0Config = tau0_training_config.policy
        model = TAU0FlowMatching(config)

        batch_size = 2
        # Create test data with known dimensions
        images = [torch.randn(batch_size, 3, 224, 224)]
        image_masks = torch.ones(1, batch_size, dtype=torch.bool)
        state = torch.randn(batch_size, config.max_state_dim)
        noisy_actions = torch.randn(batch_size, config.chunk_size, config.max_action_dim)
        frozen_actions = torch.randn(batch_size, 0, config.max_action_dim)
        frozen_action_is_pad = torch.zeros(batch_size, 0, dtype=torch.bool)
        timestep = torch.rand(batch_size)

        # Mock the embedding methods to return predictable outputs
        with (
            patch.object(model.state_proj, "forward") as mock_state_proj,
            patch.object(
                model.paligemma_with_expert.onboard_vision_encoder, "forward"
            ) as mock_vision_encoder,
            patch.object(model.action_time_mlp_out, "forward") as mock_mlp_out,
        ):
            # Create mock emb returns with known values
            hidden_size = model.paligemma_with_expert.config.gemma_expert_config.hidden_size
            num_local_image_tokens = config.num_local_image_tokens
            mock_state_emb = torch.ones(batch_size, 1, hidden_size) * 1
            mock_img_emb = torch.ones(batch_size, num_local_image_tokens, hidden_size) * 2
            mock_action_time_emb = torch.ones(batch_size, config.chunk_size, hidden_size) * 3

            # Mock state projection
            mock_state_proj.return_value = mock_state_emb.squeeze(
                1
            )  # Remove the middle dimension for the projection

            # Mock vision encoder
            mock_vision_encoder.return_value = mock_img_emb

            # Mock MLP processing
            mock_mlp_out.return_value = mock_action_time_emb

            embs, pad_masks, att_masks = model.embed_suffix(
                images,
                image_masks,
                state,
                noisy_actions,
                frozen_actions,
                frozen_action_is_pad,
                timestep,
            )

            # Verify basic shapes
            assert embs.shape[0] == batch_size
            assert pad_masks.shape[0] == batch_size
            assert att_masks.shape[0] == batch_size

            # Calculate expected sequence length: 1 (state) + num_local_image_tokens (image) + chunk_size (actions)
            expected_seq_len = 1 + num_local_image_tokens + config.frozen_actions + config.chunk_size
            assert embs.shape[1] == expected_seq_len
            assert pad_masks.shape[1] == expected_seq_len
            assert att_masks.shape[1] == expected_seq_len

            # Verify that the embeddings are in the correct order
            # State embedding should be first
            assert torch.equal(embs[:, 0:1], mock_state_emb)
            # Image embeddings should follow
            assert torch.equal(embs[:, 1 : 1 + num_local_image_tokens], mock_img_emb)
            # Action embeddings should be last
            assert torch.equal(embs[:, 1 + num_local_image_tokens :], mock_action_time_emb)

            # Verify that the mock methods were called correctly
            mock_state_proj.assert_called_once_with(state)
            mock_vision_encoder.assert_called_once_with(images[0])
            mock_mlp_out.assert_called_once()

            # Verify pad_masks structure
            # State mask should be all True (1)
            assert torch.all(pad_masks[:, 0] == 1)
            # Image masks should be all True (1)
            assert torch.all(pad_masks[:, 1 : 1 + num_local_image_tokens] == 1)
            # Action masks should be all True (1)
            assert torch.all(pad_masks[:, 1 + num_local_image_tokens :] == 1)

            # Verify attention masks structure
            # State should have att_mask = 1 (causal attention)
            # Image tokens should have att_mask = 0 (can attend to each other and state)
            # Actions should have att_mask = 1 for first action, 0 for subsequent actions
            expected_att_masks = [1] + ([0] * num_local_image_tokens) + [1] + ([0] * (config.chunk_size - 1))
            expected_att_masks = torch.tensor(expected_att_masks, dtype=torch.bool)

            # Check that the attention mask pattern is correct for each batch element
            for i in range(batch_size):
                assert torch.all(att_masks[i] == expected_att_masks)

    @pytest.mark.gpu
    @pytest.mark.slow  # 6 sec
    @run_with_fake_tensor
    def test_tau0_flow_matching_forward(self, tau0_training_config):
        """Test forward pass."""
        config = tau0_training_config.policy
        model = TAU0FlowMatching(config)

        batch_size = 2
        cloud_vlm_images = [torch.randn(batch_size, 3, 224, 224)]
        cloud_vlm_img_masks = torch.ones(1, batch_size, dtype=torch.bool)
        action_expert_images = [torch.randn(batch_size, 3, 224, 224)]
        action_expert_img_masks = torch.ones(1, batch_size, dtype=torch.bool)
        prompt_tokens = torch.randint(0, 1000, (batch_size, tau0_training_config.policy.tokenizer_max_length))
        prompt_is_pad = torch.zeros(
            batch_size, tau0_training_config.policy.tokenizer_max_length, dtype=torch.bool
        )
        response_tokens = torch.randint(
            0, 1000, (batch_size, tau0_training_config.policy.response_max_tokens)
        )
        response_is_pad = torch.zeros(
            batch_size, tau0_training_config.policy.response_max_tokens, dtype=torch.bool
        )
        state = torch.randn(batch_size, tau0_training_config.policy.max_state_dim)
        actions = torch.randn(
            batch_size, tau0_training_config.policy.chunk_size, tau0_training_config.policy.max_action_dim
        )
        frozen_actions = torch.randn(batch_size, 0, tau0_training_config.policy.max_action_dim)
        loss_type = ["MSE", "CE"] * (batch_size // 2)
        actions_is_pad = torch.zeros(batch_size, tau0_training_config.policy.chunk_size, dtype=torch.bool)
        frozen_action_is_pad = torch.zeros(batch_size, 0, dtype=torch.bool)

        losses = model.forward(
            cloud_vlm_images,
            cloud_vlm_img_masks,
            action_expert_images,
            action_expert_img_masks,
            prompt_tokens,
            prompt_is_pad,
            response_tokens,
            response_is_pad,
            state,
            actions,
            frozen_actions,
            loss_type,
            actions_is_pad,
            frozen_action_is_pad,
        )

        assert isinstance(losses, dict)
        assert "MSE" in losses
        assert "CE" in losses

    @pytest.mark.gpu
    @pytest.mark.slow  # 2 sec
    def test_tau0_flow_matching_sample_vlm_tokens(self, tau0_training_config):
        with FakeTensorContext():
            config = tau0_training_config.policy
            model = TAU0FlowMatching(config)

            batch_size = 2
            images = [torch.randn(batch_size, 3, 224, 224)]
            img_masks = torch.ones(1, batch_size, dtype=torch.bool)
            lang_tokens = torch.randint(0, 1000, (batch_size, 10))
            lang_masks = torch.ones(batch_size, 10, dtype=torch.bool)

            result = model.sample_vlm_tokens(images, img_masks, lang_tokens, lang_masks)
            assert len(result) == 4
            past_key_values, prefix_pad_masks, prefix_offsets, num_cross_att_tokens = result
            assert prefix_offsets.shape[0] == batch_size

    @pytest.mark.gpu
    @pytest.mark.slow  # 7 sec
    def test_tau0_flow_matching_sample_actions(self, tau0_training_config):
        """Test action sampling."""
        with FakeTensorContext():
            config = tau0_training_config.policy
            model = TAU0FlowMatching(config)

            batch_size = 2
            images = [torch.randn(batch_size, 3, 224, 224)]
            state = torch.randn(batch_size, tau0_training_config.policy.max_state_dim)
            kv_shape = (batch_size, 10, 1, 256)
            prefix_pad_masks = torch.ones(batch_size, 10, dtype=torch.bool)
            past_key_values = {
                17: {"key_states": torch.randn(kv_shape), "value_states": torch.randn(kv_shape)}
            }
            prefix_offsets = torch.zeros(batch_size, 1)
            num_cross_att_tokens = 10
            frozen_actions = torch.randn(batch_size, 0, tau0_training_config.policy.max_action_dim)
            frozen_action_is_pad = torch.zeros(batch_size, 0, dtype=torch.bool)

            actions = model.sample_actions(
                images,
                state,
                past_key_values,
                prefix_pad_masks,
                prefix_offsets,
                num_cross_att_tokens,
                frozen_actions,
                frozen_action_is_pad,
            )
            assert actions.shape == (batch_size, config.n_action_steps, config.max_action_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.slow  # 1.5 mins
    @pytest.mark.gpu
    def test_model_output_slicing(self, tau0_training_config):
        """Test that the correct slice of prefix_out is extracted for the action loss and language loss"""
        config = tau0_training_config.policy
        model = TAU0FlowMatching(config)

        batch_size = 2
        # Create a controlled mock prefix_out with known dimensions
        total_seq_len = 200  # Total sequence length
        response_max_tokens = config.response_max_tokens

        # Create prefix_out and suffix_out tensors with known values
        # We'll put special values at the positions we expect to extract
        mock_prefix_out = torch.zeros(
            batch_size, total_seq_len, model.paligemma_with_expert.config.paligemma_config.hidden_size
        )
        mock_suffix_out = torch.zeros(
            batch_size, config.chunk_size, model.paligemma_with_expert.config.gemma_expert_config.hidden_size
        )

        # Mark the expected slice region with special values
        expected_prefix_slice_indices = slice(total_seq_len - response_max_tokens - 1, total_seq_len - 1)
        expected_suffix_slice_indices = slice(-config.chunk_size, None)

        # Put special values in the expected slice region
        mock_prefix_out[:, expected_prefix_slice_indices, 0] = 1.0  # Mark first dimension
        mock_prefix_out[:, expected_prefix_slice_indices, -1] = 2.0  # Mark last dimension
        mock_suffix_out[:, expected_suffix_slice_indices, 0] = 1.0  # Mark first dimension
        mock_suffix_out[:, expected_suffix_slice_indices, -1] = 2.0  # Mark last dimension

        # Create mock outputs_embeds and past_key_values
        mock_outputs_embeds = [mock_prefix_out, mock_suffix_out]
        mock_past_key_values = {
            17: {
                "key_states": torch.randn(batch_size, total_seq_len, 1, 256),
                "value_states": torch.randn(batch_size, total_seq_len, 1, 1),
            }
        }

        # Mock the forward method to return our controlled values
        with patch.object(model.paligemma_with_expert, "forward") as mock_forward:
            mock_forward.return_value = (mock_outputs_embeds, mock_past_key_values)

            # Create dummy inputs for the forward method
            cloud_vlm_images = [torch.randn(batch_size, 3, 224, 224)]
            cloud_vlm_img_masks = torch.ones(1, batch_size, dtype=torch.bool)
            action_expert_images = [torch.randn(batch_size, 3, 224, 224)]
            action_expert_img_masks = torch.ones(1, batch_size, dtype=torch.bool)
            prompt_tokens = torch.randint(0, 1000, (batch_size, config.tokenizer_max_length))
            prompt_is_pad = torch.zeros(batch_size, config.tokenizer_max_length, dtype=torch.bool)
            response_tokens = torch.randint(0, 1000, (batch_size, config.response_max_tokens))
            response_is_pad = torch.zeros(batch_size, config.response_max_tokens, dtype=torch.bool)
            state = torch.randn(batch_size, config.max_state_dim)
            actions = torch.randn(batch_size, config.chunk_size, config.max_action_dim)
            frozen_actions = torch.randn(batch_size, 0, config.max_action_dim)
            loss_type = ["MSE", "CE"] * (batch_size // 2)
            actions_is_pad = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
            frozen_action_is_pad = torch.zeros(batch_size, 0, dtype=torch.bool)

            # Mock the lm_head to capture what it receives
            original_lm_head = model.paligemma_with_expert.paligemma.lm_head

            class MockLMHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.captured_input = None

                def forward(self, input_tensor):
                    self.captured_input = input_tensor
                    # Return dummy logits
                    return torch.randn(input_tensor.shape[0], input_tensor.shape[1], 257152)  # vocab size

            mock_lm_head = MockLMHead()
            model.paligemma_with_expert.paligemma.lm_head = mock_lm_head

            # Mock the action_out_proj to capture what it receives
            captured_action_input = None
            original_action_forward = model.action_out_proj.forward

            def mock_action_forward(input):
                nonlocal captured_action_input
                captured_action_input = input
                # Return dummy actions
                return torch.randn(input.shape[0], input.shape[1], config.max_action_dim)

            model.action_out_proj.forward = mock_action_forward

            try:
                losses = model.forward(
                    cloud_vlm_images,
                    cloud_vlm_img_masks,
                    action_expert_images,
                    action_expert_img_masks,
                    prompt_tokens,
                    prompt_is_pad,
                    response_tokens,
                    response_is_pad,
                    state,
                    actions,
                    frozen_actions,
                    loss_type,
                    actions_is_pad,
                    frozen_action_is_pad,
                )

                # Verify the mock was called
                mock_forward.assert_called_once()

                # Check that lm_head was called with the correct slice (prefix_out slicing)
                assert mock_lm_head.captured_input is not None

                # Verify the shape of the input to lm_head
                expected_lm_head_shape = (
                    batch_size,
                    response_max_tokens,
                    model.paligemma_with_expert.config.paligemma_config.hidden_size,
                )
                assert mock_lm_head.captured_input.shape == expected_lm_head_shape

                # Verify that the input contains our special values
                # The first and last dimensions should have our marked values
                assert torch.allclose(
                    mock_lm_head.captured_input[:, :, 0], torch.ones(batch_size, response_max_tokens)
                )
                assert torch.allclose(
                    mock_lm_head.captured_input[:, :, -1], torch.full((batch_size, response_max_tokens), 2.0)
                )

                # Verify that the slice corresponds to the expected region
                expected_slice = mock_prefix_out[:, expected_prefix_slice_indices]
                assert torch.allclose(mock_lm_head.captured_input, expected_slice)

                # Check that action_out_proj was called with the correct slice (suffix_out slicing)
                assert captured_action_input is not None

                # Verify the shape of the input to action_out_proj
                expected_action_shape = (
                    batch_size,
                    config.chunk_size,
                    model.paligemma_with_expert.config.gemma_expert_config.hidden_size,
                )
                assert captured_action_input.shape == expected_action_shape

                # Verify that the input contains our special values for suffix_out
                # The first and last dimensions should have our marked values
                assert torch.allclose(
                    captured_action_input[:, :, 0], torch.ones(batch_size, config.chunk_size)
                )
                assert torch.allclose(
                    captured_action_input[:, :, -1], torch.full((batch_size, config.chunk_size), 2.0)
                )

                # Verify that the slice corresponds to the expected region for suffix_out
                expected_suffix_slice = mock_suffix_out[:, -config.chunk_size :]
                assert torch.allclose(captured_action_input, expected_suffix_slice)

                # Verify the losses are computed correctly
                assert isinstance(losses, dict)
                assert "MSE" in losses
                assert "CE" in losses

            finally:
                # Restore the original methods
                model.paligemma_with_expert.paligemma.lm_head = original_lm_head
                model.action_out_proj.forward = original_action_forward


class TestPaliGemmaWithExpertModel:
    """Test the PaliGemmaWithExpertModel class."""

    def test_paligemma_with_expert_config_initialization(self):
        """Test PaliGemmaWithExpertConfig initialization."""
        config = PaliGemmaWithExpertConfig()
        assert config.freeze_vision_encoder is True
        assert config.train_expert_only is True
        assert config.attention_implementation == "eager"

    def test_paligemma_with_expert_config_validation(self):
        """Test config validation."""
        # Test incompatible settings
        with pytest.raises(ValueError):
            PaliGemmaWithExpertConfig(freeze_vision_encoder=False, train_expert_only=True)

        # Test invalid attention implementation
        with pytest.raises(ValueError):
            PaliGemmaWithExpertConfig(attention_implementation="invalid")

    def test_paligemma_with_expert_config_use_cache_layer_validation(self):
        """Test use_cache_layer validation logic."""
        # Test valid use_cache_layer configuration
        valid_config = PaliGemmaWithExpertConfig(use_cache_layer=[17] * 18)
        assert valid_config.use_cache_layer == [17] * 18

        # Test use_cache_layer length mismatch with gemma_expert_config.num_hidden_layers
        with pytest.raises(ValueError, match="use_cache_layer must be a list of length 18. Got 10."):
            PaliGemmaWithExpertConfig(use_cache_layer=[17] * 10)

        # Test use_cache_layer length mismatch with gemma_expert_config.num_hidden_layers (too long)
        with pytest.raises(ValueError, match="use_cache_layer must be a list of length 18. Got 20."):
            PaliGemmaWithExpertConfig(use_cache_layer=[17] * 20)

        # Test use_cache_layer index out of bounds (negative)
        with pytest.raises(ValueError, match="use_cache_layer must be between 0 and 17. Got -1."):
            PaliGemmaWithExpertConfig(use_cache_layer=[-1] + [17] * 17)

        # Test use_cache_layer index out of bounds (too high)
        with pytest.raises(ValueError, match="use_cache_layer must be between 0 and 17. Got 18."):
            PaliGemmaWithExpertConfig(use_cache_layer=[18] + [17] * 17)

        # Test use_cache_layer index out of bounds (too high) in middle of list
        with pytest.raises(ValueError, match="use_cache_layer must be between 0 and 17. Got 25."):
            PaliGemmaWithExpertConfig(use_cache_layer=[17] * 9 + [25] + [17] * 8)

        # Test multiple invalid indices
        with pytest.raises(ValueError, match="use_cache_layer must be between 0 and 17. Got -5."):
            PaliGemmaWithExpertConfig(
                use_cache_layer=[-5, 0, 5, 10, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
            )

        # Test valid mixed indices
        valid_mixed_config = PaliGemmaWithExpertConfig(
            use_cache_layer=[0, 5, 10, 15, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
        )
        assert valid_mixed_config.use_cache_layer == [
            0,
            5,
            10,
            15,
            17,
            16,
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
        ]

    def test_paligemma_with_expert_config_use_cache_layer_edge_cases(self):
        """Test use_cache_layer validation edge cases and boundary conditions."""
        # Test boundary values (0 and 17 are valid)
        boundary_config = PaliGemmaWithExpertConfig(use_cache_layer=[0] * 18)
        assert boundary_config.use_cache_layer == [0] * 18

        boundary_config_max = PaliGemmaWithExpertConfig(use_cache_layer=[17] * 18)
        assert boundary_config_max.use_cache_layer == [17] * 18

        # Test boundary values with mixed valid indices
        mixed_boundary_config = PaliGemmaWithExpertConfig(
            use_cache_layer=[0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17, 0, 17]
        )
        assert mixed_boundary_config.use_cache_layer == [
            0,
            17,
            0,
            17,
            0,
            17,
            0,
            17,
            0,
            17,
            0,
            17,
            0,
            17,
            0,
            17,
            0,
            17,
        ]

        # Test empty list (should fail length validation)
        with pytest.raises(ValueError, match="use_cache_layer must be a list of length 18. Got 0."):
            PaliGemmaWithExpertConfig(use_cache_layer=[])

        # Test single element list (should fail length validation)
        with pytest.raises(ValueError, match="use_cache_layer must be a list of length 18. Got 1."):
            PaliGemmaWithExpertConfig(use_cache_layer=[17])

        # Test all valid indices in ascending order
        ascending_config = PaliGemmaWithExpertConfig(use_cache_layer=list(range(18)))
        assert ascending_config.use_cache_layer == list(range(18))

        # Test all valid indices in descending order
        descending_config = PaliGemmaWithExpertConfig(use_cache_layer=list(range(17, -1, -1)))
        assert descending_config.use_cache_layer == list(range(17, -1, -1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.gpu
    @run_with_fake_tensor
    def test_paligemma_with_expert_model_initialization(self, tau0_training_config):
        """Test PaliGemmaWithExpertModel initialization."""
        config = PaliGemmaWithExpertConfig()

        model = PaliGemmaWithExpertModel(config)
        assert hasattr(model, "paligemma")
        assert hasattr(model, "gemma_expert")
        assert hasattr(model, "onboard_vision_encoder")

        # Test robot-only mode
        model_robot = PaliGemmaWithExpertModel(config, execution_target="robot")
        assert not hasattr(model_robot, "paligemma")
        assert hasattr(model_robot, "gemma_expert")
        assert hasattr(model_robot, "onboard_vision_encoder")

        # Test cloud-only mode
        model_cloud = PaliGemmaWithExpertModel(config, execution_target="cloud")
        assert hasattr(model_cloud, "paligemma")
        assert not hasattr(model_cloud, "gemma_expert")
        assert not hasattr(model_cloud, "onboard_vision_encoder")

    @run_with_fake_tensor
    def test_paligemma_with_expert_model_set_execution_target(self, tau0_training_config):
        """Test setting execution target."""
        config = PaliGemmaWithExpertConfig()
        model = PaliGemmaWithExpertModel(config)

        # Test setting to robot mode
        model.set_execution_target("robot")
        assert model.execution_target == "robot"
        assert not hasattr(model, "paligemma")

        # Test setting to cloud mode
        model.set_execution_target("cloud")
        assert model.execution_target == "cloud"
        assert not hasattr(model, "gemma_expert")

        # Test invalid execution target
        with pytest.raises(KeyError):
            model.set_execution_target("invalid")

    @pytest.mark.gpu
    @run_with_fake_tensor
    def test_paligemma_with_expert_model_embed_image_and_language_tokens(self, tau0_training_config):
        """Test image embedding."""
        config = PaliGemmaWithExpertConfig()
        model = PaliGemmaWithExpertModel(config)

        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)

        features = model.embed_image(image)
        assert features.shape[0] == batch_size

        tokens = torch.randint(0, 1000, (batch_size, 10))

        embeddings = model.embed_language_tokens(tokens)
        assert embeddings.shape[0] == batch_size
        assert embeddings.shape[1] == 10

    @pytest.mark.gpu
    @run_with_fake_tensor
    def test_paligemma_with_expert_model_forward(self, tau0_training_config):
        """Test forward pass."""
        config = PaliGemmaWithExpertConfig()
        model = PaliGemmaWithExpertModel(config)

        batch_size = 2
        inputs_embeds = [
            torch.randn(batch_size, 20, 2048),  # PaliGemma embeddings
            torch.randn(batch_size, 10, 1024),  # Gemma expert embeddings
        ]
        n_cross_att_tokens = 15
        vlm_attention_mask = torch.ones(batch_size, 20, 20, dtype=torch.bool)
        vlm_position_ids = torch.arange(20).unsqueeze(0).expand(batch_size, -1)
        action_expert_attention_mask = torch.ones(batch_size, 10, 25, dtype=torch.bool)  # 25 = 10 + 15
        action_expert_position_ids = torch.arange(10).unsqueeze(0).expand(batch_size, -1)

        outputs_embeds, past_key_values = model.forward(
            inputs_embeds=inputs_embeds,
            n_cross_att_tokens=n_cross_att_tokens,
            vlm_attention_mask=vlm_attention_mask,
            vlm_position_ids=vlm_position_ids,
            action_expert_attention_mask=action_expert_attention_mask,
            action_expert_position_ids=action_expert_position_ids,
        )

        assert len(outputs_embeds) == 2
        assert outputs_embeds[0] is not None  # PaliGemma output
        assert outputs_embeds[1] is not None  # Gemma expert output
        assert past_key_values is not None

    @pytest.mark.gpu
    @run_with_fake_tensor
    def test_paligemma_with_expert_model_attention_interface(self, tau0_training_config):
        """Test attention interface selection."""
        config = PaliGemmaWithExpertConfig(attention_implementation="eager")
        model = PaliGemmaWithExpertModel(config)

        interface = model.get_attention_interface()
        assert interface == model.eager_attention_forward

        # Test fa2 attention (should raise NotImplementedError)
        config_fa2 = PaliGemmaWithExpertConfig(attention_implementation="fa2")
        model_fa2 = PaliGemmaWithExpertModel(config_fa2)
        with pytest.raises(NotImplementedError):
            model_fa2.flash_attention_forward(None, None, None, None, None, None)


class TestUtilityFunctions:
    """Test utility functions from modeling_tau0.py."""

    def test_create_sinusoidal_pos_embedding(self):
        """Test sinusoidal positional embedding creation."""
        time = torch.tensor([0.0, 0.5, 1.0])
        dimension = 10
        min_period = 0.1
        max_period = 1.0
        device = torch.device("cpu")

        pos_emb = create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device)

        assert pos_emb.shape == (3, 10)
        assert pos_emb.device == device

        # Test with odd dimension (should raise ValueError)
        with pytest.raises(ValueError):
            create_sinusoidal_pos_embedding(time, 11, min_period, max_period, device)

        # Test with wrong time tensor shape
        with pytest.raises(ValueError):
            create_sinusoidal_pos_embedding(time.unsqueeze(1), dimension, min_period, max_period, device)

    def test_sample_beta(self):
        """Test beta sampling."""
        alpha = 1.5
        beta = 1.0
        bsize = 4
        device = torch.device("cpu")

        result = sample_beta(alpha, beta, bsize, device)

        assert result.shape == (bsize,)
        assert result.device == device
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    @pytest.mark.parametrize(
        "pad_masks,att_masks,n_cross_att_tokens,expected_shape,expected_pattern,test_name",
        [
            # Test case 1: Pure causal attention
            (
                torch.ones(2, 4, dtype=torch.bool),
                torch.ones(2, 4, dtype=torch.bool),
                None,
                (2, 4, 4),
                torch.tensor(
                    [
                        [
                            [True, False, False, False],
                            [True, True, False, False],
                            [True, True, True, False],
                            [True, True, True, True],
                        ],
                        [
                            [True, False, False, False],
                            [True, True, False, False],
                            [True, True, True, False],
                            [True, True, True, True],
                        ],
                    ],
                    dtype=torch.bool,
                ),
                "pure_causal_attention",
            ),
            # Test case 2: Prefix-LM attention (first 2 tokens can attend to each other, last 2 are causal)
            (
                torch.ones(2, 4, dtype=torch.bool),
                torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool),
                None,
                (2, 4, 4),
                torch.tensor(
                    [
                        [
                            [True, True, False, False],
                            [True, True, False, False],
                            [True, True, True, False],
                            [True, True, True, True],
                        ],
                        [
                            [True, True, False, False],
                            [True, True, False, False],
                            [True, True, True, False],
                            [True, True, True, True],
                        ],
                    ],
                    dtype=torch.bool,
                ),
                "prefix_lm_attention",
            ),
            # Test case 3: Block causal attention
            (
                torch.ones(2, 6, dtype=torch.bool),
                torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]], dtype=torch.bool),
                None,
                (2, 6, 6),
                torch.tensor(
                    [
                        [
                            [True, True, False, False, False, False],
                            [True, True, False, False, False, False],
                            [True, True, True, True, False, False],
                            [True, True, True, True, False, False],
                            [True, True, True, True, True, True],
                            [True, True, True, True, True, True],
                        ],
                        [
                            [True, True, False, False, False, False],
                            [True, True, False, False, False, False],
                            [True, True, True, True, False, False],
                            [True, True, True, True, False, False],
                            [True, True, True, True, True, True],
                            [True, True, True, True, True, True],
                        ],
                    ],
                    dtype=torch.bool,
                ),
                "block_causal_attention",
            ),
            # Test case 4: Mixed padding masks
            (
                torch.tensor([[True, True, False, True], [True, False, True, True]], dtype=torch.bool),
                torch.ones(2, 4, dtype=torch.bool),
                None,
                (2, 4, 4),
                torch.tensor(
                    [
                        [
                            [True, False, False, False],
                            [True, True, False, False],
                            [False, False, False, False],
                            [True, True, False, True],
                        ],
                        [
                            [True, False, False, False],
                            [False, False, False, False],
                            [True, False, True, False],
                            [True, False, True, True],
                        ],
                    ],
                    dtype=torch.bool,
                ),
                "mixed_padding_masks",
            ),
            # Test case 5: With cross attention tokens
            (
                torch.ones(1, 3, dtype=torch.bool),
                torch.ones(1, 3, dtype=torch.bool),
                2,
                (1, 3, 5),  # 3 + 2 cross attention tokens
                torch.tensor(
                    [
                        [
                            [True, False, False, True, True],
                            [True, True, False, True, True],
                            [True, True, True, True, True],
                        ]
                    ],
                    dtype=torch.bool,
                ),
                "with_cross_attention_tokens",
            ),
            # Test case 6: Single sequence
            (
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[1, 0]], dtype=torch.bool),
                None,
                (1, 2, 2),
                torch.tensor([[[True, True], [True, True]]], dtype=torch.bool),
                "single_sequence",
            ),
            # Test case 7: All zeros attention mask
            (
                torch.ones(2, 3, dtype=torch.bool),
                torch.zeros(2, 3, dtype=torch.bool),
                None,
                (2, 3, 3),
                torch.tensor(
                    [
                        [[True, True, True], [True, True, True], [True, True, True]],
                        [[True, True, True], [True, True, True], [True, True, True]],
                    ],
                    dtype=torch.bool,
                ),
                "all_zeros_attention_mask",
            ),
        ],
    )
    def test_make_att_2d_masks_parameterized(
        self, pad_masks, att_masks, n_cross_att_tokens, expected_shape, expected_pattern, test_name
    ):
        """Test 2D attention mask creation with various scenarios."""
        if n_cross_att_tokens is not None:
            cross_att_pad_masks = torch.ones(pad_masks.shape[0], n_cross_att_tokens, dtype=torch.bool)
            result = make_att_2d_masks(pad_masks, att_masks, n_cross_att_tokens, cross_att_pad_masks)
        else:
            result = make_att_2d_masks(pad_masks, att_masks)

        # Check shape
        assert result.shape == expected_shape, f"Shape mismatch in {test_name}"

        # Check pattern
        assert torch.allclose(result, expected_pattern), f"Pattern mismatch in {test_name}"

        # Check that result is boolean
        assert result.dtype == torch.bool, f"Result should be boolean in {test_name}"

        # Check device consistency
        assert result.device == pad_masks.device, f"Device mismatch in {test_name}"

    def test_make_att_2d_masks_error_cases(self):
        """Test error cases for make_att_2d_masks."""
        batch_size = 2
        seq_len = 5

        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test with wrong input shapes (3D instead of 2D)
        with pytest.raises(ValueError):
            make_att_2d_masks(pad_masks.unsqueeze(0), att_masks)
        with pytest.raises(ValueError):
            make_att_2d_masks(pad_masks, att_masks.unsqueeze(0))

        # Test with mismatched batch sizes
        with pytest.raises(RuntimeError):
            make_att_2d_masks(
                torch.ones(2, seq_len, dtype=torch.bool), torch.ones(3, seq_len, dtype=torch.bool)
            )

        # Test with mismatched sequence lengths
        with pytest.raises(RuntimeError):
            make_att_2d_masks(
                torch.ones(batch_size, seq_len, dtype=torch.bool),
                torch.ones(batch_size, seq_len + 1, dtype=torch.bool),
            )

    def test_make_att_2d_masks_edge_cases(self):
        """Test edge cases for make_att_2d_masks."""
        # Test with empty sequence
        pad_masks = torch.ones(1, 0, dtype=torch.bool)
        att_masks = torch.ones(1, 0, dtype=torch.bool)

        result = make_att_2d_masks(pad_masks, att_masks)
        assert result.shape == (1, 0, 0)

        # Test with single token
        pad_masks = torch.ones(1, 1, dtype=torch.bool)
        att_masks = torch.ones(1, 1, dtype=torch.bool)

        result = make_att_2d_masks(pad_masks, att_masks)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0].item() is True

        # Test with zero cross attention tokens
        pad_masks = torch.ones(1, 2, dtype=torch.bool)
        att_masks = torch.ones(1, 2, dtype=torch.bool)

        result = make_att_2d_masks(pad_masks, att_masks)
        assert result.shape == (1, 2, 2)

    def test_apply_rope(self):
        """Test RoPE position application."""
        batch_size = 2
        seq_len = 5
        num_heads = 4
        head_dim = 8

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        result = apply_rope(x, positions)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

        # Test with different max_wavelength
        result = apply_rope(x, positions, max_wavelength=1000)
        assert result.shape == x.shape

    @run_with_fake_tensor
    def test_utility_functions_with_fake_tensor(self):
        """Test utility functions with FakeTensorContext."""
        time = torch.tensor([0.0, 0.5, 1.0])
        dimension = 10
        min_period = 0.1
        max_period = 1.0
        device = torch.device("cpu")

        pos_emb = create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device)
        assert pos_emb.shape == (time.shape[0], dimension)

        alpha = 1.5
        beta = 1.0
        bsize = 4
        result = sample_beta(alpha, beta, bsize, device)
        assert result.shape == (bsize,)

        batch_size = 2
        seq_len = 5
        pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        att_masks = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        assert att_2d_masks.shape == (batch_size, seq_len, seq_len)

    @pytest.mark.slow  # 6 sec
    @run_with_fake_tensor
    def test_init_model(self, tau0_training_config):
        """Test the _init_model method with different initialization strategies."""

        config = tau0_training_config.policy

        # Test no_init strategy
        config.init_strategy = "no_init"
        policy = TAU0Policy(config)

        # Mock the _init_weights method to track calls
        with patch.object(policy.model, "_init_weights") as mock_init_weights:
            policy.model._init_model()
            # Should not call _init_weights for no_init strategy
            mock_init_weights.assert_not_called()

        # Test full_he_init strategy
        config.init_strategy = "full_he_init"
        policy = TAU0Policy(config)

        with patch.object(policy.model, "_init_weights") as mock_init_weights:
            policy.model._init_model()
            # Should call _init_weights for all modules
            assert mock_init_weights.call_count > 0
            # Verify it was called for different module types
            call_args = [call[0][0] for call in mock_init_weights.call_args_list]
            module_types = [type(arg) for arg in call_args]
            assert any("Linear" in str(t) for t in module_types)
            assert any("LayerNorm" in str(t) for t in module_types)

        # Test invalid init strategy
        policy = TAU0Policy(config)
        config.init_strategy = "invalid_strategy"

        with pytest.raises(ValueError, match="Invalid init strategy: invalid_strategy"):
            policy.model._init_model()

    @pytest.mark.slow  # 64 sec
    def test_init_weights(self, tau0_training_config):
        """Test the _init_weights method for different module types."""
        config = tau0_training_config.policy
        policy = TAU0Policy(config)

        # Test Linear layer initialization
        linear_layer = torch.nn.Linear(2, 2)
        original_weight = linear_layer.weight.clone()
        original_bias = linear_layer.bias.clone()

        policy.model._init_weights(linear_layer)

        # Check that weights and bias were modified
        assert not torch.allclose(linear_layer.weight, original_weight)
        assert not torch.allclose(linear_layer.bias, original_bias)

        # Test LayerNorm initialization
        layer_norm = torch.nn.LayerNorm(2)
        # Modify the weights first so we can detect the change
        layer_norm.weight.data.fill_(0.5)  # Set to 0.5 instead of default 1.0
        layer_norm.bias.data.fill_(0.3)  # Set to 0.3 instead of default 0.0
        original_weight = layer_norm.weight.clone()
        original_bias = layer_norm.bias.clone()

        policy.model._init_weights(layer_norm)

        # Check that weights and bias were modified back to expected values
        assert torch.allclose(layer_norm.weight, torch.ones_like(layer_norm.weight))  # Should be ones
        assert torch.allclose(layer_norm.bias, torch.zeros_like(layer_norm.bias))  # Should be zeros
        # Verify they changed from our modified values
        assert not torch.allclose(layer_norm.weight, original_weight)
        assert not torch.allclose(layer_norm.bias, original_bias)

        # Test other module types (should not be modified)
        conv_layer = torch.nn.Conv2d(1, 1, 1)
        original_weight = conv_layer.weight.clone()
        original_bias = conv_layer.bias.clone()

        policy.model._init_weights(conv_layer)

        # Check that weights and bias were not modified
        assert torch.allclose(conv_layer.weight, original_weight)
        assert torch.allclose(conv_layer.bias, original_bias)


class TestTAU0Integration:
    """Integration tests for the complete TAU0 pipeline."""

    def _verify_pad_masks(self, prefix_pad_masks, suffix_pad_masks, inference_mode=False):
        """Verify the pad masks are correct. This assumes all images are not padded. Language embeddings and action chunks can be padded.

        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        suffix_pad_masks: tensor with shape (batch_size, seq_len)
        inference_mode: boolean indicating if the pad masks were created using the forward method (training) or select_action method (inference)
        """
        assert prefix_pad_masks.shape[0] == 1
        assert prefix_pad_masks.shape[1] == 608 if not inference_mode else 560
        assert prefix_pad_masks.dtype == torch.bool
        assert suffix_pad_masks.shape[0] == 1
        assert suffix_pad_masks.shape[1] == 151
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
            _check_ones_before_zeros(prefix_pad_masks[i, 512:560])  # prompt tokens
            if not inference_mode:
                _check_ones_before_zeros(prefix_pad_masks[i, 560:608])  # response tokens

            assert torch.all(
                suffix_pad_masks[i, :101] == 1
            )  # state and local image tokens should not be padded
            _check_ones_before_zeros(suffix_pad_masks[i, 101:151])  # action chunks

    def _verify_position_ids(
        self,
        prefix_position_ids,
        suffix_position_ids,
        prefix_pad_masks,
        suffix_pad_masks,
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
        assert prefix_position_ids.shape[1] == 608 if not inference_mode else 560
        assert prefix_position_ids.dtype == torch.long
        assert suffix_position_ids.shape[0] == 1
        assert suffix_position_ids.shape[1] == 151
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
                assert suffix_position_ids[i, 0] == prefix_position_ids[i, 560]
            else:
                assert suffix_position_ids[i, 0] == prefix_position_ids[i, 559] + 1

    def _verify_vlm_attention_mask(self, vlm_attention_mask, prefix_pad_masks, inference_mode=False):
        """Verify the VLM attention mask is correct.

        vlm_attention_mask: tensor with shape (batch_size, seq_len, seq_len)
        prefix_pad_masks: tensor with shape (batch_size, seq_len)
        inference_mode: boolean indicating if the attention mask were created using the forward method (training) or select_action method (inference)
        """
        assert vlm_attention_mask.shape[0] == 1
        assert vlm_attention_mask.shape[1] == 608 if not inference_mode else 560
        assert vlm_attention_mask.shape[2] == 608 if not inference_mode else 560
        assert vlm_attention_mask.dtype == torch.bool

        batch_size = vlm_attention_mask.shape[0]
        for i in range(batch_size):
            # construct correct attention mask
            # see diagram here: https://drive.google.com/file/d/1EvHrKK3w072zUuMpO2f81L_7HZW1r_YC/view?usp=sharing
            correct_vlm_attention_mask = torch.ones(608, 608, dtype=torch.bool)

            # pad tokens should not be attended to or attend to any other tokens
            num_non_padded_prompt_tokens = prefix_pad_masks[i, 512:560].sum()
            num_non_padded_response_tokens = prefix_pad_masks[i, 560:608].sum()
            prompt_start_idx, response_start_idx = 512, 560
            correct_vlm_attention_mask[
                prompt_start_idx + num_non_padded_prompt_tokens : response_start_idx, :
            ] = 0
            correct_vlm_attention_mask[
                :, prompt_start_idx + num_non_padded_prompt_tokens : response_start_idx
            ] = 0
            correct_vlm_attention_mask[response_start_idx + num_non_padded_response_tokens : 608, :] = 0
            correct_vlm_attention_mask[:, response_start_idx + num_non_padded_response_tokens : 608] = 0

            # nothing should attend to response tokens (other than response tokens)
            correct_vlm_attention_mask[
                :response_start_idx, response_start_idx : response_start_idx + num_non_padded_response_tokens
            ] = 0

            # response tokens should have a causal attention mask when attending to other response tokens
            # Create causal mask: each token can attend to itself and all previous tokens
            causal_mask = torch.tril(
                torch.ones(num_non_padded_response_tokens, num_non_padded_response_tokens, dtype=torch.bool)
            )
            correct_vlm_attention_mask[
                response_start_idx : response_start_idx + num_non_padded_response_tokens,
                response_start_idx : response_start_idx + num_non_padded_response_tokens,
            ] = causal_mask

            # response tokens are not used in inference
            if inference_mode:
                correct_vlm_attention_mask = correct_vlm_attention_mask[:560, :560]

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
        assert action_expert_attention_mask.shape[1] == 151
        assert action_expert_attention_mask.shape[2] == 711
        assert action_expert_attention_mask.dtype == torch.bool

        batch_size = action_expert_attention_mask.shape[0]
        for i in range(batch_size):
            # construct correct attention mask
            # see diagram here: https://drive.google.com/file/d/181ziRlMCst1hTptWrjl5axdKwj9HiNYI/view?usp=sharing
            correct_action_expert_attention_mask = torch.ones(151, 711, dtype=torch.bool)

            # pad tokens should not be attended to or attend to any other tokens
            num_non_padded_action_tokens = suffix_pad_masks[i, 101:151].sum()
            num_non_padded_prompt_tokens = prefix_pad_masks[i, 512:560].sum()
            action_start_idx, kv_cache_start_idx, prompt_start_idx = 101, 151, 663
            correct_action_expert_attention_mask[action_start_idx + num_non_padded_action_tokens :, :] = 0
            correct_action_expert_attention_mask[
                :, action_start_idx + num_non_padded_action_tokens : kv_cache_start_idx
            ] = 0
            correct_action_expert_attention_mask[:, prompt_start_idx + num_non_padded_prompt_tokens :] = 0

            # state and local image tokens should not attend to action tokens (assume action chunk of 50)
            correct_action_expert_attention_mask[
                :action_start_idx, action_start_idx : action_start_idx + 50
            ] = 0

            assert torch.all(
                action_expert_attention_mask[i].cpu() == correct_action_expert_attention_mask.cpu()
            )

    @pytest.mark.gpu
    @pytest.mark.slow  # ~1 mins
    def test_complete_tau0_pipeline_integration(self, tau0_training_config, lerobot_dataset_metadata):
        """Test the complete TAU0 pipeline from data loading to model execution."""

        # Initialize policy with unified training mode
        config = tau0_training_config.policy
        policy = TAU0Policy(config, dataset_stats=lerobot_dataset_metadata.stats)

        # Test data preparation pipeline
        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, 3, 224, 224),
            "camera1": torch.randn(batch_size, 3, 224, 224),
            "local_camera0": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, config.max_state_dim),
            "actions": torch.randn(batch_size, config.chunk_size, config.max_action_dim),
            "frozen_actions": torch.randn(batch_size, 0, config.max_action_dim),
            "prompt": ["Pick up the red block"],
            "response": ["I will pick up the red block"],
            "loss_type": ["MSE"],
            "img_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
            "local_img_is_pad": torch.zeros(batch_size, 1, dtype=torch.bool),
            "action_is_pad": torch.cat(
                [
                    torch.zeros(batch_size, config.chunk_size // 2, dtype=torch.bool),
                    torch.ones(batch_size, config.chunk_size - config.chunk_size // 2, dtype=torch.bool),
                ],
                dim=1,
            ),
            "frozen_action_is_pad": torch.zeros(batch_size, 0, dtype=torch.bool),
        }

        policy.to("cuda")
        batch_cuda = {
            key: value.to("cuda", non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        # Capture intermediate variables for inspection by monkey-patching the paligemma_with_expert forward method
        captured_variables = {}

        def capture_variables_forward(*args, **kwargs):
            # Extract the variables we want to capture from the kwargs
            vlm_attention_mask = kwargs.get("vlm_attention_mask")
            action_expert_attention_mask = kwargs.get("action_expert_attention_mask")
            vlm_position_ids = kwargs.get("vlm_position_ids")
            action_expert_position_ids = kwargs.get("action_expert_position_ids")

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
            result = original_embed_prefix(*args, **kwargs)
            prefix_embs, prefix_pad_masks, prefix_att_masks = result
            captured_variables["prefix_pad_masks"] = prefix_pad_masks.clone()
            return result

        def capture_embed_suffix(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            suffix_embs, suffix_pad_masks, suffix_att_masks = result
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

        def capture_variables_forward_select_action(*args, **kwargs):
            # Extract the variables we want to capture from the kwargs
            vlm_attention_mask = kwargs.get("vlm_attention_mask")
            action_expert_attention_mask = kwargs.get("action_expert_attention_mask")
            vlm_position_ids = kwargs.get("vlm_position_ids")
            action_expert_position_ids = kwargs.get("action_expert_position_ids")

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
            suffix_embs, suffix_pad_masks, suffix_att_masks = result
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
            inference_mode=True,
        )
        self._verify_vlm_attention_mask(
            captured_variables_select_action["vlm_2d_attention_mask"],
            captured_variables_select_action["prefix_pad_masks"],
            inference_mode=True,
        )
        self._verify_action_expert_attention_mask(
            captured_variables_select_action["action_expert_2d_attention_mask"],
            captured_variables_select_action["prefix_pad_masks"],
            captured_variables_select_action["suffix_pad_masks"],
        )

        assert action.shape == (1, policy.config.max_action_dim)
