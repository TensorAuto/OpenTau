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

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi0.modeling_pi0 import PI0Policy


@pytest.fixture
def pi0_config():
    config = PI0Config()
    config.advantage_threshold = 0.5
    config.tokenizer_max_length = 10

    # Mock input/output features
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    return config


@patch("opentau.policies.pi0.modeling_pi0.AutoTokenizer")
@patch("opentau.policies.pi0.modeling_pi0.PI0FlowMatching")
def test_prepare_language_advantage_conditioning(mock_flow_matching, mock_auto_tokenizer, pi0_config):
    # Setup mock tokenizer
    mock_tokenizer_instance = mock_auto_tokenizer.from_pretrained.return_value

    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
    }

    # Initialize policy
    policy = PI0Policy(pi0_config)

    # Setup batch
    batch = {
        "state": torch.tensor([0.0]),  # needed for device
        "prompt": ["task1", "task2"],
        "advantage": torch.tensor([0.8, 0.2]),
    }

    # Call method
    policy.prepare_language(batch)

    # Verify tokenizer call
    expected_texts = ["task1\nAdvantage: positive\n", "task2\nAdvantage: negative\n"]

    mock_tokenizer_instance.assert_called_with(
        expected_texts,
        padding="max_length",
        padding_side="right",
        max_length=pi0_config.tokenizer_max_length,
        return_tensors="pt",
    )


@patch("opentau.policies.pi0.modeling_pi0.AutoTokenizer")
@patch("opentau.policies.pi0.modeling_pi0.PI0FlowMatching")
def test_prepare_language_no_advantage(mock_flow_matching, mock_auto_tokenizer, pi0_config):
    # Setup mock tokenizer
    mock_tokenizer_instance = mock_auto_tokenizer.from_pretrained.return_value

    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
    }

    # Initialize policy
    pi0_config = deepcopy(pi0_config)
    pi0_config.advantage = "ignore"
    policy = PI0Policy(pi0_config)

    # Setup batch without advantage
    batch = {"state": torch.tensor([0.0]), "prompt": ["task1", "task2"]}

    policy.prepare_language(batch)

    # Verify tokenizer call - should just append \n if not present
    expected_texts = ["task1\n", "task2\n"]

    mock_tokenizer_instance.assert_called_with(
        expected_texts,
        padding="max_length",
        padding_side="right",
        max_length=pi0_config.tokenizer_max_length,
        return_tensors="pt",
    )


@patch("opentau.policies.pi0.modeling_pi0.AutoTokenizer")
@patch("opentau.policies.pi0.modeling_pi0.PI0FlowMatching")
def test_prepare_language_existing_newline(mock_flow_matching, mock_auto_tokenizer, pi0_config):
    # Test case where prompt already ends with newline
    mock_tokenizer_instance = mock_auto_tokenizer.from_pretrained.return_value
    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[1]]),
    }

    policy = PI0Policy(pi0_config)

    batch = {"state": torch.tensor([0.0]), "prompt": ["task1\n"], "advantage": torch.tensor([1.0])}

    policy.prepare_language(batch)

    expected_texts = ["task1\nAdvantage: positive\n"]
    mock_tokenizer_instance.assert_called_with(
        expected_texts,
        padding="max_length",
        padding_side="right",
        max_length=pi0_config.tokenizer_max_length,
        return_tensors="pt",
    )
