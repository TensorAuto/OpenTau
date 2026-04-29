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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.policies.pi0.configuration_pi0 import PI0Config
from opentau.policies.pi0.modeling_pi0 import PI0Policy
from opentau.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)


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


# Engine config — SDPA + gradient checkpointing knobs.
#
# Note: `PaliGemmaWithExpertConfig.__post_init__` is dead code in pi0 (and
# pi05) — `transformers.PretrainedConfig` has no `__post_init__`, so the
# `super().__post_init__()` call inside it always raises AttributeError if
# anyone tries to invoke it. The validation it contains never runs in
# production. The tests below only exercise the constructor + the dispatch
# path that actually executes at forward time.


class TestPaliGemmaWithExpertConfig:
    def test_default_attention_implementation_is_eager(self):
        cfg = PaliGemmaWithExpertConfig()
        assert cfg.attention_implementation == "eager"

    def test_default_gradient_checkpointing_is_false(self):
        cfg = PaliGemmaWithExpertConfig()
        assert cfg.gradient_checkpointing is False

    def test_sdpa_attention_implementation_stored(self):
        cfg = PaliGemmaWithExpertConfig(attention_implementation="sdpa")
        assert cfg.attention_implementation == "sdpa"

    def test_gradient_checkpointing_field_set_via_kwarg(self):
        cfg = PaliGemmaWithExpertConfig(gradient_checkpointing=True)
        assert cfg.gradient_checkpointing is True


class TestPi0ConfigSdpaCkpt:
    def test_default_gradient_checkpointing_is_false(self):
        assert PI0Config().gradient_checkpointing is False

    def test_gradient_checkpointing_can_be_enabled(self):
        assert PI0Config(gradient_checkpointing=True).gradient_checkpointing is True

    def test_attention_implementation_sdpa_passes_through(self):
        cfg = PI0Config(attention_implementation="sdpa")
        assert cfg.attention_implementation == "sdpa"


# SDPA equivalence — direct math test on `eager_attention_forward` vs `sdpa_attention_forward`.
#
# Constructing a real ``PaliGemmaWithExpertModel`` is too expensive for a unit
# test (the default config drives a multi-billion-parameter PaliGemma + Gemma
# expert), and the engine constructor's ``elif isinstance(self.paligemma_config,
# dict):`` branch is dead code, so a tiny config can't be threaded through.
# Both attention functions only read scalars off ``self.config.paligemma_config.
# text_config``, so we build a fake-self with just those scalars and fake
# Q/K/V tensors. This is the same equivalence relation pi05 validates with a
# 1k-step GPU loss-equivalence run (see PR #182).


def _make_fake_self_pi0(num_attention_heads: int, num_key_value_heads: int, head_dim: int):
    text_config = SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    paligemma_config = SimpleNamespace(text_config=text_config)
    config = SimpleNamespace(paligemma_config=paligemma_config)
    return SimpleNamespace(config=config)


class TestPi0SdpaEquivalence:
    @pytest.mark.parametrize(
        "num_att,num_kv,head_dim,seq_len",
        [
            (4, 2, 16, 16),  # GQA case
            (4, 4, 16, 16),  # MHA case (no group expansion)
        ],
    )
    def test_eager_vs_sdpa_close(self, num_att, num_kv, head_dim, seq_len):
        torch.manual_seed(0)
        fake = _make_fake_self_pi0(num_att, num_kv, head_dim)
        b = 2

        # Use float32 for tight tolerance — the math is the only thing we're
        # comparing; the bf16 cast happens upstream in the loop body.
        q = torch.randn(b, seq_len, num_att, head_dim, dtype=torch.float32)
        k = torch.randn(b, seq_len, num_kv, head_dim, dtype=torch.float32)
        v = torch.randn(b, seq_len, num_kv, head_dim, dtype=torch.float32)
        # Block-causal-ish mask: first half attends to all, second half causal
        mask = torch.zeros(b, seq_len, seq_len, dtype=torch.bool)
        mask[:, : seq_len // 2, :] = True
        for i in range(seq_len // 2, seq_len):
            mask[:, i, : i + 1] = True

        eager_out = PaliGemmaWithExpertModel.eager_attention_forward(
            fake, mask, b, head_dim, q.clone(), k.clone(), v.clone()
        )
        sdpa_out = PaliGemmaWithExpertModel.sdpa_attention_forward(
            fake, mask, b, head_dim, q.clone(), k.clone(), v.clone()
        )

        assert eager_out.shape == sdpa_out.shape
        # SDPA does fp32 accumulation in softmax; eager does explicit fp32
        # upcast. Match within tight float32 tolerance.
        assert torch.allclose(eager_out, sdpa_out, atol=1e-4, rtol=1e-4)

    @staticmethod
    def _fake_for_dispatch(impl: str):
        fake = SimpleNamespace(
            config=SimpleNamespace(attention_implementation=impl),
            sdpa_attention_forward="<sdpa>",
            eager_attention_forward="<eager>",
        )
        return fake

    def test_dispatcher_returns_sdpa_for_sdpa_impl(self):
        fake = self._fake_for_dispatch("sdpa")
        assert PaliGemmaWithExpertModel.get_attention_interface(fake) == "<sdpa>"

    def test_dispatcher_returns_eager_for_eager_impl(self):
        fake = self._fake_for_dispatch("eager")
        assert PaliGemmaWithExpertModel.get_attention_interface(fake) == "<eager>"

    def test_dispatcher_falls_back_to_eager_for_fa2(self):
        # The dispatch falls back to eager regardless of whether the dead-code
        # __post_init__ validation ever ran. The safety net is here in
        # get_attention_interface itself.
        fake = self._fake_for_dispatch("fa2")
        assert PaliGemmaWithExpertModel.get_attention_interface(fake) == "<eager>"
