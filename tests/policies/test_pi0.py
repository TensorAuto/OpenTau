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
from transformers.models.auto import CONFIG_MAPPING

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
# Validation lives in `PaliGemmaWithExpertConfig.__init__` (PretrainedConfig
# has no `__post_init__`, so a method by that name on a subclass would never
# run). Tests below cover both the constructor-time validation/warning paths
# and the runtime dispatch path.


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

    def test_bad_attention_implementation_raises(self):
        with pytest.raises(ValueError, match="attention_implementation"):
            PaliGemmaWithExpertConfig(attention_implementation="garbage")

    def test_fa2_falls_back_to_eager_with_warning(self, caplog):
        # Mirrors pi06's parallel test — fa2 has never been implemented in
        # PaliGemmaWithExpertModel, so the constructor accepts the value but
        # warns. The runtime dispatcher then maps fa2 → eager_attention_forward.
        with caplog.at_level("WARNING"):
            cfg = PaliGemmaWithExpertConfig(attention_implementation="fa2")
        assert cfg.attention_implementation == "fa2"
        assert any("falling back to 'eager'" in record.message for record in caplog.records)

    def test_train_expert_only_incompatible_with_unfrozen_vision(self):
        with pytest.raises(ValueError, match="not compatible"):
            PaliGemmaWithExpertConfig(freeze_vision_encoder=False, train_expert_only=True)


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
        # Dispatcher safety net — independent of whether the constructor
        # warning fired (e.g. for configs reloaded from a checkpoint where
        # __init__ validation isn't re-run).
        fake = self._fake_for_dispatch("fa2")
        assert PaliGemmaWithExpertModel.get_attention_interface(fake) == "<eager>"


# Gradient-checkpointing forward equivalence — locks in that the `_run_layer`
# extraction is bit-identical to the original inlined loop body. Mirrors
# pi06's `TestPi06GradCkptEquivalence::test_grad_ckpt_forward_matches_no_ckpt`.
#
# `PaliGemmaWithExpertConfig`'s ``elif isinstance(self.paligemma_config, dict)``
# branch is broken (refs an attribute set only inside the `if` branch above),
# so we can't pass tiny configs as dicts the way pi06 does. We instead build
# with default sub-configs and overwrite them post-hoc with tiny PaliGemma /
# Gemma configs before constructing the model.


def _make_tiny_pi0_engine_config():
    """Builds a `PaliGemmaWithExpertConfig` with tiny sub-configs for fast tests."""
    cfg = PaliGemmaWithExpertConfig(
        freeze_vision_encoder=False,
        train_expert_only=False,
        dropout=0.0,
    )
    text_kwargs = {
        "model_type": "gemma",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "torch_dtype": "float32",
        "hidden_activation": "gelu_pytorch_tanh",
    }
    vision_kwargs = {
        "model_type": "siglip_vision_model",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_image_tokens": 4,
        "patch_size": 14,
        "projection_dim": 32,
        "projector_hidden_act": "gelu_fast",
        "torch_dtype": "float32",
        "vision_use_head": False,
    }
    cfg.paligemma_config = CONFIG_MAPPING["paligemma"](
        bos_token_id=2,
        eos_token_id=1,
        hidden_size=32,
        image_token_index=127,
        pad_token_id=0,
        projection_dim=32,
        text_config=text_kwargs,
        vision_config=vision_kwargs,
    )
    cfg.gemma_expert_config = CONFIG_MAPPING["gemma"](
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=16,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        torch_dtype="float32",
        vocab_size=128,
    )
    return cfg


class TestPi0GradCkptEquivalence:
    def test_grad_ckpt_forward_matches_no_ckpt(self):
        """gradient_checkpointing=True must not change the forward output —
        it only trades activation memory for recompute. Locks in that
        ``_run_layer`` is bit-identical to the original inlined loop body.

        Runs in the engine's native bfloat16 (the default after
        ``to_bfloat16_like_physical_intelligence``) — both models do exactly
        the same op sequence on the same weights/inputs, so equality is bit-
        identical even at low precision.
        """
        cfg = _make_tiny_pi0_engine_config()

        torch.manual_seed(0)
        model_no_ckpt = PaliGemmaWithExpertModel(cfg)
        model_no_ckpt.train()

        cfg_ckpt = _make_tiny_pi0_engine_config()
        cfg_ckpt.gradient_checkpointing = True
        torch.manual_seed(0)
        model_ckpt = PaliGemmaWithExpertModel(cfg_ckpt)
        model_ckpt.load_state_dict(model_no_ckpt.state_dict(), strict=True)
        model_ckpt.train()

        batch, seq_len = 1, 4
        hidden_dim = cfg.paligemma_config.text_config.hidden_size
        # bf16 input matches the engine's internal hidden dtype after the
        # explicit `hidden_states.to(bfloat16)` cast in `_run_layer`.
        hidden = torch.randn(batch, seq_len, hidden_dim, dtype=torch.bfloat16)
        position_ids = torch.arange(seq_len)[None, :]
        attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None]

        out_a, _ = model_no_ckpt(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden, None],
            use_cache=False,
            fill_kv_cache=False,
        )
        out_b, _ = model_ckpt(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[hidden, None],
            use_cache=False,
            fill_kv_cache=False,
        )

        assert out_a[0] is not None and out_b[0] is not None
        # Identical op sequence + identical weights + dropout=0.0 ⇒ bit-equal.
        assert torch.equal(out_a[0], out_b[0])


class TestPI0ExecutionHorizon:
    """Regression coverage for ``n_action_steps`` as a short execution horizon.

    ``chunk_size`` is the trained prediction horizon (always decoded);
    ``n_action_steps`` (<= chunk_size) is how many actions are executed before
    re-querying. ``n_action_steps < chunk_size`` used to broadcast-crash the
    denoise/MSE paths. pi0 has no real-time-delay path; its analog of the guard
    is ``safety_buffer`` (the overlap-refill knob), and ``select_action`` refills
    via ``safety_buffer`` rather than a delay prefix.
    """

    MAX_STATE_DIM = 32
    MAX_ACTION_DIM = 32

    @classmethod
    def _config(cls, chunk_size=10, n_action_steps=3, safety_buffer=0):
        return PI0Config(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            safety_buffer=safety_buffer,
            max_state_dim=cls.MAX_STATE_DIM,
            max_action_dim=cls.MAX_ACTION_DIM,
        )

    def test_guard_rejects_short_horizon_with_safety_buffer(self):
        # A shortened execution horizon is not compatible with a non-zero safety
        # buffer; the config must reject it loudly.
        with pytest.raises(ValueError, match="safety_buffer"):
            self._config(chunk_size=10, n_action_steps=3, safety_buffer=2)

    def test_guard_allows_short_horizon_without_safety_buffer(self):
        cfg = self._config(chunk_size=10, n_action_steps=3, safety_buffer=0)
        assert (cfg.chunk_size, cfg.n_action_steps, cfg.safety_buffer) == (10, 3, 0)

    def test_guard_allows_full_horizon_with_safety_buffer(self):
        # n_action_steps == chunk_size keeps the safety-buffer overlap available.
        cfg = self._config(chunk_size=10, n_action_steps=10, safety_buffer=2)
        assert cfg.safety_buffer == 2

    def test_select_action_executes_first_n_then_requeries(self):
        """Model decodes the full ``chunk_size`` chunk, but ``select_action``
        executes only the first ``n_action_steps`` before re-querying.
        """
        chunk_size, n_steps, bsz = 10, 3, 2
        cfg = self._config(chunk_size=chunk_size, n_action_steps=n_steps, safety_buffer=0)

        policy = object.__new__(PI0Policy)
        policy.config = cfg
        policy.eval = lambda: None  # bypass nn.Module.eval (no __init__ was run)
        PI0Policy.reset(policy)

        calls = {"n": 0}

        def fake_sample_actions(batch, noise=None):
            # Return a full chunk_size chunk; element value encodes
            # (call_index * 1000 + timestep) so we can assert which timesteps run.
            calls["n"] += 1
            ts = torch.arange(chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
            return (calls["n"] * 1000 + ts).expand(bsz, chunk_size, self.MAX_ACTION_DIM).clone()

        policy.sample_actions = fake_sample_actions
        batch = {"state": torch.zeros(bsz, self.MAX_STATE_DIM)}

        # First n_steps actions all come from a single decode (call 1), in order.
        acts = [PI0Policy.select_action(policy, batch) for _ in range(n_steps)]
        assert calls["n"] == 1
        assert [tuple(a.shape) for a in acts] == [(bsz, self.MAX_ACTION_DIM)] * n_steps
        assert [a[0, 0].item() for a in acts] == [1000.0, 1001.0, 1002.0]

        # Queue drained after n_action_steps -> next call re-queries (call 2).
        a_next = PI0Policy.select_action(policy, batch)
        assert calls["n"] == 2
        assert a_next[0, 0].item() == 2000.0

    def test_embed_suffix_attn_mask_spans_full_chunk(self, monkeypatch):
        """Real-path guard the mocked select_action test above misses.

        ``embed_suffix`` must build the action attention mask over the full
        ``chunk_size``, not ``n_action_steps`` -- otherwise the ``att_masks`` and the
        ``chunk_size``-length ``pad_masks`` mismatch and ``make_att_2d_masks`` crashes
        for ``n_action_steps < chunk_size``. pi0 also builds its inference noise at
        ``chunk_size`` now (so the decoded chunk, this mask, and the ``denoise_step``
        output slice all align). pi0's ``embed_suffix`` hardcodes bfloat16; only the
        sinusoidal time embedding is stubbed, with real (tiny) projections so the
        state/action/time feature dims stay consistent through the concats.
        """
        import torch.nn as nn

        from opentau.policies.pi0.modeling_pi0 import PI0FlowMatching

        mod = "opentau.policies.pi0.modeling_pi0"
        monkeypatch.setattr(
            f"{mod}.create_sinusoidal_pos_embedding",
            lambda timestep, dim, **kw: torch.zeros(1, dim),
        )

        cfg = PI0Config(
            chunk_size=10,
            n_action_steps=3,
            safety_buffer=0,
            max_state_dim=self.MAX_STATE_DIM,
            max_action_dim=self.MAX_ACTION_DIM,
            proj_width=16,
        )
        pw, dt = cfg.proj_width, torch.bfloat16
        fm = object.__new__(PI0FlowMatching)
        nn.Module.__init__(fm)  # set up _modules so we can attach the stub layers
        fm.config = cfg
        fm.state_proj = nn.Linear(cfg.max_state_dim, pw).to(dt)
        fm.action_in_proj = nn.Linear(cfg.max_action_dim, pw).to(dt)
        fm.action_time_mlp_in = nn.Linear(pw * 2, pw).to(dt)
        fm.action_time_mlp_out = nn.Linear(pw, pw).to(dt)

        out = PI0FlowMatching.embed_suffix(
            fm,
            torch.zeros(1, cfg.max_state_dim, dtype=dt),
            torch.zeros(1, cfg.chunk_size, cfg.max_action_dim),
            torch.zeros(1),
        )
        pad_masks, att_masks = out[1], out[2]
        # one state token + chunk_size action tokens.
        assert pad_masks.shape[1] == att_masks.shape[1] == 1 + cfg.chunk_size
