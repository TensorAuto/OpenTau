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

"""End-to-end coverage for ``train_state_action_representation_only`` on the
*real* policy modules.

``test_state_action_representation_only_helper.py`` pins the helpers' semantics
against hand-rolled fakes. That is not enough on its own: the fakes cannot catch
a wrapper whose layout differs from the fake (pi07 reparents its decoder layers
into ``InterleavedDecoderLayer``, so eval-ing ``self.gemma3`` by name would miss
them), nor a policy that simply forgets to thread the flag into its with-expert
config — which is a silent no-op on the whole VLM side.

So these tests construct the actual ``*WithExpertModel`` wrappers and the actual
flow-matching modules with tiny configs, and assert the observed trainable set
and eval pinning.
"""

from __future__ import annotations

import logging

import draccus
import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING

from opentau.policies.factory import make_policy_config
from opentau.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig as PI0WithExpertConfig,
)
from opentau.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertModel as PI0WithExpertModel,
)
from opentau.policies.pi05 import modeling_pi05
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import PI05FlowMatching
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertConfig as PI05WithExpertConfig,
)
from opentau.policies.pi05.paligemma_with_expert import (
    PaliGemmaWithExpertModel as PI05WithExpertModel,
)
from opentau.policies.pi06.gemma3_with_expert import (
    Gemma3WithExpertConfig as PI06WithExpertConfig,
)
from opentau.policies.pi06.gemma3_with_expert import (
    Gemma3WithExpertModel as PI06WithExpertModel,
)
from opentau.policies.pi07.gemma3_with_expert import (
    Gemma3WithExpertConfig as PI07WithExpertConfig,
)
from opentau.policies.pi07.gemma3_with_expert import (
    Gemma3WithExpertModel as PI07WithExpertModel,
)

DISCRETE_PARAMS = {
    "da_head.bias",
    "da_head.weight",
    "discrete_action_embedding.weight",
}


# --------------------------------------------------------------------------- tiny configs


def _tiny_paligemma_sub_configs(cfg):
    """Overwrite a ``PaliGemmaWithExpertConfig``'s sub-configs with tiny ones.

    Mirrors ``_make_tiny_pi0_engine_config`` in ``test_pi0.py``: the config's
    ``elif isinstance(self.paligemma_config, dict)`` branch is broken, so tiny
    sub-configs have to be assigned after construction.
    """
    cfg.paligemma_config = CONFIG_MAPPING["paligemma"](
        _vocab_size=128,
        bos_token_id=2,
        eos_token_id=1,
        hidden_size=32,
        image_token_index=127,
        pad_token_id=0,
        projection_dim=32,
        text_config={
            "model_type": "gemma",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 128,
            "max_position_embeddings": 128,
            "hidden_activation": "gelu_pytorch_tanh",
            "use_adarms": False,
            "adarms_cond_dim": None,
            "num_image_tokens": 4,
        },
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_image_tokens": 4,
            "patch_size": 14,
            "projection_dim": 32,
            "vision_use_head": False,
            "image_size": 28,
        },
    )
    cfg.gemma_expert_config = CONFIG_MAPPING["gemma"](
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=128,
        hidden_activation="gelu_pytorch_tanh",
        use_adarms=True,
        adarms_cond_dim=16,
    )
    return cfg


_TINY_GEMMA3 = {
    "text_config": {
        "model_type": "gemma3_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "sliding_window": 2,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "query_pre_attn_scalar": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_activation": "gelu_pytorch_tanh",
        "sliding_window_pattern": 2,
        "torch_dtype": "float32",
        "layer_types": ["sliding_attention", "full_attention"],
    },
    "vision_config": {
        "model_type": "siglip_vision_model",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "patch_size": 14,
        "image_size": 448,
        "projection_dim": 32,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
        "torch_dtype": "float32",
        "layer_norm_eps": 1e-6,
    },
    "image_token_index": 127,
    "mm_tokens_per_image": 4,
    "boi_token_index": 125,
    "eoi_token_index": 126,
}

_TINY_EXPERT = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "head_dim": 16,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 16,
    "intermediate_size": 32,
    "max_position_embeddings": 512,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "use_adarms": True,
    "adarms_cond_dim": 16,
    "vocab_size": 128,
}


def _build_pi0_wrapper():
    cfg = _tiny_paligemma_sub_configs(
        PI0WithExpertConfig(
            freeze_vision_encoder=False,
            train_expert_only=False,
            train_state_action_representation_only=True,
            dropout=0.1,
        )
    )
    return PI0WithExpertModel(cfg)


def _build_pi05_wrapper():
    cfg = _tiny_paligemma_sub_configs(
        PI05WithExpertConfig(
            freeze_vision_encoder=False,
            train_expert_only=False,
            train_state_action_representation_only=True,
            discrete_action_vocab_size=16,
            dropout=0.1,
        )
    )
    return PI05WithExpertModel(cfg)


def _build_pi06_wrapper():
    return PI06WithExpertModel(
        PI06WithExpertConfig(
            gemma3_config=_TINY_GEMMA3,
            gemma_expert_config=_TINY_EXPERT,
            discrete_action_vocab_size=32,
            freeze_vision_encoder=False,
            train_expert_only=False,
            train_state_action_representation_only=True,
            dropout=0.1,
        )
    )


def _build_pi07_wrapper():
    return PI07WithExpertModel(
        PI07WithExpertConfig(
            gemma3_config=_TINY_GEMMA3,
            gemma_expert_config=_TINY_EXPERT,
            discrete_action_vocab_size=32,
            freeze_vision_encoder=False,
            train_expert_only=False,
            train_state_action_representation_only=True,
            dropout=0.1,
        )
    )


#: (label, builder, whether the wrapper owns a discrete-action pathway).
WRAPPERS = [
    ("pi0", _build_pi0_wrapper, False),
    ("pi05", _build_pi05_wrapper, True),
    ("pi06", _build_pi06_wrapper, True),
    ("pi07", _build_pi07_wrapper, True),
]
WRAPPER_IDS = [label for label, _, _ in WRAPPERS]


# --------------------------------------------------------------------------- wrapper half


@pytest.mark.parametrize(("label", "build", "has_discrete"), WRAPPERS, ids=WRAPPER_IDS)
def test_wrapper_trains_only_the_discrete_action_representation(label, build, has_discrete):
    """The VLM, the vision pathway (tower *and* multimodal projector) and the
    action expert are all frozen; only the discrete-action embedding + head remain."""
    model = build()

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    assert trainable == (DISCRETE_PARAMS if has_discrete else set())


@pytest.mark.parametrize(("label", "build", "has_discrete"), WRAPPERS, ids=WRAPPER_IDS)
def test_wrapper_pins_the_frozen_trunk_to_eval_before_any_train_call(label, build, has_discrete):
    """``__init__`` leaves modules in training mode. The frozen trunk must already
    be in eval when the constructor returns — an entry point that runs a forward
    without calling ``train()``/``eval()`` first would otherwise get dropout noise
    through a trunk that cannot learn from it."""
    model = build()

    leaking = [n for n, m in model.named_modules() if isinstance(m, nn.Dropout) and m.training]

    assert leaking == []


@pytest.mark.parametrize(("label", "build", "has_discrete"), WRAPPERS, ids=WRAPPER_IDS)
def test_wrapper_keeps_the_trunk_in_eval_across_train_calls(label, build, has_discrete):
    """``update_policy`` calls ``policy.train()`` every step, so the ``train()``
    override — not just ``set_requires_grad`` — has to re-pin the trunk."""
    model = build()

    model.train(True)

    leaking = [n for n, m in model.named_modules() if isinstance(m, nn.Dropout) and m.training]
    assert leaking == []
    if has_discrete:
        assert model.discrete_action_embedding.training
        assert model.da_head.training


@pytest.mark.parametrize(("label", "build", "has_discrete"), WRAPPERS, ids=WRAPPER_IDS)
def test_wrapper_freezes_the_multimodal_projector(label, build, has_discrete):
    """``freeze_vision_encoder`` freezes only the tower, so the projector is the
    classic miss. These wrappers are built with ``freeze_vision_encoder=False``,
    which makes the check meaningful: nothing but the flag can be freezing it."""
    model = build()

    projector_params = [
        p for n, p in model.named_parameters() if "multi_modal_projector" in n or "mm_input" in n
    ]

    assert projector_params, f"{label}: no projector parameters found — test is not checking anything"
    assert not any(p.requires_grad for p in projector_params)


@pytest.mark.parametrize(("label", "build", "has_discrete"), WRAPPERS, ids=WRAPPER_IDS)
def test_wrapper_freezes_the_language_model_head(label, build, has_discrete):
    """The VLM's ``lm_head`` is tied to the token embedding table and is never the
    discrete-action head. Freezing ``self.gemma3.model`` instead of ``self.gemma3``
    (or ``backbone.model`` instead of ``backbone``) would leave it trainable."""
    model = build()

    lm_head_params = [p for n, p in model.named_parameters() if "lm_head" in n]

    assert not any(p.requires_grad for p in lm_head_params)


def test_flag_off_leaves_the_wrapper_untouched():
    """The no-op guarantee: with the flag off, the trainable set is whatever it was
    before this flag existed."""
    cfg = _tiny_paligemma_sub_configs(
        PI05WithExpertConfig(
            freeze_vision_encoder=False,
            train_expert_only=False,
            train_state_action_representation_only=False,
            discrete_action_vocab_size=16,
            dropout=0.1,
        )
    )
    model = PI05WithExpertModel(cfg)

    assert all(p.requires_grad for p in model.parameters())


# --------------------------------------------------------------------------- outer half


class _StubTokenizer:
    """Minimal stand-in for the PaliGemma tokenizer.

    ``PI05FlowMatching.__init__`` otherwise calls
    ``AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")``, which is a
    live-Hub fetch — that would force these tests onto the ``network`` marker and
    off the gating CPU run. Only ``ensure_loc_tokens`` touches the tokenizer during
    construction, and it needs just ``__len__`` and ``add_tokens``.
    """

    def __init__(self):
        self._len = 257_024

    def __len__(self):
        return self._len

    def add_tokens(self, tokens, special_tokens=False):  # noqa: ARG002
        # PaliGemma already reserves the <locNNNN> ids, so a real tokenizer adds 0.
        return 0


@pytest.fixture
def build_pi05_flow_matching(monkeypatch):
    """Build a real ``PI05FlowMatching`` with tiny sub-configs.

    Without this the module builds the production 3B PaliGemma — ~65s per
    construction, which is why the rest of the pi05 CPU suite uses
    ``object.__new__(PI05FlowMatching)`` shells instead. Patching the config class
    the module resolves at construction time keeps the real wiring (so the flag's
    plumbing is genuinely exercised) while making the build cheap.
    """

    def _tiny_config(**kwargs):
        return _tiny_paligemma_sub_configs(PI05WithExpertConfig(**kwargs))

    monkeypatch.setattr(modeling_pi05, "PaliGemmaWithExpertConfig", _tiny_config)

    def _build(**overrides):
        # Continuous state by default so `state_proj` exists; the discrete-state
        # case overrides it to assert the opposite.
        overrides.setdefault("state_type", "continuous")
        cfg = PI05Config(**overrides)
        return PI05FlowMatching(cfg, discrete_action_vocab_size=16, language_tokenizer=_StubTokenizer())

    return _build


def test_policy_trains_exactly_the_state_action_representation(build_pi05_flow_matching):
    """The whole contract, on a real policy: both halves together."""
    model = build_pi05_flow_matching(train_state_action_representation_only=True)

    trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)

    assert trainable == [
        "action_in_proj.bias",
        "action_in_proj.weight",
        "action_out_proj.bias",
        "action_out_proj.weight",
        "paligemma_with_expert.da_head.bias",
        "paligemma_with_expert.da_head.weight",
        "paligemma_with_expert.discrete_action_embedding.weight",
        "state_proj.bias",
        "state_proj.weight",
    ]


def test_policy_freezes_the_time_mlps(build_pi05_flow_matching):
    """The time MLPs are hidden-size-only and condition the frozen expert; they are
    not part of the state/action representation. This is the line
    ``per_group_projection`` already draws."""
    model = build_pi05_flow_matching(train_state_action_representation_only=True)

    assert not any(p.requires_grad for p in model.time_mlp_in.parameters())
    assert not any(p.requires_grad for p in model.time_mlp_out.parameters())


def test_policy_freezes_the_modality_embeddings_when_enabled(build_pi05_flow_matching):
    model = build_pi05_flow_matching(train_state_action_representation_only=True, use_modality_embedding=True)

    assert not any(p.requires_grad for p in model.modality_embedding.parameters())
    assert not any(p.requires_grad for p in model.action_modality_embedding.parameters())


def test_policy_flag_off_is_a_no_op(build_pi05_flow_matching):
    """With the flag off, everything this mode would freeze is still trainable.

    Checked against the LLM backbone and the expert rather than the vision tower:
    ``freeze_vision_encoder`` defaults to True, so the tower is frozen either way
    and would make this assertion pass for the wrong reason.
    """
    model = build_pi05_flow_matching()

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    assert "time_mlp_in.weight" in trainable
    assert any(n.startswith("paligemma_with_expert.gemma_expert.") for n in trainable)
    assert any(n.startswith("paligemma_with_expert.paligemma.model.language_model.") for n in trainable)


def test_policy_backward_reaches_only_the_representation_params(build_pi05_flow_matching):
    """A real gradient step: the frozen trunk transmits gradient to the trainable
    projections without accumulating any of its own."""
    model = build_pi05_flow_matching(train_state_action_representation_only=True)
    model.train(True)

    actions = torch.randn(2, model.config.chunk_size, model.config.max_action_dim)
    model.action_out_proj(model.action_in_proj(actions)).sum().backward()

    assert model.action_in_proj.weight.grad is not None
    assert model.action_out_proj.weight.grad is not None
    assert model.time_mlp_in.weight.grad is None
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())


# --------------------------------------------------------------------------- config surface


def test_pi05_discrete_state_warns_that_there_is_no_state_proj(caplog):
    with caplog.at_level(logging.WARNING):
        PI05Config(train_state_action_representation_only=True, state_type="discrete")

    assert "no state_proj to train" in caplog.text


def test_pi05_discrete_state_has_no_state_proj_to_train(build_pi05_flow_matching):
    """pi06 and discrete-state pi05 have no ``state_proj`` at all: the mode still
    works, it just trains one projection fewer."""
    model = build_pi05_flow_matching(train_state_action_representation_only=True, state_type="discrete")

    trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)

    assert not any(n.startswith("state_proj") for n in trainable)
    assert "action_in_proj.weight" in trainable
    assert "paligemma_with_expert.discrete_action_embedding.weight" in trainable


# ------------------------------------------------------- pi07's two-sources-of-truth hazard


def _pi07_config(**overrides):
    return make_policy_config("pi07_low_level", **overrides)


def test_pi07_pushes_the_flag_into_vlm_config():
    """pi07 is the only policy whose with-expert config is a serialized *field*, so
    the flag has to reach ``vlm_config`` — ``Gemma3WithExpertModel.set_requires_grad``
    reads it from there, and a missing push-down is a silent no-op on the VLM side."""
    cfg = _pi07_config(train_state_action_representation_only=True)

    assert cfg.vlm_config.train_state_action_representation_only is True


def test_pi07_reloading_an_adapter_run_config_can_clear_the_flag():
    """Regression: the push-down must be two-way.

    ``vlm_config`` is serialized into the saved config.json, so after an adapter run
    it carries the flag. Reloading that checkpoint to fully finetune —
    ``--policy.train_state_action_representation_only=false`` — must clear BOTH
    halves. A True-only push-down would leave the VLM and expert frozen while the
    outer projections unfroze: a mode nobody asked for, with no error.
    """
    trained = _pi07_config(train_state_action_representation_only=True)
    payload = draccus.encode(trained)
    assert payload["vlm_config"]["train_state_action_representation_only"] is True

    payload["train_state_action_representation_only"] = False
    reloaded = draccus.decode(type(trained), payload)

    assert reloaded.train_state_action_representation_only is False
    assert reloaded.vlm_config.train_state_action_representation_only is False


def test_pi07_inner_only_override_is_cleared_with_a_warning(caplog):
    """Setting only ``--policy.vlm_config.train_state_action_representation_only=true``
    would freeze the trunk while leaving every outer parameter trainable. The
    policy-level flag wins, and the override is reported rather than silently dropped."""
    with caplog.at_level(logging.WARNING):
        cfg = _pi07_config(
            vlm_config=PI07WithExpertConfig(
                freeze_vision_encoder=True,
                train_expert_only=False,
                train_state_action_representation_only=True,
            )
        )

    assert cfg.vlm_config.train_state_action_representation_only is False
    assert "is cleared because the policy-level" in caplog.text


def test_pi07_normal_push_down_is_quiet(caplog):
    """The ordinary False -> True push-down must not log — a warning on every
    enabling run trains readers to ignore it."""
    with caplog.at_level(logging.WARNING):
        _pi07_config(train_state_action_representation_only=True)

    assert "is cleared because the policy-level" not in caplog.text


def test_pi07_rejects_conflicting_vlm_config_modes():
    """The sibling "train only X" flags live on ``vlm_config`` for pi07, so the
    shared validator (which reads the policy config) cannot see them."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _pi07_config(
            train_state_action_representation_only=True,
            # `freeze_vision_encoder=True` because the wrapper config independently
            # rejects unfreezing the tower under `train_expert_only`.
            vlm_config=PI07WithExpertConfig(freeze_vision_encoder=True, train_expert_only=True),
        )
