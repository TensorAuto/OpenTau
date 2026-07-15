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

"""CPU smoke tests for the cosmos3_nano policy.

cosmos3_nano shares all modeling code with cosmos3 -- see ``test_cosmos3_cpu.py`` for the
full architectural coverage (masks, single-layer conditioning, freezing, per-sample loss,
prompt truncation, etc.). These tests lock the nano-specific surface only: factory/draccus
registration, the nano defaults (the Qwen3-VL-8B tower geometry), inheritance of the
config validators, a tiny end-to-end forward + determinism smoke through the
``Cosmos3NanoConfig`` / ``Cosmos3NanoPolicy`` plumbing, and the sub-1B expert parameter
budget at nano depth.
"""

import dataclasses
from pathlib import Path

import pytest
import torch
from transformers import Qwen3VLConfig

from opentau.policies.cosmos3.configuration_cosmos3 import Cosmos3Config
from opentau.policies.cosmos3.modeling_cosmos3 import Cosmos3FlowMatching, Cosmos3Policy
from opentau.policies.cosmos3_nano.configuration_cosmos3_nano import Cosmos3NanoConfig
from opentau.policies.cosmos3_nano.modeling_cosmos3_nano import Cosmos3NanoPolicy

# Small vision/text special-token ids that fit inside the tiny vocab below.
IMAGE_TOKEN_ID = 10
VIDEO_TOKEN_ID = 11
VISION_START_ID = 12
VISION_END_ID = 13
VOCAB = 200

CHUNK = 4
STATE_DIM = 5
MAX_ACTION_DIM = 8
MAX_STATE_DIM = 8


def _tiny_qwen3vl_config(num_hidden_layers: int = 2) -> Qwen3VLConfig:
    """A minimally-sized Qwen3-VL config (head_dim/kv-heads/layers must match the expert)."""
    return Qwen3VLConfig(
        text_config={
            "model_type": "qwen3_vl_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
        },
        vision_config={
            "model_type": "qwen3_vl",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "depth": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [1, 2],
            "num_position_embeddings": 256,
            "in_channels": 3,
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=False,
    )


def _tiny_nano_config(**overrides) -> Cosmos3NanoConfig:
    kwargs = {
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "max_action_dim": MAX_ACTION_DIM,
        "max_state_dim": MAX_STATE_DIM,
        "proj_width": 16,
        "num_steps": 3,
        "attention_implementation": "eager",
        "load_pretrained_backbone": False,
        "dropout": 0.0,
        "expert_hidden_size": 32,
        "expert_intermediate_size": 64,
        "expert_num_hidden_layers": 2,
        "expert_num_attention_heads": 4,
        "expert_num_key_value_heads": 2,
        "expert_head_dim": 16,
        "expert_adarms_cond_dim": 16,
    }
    kwargs.update(overrides)
    return Cosmos3NanoConfig(**kwargs)


def _text_batch(bsize: int = 2, n_text: int = 6):
    input_ids = torch.randint(20, VOCAB, (bsize, n_text))
    attention_mask = torch.ones(bsize, n_text, dtype=torch.long)
    state = torch.randn(bsize, STATE_DIM)
    state = torch.nn.functional.pad(state, (0, MAX_STATE_DIM - STATE_DIM))
    actions = torch.randn(bsize, CHUNK, MAX_ACTION_DIM)
    return input_ids, attention_mask, state, actions


def test_cosmos3_nano_factory_registration():
    from opentau.policies.factory import get_policy_class, make_policy_config

    cfg = make_policy_config("cosmos3_nano")
    assert isinstance(cfg, Cosmos3NanoConfig)
    assert cfg.type == "cosmos3_nano"

    policy_cls = get_policy_class("cosmos3_nano")
    assert policy_cls is Cosmos3NanoPolicy
    assert policy_cls.name == "cosmos3_nano"
    assert policy_cls.config_class is Cosmos3NanoConfig
    # nano is a strict specialization of cosmos3 (all modeling code is shared).
    assert issubclass(Cosmos3NanoPolicy, Cosmos3Policy)
    assert issubclass(Cosmos3NanoConfig, Cosmos3Config)


def test_cosmos3_nano_defaults_only_two_fields_differ():
    """nano == cosmos3 defaults except the backbone repo and the expert depth.

    The Nano reasoner keeps the exact KV interface the expert consumes (8 KV heads x
    head_dim 128 -- identical to the 32B tower), so no other default may drift; a third
    differing field means an unintended change to one of the packages.
    """
    nano, base = Cosmos3NanoConfig(), Cosmos3Config()
    # Field-name sets must match exactly, else the per-field loop below would silently
    # skip a field declared only on Cosmos3NanoConfig.
    assert {f.name for f in dataclasses.fields(Cosmos3NanoConfig)} == {
        f.name for f in dataclasses.fields(Cosmos3Config)
    }, "cosmos3_nano must not add new config fields silently -- extend this test if intentional"
    overridden = {"pretrained_backbone_repo_id", "expert_num_hidden_layers"}
    for f in dataclasses.fields(Cosmos3Config):
        if f.name in overridden:
            continue
        assert getattr(nano, f.name) == getattr(base, f.name), f"unexpected nano override: {f.name}"

    assert nano.pretrained_backbone_repo_id == "TensorAuto/cosmos3-reason-8b"
    assert nano.expert_num_hidden_layers == 36  # Qwen3-VL-8B text-tower depth
    assert nano.expert_num_key_value_heads == 8
    assert nano.expert_head_dim == 128


def test_cosmos3_nano_config_validation_inherited():
    with pytest.raises(ValueError):
        Cosmos3NanoConfig(attention_implementation="flash_cuda")
    with pytest.raises(ValueError):
        Cosmos3NanoConfig(n_action_steps=100, chunk_size=50)
    with pytest.raises(ValueError):
        # query heads not a multiple of kv heads
        Cosmos3NanoConfig(expert_num_attention_heads=15, expert_num_key_value_heads=8)


def test_cosmos3_nano_inner_forward_and_determinism():
    """Tiny forward through the nano config plumbing: loss-dict compat + bit-identical repeats."""
    torch.manual_seed(0)
    model = Cosmos3FlowMatching(_tiny_nano_config(), qwen3vl_config=_tiny_qwen3vl_config())
    input_ids, attention_mask, state, actions = _text_batch()
    noise = torch.randn_like(actions)
    time = torch.rand(actions.shape[0])

    losses = []
    for _ in range(2):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=None,
            state=state,
            actions=actions,
            noise=noise,
            time=time,
        )
        assert set(out.keys()) == {"MSE", "CE"}
        assert out["CE"].item() == 0.0
        assert torch.isfinite(out["MSE"])
        losses.append(out["MSE"].item())
    assert losses[0] == losses[1], f"non-deterministic loss: {losses}"


def test_cosmos3_nano_default_depth_exercises_build_validator():
    """The nano default expert depth (36) must build against a 36-layer backbone via the
    build-time depth-equality validator, and be rejected against any other depth."""
    torch.manual_seed(0)
    default_depth = Cosmos3NanoConfig().expert_num_hidden_layers
    cfg = _tiny_nano_config(expert_num_hidden_layers=default_depth)
    model = Cosmos3FlowMatching(cfg, qwen3vl_config=_tiny_qwen3vl_config(num_hidden_layers=default_depth))
    assert len(model.qwen3vl_with_expert.expert.layers) == 36
    with pytest.raises(ValueError, match="expert_num_hidden_layers"):
        Cosmos3FlowMatching(cfg, qwen3vl_config=_tiny_qwen3vl_config(num_hidden_layers=2))


def test_cosmos3_nano_example_config_parses():
    """The shipped example config must parse into a TrainPipelineConfig with a
    Cosmos3NanoConfig policy (a typo'd policy field would crash draccus here, offline,
    rather than at training launch)."""
    import draccus

    from opentau.configs.train import TrainPipelineConfig

    path = Path(__file__).resolve().parents[2] / "configs" / "examples" / "cosmos3_nano_training_config.json"
    cfg = draccus.parse(TrainPipelineConfig, config_path=str(path), args=[])
    assert isinstance(cfg.policy, Cosmos3NanoConfig)
    assert cfg.policy.type == "cosmos3_nano"
    assert cfg.policy.pretrained_backbone_repo_id == "TensorAuto/cosmos3-reason-8b"
    assert cfg.policy.expert_num_hidden_layers == 36
    assert cfg.batch_size == cfg.dataloader_batch_size * cfg.gradient_accumulation_steps


def test_cosmos3_nano_policy_wrapper_builds():
    """``Cosmos3NanoPolicy`` constructs offline with a tiny config and inherits the wrapper."""
    policy = Cosmos3NanoPolicy(_tiny_nano_config(), qwen3vl_config=_tiny_qwen3vl_config())
    assert policy.processor is None  # load_pretrained_backbone=False
    assert isinstance(policy.model, Cosmos3FlowMatching)
    assert isinstance(policy.config, Cosmos3NanoConfig)
    policy.reset()
    assert len(policy._action_queue) == 0


def test_cosmos3_nano_expert_under_1b_params():
    """The default (real) nano expert + projections must be < 1B trainable params.

    Same arithmetic as ``test_cosmos3_expert_under_1b_params``; at the nano depth (36
    layers vs 64) the default widths land around ~0.51B, so also pin a tighter bound to
    catch an accidental depth/width bump.
    """
    cfg = Cosmos3NanoConfig()  # real defaults; head geometry from Qwen3-VL-8B
    h = cfg.expert_hidden_size
    inter = cfg.expert_intermediate_size
    hq, hkv, hd = cfg.expert_num_attention_heads, cfg.expert_num_key_value_heads, cfg.expert_head_dim
    c = cfg.expert_adarms_cond_dim
    per_layer = (
        h * (hq * hd)  # q_proj
        + 2 * h * (hkv * hd)  # k_proj + v_proj
        + (hq * hd) * h  # o_proj
        + 2 * hd  # q_norm + k_norm
        + 3 * h * inter  # gated MLP
        + 2 * (c * 3 * h + 3 * h)  # two AdaRMS dense (weight + bias)
    )
    total = per_layer * cfg.expert_num_hidden_layers + (c * 3 * h + 3 * h)  # + final norm
    assert total < 1_000_000_000, f"expert has {total:,} params (>= 1B)"
    assert total < 600_000_000, f"nano expert unexpectedly large: {total:,} params"

    # Pin the analytic formula to the actual module: a 2-layer expert at the real
    # widths must count exactly per_layer*2 + final norm, so module drift (a new
    # sublayer, a bias change) fails this test instead of silently invalidating the
    # budget arithmetic above.
    from opentau.policies.cosmos3.qwen3vl_with_expert import Qwen3ActionExpert

    expert = Qwen3ActionExpert(
        num_hidden_layers=2,
        hidden_size=h,
        intermediate_size=inter,
        num_attention_heads=hq,
        num_key_value_heads=hkv,
        head_dim=hd,
        adarms_cond_dim=c,
        rms_norm_eps=cfg.expert_rms_norm_eps,
        dropout=0.0,
        attention_implementation="eager",
    )
    actual = sum(p.numel() for p in expert.parameters())
    expected = per_layer * 2 + (c * 3 * h + 3 * h)
    assert actual == expected, f"analytic formula drifted from Qwen3ActionExpert: {actual:,} != {expected:,}"
