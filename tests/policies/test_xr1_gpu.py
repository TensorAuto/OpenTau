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

"""GPU parity gates for ``xr1`` against the reference Xiaomi-Robotics-1 checkpoint.

These are the P6-P12 gates of the port's parity ladder, and they are the reason to trust
the port at all: because the backbone is byte-equivalent to stock ``transformers`` and the
DiT is a line-for-line translation, the default gate is **bit-identity**
(``torch.equal``), not a tolerance. A tolerance appears only where the docstring names the
op responsible.

| Gate | What it pins |
|------|--------------|
| P6   | The remap loads ``strict=True`` with zero missing / unexpected keys |
| P7   | Every parameter is bit-identical to the captured manifest |
| P8   | Our crop + patchify reproduce the reference's ``pixel_values_videos`` |
| P9   | All per-layer prefix K/V tensors match |
| P10  | Each Euler step's velocity matches, with the noise **and** ``x_in`` injected |
| P11  | The full 5-step chunk matches, injecting only the noise |
| P12  | A 250-step replay trace stays within the recorded tolerance |

Both the checkpoint and the heavy fixtures live outside the repo (multi-GB), so the tests
skip loudly when the two environment variables below are unset rather than silently
passing:

* ``XR1_REFERENCE_CHECKPOINT`` -- a local ``Xiaomi-Robotics-1-RoboCasa365`` directory.
* ``XR1_FIXTURE_DIR`` -- the ``--out-heavy`` directory of
  ``tests/artifacts/policies/xr1/capture_reference_fixtures.py``.

The JSON fixtures those tests read (manifest, prompt tokens, per-step fingerprints) are
in-repo, so a mismatch reports *which* tensor diverged without needing the heavy bundle.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from opentau.policies.xr1.configuration_xr1 import XR1Config
from opentau.policies.xr1.modeling_xr1 import XR1FlowMatching, resolve_qwen3vl_config
from opentau.policies.xr1.obs_adapter import adapt_state
from opentau.policies.xr1.processing_xr1 import (
    build_action_mask,
    center_crop_resize,
    expand_video_placeholders,
    patchify_videos,
    render_chat_text,
)
from opentau.policies.xr1.state_dict_remap import remap_reference_state_dict
from tests.utils import require_vram_gib

pytestmark = pytest.mark.gpu

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "policies" / "xr1"
CAMERA_LABELS = ("Left camera: ", "\nRight camera: ", "\nWrist camera: ")


def _golden(name: str):
    return json.loads((ARTIFACTS / name).read_text())


def _capture_module():
    """Import the capture script so the synthetic observation stream is single-sourced."""
    path = ARTIFACTS / "capture_reference_fixtures.py"
    spec = importlib.util.spec_from_file_location("xr1_capture_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_dir() -> Path:
    value = os.environ.get("XR1_REFERENCE_CHECKPOINT")
    if not value:
        pytest.skip(
            "XR1_REFERENCE_CHECKPOINT is unset; point it at a local "
            "Xiaomi-Robotics-1-RoboCasa365 directory to run the reference-parity gates."
        )
    return Path(value).expanduser()


def _fixture_dir() -> Path:
    value = os.environ.get("XR1_FIXTURE_DIR")
    if not value:
        pytest.skip(
            "XR1_FIXTURE_DIR is unset; point it at the --out-heavy directory produced by "
            "tests/artifacts/policies/xr1/capture_reference_fixtures.py."
        )
    return Path(value).expanduser()


def _require_matching_platform() -> None:
    """Skip (loudly) rather than loosen when the capture platform differs.

    Bit-identity is only meaningful against the same torch / CUDA / device: a different
    GPU picks different kernels, and a tolerance quietly substituted for equality would
    turn the whole ladder into a rounding check.
    """
    manifest = _golden("checkpoint_manifest.json")
    actual = (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    expected = (manifest["torch_version"], manifest["cuda_version"], manifest["device_name"])
    if actual != expected:
        pytest.skip(
            f"Fixtures were captured on {expected} but this box is {actual}. Bit-identity is "
            "kernel-dependent; re-capture the fixtures here rather than loosening the gate."
        )


def _load_reference_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Load the (sharded) reference safetensors into one dict, keys unchanged."""
    from safetensors.torch import load_file

    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(checkpoint / shard))
    return state_dict


def _real_config(checkpoint: Path) -> XR1Config:
    config = XR1Config(pretrained_backbone_repo_id=str(checkpoint), load_pretrained_backbone=True)
    config.validate_features()
    return config


@pytest.fixture(scope="module")
def reference_model():
    """The port, loaded with the reference weights, in eval mode on CUDA (bf16)."""
    require_vram_gib(20)
    _require_matching_platform()
    checkpoint = _checkpoint_dir()
    config = _real_config(checkpoint)
    qwen3vl_config = resolve_qwen3vl_config(str(checkpoint))
    model = XR1FlowMatching(config, qwen3vl_config=qwen3vl_config)
    # The inner module's children ARE the reference's top-level names, so it takes the raw
    # keys; `remap_reference_state_dict`'s `model.` prefix targets the XR1Policy wrapper
    # that holds this module as `self.model` (P6 exercises that path).
    state_dict = _load_reference_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # P6 lives in its own test; here just refuse to hand out a partially-loaded model.
    assert [k for k in missing if k != "vlm.lm_head.weight"] == [], missing[:10]
    assert unexpected == [], unexpected[:10]
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    return model


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_checkpoint_dir()))


def _sha256_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes() if t.dtype == torch.bool else t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# =====================================================================================
# P6 / P7 -- weights
# =====================================================================================


def test_p6_reference_checkpoint_loads_with_zero_missing_or_unexpected_keys():
    """One remap rule, applied to the whole checkpoint, with nothing left over.

    ``load_state_dict(strict=False)`` reports a wrongly-remapped key as *missing*, which is
    indistinguishable from a successful load until the numbers are wrong -- so the gate is
    the key sets, not the absence of an exception.
    """
    require_vram_gib(20)
    from opentau.policies.xr1.modeling_xr1 import XR1Policy
    from opentau.policies.xr1.state_dict_remap import assert_full_coverage

    checkpoint = _checkpoint_dir()
    config = _real_config(checkpoint)
    policy = XR1Policy(config, qwen3vl_config=resolve_qwen3vl_config(str(checkpoint)))
    state_dict = remap_reference_state_dict(_load_reference_state_dict(checkpoint))
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    # `model.vlm.lm_head.weight` is absent from the checkpoint because Qwen3-VL ties it to
    # the input embedding table; every other key must land, and nothing may be left over.
    assert_full_coverage(missing, unexpected)
    assert missing == ["model.vlm.lm_head.weight"], missing[:10]
    assert unexpected == []


def test_p7_every_parameter_is_bit_identical_to_the_manifest(reference_model):
    """Global sha256 over the sorted parameter bytes, reporting the first divergence."""
    manifest = _golden("checkpoint_manifest.json")
    state_dict = reference_model.state_dict()
    assert len(state_dict) == manifest["num_tensors"], (
        f"tensor count {len(state_dict)} != captured {manifest['num_tensors']}"
    )

    digest = hashlib.sha256()
    first_bad = None
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        entry = manifest["tensors"].get(name)
        if entry is None:
            first_bad = first_bad or f"{name}: not in the manifest"
        elif list(tensor.shape) != list(entry["shape"]) and first_bad is None:
            first_bad = f"{name}: shape {list(tensor.shape)} != {entry['shape']}"
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    assert first_bad is None, first_bad
    assert digest.hexdigest() == manifest["global_sha256"]


# =====================================================================================
# Scenario plumbing
# =====================================================================================


SCENARIOS = ["A_start", "B_midrun", "C_long_instruction", "D0", "D3", "D7"]


def _build_our_inputs(scenario: str, tokenizer, device):
    """Run the *port's* front end over the same synthetic observations the capture used."""
    capture = _capture_module()
    stream_seed, step, instruction = capture.SCENARIOS[scenario]
    images, states, _ = capture.build_history(stream_seed, step)

    # (num_cams, T, C, H, W) in [0, 1], camera order Left / Right / Wrist.
    frames = torch.stack(
        [torch.from_numpy(images[key]).permute(0, 3, 1, 2).float() / 255.0 for key in capture.CAMERA_KEYS]
    )
    cropped = center_crop_resize(frames, 0.95)
    pixel_values_videos, video_grid_thw = patchify_videos(cropped, 16, 2, 2)

    text = render_chat_text(instruction, CAMERA_LABELS)
    text = expand_video_placeholders(text, video_grid_thw, 2, [[0, 1, 2, 3]] * 3, video_fps=24.0)
    encoded = tokenizer([text], return_tensors="pt", padding=True, add_special_tokens=False)

    state = torch.from_numpy(states).float()  # (T, 14) -- already the adapted layout
    state = adapt_state(state[None], state_adapter="identity", state_token_dim=60)
    action_mask = build_action_mask(1, 16, 60, 12, device=device, dtype=torch.bfloat16)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        # Kept float32, as the reference video processor emits it; the cast to bf16 happens
        # at the model boundary (their serving loop does the same). Comparing a bf16 copy
        # against the float32 fixture would measure bf16's ULP, not the crop.
        "pixel_values_videos": pixel_values_videos.to(device),
        "video_grid_thw": video_grid_thw.to(device),
        "state": state.to(device=device, dtype=torch.bfloat16),
        "action_mask": action_mask,
    }


def _load_scenario(scenario: str, device):
    payload = torch.load(_fixture_dir() / f"scenario_{scenario}.pt", map_location=device, weights_only=False)
    return payload


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_p1_tokenized_prompt_is_bit_identical(scenario, tokenizer):
    """The port's own tokenizer output vs the reference processor's, ids and mask."""
    golden = _golden("prompt_tokens.json")[scenario]
    inputs = _build_our_inputs(scenario, tokenizer, device="cpu")
    assert inputs["input_ids"][0].tolist() == golden["input_ids"]
    assert inputs["attention_mask"][0].tolist() == golden["attention_mask"]
    assert inputs["video_grid_thw"].tolist() == golden["video_grid_thw"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_p8_pixel_values_videos_match_the_reference(scenario, tokenizer):
    """Our crop + patchify against the reference video processor's output.

    A tolerance rather than ``torch.equal``: the reference crops a uint8 ``PIL.Image`` and
    ours crops the float tensor with an 8-bit round-trip, so the two can differ by one
    8-bit level per pixel. One level is ``2/255`` after Qwen3-VL's ``(x - 0.5) / 0.5``
    normalization, which is the bound asserted here.
    """
    fixture = _load_scenario(scenario, device="cpu")
    reference = fixture["inputs"]["pixel_values_videos"].float()
    ours = _build_our_inputs(scenario, tokenizer, device="cpu")["pixel_values_videos"].float()
    assert ours.shape == reference.shape
    delta = (ours - reference).abs()
    # One 8-bit level is 2/255 after Qwen3-VL's ``(x - 0.5) / 0.5``. Measured on the capture
    # fixtures: the median pixel is exact, ~18 % differ by exactly one level, and nothing
    # differs by more -- PIL resamples uint8 through fixed-point coefficients while torch
    # uses float bilinear, and the two round apart on the boundary pixels. Running PIL per
    # frame on the eval path to close that gap would cost a CPU round-trip per camera per
    # step; `test_p8b_our_pixels_move_the_action_chunk_by_less_than_the_sim_resolves`
    # measures what the difference is actually worth downstream.
    one_level = 2.0 / 255.0
    assert delta.max().item() <= one_level + 1e-6, delta.max().item()
    fraction_off_by_a_level = (delta > one_level / 2).float().mean().item()
    assert fraction_off_by_a_level < 0.25, fraction_off_by_a_level


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_p9_prefix_kv_cache_is_bit_identical(reference_model, tokenizer, scenario):
    """All 72 cached tensors (36 layers x K/V) must match by content hash.

    Fed the reference's *own* ``pixel_values_videos`` so a P8 one-bit pixel difference
    cannot masquerade as a backbone divergence -- this gate is about the backbone.
    """
    fixture = _load_scenario(scenario, device="cuda")
    golden = _golden("prefix_kv_fingerprints.json")[scenario]
    inputs = fixture["inputs"]
    position_ids, _ = reference_model.get_rope_index(
        input_ids=inputs["input_ids"].cuda(),
        video_grid_thw=inputs["video_grid_thw"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
    )
    assert torch.equal(position_ids, fixture["vlm_position_ids"].cuda())

    with torch.no_grad():
        cached = reference_model.run_prefix(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            position_ids=position_ids,
            pixel_values_videos=inputs["pixel_values_videos"].to(device="cuda", dtype=torch.bfloat16),
            video_grid_thw=inputs["video_grid_thw"].cuda(),
        )
    assert len(cached) == golden["num_layers"]
    mismatched = [
        i
        for i, (key, value) in enumerate(cached)
        if _sha256_tensor(key) != golden["layers"][i]["k"]["sha256"]
        or _sha256_tensor(value) != golden["layers"][i]["v"]["sha256"]
    ]
    assert not mismatched, f"prefix KV diverged first at layer {mismatched[0]} (of {len(cached)})"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_p10_each_euler_step_velocity_is_bit_identical(reference_model, scenario):
    """Per-step ``v_k`` with **both** the noise and each step's ``x_in`` injected.

    Injecting ``x_in`` decouples the steps, so a divergence points at one DiT evaluation
    rather than at the accumulation -- which is what makes a failure bisectable.
    """
    fixture = _load_scenario(scenario, device="cuda")
    inputs = fixture["inputs"]
    with torch.no_grad():
        cached_kv, attn_mask, position_ids, state_embed = reference_model._run_prefix_and_geometry(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            pixel_values_videos=inputs["pixel_values_videos"].to(device="cuda", dtype=torch.bfloat16),
            video_grid_thw=inputs["video_grid_thw"].cuda(),
            state=inputs["state"].to(device="cuda", dtype=torch.bfloat16),
            n_prefix_rows=0,
            dtype=torch.bfloat16,
        )
        assert torch.equal(attn_mask, fixture["attn_mask"].cuda())
        action_mask = inputs["action_mask"].to(device="cuda", dtype=torch.bfloat16)
        position_embeds = reference_model.compute_rope(
            position_ids, dtype=action_mask.dtype, device=action_mask.device
        )
        for index, step in enumerate(fixture["steps"]):
            v = reference_model.dit_forward(
                noisy_action=step["x_in"].cuda(),
                t=step["t"].cuda(),
                action_mask=action_mask,
                state_embed=state_embed,
                position_embeds=position_embeds,
                past_key_values=cached_kv,
                attn_mask=attn_mask,
            )
            assert torch.equal(v, step["v"].cuda()), f"velocity diverged at Euler step {index}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_p11_full_chunk_is_bit_identical_with_injected_noise(reference_model, scenario):
    """The compounded 5-step rollout, injecting only the starting noise."""
    fixture = _load_scenario(scenario, device="cuda")
    inputs = fixture["inputs"]
    with torch.no_grad():
        chunk = reference_model.sample_actions(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            pixel_values_videos=inputs["pixel_values_videos"].to(device="cuda", dtype=torch.bfloat16),
            video_grid_thw=inputs["video_grid_thw"].cuda(),
            state=inputs["state"].to(device="cuda", dtype=torch.bfloat16),
            action_mask=inputs["action_mask"].to(device="cuda", dtype=torch.bfloat16),
            noise=fixture["noise"].cuda(),
        )
    assert torch.equal(chunk, fixture["actions"].cuda())


def test_p11_tau_schedule_is_the_quantized_ascending_one(reference_model):
    """The Euler loop builds tau in ``x``'s own dtype, so on bf16 it is
    ``[0, 0.2002, 0.4004, 0.6016, 0.8008]`` -- not the exact fifths.

    Building the schedule in float32 and casting afterwards gives different timestep
    embeddings, and nothing downstream complains.
    """
    golden = _golden("dit_step_fingerprints.json")["A_start"]
    taus = [step["tau"] for step in golden["steps"]]
    rebuilt = [float((torch.ones(1, dtype=torch.bfloat16) * step / 5).item()) for step in range(5)]
    assert taus == rebuilt
    assert taus != [0.0, 0.2, 0.4, 0.6, 0.8]  # the un-quantized schedule is a different one


def test_p12_replay_trace_stays_within_tolerance(reference_model):
    """The headline model-level gate: 16 chunks over a recorded observation stream.

    Replaying recorded observations decouples model parity from simulator chaos -- over a
    450-900-step RoboCasa episode a 1e-3 action difference diverges the trajectory even
    with a perfect port, so a sim success rate cannot be used to certify the model.
    """
    trace = torch.load(_fixture_dir() / "replay_trace.pt", map_location="cpu", weights_only=False)
    deltas = []
    with torch.no_grad():
        for record in trace:
            inputs = record["inputs"]
            chunk = reference_model.sample_actions(
                input_ids=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                pixel_values_videos=inputs["pixel_values_videos"].to(device="cuda", dtype=torch.bfloat16),
                video_grid_thw=inputs["video_grid_thw"].cuda(),
                state=inputs["state"].to(device="cuda", dtype=torch.bfloat16),
                action_mask=inputs["action_mask"].to(device="cuda", dtype=torch.bfloat16),
                noise=record["noise"].cuda(),
            )
            deltas.append((chunk[0, :, :12].float().cpu() - record["chunk"]).abs())
    stacked = torch.cat([d.flatten() for d in deltas])
    p99 = torch.quantile(stacked, 0.99).item()
    assert p99 <= 2e-3, f"replay p99 abs-delta {p99}"
    assert stacked.max().item() <= 1e-2, f"replay max abs-delta {stacked.max().item()}"


# =====================================================================================
# Training-path checks at full scale
# =====================================================================================


def test_use_cache_survives_gradient_checkpointing_at_full_scale():
    """The trap that returns ``past_key_values=None`` with only a ``warning_once``.

    Pinned on the tiny model too, but it is a *transformers* interaction -- worth one
    real-geometry run so a library upgrade that changes the predicate is caught.
    """
    require_vram_gib(24)
    checkpoint = _checkpoint_dir()
    config = _real_config(checkpoint)
    config.gradient_checkpointing = True
    config.train_expert_only = False
    model = XR1FlowMatching(config, qwen3vl_config=resolve_qwen3vl_config(str(checkpoint)))
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    input_ids = torch.randint(100, 1000, (1, 32), device="cuda")
    attention_mask = torch.ones(1, 32, dtype=torch.long, device="cuda")
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, video_grid_thw=None, attention_mask=attention_mask
    )
    cached = model.run_prefix(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    assert len(cached) == model.num_layers
    assert all(k.shape[2] == 32 for k, _ in cached)


def test_frozen_embedding_table_is_excluded_from_the_optimizer_params():
    """~389 M parameters; a fused AdamW handed them still allocates ~4.7 GB of state."""
    require_vram_gib(20)
    checkpoint = _checkpoint_dir()
    from opentau.policies.xr1.modeling_xr1 import XR1Policy

    policy = XR1Policy(_real_config(checkpoint), qwen3vl_config=resolve_qwen3vl_config(str(checkpoint)))
    embedding = policy.model.vlm.get_input_embeddings().weight
    assert not embedding.requires_grad
    trainable = policy.get_optim_params()
    assert not any(p is embedding for p in trainable)
    assert sum(p.numel() for p in trainable) < sum(p.numel() for p in policy.parameters())


def test_train_expert_only_gradients_reach_the_dit_and_not_the_backbone():
    """The one-GPU fine-tune mode: 0.60 B trainable, ~19 GB with the frozen weights."""
    require_vram_gib(24)
    checkpoint = _checkpoint_dir()
    config = _real_config(checkpoint)
    config.train_expert_only = True
    model = XR1FlowMatching(config, qwen3vl_config=resolve_qwen3vl_config(str(checkpoint)))
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    assert all(not p.requires_grad for p in model.vlm.parameters())
    assert all(p.requires_grad for p in model.dit.parameters())

    batch = _tiny_train_batch(model)
    out = model(**batch)
    out["MSE"].backward()
    assert model.dit.layers[0].attn.qkv_proj.weight.grad is not None
    assert model.action_output_layer.layers[0].weight.grad is not None
    assert all(p.grad is None for p in model.vlm.parameters())


def test_full_finetune_gradients_reach_the_vit_and_the_llm_mlp_but_not_the_embeddings():
    """The freezing contract at real geometry.

    The embedding table is the one deliberately-frozen tensor, and it is frozen for a
    memory reason (~4.7 GB of optimizer state) rather than a modelling one -- so it is
    worth asserting that everything *else* still receives gradient.
    """
    require_vram_gib(40)
    checkpoint = _checkpoint_dir()
    config = _real_config(checkpoint)
    config.train_expert_only = False
    config.freeze_vision_encoder = False
    config.gradient_checkpointing = True
    model = XR1FlowMatching(config, qwen3vl_config=resolve_qwen3vl_config(str(checkpoint)))
    model = model.to(device="cuda", dtype=torch.bfloat16).train()

    batch = _tiny_train_batch(model)
    out = model(**batch)
    out["MSE"].backward()
    assert model.vlm.model.visual.blocks[0].mlp.linear_fc1.weight.grad is not None
    assert model.text_model.layers[0].mlp.gate_proj.weight.grad is not None
    assert model.vlm.get_input_embeddings().weight.grad is None


def _tiny_train_batch(model):
    """A minimal text-only training batch at the model's real geometry."""
    config = model.config
    device = "cuda"
    seq = 24
    input_ids = torch.randint(100, 1000, (1, seq), device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, seq, dtype=torch.long, device=device),
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "state": torch.randn(
            1, config.n_obs_steps, config.state_token_dim, device=device, dtype=torch.bfloat16
        ),
        "actions": torch.randn(
            1, config.chunk_size, config.max_action_dim, device=device, dtype=torch.bfloat16
        ),
        "action_mask": build_action_mask(
            1, config.chunk_size, config.max_action_dim, 12, device=device, dtype=torch.bfloat16
        ),
    }


def test_opentau_task_groups_cover_the_references_task_set_exactly():
    """``atomic_seen | composite_seen | composite_unseen`` must equal ``target50``.

    The reference evaluates a ``target50`` task set that OpenTau has no name for; the three
    task groups stand in for it, and the substitution is only sound if the union matches in
    *both* directions. A missing task quietly shrinks the sweep; an extra one contaminates
    the headline average.
    """
    robocasa = pytest.importorskip("robocasa.utils.dataset_registry")
    from opentau.envs.robocasa import _resolve_tasks

    golden = _golden("scene_seeds.json")
    target50 = set(robocasa.TASK_SET_REGISTRY["target50"])
    union: set[str] = set()
    for group, expected_size in golden["task_group_sizes"].items():
        tasks, _ = _resolve_tasks(group)
        assert len(tasks) == expected_size, group
        union |= set(tasks)
    assert union == target50, (sorted(target50 - union), sorted(union - target50))


def test_reference_scene_seeds_are_reproduced_from_the_task_registry():
    """The seed lists shipped for the parity ladder must match ``7 + index * 50 + episode``
    with ``index`` read live out of ``target50`` -- so a registry reorder is caught here
    rather than by an inexplicable success-rate gap."""
    robocasa = pytest.importorskip("robocasa.utils.dataset_registry")

    golden = _golden("scene_seeds.json")
    tasks = list(robocasa.TASK_SET_REGISTRY["target50"])
    for task, index in golden["task_indices_in_target50"].items():
        assert tasks.index(task) == index, task
    for key, seed_list in golden["ladder_seed_lists"].items():
        task, count = key.rsplit("_n", 1)
        index = golden["task_indices_in_target50"][task]
        expected = [golden["base_seed"] + index * golden["num_trials"] + i for i in range(int(count))]
        assert [int(s) for s in seed_list.split(",")] == expected, key
