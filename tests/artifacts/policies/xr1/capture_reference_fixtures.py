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

r"""Capture golden fixtures from the reference Xiaomi-Robotics-1 checkpoint.

This script is **not** part of the OpenTau package (it lives under ``tests/artifacts``
so it never widens ``scripts/``'s AST-scanned entry-point surface -- see
``tests/policies/test_siglip_embedding_dtype.py``). It runs on a CUDA box that has the
reference checkpoint on disk and writes the JSON fixtures the ``xr1`` parity tests
assert against, plus the heavy tensors those tests fetch out of band.

Usage::

    python capture_reference_fixtures.py \
        --checkpoint ~/xiaomi_r1/ckpt_robocasa365 \
        --out-json   <repo>/tests/artifacts/policies/xr1 \
        --out-heavy  ~/xiaomi_r1/fixtures

Everything is driven off the reference code itself (``trust_remote_code=True``): the
prompt is assembled by *their* processor and the actions are produced by *their*
``MiBoTForActionGeneration.forward``. The only intervention is that
``torch.randn_like`` is monkeypatched for the duration of a rollout so the flow noise
is a recorded tensor rather than a draw off the global RNG -- OpenTau's
``sample_actions`` accepts an explicit ``noise=`` for exactly this reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

# --------------------------------------------------------------------------------------
# Reference constants, copied from eval_robocasa365/entry.py (Xiaomi-Robotics-1 @4da1db0)
# --------------------------------------------------------------------------------------
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
CAMERA_LABELS = ("Left camera: ", "\nRight camera: ", "\nWrist camera: ")
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12
OBS_HISTORY = 4
OBS_INTERVAL = 2
CROP_RATIO = 0.95
NUM_STEPS = 5
BASE_SEED = 7
NUM_TRIALS = 50


def sha256_tensor(t: torch.Tensor) -> str:
    """Stable content hash of a tensor: dtype + shape + raw little-endian bytes."""
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes())
    return h.hexdigest()


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    f = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": sha256_tensor(t),
        "mean": float(f.mean()),
        "std": float(f.std()) if f.numel() > 1 else 0.0,
        "min": float(f.min()),
        "max": float(f.max()),
        "abs_sum": float(f.abs().sum()),
    }


# --------------------------------------------------------------------------------------
# Reference helpers (verbatim from entry.py)
# --------------------------------------------------------------------------------------
def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(observation: dict[str, Any]) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(observation["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["state.end_effector_rotation_relative"]),
            np.asarray(observation["state.gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["state.base_position"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["state.base_rotation"]),
        ]
    ).astype(np.float32)
    assert state.shape == (14,), state.shape
    return state


def sample_history_indices(length_of_items: int, length: int, interval: int) -> list[int]:
    return [max(0, length_of_items - 1 - (length - 1 - i) * interval) for i in range(length)]


def center_crop(image: np.ndarray, crop_ratio: float):
    from PIL import Image

    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def build_messages(videos: dict[str, list], instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAMERA_LABELS[0]},
                {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                {"type": "text", "text": CAMERA_LABELS[1]},
                {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                {"type": "text", "text": CAMERA_LABELS[2]},
                {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                {
                    "type": "text",
                    "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


# --------------------------------------------------------------------------------------
# Deterministic synthetic observation stream
# --------------------------------------------------------------------------------------
def _smooth_frame(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """A low-frequency, image-like RGB frame (not white noise, which is far
    off-distribution for a ViT and would saturate its activations)."""
    coarse = rng.integers(0, 256, size=(8, 8, 3), dtype=np.int64).astype(np.float64)
    ys = np.linspace(0, 7, size)
    xs = np.linspace(0, 7, size)
    y0 = np.clip(np.floor(ys).astype(int), 0, 7)
    x0 = np.clip(np.floor(xs).astype(int), 0, 7)
    y1 = np.clip(y0 + 1, 0, 7)
    x1 = np.clip(x0 + 1, 0, 7)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
    bot = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
    img = top * (1 - wy) + bot * wy
    img = img + rng.normal(0.0, 6.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion in xyzw order."""
    q = rng.normal(size=4)
    return (q / np.linalg.norm(q)).astype(np.float64)


def synthetic_observation(step: int, stream_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng((stream_seed << 20) + step)
    obs: dict[str, Any] = {key: _smooth_frame(rng) for key in CAMERA_KEYS}
    obs["state.end_effector_position_relative"] = rng.uniform(-0.6, 0.6, size=3)
    obs["state.end_effector_rotation_relative"] = _random_unit_quat(rng)
    obs["state.gripper_qpos"] = rng.uniform(-0.04, 0.04, size=2)
    obs["state.base_position"] = rng.uniform(-1.5, 1.5, size=3)
    obs["state.base_rotation"] = _random_unit_quat(rng)
    return obs


# --------------------------------------------------------------------------------------
# Instrumented reference rollout
# --------------------------------------------------------------------------------------
class ReferenceRunner:
    def __init__(self, checkpoint: str, attn_implementation: str = "eager") -> None:
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
        self.model = (
            AutoModel.from_pretrained(
                checkpoint,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16,
            )
            .cuda()
            .to(torch.bfloat16)
            .eval()
        )
        self.attn_implementation = attn_implementation

    def build_inputs(self, image_history: dict[str, np.ndarray], state_history: np.ndarray, instruction: str):
        videos = {key: [center_crop(f, CROP_RATIO) for f in image_history[key]] for key in CAMERA_KEYS}
        state = np.zeros((1, state_history.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, : state_history.shape[-1]] = state_history
        inputs = self.processor.apply_chat_template(
            build_messages(videos, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=state,
            robot_type=ROBOT_TYPE,
        )
        return dict(inputs)

    def to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        dev, dt = self.model.device, self.model.dtype
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device=dev, dtype=dt) if v.is_floating_point() else v.to(device=dev)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def run(self, inputs: dict[str, Any], noise: torch.Tensor | None = None, capture_layers: bool = False):
        """Run the reference forward, returning the actions plus every intermediate the
        parity gates compare against."""
        model = self.model
        payload = self.to_device(inputs)
        state = payload.pop("state")
        action_mask = payload.pop("action_mask")

        cap: dict[str, Any] = {"dit_calls": [], "layer_hidden": {}}

        if noise is None:
            noise = torch.randn_like(action_mask)
        noise = noise.to(device=action_mask.device, dtype=action_mask.dtype)
        cap["noise"] = noise.clone()

        real_randn_like = torch.randn_like
        drawn = {"n": 0}

        def fake_randn_like(t, *a, **kw):
            drawn["n"] += 1
            assert t.shape == noise.shape and t.dtype == noise.dtype, (t.shape, t.dtype)
            return noise.clone()

        # Capture the DiT call boundary: the (masked) input, tau, the cache, the mask.
        real_dit_forward = model.dit_forward

        def spy_dit_forward(
            noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask
        ):
            rec = {
                "x_in": noisy_action.clone(),
                "t": t.clone(),
                "state_embed": state_embed.clone(),
            }
            if not cap["dit_calls"]:
                cap["attn_mask"] = attn_mask.clone()
                cap["position_embeds"] = (position_embeds[0].clone(), position_embeds[1].clone())
                cap["past_key_values"] = [(k.clone(), v.clone()) for k, v in past_key_values]
            out = real_dit_forward(
                noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask
            )
            rec["v"] = out.clone()
            cap["dit_calls"].append(rec)
            return out

        handles = []
        if capture_layers:
            for idx in (0, 17, 35):
                layer = model.dit.layers[idx]

                def make_hook(i):
                    def hook(_module, _inp, out):
                        if len(cap["dit_calls"]) == 0:  # first Euler step only
                            cap["layer_hidden"][i] = out.detach().clone()

                    return hook

                handles.append(layer.register_forward_hook(make_hook(idx)))

        # Capture the VLM's own outputs (position_ids / attention_mask are the two extra
        # fields their vendored Qwen3-VL adds on top of stock transformers).
        real_vlm_forward = model.vlm.forward

        def spy_vlm_forward(*a, **kw):
            out = real_vlm_forward(*a, **kw)
            cap["vlm_position_ids"] = out.position_ids.clone()
            cap["vlm_attention_mask"] = out.attention_mask.clone()
            return out

        torch.randn_like = fake_randn_like
        model.dit_forward = spy_dit_forward
        model.vlm.forward = spy_vlm_forward
        try:
            out = model(state=state, action_mask=action_mask, num_steps=NUM_STEPS, **payload)
        finally:
            torch.randn_like = real_randn_like
            model.dit_forward = real_dit_forward
            model.vlm.forward = real_vlm_forward
            for h in handles:
                h.remove()

        assert drawn["n"] == 1, f"expected exactly one randn_like draw, got {drawn['n']}"
        cap["actions"] = out.actions.clone()
        cap["action_mask"] = action_mask.clone()
        cap["state"] = state.clone()
        return cap


# --------------------------------------------------------------------------------------
# Scenario assembly
# --------------------------------------------------------------------------------------
def build_history(stream_seed: int, step: int) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    """Reproduce the reference's deque + ``sample_history`` for a stream at ``step``.

    The deque holds ``(OBS_HISTORY - 1) * OBS_INTERVAL + 1`` frames; at ``step`` it has
    seen ``step + 1`` observations, so its length is ``min(step + 1, maxlen)``.
    """
    maxlen = (OBS_HISTORY - 1) * OBS_INTERVAL + 1
    seen = step + 1
    buf_len = min(seen, maxlen)
    first_in_buffer = seen - buf_len  # absolute index of deque slot 0
    indices = sample_history_indices(buf_len, OBS_HISTORY, OBS_INTERVAL)
    abs_indices = [first_in_buffer + i for i in indices]
    obs = [synthetic_observation(i, stream_seed) for i in abs_indices]
    images = {key: np.ascontiguousarray(np.stack([o[key] for o in obs], axis=0)) for key in CAMERA_KEYS}
    states = np.ascontiguousarray(np.stack([observation_to_state(o) for o in obs], axis=0))
    return images, states, abs_indices


SCENARIOS = {
    # name: (stream_seed, step, instruction)
    "A_start": (1001, 0, "close the fridge"),
    "B_midrun": (1001, 300, "close the fridge"),
    "C_long_instruction": (
        2002,
        7,
        "pick up the mug from the counter and place it inside the microwave, then close the door",
    ),
    "D0": (3003, 0, "turn on the microwave"),
    "D1": (3003, 1, "turn on the microwave"),
    "D2": (3003, 2, "turn on the microwave"),
    "D3": (3003, 3, "turn on the microwave"),
    "D4": (3003, 4, "turn on the microwave"),
    "D5": (3003, 5, "turn on the microwave"),
    "D6": (3003, 6, "turn on the microwave"),
    "D7": (3003, 7, "turn on the microwave"),
}


def capture_static_fixtures(out_json: Path) -> None:
    """Fixtures that need neither GPU nor checkpoint (P3 crop, P4 tau/sinusoid, P5 history)."""
    # -- P5: history index tables -------------------------------------------------------
    history = {
        "length": OBS_HISTORY,
        "interval": OBS_INTERVAL,
        "buffer_maxlen": (OBS_HISTORY - 1) * OBS_INTERVAL + 1,
        "indices_by_buffer_length": {
            str(n): sample_history_indices(n, OBS_HISTORY, OBS_INTERVAL) for n in range(1, 11)
        },
    }
    (out_json / "history_indices.json").write_text(json.dumps(history, indent=2) + "\n")

    # -- P3: crop box + a sha256 over a fixed synthetic pattern --------------------------
    rng = np.random.default_rng(0xC0FFEE)
    pattern = _smooth_frame(rng)
    cropped = np.asarray(center_crop(pattern, CROP_RATIO), dtype=np.uint8)
    size = 256
    crop_side = int(size * CROP_RATIO)
    crop = {
        "crop_ratio": CROP_RATIO,
        "input_size": [size, size],
        "crop_side": crop_side,
        "box_top_left_h_w": [(size - crop_side) // 2, (size - crop_side) // 2, crop_side, crop_side],
        "resample": "PIL.Image.Resampling.BILINEAR",
        "input_sha256": hashlib.sha256(pattern.tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(cropped.tobytes()).hexdigest(),
        "output_mean": float(cropped.mean()),
        "note": (
            "int(256 * 0.95) = 243 is odd, so 6 columns are dropped on the left and 7 on the "
            "right -- the crop box is asymmetric. PIL BILINEAR (no antialias) is not "
            "interchangeable with torchvision antialias=True or cv2 INTER_LINEAR."
        ),
    }
    (out_json / "crop_resize.json").write_text(json.dumps(crop, indent=2) + "\n")

    # -- P4 (partial): tau schedule + the DiT timestep sinusoid --------------------------
    def ref_timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    taus = [step / NUM_STEPS for step in range(NUM_STEPS)]
    sin_t = torch.tensor(taus, dtype=torch.float32) * 1000.0
    emb = ref_timestep_embedding(sin_t)
    geometry = {
        "num_steps": NUM_STEPS,
        "dt": 1.0 / NUM_STEPS,
        "tau_schedule": taus,
        "tau_scale_before_sinusoid": 1000.0,
        "sinusoid": {
            "dim": 256,
            "max_period": 10000,
            "concat_order": ["cos", "sin"],
            "values": [[float(x) for x in row] for row in emb.tolist()],
        },
        "query_layout": ["sink", "state x 4", "noisy_action x 16"],
        "dit_query_length": 1 + OBS_HISTORY + 16,
        "self_mask_is_lower_triangular": True,
        "note": (
            "tau ASCENDS 0 -> 1 with dt = +1/num_steps (x <- x + v*dt); pi05/cosmos3 descend. "
            "The sinusoid is the classic DiT 1/10000^(2i/d) ladder with [cos, sin] concat, NOT "
            "opentau.policies.cosmos3.modeling_cosmos3.create_sinusoidal_pos_embedding."
        ),
    }
    (out_json / "dit_geometry_static.json").write_text(json.dumps(geometry, indent=2) + "\n")

    # -- P2: state builder, at float64 ---------------------------------------------------
    cases = []
    for i, seed in enumerate((11, 22, 33, 44, 55)):
        obs = synthetic_observation(i, seed)
        cases.append(
            {
                "raw": {
                    k: [float(x) for x in np.asarray(v).reshape(-1)]
                    for k, v in obs.items()
                    if k.startswith("state.")
                },
                "state14": [float(x) for x in observation_to_state(obs)],
                "agent_pos16": [
                    float(x)
                    for x in np.concatenate(
                        [
                            obs["state.base_position"],
                            obs["state.base_rotation"],
                            obs["state.end_effector_position_relative"],
                            obs["state.end_effector_rotation_relative"],
                            obs["state.gripper_qpos"],
                        ]
                    )
                ],
            }
        )
    # Degenerate quaternions the axis-angle conversion must survive.
    edge = {
        "identity_xyzw": [float(x) for x in quat_xyzw_to_axis_angle(np.array([0.0, 0.0, 0.0, 1.0]))],
        "negated_identity_xyzw": [float(x) for x in quat_xyzw_to_axis_angle(np.array([0.0, 0.0, 0.0, -1.0]))],
        "zero_quat": [float(x) for x in quat_xyzw_to_axis_angle(np.zeros(4))],
        "unnormalized": [float(x) for x in quat_xyzw_to_axis_angle(np.array([0.0, 0.0, 2.0, 2.0]))],
        "sign_flipped_pair": [
            [float(x) for x in quat_xyzw_to_axis_angle(np.array([0.1, 0.2, 0.3, -0.9]))],
            [float(x) for x in quat_xyzw_to_axis_angle(-np.array([0.1, 0.2, 0.3, -0.9]))],
        ],
    }
    (out_json / "state_from_env.json").write_text(
        json.dumps(
            {
                "agent_pos16_layout": [
                    "base_position(3)",
                    "base_rotation_xyzw(4)",
                    "end_effector_position_relative(3)",
                    "end_effector_rotation_relative_xyzw(4)",
                    "gripper_qpos(2)",
                ],
                "state14_layout": [
                    "ee_pos_rel(3)",
                    "ee_rot_rel_axis_angle(3)",
                    "gripper_qpos(2)",
                    "base_pos(3)",
                    "base_rot_axis_angle(3)",
                ],
                "state_token_dim": STATE_DIM,
                "cases": cases,
                "quaternion_edge_cases": edge,
            },
            indent=2,
        )
        + "\n"
    )

    # -- scene seeds: the reference's exact per-episode seed scheme -----------------------
    seeds = {
        "scheme": "episode_seed = base_seed + task_index * num_trials + episode_index",
        "base_seed": BASE_SEED,
        "num_trials": NUM_TRIALS,
        "env_constructor_seed": BASE_SEED,
        "note": (
            "task_index is the position of the task in "
            "robocasa.utils.dataset_registry.TASK_SET_REGISTRY['target50']. OpenTau's eval uses "
            "start_seed + episode_index, so feed the reference's seeds through cfg.eval.seed_list "
            "to make the scene sets identical by construction."
        ),
    }
    (out_json / "scene_seeds.json").write_text(json.dumps(seeds, indent=2) + "\n")


def capture_model_fixtures(runner: ReferenceRunner, out_json: Path, out_heavy: Path) -> None:
    # -- P6/P7: checkpoint manifest -------------------------------------------------------
    sd = runner.model.state_dict()
    names = sorted(sd)
    global_hash = hashlib.sha256()
    entries = {}
    for name in names:
        t = sd[name]
        entries[name] = {"shape": list(t.shape), "dtype": str(t.dtype)}
        global_hash.update(name.encode())
        global_hash.update(str(t.dtype).encode())
        global_hash.update(str(tuple(t.shape)).encode())
        global_hash.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    manifest = {
        "num_tensors": len(names),
        "global_sha256": global_hash.hexdigest(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "attn_implementation": runner.attn_implementation,
        "tensors": entries,
    }
    (out_json / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    prompt_fixtures: dict[str, Any] = {}
    dit_fixtures: dict[str, Any] = {}
    kv_fixtures: dict[str, Any] = {}

    for name, (stream_seed, step, instruction) in SCENARIOS.items():
        images, states, abs_indices = build_history(stream_seed, step)
        inputs = runner.build_inputs(images, states, instruction)

        ids = inputs["input_ids"]
        decoded = runner.processor.tokenizer.decode(ids[0].tolist())
        prompt_fixtures[name] = {
            "instruction": instruction,
            "stream_seed": stream_seed,
            "step": step,
            "history_absolute_indices": abs_indices,
            "num_tokens": int(ids.shape[1]),
            "input_ids": ids[0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "video_grid_thw": inputs["video_grid_thw"].tolist(),
            "decoded": decoded,
            "state": [[float(x) for x in row] for row in inputs["state"][0].float().tolist()],
            "state_dtype": str(inputs["state"].dtype),
            "action_mask_shape": list(inputs["action_mask"].shape),
            "action_mask_dtype": str(inputs["action_mask"].dtype),
            "action_mask_num_active": int(inputs["action_mask"].float().sum().item()),
            "action_mask_active_dims": sorted(
                {int(c) for c in inputs["action_mask"][0].float().nonzero()[:, 1].tolist()}
            ),
            "pixel_values_videos": tensor_stats(inputs["pixel_values_videos"]),
        }

        capture_layers = name in ("A_start", "C_long_instruction")
        cap = runner.run(inputs, noise=None, capture_layers=capture_layers)

        pos_ids = cap["vlm_position_ids"]
        dit_len = 1 + OBS_HISTORY + 16
        expected_offset = pos_ids.max(dim=-1)[0]  # (3, B)
        dit_fixtures[name] = {
            "vlm_position_ids_shape": list(pos_ids.shape),
            "vlm_position_ids_per_axis_max": expected_offset.tolist(),
            "dit_position_ids_first_row": [
                int(expected_offset[a, 0].item()) + 1 for a in range(pos_ids.shape[0])
            ],
            "dit_query_length": dit_len,
            "attn_mask_shape": list(cap["attn_mask"].shape),
            "attn_mask_sha256": sha256_tensor(cap["attn_mask"]),
            "attn_mask_self_block_is_tril": bool(
                torch.equal(
                    cap["attn_mask"][0, 0, :, -dit_len:].cpu(),
                    torch.tril(torch.ones(dit_len, dit_len, dtype=torch.bool)),
                )
            ),
            "noise": tensor_stats(cap["noise"]),
            "steps": [
                {
                    "tau": float(rec["t"][0, 0, 0].item()),
                    "x_in": tensor_stats(rec["x_in"]),
                    "x_in_padded_dims_are_nonzero": bool(
                        rec["x_in"][..., ACTION_DIM:].abs().sum().item() > 0
                    ),
                    "v": tensor_stats(rec["v"]),
                }
                for rec in cap["dit_calls"]
            ],
            "actions": tensor_stats(cap["actions"]),
            "actions_first_row": [float(x) for x in cap["actions"][0, 0, :ACTION_DIM].float().tolist()],
            "executed_chunk": [
                [float(x) for x in row] for row in cap["actions"][0, :, :ACTION_DIM].float().tolist()
            ],
        }
        if capture_layers:
            dit_fixtures[name]["layer_hidden"] = {
                str(k): tensor_stats(v) for k, v in cap["layer_hidden"].items()
            }

        kv_fixtures[name] = {
            "num_layers": len(cap["past_key_values"]),
            "layers": [
                {"index": i, "k": tensor_stats(k), "v": tensor_stats(v)}
                for i, (k, v) in enumerate(cap["past_key_values"])
            ],
        }

        torch.save(
            {
                "inputs": {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
                "noise": cap["noise"].cpu(),
                "actions": cap["actions"].cpu(),
                "attn_mask": cap["attn_mask"].cpu(),
                "vlm_position_ids": cap["vlm_position_ids"].cpu(),
                "vlm_attention_mask": cap["vlm_attention_mask"].cpu(),
                "past_key_values": [(k.cpu(), v.cpu()) for k, v in cap["past_key_values"]],
                "steps": [
                    {"t": r["t"].cpu(), "x_in": r["x_in"].cpu(), "v": r["v"].cpu()} for r in cap["dit_calls"]
                ],
                "layer_hidden": {k: v.cpu() for k, v in cap["layer_hidden"].items()},
            },
            out_heavy / f"scenario_{name}.pt",
        )
        print(f"[capture] scenario {name}: {ids.shape[1]} tokens, actions {tuple(cap['actions'].shape)}")

    (out_json / "prompt_tokens.json").write_text(json.dumps(prompt_fixtures, indent=2) + "\n")
    (out_json / "dit_step_fingerprints.json").write_text(json.dumps(dit_fixtures, indent=2) + "\n")
    (out_json / "prefix_kv_fingerprints.json").write_text(json.dumps(kv_fixtures, indent=2) + "\n")


def capture_replay_trace(runner: ReferenceRunner, out_json: Path, out_heavy: Path, steps: int) -> None:
    """Scenario E: a long open-loop replay over a fixed observation stream.

    Each chunk is produced from a *recorded* observation window, so the trace decouples
    model parity from simulator chaos (P12).
    """
    stream_seed = 4004
    instruction = "close the fridge"
    records = []
    heavy = []
    for step in range(0, steps, 16):
        images, states, abs_indices = build_history(stream_seed, step)
        inputs = runner.build_inputs(images, states, instruction)
        noise = torch.randn(
            inputs["action_mask"].shape, generator=torch.Generator().manual_seed(90000 + step)
        ).to(dtype=inputs["action_mask"].dtype)
        cap = runner.run(inputs, noise=noise)
        chunk = cap["actions"][0, :, :ACTION_DIM].float().cpu()
        records.append(
            {
                "step": step,
                "history_absolute_indices": abs_indices,
                "num_tokens": int(inputs["input_ids"].shape[1]),
                "chunk_sha256": sha256_tensor(chunk),
                "chunk": [[float(x) for x in row] for row in chunk.tolist()],
            }
        )
        heavy.append(
            {
                "step": step,
                "inputs": {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
                "noise": cap["noise"].cpu(),
                "chunk": chunk,
            }
        )
    torch.save(heavy, out_heavy / "replay_trace.pt")
    (out_json / "replay_trace.json").write_text(
        json.dumps(
            {
                "stream_seed": stream_seed,
                "instruction": instruction,
                "obs_history": OBS_HISTORY,
                "obs_interval": OBS_INTERVAL,
                "replan_steps": 16,
                "num_chunks": len(records),
                "noise_generator": "torch.Generator().manual_seed(90000 + step), cast to bfloat16",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[capture] replay trace: {len(records)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="~/xiaomi_r1/ckpt_robocasa365")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-heavy", required=True)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--replay-steps", type=int, default=256)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json).expanduser()
    out_heavy = Path(args.out_heavy).expanduser()
    out_json.mkdir(parents=True, exist_ok=True)
    out_heavy.mkdir(parents=True, exist_ok=True)

    capture_static_fixtures(out_json)
    print(f"[capture] static fixtures -> {out_json}")
    if args.static_only:
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    runner = ReferenceRunner(str(Path(args.checkpoint).expanduser()), args.attn_implementation)
    capture_model_fixtures(runner, out_json, out_heavy)
    capture_replay_trace(runner, out_json, out_heavy, args.replay_steps)
    print("[capture] done")


if __name__ == "__main__":
    main()
