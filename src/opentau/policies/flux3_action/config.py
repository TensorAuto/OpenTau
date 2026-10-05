# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Explicit policy contract; training-loop configuration is a separate concern."""

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from .processing import packing

# Content streams of the full generative trunk, canonical order. The action policy feeds only ``video`` (predicted
# frames) and ``video_cond`` (the observed frame); the others get dummy tokens that never enter the joint attention,
# so a trunk without the image and audio streams predicts identical actions.
CONTENT_STREAMS = ("video", "video_cond", "image", "image_cond", "audio", "audio_cond")
REQUIRED_CONTENT_STREAMS = ("video", "video_cond")
ACTION_PARAMETERIZATIONS = ("absolute", "joint_delta")

# Inference settings the selected DROID checkpoint was chosen and benchmarked with: Cosmos UniPC,
# four denoising steps, inference timestep shift 5 (distinct from the training shift 42), guidance
# 4.0 on video tokens and 1.0 (none) on action tokens, and a 32-action open loop. They are not
# defaults: apply them explicitly, e.g. PolicyConfig(**DROID_INFERENCE_SETTINGS, ...).
DROID_INFERENCE_SETTINGS: dict[str, object] = {
    "sampler": "cosmos_unipc",
    "num_inference_steps": 4,
    "sampler_shift": 5.0,
    "guidance_scale": 4.0,
    "guidance_scale_action": 1.0,
    "n_action_steps": 32,
}


@dataclass
class PolicyConfig:
    action_dim: int = 8
    action_modality: str = "action"
    # How the cameras are composited onto the canvas: "droid" (three 360x640 cameras [wrist, left exterior,
    # right exterior]: wrist full-res on top, exteriors half-res below, reflect-padded), "single" (one resized
    # camera), "side_by_side" (two cameras, scene then wrist, each half the width) or "grid" (any number of
    # cameras, any resolutions, each resized into a cell of a near-square grid in camera_keys order). The first
    # three are the layouts our checkpoints were trained with; "grid" runs for any setup but no checkpoint was
    # trained on it. camera_keys name the batch streams ("images.<camera>") in layout order.
    camera_layout: str = "droid"
    camera_keys: tuple[str, ...] = ("images.wrist", "images.left", "images.right")
    canvas_hw: tuple[int, int] = (544, 736)
    chunk_size: int = 32
    n_action_steps: int = 32
    fps: float = 15.0
    action_scale: float = 2.0
    gripper_flip_dims: tuple[int, ...] = (-1,)
    # What the action tokens carry, relative to the dataset's action column. "absolute": the commands as
    # stored (DROID). "joint_delta": per-frame differences a[t] - a[t-1] on every channel except
    # absolute_action_dims, which stay absolute (SO-101: the gripper); a training window then needs the
    # command before its first one (``action_prev`` in the batch), and inference integrates the predicted
    # deltas onto the observed state (target[k] = state + sum(delta[:k + 1])), open loop.
    action_parameterization: str = "absolute"
    absolute_action_dims: tuple[int, ...] = ()
    # Range normalization of the action targets and of the state token, per channel:
    # x_norm = clip(2 (x - q01) / span - 1, -normalization_clip, normalization_clip) with span = q99 - q01, or 1
    # where that is at most 1e-6. {"q01": [...], "q99": [...]} with action_dim entries each, or None for the
    # identity (DROID radians and closed fractions need none). ``flux-action index-lerobot`` computes them; the
    # trainer fills empty fields from the index and refuses bounds that disagree with it. Exports carry them.
    action_normalization: dict[str, list[float]] | None = None
    state_normalization: dict[str, list[float]] | None = None
    normalization_clip: float = 6.0
    # Training-time camera dropout: per window, the probability that a camera's frames are replaced by a flat
    # mid-gray tile (level 128), drawn from the window's generator after the caption dropout; keys are
    # camera_keys entries. The SO-101 recipe blanks the wrist camera in 20% of the windows.
    camera_dropout: dict[str, float] = field(default_factory=dict)
    trunk_weights: str | None = None
    video_vae_id: str | None = None
    text_encoder_id: str | None = None
    dit_config: dict = field(default_factory=dict)
    # Content streams the DiT is built with (besides the action modality). None =
    # video/video_cond only. Explicit stream lists in existing exports are preserved on restore.
    content_streams: tuple[str, ...] | None = None
    # Seed of the xavier draw for fresh embodiment heads (their final layers are zero). The trainer sets it
    # to the run seed so every rank builds identical heads.
    head_init_seed: int = 0
    torch_dtype: str = "bfloat16"
    quantization: Literal[None, "fp8r"] = None
    # Reference/training attention backend; prepared BF16/FP8r inference always uses PyTorch SDPA.
    attn_mode: str = "torch"
    # torch.compile the frozen video VAE and text encoder (the inference-side components; the DiT is not compiled).
    compile_model: bool = False
    # Inference encodes the conditioning frame alone. False retains the repeated-frame reference path;
    # prepare_inference() always selects single-frame encoding. Training still encodes full videos.
    single_frame_encode: bool = True
    # Select the token layout explicitly; never infer it from the action dimensions.
    inference_profile: str = "default"
    # History layout shared by training and inference; exports own these settings.
    n_obs_steps: int = 1
    history_snapshots: int = 1
    condition_on_past_actions: bool = False
    video_position_fps: float = 24.0
    text_fixed_length: int = 320
    # No inference preset is selected by the training recipe. These are explicit choices.
    sampler: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    guidance_scale_action: float | None = None
    sampler_shift: float | None = None
    inference_seed: int = 0
    # Training recipe; inference does not use these settings.
    train_timestep_width: float = 0.75
    train_timestep_shift: float = 42.0
    separate_timesteps: bool = False
    video_logit_mean: float = 1.08
    video_logit_std: float = 1.0
    conditioning_noise_max: float = 0.0
    loss_reduction: str = "joint_tokens"
    action_channel_weights: list[float] | None = None
    action_loss_weight: float = 50.0
    video_loss_weight: float = 1.0
    # The winning GB200 run blanked the caption of 10% of the windows in its DROID loader (a hardcoded
    # CFG_DROPOUT_RATE = 0.1, empty string through the same chat template); its P_UC_ACTION_PREDICTION = 0.0
    # only switched off a second, packer-level dropout on top of that. Verified 2026-09-13.
    caption_dropout: float = 0.1
    augment: bool = True
    # Training windows encoded per video-VAE call (the VAE treats them independently; this trades
    # activation memory for fewer kernel launches).
    vae_batch_windows: int = 8
    # Peak trunk LR of the winning recipe: 2e-4 x LR_SCALE 0.96. Heads run at 5x.
    optimizer_lr: float = 1.92e-4
    optimizer_lr_heads_multiplier: float = 5.0

    def __post_init__(self):
        if self.inference_profile not in ("default", "history"):
            raise ValueError("inference_profile must be default or history")
        if not 1 <= self.history_snapshots <= self.n_obs_steps:
            raise ValueError("history_snapshots must fit n_obs_steps")
        if not math.isfinite(self.video_position_fps) or self.video_position_fps <= 0:
            raise ValueError("video_position_fps must be finite and positive")
        if not 1 <= self.text_fixed_length <= 8192:
            raise ValueError("text_fixed_length must be between 1 and 8192")
        if self.inference_profile != "history" and (self.n_obs_steps != 1 or self.condition_on_past_actions):
            raise ValueError("observation/past-command history requires the history inference profile")
        if self.inference_profile == "history" and self.gripper_flip_dims:
            raise ValueError("history profiles require checkpoint-native channels without gripper flips")
        if self.loss_reduction not in ("joint_tokens", "modalities"):
            raise ValueError("loss_reduction must be joint_tokens or modalities")
        if not 0 <= self.conditioning_noise_max <= 1:
            raise ValueError("conditioning_noise_max must be between zero and one")
        if (
            not math.isfinite(self.video_logit_mean)
            or not math.isfinite(self.video_logit_std)
            or self.video_logit_std < 0
        ):
            raise ValueError("video logit parameters must be finite with nonnegative std")
        if self.action_channel_weights is not None:
            weights = self.action_channel_weights
            if (
                len(weights) != self.action_dim
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not any(weights)
            ):
                raise ValueError(
                    "action_channel_weights must contain action_dim nonnegative finite weights, at least one positive"
                )
        self.camera_keys = tuple(self.camera_keys)
        self.canvas_hw = tuple(self.canvas_hw)
        self.gripper_flip_dims = tuple(self.gripper_flip_dims)
        assert self.action_dim >= 1 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.action_modality), (
            "invalid action modality"
        )
        assert self.action_modality not in (
            "video",
            "image",
            "text",
            "audio",
            "action_prediction",
        ), "reserved action modality"
        if self.content_streams is not None:
            self.content_streams = tuple(self.content_streams)
            unknown = [s for s in self.content_streams if s not in CONTENT_STREAMS]
            assert not unknown and len(set(self.content_streams)) == len(self.content_streams), (
                "invalid content streams"
            )
            assert all(s in self.content_streams for s in REQUIRED_CONTENT_STREAMS), (
                "missing required content streams"
            )
        assert self.camera_layout in packing.CAMERA_LAYOUTS, "invalid camera layout"
        expected = {"droid": 3, "single": 1, "side_by_side": 2}.get(self.camera_layout)  # grid: any number
        n_keys = len(self.camera_keys)
        assert not (
            n_keys == 0
            or len(set(self.camera_keys)) != n_keys
            or (expected is not None and n_keys != expected)
        ), "invalid camera keys"
        assert len(self.canvas_hw) == 2 and all(x >= 32 and x % 32 == 0 for x in self.canvas_hw), (
            "invalid canvas size"
        )
        assert self.camera_layout != "droid" or self.canvas_hw == (544, 736), "invalid DROID canvas"
        assert self.chunk_size >= 1 and (self.inference_profile == "history" or self.chunk_size % 4 == 0), (
            "invalid chunk size"
        )
        assert 1 <= self.n_action_steps <= self.chunk_size, "invalid action step count"
        assert all(math.isfinite(x) and x > 0 for x in (self.fps, self.action_scale)), "invalid action timing"
        assert all(-self.action_dim <= d < self.action_dim for d in self.gripper_flip_dims), (
            "invalid gripper dimension"
        )
        assert len({d % self.action_dim for d in self.gripper_flip_dims}) == len(self.gripper_flip_dims), (
            "duplicate gripper dimension"
        )
        self.absolute_action_dims = tuple(self.absolute_action_dims)
        assert self.action_parameterization in ACTION_PARAMETERIZATIONS, "invalid action parameterization"
        assert all(-self.action_dim <= d < self.action_dim for d in self.absolute_action_dims), (
            "invalid absolute action dimension"
        )
        assert len({d % self.action_dim for d in self.absolute_action_dims}) == len(
            self.absolute_action_dims
        ), "duplicate absolute action dimension"
        assert self.action_parameterization != "absolute" or not self.absolute_action_dims, (
            "absolute actions cannot mix delta dimensions"
        )
        if self.action_parameterization == "joint_delta":
            absolute = {d % self.action_dim for d in self.absolute_action_dims}
            assert all(d % self.action_dim in absolute for d in self.gripper_flip_dims), (
                "gripper dimensions must stay absolute"
            )
        for name in ("action_normalization", "state_normalization"):
            value = getattr(self, name)
            if value is None:
                continue
            assert set(value) == {"q01", "q99"}, "normalization requires q01 and q99"
            q01, q99 = (list(map(float, value[k])) for k in ("q01", "q99"))
            assert len(q01) == len(q99) == self.action_dim, "normalization width mismatch"
            assert all(math.isfinite(x) for x in q01 + q99), "nonfinite normalization bound"
            assert all(hi >= lo for lo, hi in zip(q01, q99, strict=True)), "reversed normalization bound"
            setattr(self, name, {"q01": q01, "q99": q99})
        assert math.isfinite(self.normalization_clip) and self.normalization_clip > 0, (
            "invalid normalization clip"
        )
        self.camera_dropout = dict(self.camera_dropout)
        assert self.torch_dtype in ("float32", "bfloat16"), "invalid torch dtype"
        assert self.attn_mode in ("torch", "flash", "cudnn"), "invalid attention mode"
        assert self.quantization in (None, "fp8r"), "invalid quantization"
        assert self.quantization is None or self.torch_dtype == "bfloat16", "FP8r requires BF16"

    def validate_training(self):
        """Training uses the reference model and its augmentation settings."""
        assert self.quantization is None, "training requires unquantized weights"
        assert 0 <= self.caption_dropout <= 1, "invalid caption dropout"
        assert self.vae_batch_windows >= 1, "invalid VAE batch size"
        for key, probability in self.camera_dropout.items():
            assert key in self.camera_keys, "unknown dropout camera"
            assert 0 <= probability <= 1, "invalid camera dropout"

    def validate_inference(self):
        """Inference uses the checkpoint's sampler settings, without training augmentation."""
        assert self.inference_profile in ("default", "history"), "invalid inference profile"
        assert self.sampler in ("euler", "cosmos_unipc"), "invalid sampler"
        assert self.num_inference_steps is not None and self.num_inference_steps >= 1, (
            "invalid inference step count"
        )
        assert self.guidance_scale is not None and math.isfinite(self.guidance_scale), (
            "invalid guidance scale"
        )
        assert self.sampler_shift is not None and math.isfinite(self.sampler_shift), "invalid sampler shift"
        assert self.sampler_shift > 0, "sampler shift must be positive"
        assert self.guidance_scale_action is None or math.isfinite(self.guidance_scale_action), (
            "invalid action guidance"
        )

    @property
    def conditioning_channels(self) -> int:
        """Width of the ``<modality>_cond`` stream: the state, plus the previous command under past-action history."""
        return self.action_dim * (2 if self.condition_on_past_actions else 1)

    @property
    def camera_order(self):
        return self.camera_keys

    @property
    def window_frames(self):
        return self.chunk_size + (self.n_obs_steps if self.inference_profile == "history" else 1)

    @property
    def latent_hw(self):
        return packing.latent_hw(self.camera_layout, self.canvas_hw)

    def to_dict(self):
        return asdict(self)
