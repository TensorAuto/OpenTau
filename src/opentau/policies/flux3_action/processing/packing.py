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
"""Observation window -> the exact DiT inputs, plus the flow-matching loss.

DROID defaults and reusable token construction. The policy config overrides camera layout, canvas,
action dimensions, chunk length, frame rate and representation scale. Recipe fidelity and limits are described
in docs/droid-finetune.md.

Pipeline for one training window (``chunk + 1`` frames: 1 conditioning + ``chunk`` predicted):

    cameras uint8 (n_cams, T, 3, H, W)
      -> materialize_video      (augment, compose one canvas, [-1, 1])
      -> encode_videos          (video VAE, pad to 45 frames, crop the latent to the content)
      -> pack_video             (latent frame 0 = x_video_cond, frames 1.. = x_video)
    state[s], action[s .. s+chunk-1]
      -> pack_actions           (x action_scale, audio-style ids on the shared 10 ms clock)
    caption
      -> text context           (see text_encoder.text_context), pack_text
    -> sample_timesteps + add_noise (one t per sample, shared by video and action)
    -> build_forward_kwargs / flow_loss

The steps above are batched over the windows of a micro-batch ``(B, ...)``. ``pack_windows`` then
concatenates the ``B`` windows into one sequence of batch size 1 with per-window token counts, the
form the DiT consumes in training (``JointSingleSeq.forward(seqlens=...)``): one forward per
micro-batch, attention confined to each window, captions of any length side by side.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from einops import repeat
from torch import Tensor

from ..models.positional import batched_prc_action, batched_prc_txt, batched_prc_vid, times_to_ids

# ---- DROID reference recipe constants -------------------------------------------------------
FPS = 15.0
CHUNK = 32  # predicted actions per window
CAMERA_HW = (360, 640)  # DROID cameras
COMPOSITE_HW = (540, 640)  # wrist full-res on top, two exteriors half-res below
CANVAS_HW = (544, 736)  # Cosmos 480p 4:3 bucket; reflect-pad right 96 / bottom 4
LATENT_HW = (17, 20)  # post-VAE crop for the DROID composite (padded latent columns dropped)
LATENT_CHANNELS = 96
SPATIAL_DOWNSAMPLE = 32
TEMPORAL_DOWNSAMPLE = 4
VAE_CHUNK_FRAMES = 45  # the video VAE encodes 45-frame chunks
VAE_PAD_VALUE = -1.0  # short clips are padded with black to the chunk length
ACTION_DIM = 8  # DROID: 7 joints + gripper
ACTION_SCALE = 2.0  # representation scale, applied before BOTH action embedders
TIMESTEP_WIDTH = 0.75  # logit-logistic scale
TIMESTEP_SHIFT = 42.0  # rational time shift ("alpha") of the training timestep distribution
VIDEO_LOSS_WEIGHT = 1.0
# Joint token mean: ONE mean over all target tokens of a window (2720 video + 32 action for DROID),
# with this weight on the action tokens. The base value 100 undoes the dilution of 32 action tokens
# among 2752 (2720 / 32 = 85, rounded up to tilt toward actions) and is divided once by the
# representation scale. Effective per-modality ratio: 32 * 50 / 2720 = 0.588. A per-modality-pooled
# loss would need 0.588 here, not 50.
ACTION_LOSS_WEIGHT = 100.0 / ACTION_SCALE
TEXT_PAD_MULTIPLE = 80
VEC_DIM = 768
AUG_CROP_FRACTION = 0.95
AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION, AUG_HUE = 0.3, 0.4, 0.5, 0.08

CAMERA_LAYOUTS = ("droid", "single", "side_by_side", "grid")

assert math.isclose(ACTION_LOSS_WEIGHT, 50.0), ACTION_LOSS_WEIGHT


def latent_frames(num_frames: int, temporal_downsample: int = TEMPORAL_DOWNSAMPLE) -> int:
    return 1 + (num_frames - 1) // temporal_downsample


def padded_chunk_length(num_frames: int, chunk: int, overlap: int = 1) -> int:
    """Smallest ``T' >= num_frames`` satisfying ``T' = chunk + n * (chunk - overlap)``."""
    stride = chunk - overlap
    n = max(0, -(-(num_frames - chunk) // stride))  # ceil division
    return chunk + n * stride


def content_hw(layout: str, canvas_hw: tuple[int, int]) -> tuple[int, int]:
    """Pixels of the canvas that carry image content (the rest is reflect padding)."""
    if layout == "droid":
        return COMPOSITE_HW
    if layout in ("single", "side_by_side", "grid"):
        return tuple(canvas_hw)
    raise ValueError(f"unknown camera layout {layout!r}; choose from {CAMERA_LAYOUTS}")


def latent_hw(layout: str, canvas_hw: tuple[int, int]) -> tuple[int, int]:
    """Latent grid kept after the VAE: ceil(content / 32); DROID composite 540x640 -> 17x20."""
    h, w = content_hw(layout, canvas_hw)
    return (-(-h // SPATIAL_DOWNSAMPLE), -(-w // SPATIAL_DOWNSAMPLE))


# ---- 1. pixels ------------------------------------------------------------------------------
def sample_augmentation(
    generator: torch.Generator | None = None, camera_hw: tuple[int, int] = CAMERA_HW
) -> dict:
    """Draw the Cosmos crop + color-jitter parameters for one window.

    One draw is shared by all cameras and all frames (so the model cannot read the augmentation as
    motion). Draw order mirrors torchvision's ColorJitter: crop top, crop left, function permutation,
    then the factors.
    """
    h, w = camera_hw
    crop_h, crop_w = int(h * AUG_CROP_FRACTION), int(w * AUG_CROP_FRACTION)
    top = int(torch.randint(0, h - crop_h + 1, (), generator=generator))
    left = int(torch.randint(0, w - crop_w + 1, (), generator=generator))
    order = [int(i) for i in torch.randperm(4, generator=generator).tolist()]

    def uniform(lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=generator).item())

    return {
        "crop_top": top,
        "crop_left": left,
        "crop_height": crop_h,
        "crop_width": crop_w,
        "color_fn_order": order,
        "brightness_factor": uniform(max(0.0, 1 - AUG_BRIGHTNESS), 1 + AUG_BRIGHTNESS),
        "contrast_factor": uniform(max(0.0, 1 - AUG_CONTRAST), 1 + AUG_CONTRAST),
        "saturation_factor": uniform(max(0.0, 1 - AUG_SATURATION), 1 + AUG_SATURATION),
        "hue_factor": uniform(-AUG_HUE, AUG_HUE),
    }


def apply_augmentation(x: Tensor, a: dict) -> Tensor:
    """``x`` float ``(N, 3, H, W)`` in ``[0, 1]`` -> same shape, cropped/resized back + color jitter."""
    import torchvision.transforms.v2.functional as tvf

    h, w = x.shape[-2:]
    x = tvf.crop(x, top=a["crop_top"], left=a["crop_left"], height=a["crop_height"], width=a["crop_width"])
    x = tvf.resize(x, size=[h, w], antialias=True)
    for fn in a["color_fn_order"]:
        if fn == 0:
            x = tvf.adjust_brightness(x, a["brightness_factor"])
        elif fn == 1:
            x = tvf.adjust_contrast(x, a["contrast_factor"])
        elif fn == 2:
            x = tvf.adjust_saturation(x, a["saturation_factor"])
        else:
            x = tvf.adjust_hue(x, a["hue_factor"])
    return x


def grid_shape(n_cams: int) -> tuple[int, int]:
    """Near-square ``(rows, cols)``, wider than tall: 1 -> 1x1, 2 -> 1x2, 3 and 4 -> 2x2, 5 and 6 -> 2x3."""
    cols = math.ceil(math.sqrt(n_cams))
    return math.ceil(n_cams / cols), cols


def compose_grid(cams: Tensor, canvas_hw: tuple[int, int]) -> Tensor:
    """Any number of cameras, each resized into one cell of a near-square grid (row-major); unused cells black.

    The generic layout for camera setups outside the validated ones. It runs for any count and resolution,
    but no checkpoint was trained on it, so finetune quality away from the named layouts is unmeasured.
    """
    n_cams, t, c = cams.shape[:3]
    ch, cw = canvas_hw
    rows, cols = grid_shape(n_cams)
    cell_h, cell_w = ch // rows, cw // cols
    canvas = cams.new_zeros(t, c, ch, cw)
    for i, cam in enumerate(cams.unbind(0)):
        r, k = divmod(i, cols)
        cell = F.interpolate(cam, size=(cell_h, cell_w), mode="bilinear", align_corners=False, antialias=True)
        canvas[:, :, r * cell_h : (r + 1) * cell_h, k * cell_w : (k + 1) * cell_w] = cell
    return canvas


def compose_canvas(cams: Tensor, layout: str, canvas_hw: tuple[int, int] = CANVAS_HW) -> Tensor:
    """``cams`` float ``(n_cams, T, 3, H, W)`` in ``[0, 1]`` -> canvas ``(3, T, Hc, Wc)`` in ``[-1, 1]``.

    ``droid``: three 360x640 cameras [wrist, left, right] -> wrist full-res on top, exteriors half-res
    side by side below (540x640) -> reflect-pad right/bottom to the canvas.
    ``single``: one camera, resized to the canvas. ``side_by_side``: two cameras, each half the width.
    ``grid``: any number of cameras in a near-square grid (:func:`compose_grid`).
    What binds a checkpoint is that training and deployment use the same layout and canvas.
    """
    n_cams, _, _, h, w = cams.shape
    ch, cw = canvas_hw
    if layout == "droid":
        if n_cams != 3 or (h, w) != CAMERA_HW:
            raise ValueError(f"droid layout needs 3 cameras of {CAMERA_HW}, got {n_cams} of {(h, w)}")
        wrist, left, right = cams.unbind(0)
        half = (h // 2, w // 2)
        left = F.interpolate(left, size=half, mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=half, mode="bilinear", align_corners=False)
        composite = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)  # (T, 3, 540, 640)
        return pad_composite(composite, canvas_hw)
    elif layout == "single":
        if n_cams != 1:
            raise ValueError(f"single layout needs exactly one camera, got {n_cams}")
        canvas = F.interpolate(cams[0], size=(ch, cw), mode="bilinear", align_corners=False, antialias=True)
    elif layout == "side_by_side":
        if n_cams != 2 or cw % 2:
            raise ValueError(f"side_by_side layout needs two cameras and an even canvas width, got {n_cams}")
        canvas = torch.cat(
            [
                F.interpolate(cam, size=(ch, cw // 2), mode="bilinear", align_corners=False, antialias=True)
                for cam in cams
            ],
            dim=-1,
        )
    elif layout == "grid":
        canvas = compose_grid(cams, canvas_hw)
    else:
        raise ValueError(f"unknown camera layout {layout!r}; choose from {CAMERA_LAYOUTS}")
    return canvas.permute(1, 0, 2, 3).contiguous().mul_(2.0).sub_(1.0)


def pad_composite(composite: Tensor, canvas_hw: tuple[int, int] = CANVAS_HW) -> Tensor:
    """DROID composite ``(T, 3, 540, 640)`` in ``[0, 1]`` -> canvas ``(3, T, Hc, Wc)`` in ``[-1, 1]``.

    Reflect-pads right and bottom, the placement the three-camera path and the RoboLab/Cosmos
    composite input share (wrist on top, the two exteriors at half resolution below).
    """
    if composite.ndim != 4 or composite.shape[1] != 3 or tuple(composite.shape[-2:]) != COMPOSITE_HW:
        raise ValueError(
            f"composite must be (T, 3, {COMPOSITE_HW[0]}, {COMPOSITE_HW[1]}), got {tuple(composite.shape)}"
        )
    ch, cw = canvas_hw
    pad_right, pad_bottom = cw - composite.shape[-1], ch - composite.shape[-2]
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(f"canvas {canvas_hw} smaller than the DROID composite {COMPOSITE_HW}")
    canvas = F.pad(composite.float(), (0, pad_right, 0, pad_bottom), mode="reflect")
    return canvas.permute(1, 0, 2, 3).contiguous().mul_(2.0).sub_(1.0)


def materialize_video(
    cameras: Tensor,
    augmentation: dict | None,
    device: torch.device | str,
    *,
    layout: str = "droid",
    canvas_hw: tuple[int, int] = CANVAS_HW,
) -> Tensor:
    """``cameras`` uint8 or float ``(n_cams, T, 3, H, W)`` -> canvas ``(3, T, Hc, Wc)`` in ``[-1, 1]``.

    Deterministic given ``augmentation`` (None = no augmentation, the inference path). uint8 is
    scaled by 255; float input is taken as already in ``[0, 1]``.
    """
    if cameras.ndim != 5:
        raise ValueError(f"cameras must be (n_cams, T, 3, H, W), got {tuple(cameras.shape)}")
    n_cams, t, c, h, w = cameras.shape
    x = cameras.to(device=device, non_blocking=True).reshape(-1, c, h, w)
    x = x.float().div_(255.0) if cameras.dtype == torch.uint8 else x.float()
    if augmentation is not None:
        x = apply_augmentation(x, augmentation)
    return compose_canvas(x.reshape(n_cams, t, c, h, w), layout, canvas_hw)


# ---- 2. video VAE ---------------------------------------------------------------------------
@torch.no_grad()
def encode_videos(
    vae,
    videos: Tensor,
    latent_hw: tuple[int, int] = LATENT_HW,
    *,
    chunk_frames: int = VAE_CHUNK_FRAMES,
    pad_value: float = VAE_PAD_VALUE,
) -> Tensor:
    """``(B, 3, T, Hc, Wc)`` in ``[-1, 1]`` -> latents ``(B, 96, latent_frames(T), *latent_hw)``.

    Each clip is padded at the END with black to the VAE's 45-frame chunk, the whole padded canvas is
    encoded (the VAE sees the mirrored borders), then the padded latent frames and the padded latent
    columns/rows are dropped. The VAE treats the ``B`` clips independently; batching them only saves
    kernel launches.
    """
    b, c, t, h, w = videos.shape
    total = chunk_frames if t < chunk_frames else padded_chunk_length(t, chunk_frames)
    if total > t:
        pad = torch.full((b, c, total - t, h, w), pad_value, dtype=videos.dtype, device=videos.device)
        videos = torch.cat([videos, pad], dim=2)
    lat = vae.encode(videos.to(torch.bfloat16))  # (B, 96, latent_frames(total), H/32, W/32)
    lat = lat[:, :, : latent_frames(t), : latent_hw[0], : latent_hw[1]]
    return lat.clone()  # leave the VAE's inference_mode so autograd may consume it


def encode_video(
    vae,
    video: Tensor,
    latent_hw: tuple[int, int] = LATENT_HW,
    *,
    chunk_frames: int = VAE_CHUNK_FRAMES,
    pad_value: float = VAE_PAD_VALUE,
) -> Tensor:
    """One clip ``(3, T, Hc, Wc)`` -> ``(1, 96, latent_frames(T), *latent_hw)``; see :func:`encode_videos`."""
    return encode_videos(vae, video[None], latent_hw, chunk_frames=chunk_frames, pad_value=pad_value)


@torch.no_grad()
def encode_single_frame(
    vae, frame: Tensor, latent_hw: tuple[int, int] = LATENT_HW, *, single_frame: bool = False
) -> Tensor:
    """``(3, Hc, Wc)`` -> ``(1, 96, 1, *latent_hw)``: the inference-time conditioning latent.

    Only the current frame is supplied. By default the VAE repeats it to a 45-frame chunk, encodes that chunk
    and retains its first latent frame (the reference inference behavior). ``single_frame=True`` encodes the
    frame alone through ``vae.encode_frame`` when the VAE has one: about 0.5% relative difference in the
    latent, a fifth of the time.
    """
    x = frame[None, :, None].to(torch.bfloat16)  # (1, 3, 1, H, W)
    if single_frame and hasattr(vae, "encode_frame"):
        lat = vae.encode_frame(x[:, :, 0])
    else:
        lat = vae.encode(x)  # (1, 96, 1, H/32, W/32)
    return lat[..., : latent_hw[0], : latent_hw[1]].clone()


# ---- 3. token packing (position ids on one 10 ms clock) -------------------------------------
def video_time_ids(
    n_latent: int,
    first_latent: int,
    batch: int,
    *,
    fps: float = FPS,
    temporal_downsample: int = TEMPORAL_DOWNSAMPLE,
) -> Tensor:
    """Latent frame ``i`` sits at time ``i * 4 / fps`` s (frame ``i*4`` of the window)."""
    seconds = torch.arange(first_latent, first_latent + n_latent).float() * temporal_downsample / fps
    return times_to_ids(repeat(seconds, "t -> b t", b=batch))


def pack_video(latents: Tensor, *, fps: float = FPS) -> dict[str, Tensor]:
    """``(B, 96, n, h, w)`` -> ``x_video_cond`` (frame 0) + ``x_video`` (frames 1..) with ids."""
    b, _, n, _, _ = latents.shape
    latents = latents.to(torch.bfloat16)
    cond, ids_c = batched_prc_vid(latents[:, :, :1], video_time_ids(1, 0, b, fps=fps))
    pred, ids_p = batched_prc_vid(latents[:, :, 1:], video_time_ids(n - 1, 1, b, fps=fps))
    return {"x_video_cond": cond, "x_video_cond_ids": ids_c, "x_video": pred, "x_video_ids": ids_p}


def pack_actions(
    state: Tensor,
    actions: Tensor,
    action_times_s: Tensor,
    modality: str,
    *,
    scale: float = ACTION_SCALE,
) -> dict[str, Tensor]:
    """``state (B, 1, D)``, ``actions (B, K, D)``, ``times (B, K)`` s since the frame -> scaled tokens + ids.

    Audio-style packing: ids ``(t, 0, 0, 0)``. The state token sits at ``t = 0``, each action at the
    frame it produces (DROID: ``k / 15`` s -> ids 6, 13, 20, 26, ...).
    """
    b = state.shape[0]
    x_cond, ids_cond = batched_prc_action(
        (state * scale).transpose(1, 2), times_to_ids(torch.zeros(b, 1, device=state.device))
    )
    x_act, ids_act = batched_prc_action(
        (actions * scale).transpose(1, 2), times_to_ids(action_times_s.to(actions.device))
    )
    return {
        f"x_{modality}_cond": x_cond.float(),
        f"x_{modality}_cond_ids": ids_cond,
        f"x_{modality}": x_act.to(torch.bfloat16),
        f"x_{modality}_ids": ids_act,
    }


def default_action_times(batch: int, chunk: int = CHUNK, fps: float = FPS) -> Tensor:
    """Action ``k`` (0-based) produces frame ``k + 1``: times ``(k + 1) / fps``."""
    return repeat((torch.arange(chunk).float() + 1) / fps, "t -> b t", b=batch)


def pack_text(ctx: Tensor, vec_dim: int = VEC_DIM) -> dict[str, Tensor]:
    """``ctx (B, L, 20480)`` -> ctx ids on the l axis (t = h = w = 0), zero ctx timesteps, zero vector."""
    _, ctx_ids = batched_prc_txt(ctx)
    b = ctx.shape[0]
    return {
        "ctx": ctx,
        "ctx_ids": ctx_ids.to(ctx.device),
        "timesteps_ctx": torch.zeros(ctx.shape[:2], device=ctx.device),
        "vector": torch.zeros(b, vec_dim, device=ctx.device),
    }


# ---- 4. flow matching -----------------------------------------------------------------------
def rational_time_shift(t: Tensor, shift: float) -> Tensor:
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_timesteps(
    batch: int,
    generator: torch.Generator | None = None,
    width: float = TIMESTEP_WIDTH,
    shift: float = TIMESTEP_SHIFT,
) -> Tensor:
    """Training timestep: ``logit(t) ~ Logistic(0, width)``, then the rational shift.

    ONE ``t`` per sample, shared by the video tokens and the action tokens.
    """
    eps = torch.finfo(torch.float32).eps
    u = torch.rand(batch, generator=generator).clamp(eps, 1 - eps)
    t = torch.sigmoid(width * (torch.log(u) - torch.log1p(-u)))
    return rational_time_shift(t, shift)


def add_noise(x0: Tensor, t: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """``x_t = t * eps + (1 - t) * x0``; velocity target ``eps - x0``. ``t`` is ``(B,)``.

    Everything is computed in ``x0.dtype`` after rounding ``t`` and ``eps`` to it: the reference packs
    latents, noise and timesteps in bfloat16 and noises in that dtype, so a bf16 window sees ``t`` on
    the bf16 grid (spacing 2^-8 just below 1; ``t > 1 - 2^-9`` becomes exactly 1).
    """
    eps_ = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(
        device=x0.device, dtype=x0.dtype
    )
    tt = t.to(device=x0.device, dtype=x0.dtype).view(-1, *([1] * (x0.ndim - 1)))
    x_t = tt * eps_ + (1 - tt) * x0
    return x_t, eps_ - x0


def build_forward_kwargs(
    video: dict[str, Tensor],
    action: dict[str, Tensor],
    text: dict[str, Tensor],
    t: Tensor,
    modality: str,
    generator: torch.Generator | None = None,
    *,
    action_timesteps: Tensor | None = None,
    conditioning_noise_max: float = 0.0,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Assemble one training call: ``(forward_kwargs, flow_targets, clean_latents)``.

    Conditioning streams (``x_video_cond``, ``x_<modality>_cond``) stay clean at timestep 0. Predicted
    streams use ``t`` and optionally separate ``action_timesteps``, rounded to their latent dtype.
    Optional conditioning noise applies only to visual snapshots; state/command history stays clean.
    The timesteps handed to the model are fp32 tensors holding the rounded values.
    """
    ak = f"x_{modality}"
    xv, tv = add_noise(video["x_video"], t, generator)
    t_action = t if action_timesteps is None else action_timesteps
    xa, ta = add_noise(action[ak], t_action, generator)
    dev = xv.device
    tt = t.to(device=dev, dtype=xv.dtype).float()  # the rounded value the noising used
    tt_action = t_action.to(device=dev, dtype=xa.dtype).float()
    kw = {
        "x_video": xv,
        "x_video_ids": video["x_video_ids"].to(dev),
        "x_video_timesteps": tt[:, None] * torch.ones(xv.shape[:2], device=dev),
        ak: xa,
        f"{ak}_ids": action[f"{ak}_ids"].to(dev),
        f"{ak}_timesteps": tt_action[:, None] * torch.ones(xa.shape[:2], device=dev),
        "x_video_cond": video["x_video_cond"],
        "x_video_cond_ids": video["x_video_cond_ids"].to(dev),
        "x_video_cond_timesteps": torch.zeros(video["x_video_cond"].shape[:2], device=dev),
        f"{ak}_cond": action[f"{ak}_cond"],
        f"{ak}_cond_ids": action[f"{ak}_cond_ids"].to(dev),
        f"{ak}_cond_timesteps": torch.zeros(action[f"{ak}_cond"].shape[:2], device=dev),
        **text,
    }
    if conditioning_noise_max:
        cond_t = torch.rand(t.shape, generator=generator) * conditioning_noise_max
        kw["x_video_cond"], _ = add_noise(video["x_video_cond"], cond_t, generator)
        cond_tt = cond_t.to(device=dev, dtype=kw["x_video_cond"].dtype).float()
        kw["x_video_cond_timesteps"] = cond_tt[:, None].expand(video["x_video_cond"].shape[:2])
    targets = {"x_video": tv, ak: ta}
    clean = {"x_video": video["x_video"], ak: action[ak]}
    return kw, targets, clean


def pack_windows(
    kwargs: dict[str, Tensor], ctxs: list[Tensor], vec_dim: int = VEC_DIM
) -> tuple[dict[str, Tensor], dict[str, list[int]]]:
    """``B`` windows -> one DiT call of batch size 1 plus the per-window token counts it needs.

    ``kwargs`` are the stream tensors of :func:`build_forward_kwargs` for ``B`` windows (``(B, L, ...)``:
    the windows share every stream length), ``ctxs`` their text contexts ``(1, L_i, ctx_dim)`` of any
    lengths. Streams become ``(1, B * L, ...)`` with window ``i`` at ``[i * L, (i + 1) * L)``; the contexts
    are concatenated with their own position ids (``l`` restarts at 0 in every window). ``seqlens``
    (``"ctx"`` and each ``x_<stream>`` -> ``B`` token counts) is what ``JointSingleSeq.forward`` takes.
    """
    b = len(ctxs)
    if b == 0:
        raise ValueError("pack_windows needs at least one window")
    packed: dict[str, Tensor] = {}
    seqlens: dict[str, list[int]] = {}
    for key, value in kwargs.items():
        if value.shape[0] != b:
            raise ValueError(f"{key}: expected {b} windows, got {value.shape[0]}")
        packed[key] = value.reshape(1, b * value.shape[1], *value.shape[2:])
        if not key.endswith(("_ids", "_timesteps")):
            seqlens[key] = [int(value.shape[1])] * b
    if any(c.ndim != 3 or c.shape[0] != 1 for c in ctxs):
        raise ValueError("each context must be (1, L_i, ctx_dim)")
    ctx = torch.cat(ctxs, dim=1)
    ctx_ids = torch.cat([batched_prc_txt(c)[1].to(ctx.device) for c in ctxs], dim=1)
    packed.update(
        ctx=ctx,
        ctx_ids=ctx_ids,
        timesteps_ctx=torch.zeros(ctx.shape[:2], device=ctx.device),
        vector=torch.zeros(1, vec_dim, device=ctx.device),
    )
    seqlens["ctx"] = [int(c.shape[1]) for c in ctxs]
    return packed, seqlens


def flatten_windows(tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    """``(B, L, ...)`` -> ``(1, B * L, ...)`` for every entry, the layout of :func:`pack_windows` outputs."""
    return {k: v.reshape(1, v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in tensors.items()}


def flow_loss(
    pred: dict[str, Tensor],
    targets: dict[str, Tensor],
    modality: str,
    action_weight: float = ACTION_LOSS_WEIGHT,
    video_weight: float = VIDEO_LOSS_WEIGHT,
    *,
    action_mask: Tensor | None = None,
    reduction: str = "joint_tokens",
    channel_weights: list[float] | None = None,
) -> dict[str, Tensor]:
    """Joint-token mean (DROID) or weighted per-modality means (history), with optional channel weights.

    The default action weight (50) applies to joint-token pooling. Channel weights are squared and
    normalized by their mean, matching collab's task loss. Per-modality
    MSEs are returned for logging. Over the windows of a micro-batch (batched or packed, see
    :func:`pack_windows`) this equals the mean of the per-window losses, since every window holds the
    same number of video and action tokens.

    ``action_mask`` ``(n, D)`` (1 = trained dim) restricts the per-token action MSE to the given dims of each
    window, for a head shared by embodiments of different action widths (padded dims carry no loss). The
    action tokens are laid out window-major in the packed sequence, ``n`` windows x ``K`` tokens each.
    """
    ak = f"x_{modality}"
    video = ((pred["x_video"].float() - targets["x_video"].float()) ** 2).mean(-1)  # (B, Nv)
    sq = (pred[ak].float() - targets[ak].float()) ** 2  # (B, Na, D): batched (n, K, D) or packed (1, n*K, D)
    if action_mask is None:
        action = sq.mean(-1)  # (B, Na)
    else:
        n = action_mask.shape[0]
        m = action_mask.float()  # (n, D)
        if sq.shape[0] == n:  # batched: one mask row per window
            m = m[:, None, :].expand_as(sq)
        else:  # packed: window-major tokens, n windows x K tokens each
            m = m.repeat_interleave(sq.shape[1] // n, dim=0).reshape(sq.shape)
        action = (sq * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    weighted_action = action
    if channel_weights is not None:
        weights = torch.tensor(channel_weights, device=sq.device, dtype=torch.float32).square()
        weights = weights / weights.mean()
        weighted_action = (
            (sq * weights).mean(-1)
            if action_mask is None
            else (sq * weights * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        )
    if reduction == "modalities":
        total = video_weight * video.mean() + action_weight * weighted_action.mean()
    elif reduction == "joint_tokens":
        total = (video_weight * video.sum() + action_weight * weighted_action.sum()) / (
            video.numel() + action.numel()
        )
    else:
        raise ValueError(f"unknown loss reduction {reduction!r}")
    return {"loss": total, "video_mse": video.mean().detach(), "action_mse": action.mean().detach()}


def token_budget(
    chunk: int = CHUNK, latent_hw: tuple[int, int] = LATENT_HW, text_tokens: int = TEXT_PAD_MULTIPLE
) -> dict[str, int]:
    """Token counts of one window (for sanity checks and batch sizing)."""
    hw = latent_hw[0] * latent_hw[1]
    n_pred = latent_frames(chunk + 1) - 1
    return {
        "x_video_cond": hw,
        "x_video": n_pred * hw,
        "x_action_cond": 1,
        "x_action": chunk,
        "ctx_min": text_tokens,
    }
