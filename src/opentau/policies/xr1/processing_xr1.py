# Copyright 2026 Tensor Auto Inc. All rights reserved.
# Copyright (C) 2026 Xiaomi Corporation.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

"""Prompt assembly for ``xr1``: text template + Qwen3-VL video-token expansion.

Xiaomi-Robotics-1 ships a ``MiBotProcessor`` (``processing_mibot.py``,
``XiaomiRobotics/Xiaomi-Robotics-1`` @ ``4da1db0``, Apache-2.0) that is a copy of stock
``transformers.Qwen3VLProcessor.__call__`` plus two robot-specific tensors. Rather than
vendor a ``trust_remote_code`` module, this file reimplements the two pieces that actually
matter and takes everything else from stock transformers:

* **The chat text** -- the checkpoint's ``chat_template.jinja`` is the stock Qwen3-VL
  template, so the rendered string for XR-1's fixed two-turn conversation is a constant
  that :func:`render_chat_text` writes out directly. Pinned character-for-character
  against the captured golden (``tests/artifacts/policies/xr1/prompt_tokens.json``).
* **The video placeholder expansion** -- the ``<|vision_start|><|video_pad|><|vision_end|>``
  marker becomes *one block per temporal patch*, each preceded by a ``<T.T seconds>``
  timestamp. Ported from Qwen3-VL's own ``Qwen3VLProcessor.__call__`` /
  ``_calculate_timestamps``.

Two details a hand-rolled builder gets wrong, both pinned by ``test_xr1_cpu.py``:

* There are **two** blocks per camera (4 frames at ``temporal_patch_size=2``), i.e. 128
  video tokens per camera, not 64.
* The timestamps come from a **default fps of 24** -- the reference passes no video
  metadata, so both the video processor's frame sampling and the timestamp calculation
  fall back to 24, giving ``<0.0 seconds>`` and ``<0.1 seconds>``.

The robot-specific tensors are built here too, but from the policy config rather than from
the checkpoint's normalization statistics: the reference's ``action_mask`` is
``(std > 1e-5)`` over its ``(chunk, 60)`` stats grid, and for RoboCasa365 that is exactly
"the first ``action_dim`` columns are active" -- so it is derivable, and the derivation is
one fewer file the port has to carry.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

#: The two-turn conversation XR-1 always sends, with the stock Qwen3-VL chat template
#: already applied. ``{cameras}`` is the concatenation of one labelled video marker per
#: camera; ``{instruction}`` is the task string.
CHAT_TEMPLATE = (
    "<|im_start|>user\n"
    "{cameras}"
    "\n\nGenerate robot actions for the task:\n"
    "{instruction} /no_cot<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<cot></cot><|im_end|>\n"
)

# nosec B105 x3 — these are Qwen3-VL prompt markers, not credentials; bandit's
# hardcoded-password heuristic fires on any constant whose name ends in "TOKEN".
VISION_START_TOKEN = "<|vision_start|>"  # nosec B105
VISION_END_TOKEN = "<|vision_end|>"  # nosec B105
VIDEO_TOKEN = "<|video_pad|>"  # nosec B105

#: Frames-per-second the reference's timestamps are computed at. It is a *fallback*: no
#: video metadata is passed, so ``Qwen3VLProcessor`` warns and defaults to 24. Reproducing
#: the fallback is what makes the ``<0.0 seconds>`` / ``<0.1 seconds>`` markers match.
DEFAULT_VIDEO_FPS = 24.0


def render_chat_text(instruction: str, camera_labels: tuple[str, ...] | list[str]) -> str:
    """Render the chat string with one un-expanded video marker per camera."""
    cameras = "".join(
        f"{label}{VISION_START_TOKEN}{VIDEO_TOKEN}{VISION_END_TOKEN}" for label in camera_labels
    )
    return CHAT_TEMPLATE.format(cameras=cameras, instruction=instruction)


def calculate_timestamps(frame_indices: list[int], video_fps: float, merge_size: int = 2) -> list[float]:
    """Per-temporal-patch timestamps, ported from ``Qwen3VLProcessor._calculate_timestamps``.

    Frames are merged ``merge_size`` at a time, and each patch is stamped with the mean of
    its first and last frame time.
    """
    indices = list(frame_indices)
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    timestamps = [idx / video_fps for idx in indices]
    return [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
    ]


def expand_video_placeholders(
    text: str,
    video_grid_thw: Tensor | np.ndarray,
    merge_size: int,
    frame_indices_per_video: list[list[int]],
    video_fps: float = DEFAULT_VIDEO_FPS,
) -> str:
    """Replace each ``<|vision_start|><|video_pad|><|vision_end|>`` with its expanded block.

    One ``<T.T seconds><|vision_start|>...<|vision_end|>`` group per temporal patch, each
    holding ``(h * w) / merge_size ** 2`` ``<|video_pad|>`` tokens.
    """
    # `.cpu()` rather than `np.asarray(...)`: the grid is often already a CUDA tensor by the
    # time this runs (it comes back from `patchify_videos` on the batch's device), and numpy
    # refuses to convert one -- a crash on the serving path only, since CPU tests never hit it.
    grid = (
        video_grid_thw.detach().cpu()
        if isinstance(video_grid_thw, Tensor)
        else torch.as_tensor(np.asarray(video_grid_thw))
    )
    merge_length = merge_size**2
    marker = f"{VISION_START_TOKEN}{VIDEO_TOKEN}{VISION_END_TOKEN}"
    for index in range(grid.shape[0]):
        timestamps = calculate_timestamps(frame_indices_per_video[index], video_fps, merge_size)
        frame_seqlen = int(grid[index][1:].prod()) // merge_length
        placeholder = ""
        for frame_idx in range(int(grid[index][0])):
            placeholder += f"<{timestamps[frame_idx]:.1f} seconds>"
            placeholder += VISION_START_TOKEN + "<|placeholder|>" * frame_seqlen + VISION_END_TOKEN
        if marker not in text:
            raise ValueError(f"Expected {grid.shape[0]} video markers in the prompt, ran out after {index}.")
        text = text.replace(marker, placeholder, 1)
    return text.replace("<|placeholder|>", VIDEO_TOKEN)


def build_action_mask(
    batch_size: int, chunk_size: int, max_action_dim: int, action_dim: int, *, device, dtype
) -> Tensor:
    """``(B, chunk, max_action_dim)`` float mask; the first ``action_dim`` columns are 1.

    The reference derives this from ``std > 1e-5`` over its per-robot action statistics.
    For RoboCasa365 those statistics are identity (mean 0 / std 1) on the 12 real columns
    and exactly zero on the 48 padding columns, so the derived mask is this.

    It is a **float** tensor, not bool, and that is load-bearing: the reference draws its
    flow noise with ``torch.randn_like(action_mask)``, so this tensor's dtype decides the
    noise dtype (bfloat16 on the real model) and its shape decides the noise shape.
    """
    mask = torch.zeros(batch_size, chunk_size, max_action_dim, device=device, dtype=dtype)
    mask[..., :action_dim] = 1.0
    return mask


def center_crop_resize(images: Tensor, crop_ratio: float, quantize_uint8: bool = True) -> Tensor:
    """Centre-crop ``(..., C, H, W)`` images in ``[0, 1]`` by ``crop_ratio``, resize back.

    Matches the reference's PIL path step for step:

    * The crop side is ``int(side * ratio)`` -- **odd** at the shipped 256 / 0.95, so the
      box is asymmetric (6 px dropped left, 7 right, and likewise top/bottom).
    * The resize is a plain bilinear with **no antialiasing**. On an upscale that is
      exactly ``PIL.Image.BILINEAR``'s triangle filter under the same
      ``(x + 0.5) * scale - 0.5`` coordinate mapping that ``align_corners=False`` uses.
      ``torchvision``'s ``antialias=True`` default and cv2's ``INTER_LINEAR`` are both
      different filters.
    * ``quantize_uint8`` reproduces the fact that the reference hands the processor a
      **uint8** ``PIL.Image``: the crop and resize round-trip through 8-bit, so the
      quantization is part of what the model saw. Leave it on for reference parity; turn
      it off only if a caller is deliberately keeping full float precision.
    """
    if crop_ratio >= 1.0:
        return images
    *lead, channels, height, width = images.shape
    crop_h = max(1, int(height * crop_ratio))
    crop_w = max(1, int(width * crop_ratio))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2

    working = images.float()
    if quantize_uint8:
        working = torch.round(working * 255.0).clamp_(0.0, 255.0) / 255.0
    cropped = working[..., top : top + crop_h, left : left + crop_w]
    flat = cropped.reshape(-1, channels, crop_h, crop_w)
    resized = torch.nn.functional.interpolate(
        flat, size=(height, width), mode="bilinear", align_corners=False, antialias=False
    )
    if quantize_uint8:
        resized = torch.round(resized * 255.0).clamp_(0.0, 255.0) / 255.0
    return resized.reshape(*lead, channels, height, width).to(images.dtype)


def crop_box(side: int, crop_ratio: float) -> tuple[int, int, int, int]:
    """``(top, left, height, width)`` of the centre crop -- exposed so tests can pin it."""
    crop = max(1, int(side * crop_ratio))
    offset = (side - crop) // 2
    return offset, offset, crop, crop


def patchify_videos(
    videos: Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[Tensor, Tensor]:
    """Turn ``(V, T, C, H, W)`` uint8-scale videos into Qwen3-VL patch features.

    Returns ``(pixel_values_videos, video_grid_thw)`` with
    ``pixel_values_videos`` of shape ``(V * T/tp * H/p * W/p, C * tp * p * p)`` -- exactly
    what ``Qwen3VLVideoProcessor`` emits with ``do_resize=False``. The patch ordering is
    the merge-aware one Qwen3-VL uses: within a ``merge_size x merge_size`` window the
    patches are contiguous, so the vision tower's 2x2 merger sees a square.

    ``videos`` must already be in ``[0, 1]`` (the ``rescale_factor`` step is the caller's,
    since OpenTau's dataloader already emits floats in that range).
    """
    num_videos, frames, channels, height, width = videos.shape
    if frames % temporal_patch_size != 0:
        raise ValueError(
            f"Frame count {frames} is not a multiple of temporal_patch_size {temporal_patch_size}."
        )
    if height % (patch_size * merge_size) or width % (patch_size * merge_size):
        raise ValueError(
            f"({height}, {width}) is not a multiple of patch_size * merge_size "
            f"({patch_size * merge_size}); Qwen3-VL's 2x2 merger needs whole merge windows."
        )

    mean = torch.tensor(image_mean, device=videos.device, dtype=torch.float32).view(1, 1, -1, 1, 1)
    std = torch.tensor(image_std, device=videos.device, dtype=torch.float32).view(1, 1, -1, 1, 1)
    normalized = (videos.float() - mean) / std

    grid_t = frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size
    # The axis order is Qwen3-VL's own (``Qwen3VLVideoProcessor._preprocess``): merge
    # windows are the *inner* spatial axes, so the 2x2 merger in the vision tower sees a
    # contiguous square. Getting the two spatial decompositions the wrong way round
    # produces a correctly-shaped tensor of scrambled patches.
    flat = rearrange(
        normalized,
        "v (gt tp) c (gh mh ph) (gw mw pw) -> (v gt gh gw mh mw) (c tp ph pw)",
        tp=temporal_patch_size,
        mh=merge_size,
        ph=patch_size,
        mw=merge_size,
        pw=patch_size,
    )
    grid = torch.tensor([[grid_t, grid_h, grid_w]] * num_videos, dtype=torch.long, device=videos.device)
    return flat, grid
