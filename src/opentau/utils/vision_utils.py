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

"""Vision-tower helpers reused across policies (pi06, pi07, ...).

Residents:

- :func:`bilinear_resample_pos_embed` adapts SigLIP / ViT-style learned
  position-embedding tables to a different patch count when the policy runs
  the vision tower at a non-published resolution (offline weight surgery).
- :func:`patch_grid_hw` / :func:`pad_to_patch_multiple` support running the
  SigLIP tower at native (non-square, non-patch-multiple) input resolutions
  at forward time: the stride-``patch_size`` conv patch embedding silently
  floor-crops any sub-patch remainder (e.g. 180×320 with patch 14 drops 12
  pixel rows and 12 columns), so callers pad up to the next patch multiple
  and enable ``interpolate_pos_encoding`` instead.

Why this is its own module rather than inline in a policy: π0.6 uses Gemma
3-4B at 448×448 (1024 patches) but `google/gemma-3-4b-pt` ships at 896×896
(4096 patches), so any script that bootstraps an "untrained" π0.6
checkpoint from the public VLM weights must resample. π0.7 uses the same
backbone family and will inherit the same need — sharing the helper here
keeps the recipe identical across them and avoids drift between
re-implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def patch_grid_hw(height: int, width: int, patch_size: int) -> tuple[int, int]:
    """Patch-grid shape ``(grid_h, grid_w)`` that fully covers an image.

    Uses ceiling division, so a resolution that does not divide
    ``patch_size`` still gets a grid covering every pixel — the caller is
    expected to pad the image up to ``grid * patch_size`` with
    :func:`pad_to_patch_multiple` before the conv patch embedding (which
    would otherwise floor-crop the remainder).

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        patch_size: ViT patch size (14 for SigLIP so400m).

    Returns:
        ``(grid_h, grid_w)`` patch counts per axis.
    """
    return -(-height // patch_size), -(-width // patch_size)


def pad_to_patch_multiple(img: torch.Tensor, patch_size: int, pad_value: float = -1.0) -> torch.Tensor:
    """Pad the trailing two (H, W) dims up to the next multiple of ``patch_size``.

    Padding goes on the bottom and right so the image origin stays
    pixel-aligned: every patch that contains real content keeps exactly the
    pixels it would have at a divisible resolution, and only the last patch
    row/column mixes in padding. The default ``pad_value=-1.0`` is black in
    the ``[-1, 1]`` range SigLIP consumes; callers padding ``[0, 1]``-range
    pixels should pass ``0.0``.

    Args:
        img: Image tensor of shape ``(..., H, W)``.
        patch_size: ViT patch size to pad up to a multiple of.
        pad_value: Fill value for the padded band.

    Returns:
        The input unchanged (no copy) when ``H`` and ``W`` are already
        multiples of ``patch_size``, else a padded copy of shape
        ``(..., ceil(H/p)*p, ceil(W/p)*p)``.
    """
    height, width = img.shape[-2:]
    pad_h = -height % patch_size
    pad_w = -width % patch_size
    if pad_h == 0 and pad_w == 0:
        return img
    return F.pad(img, (0, pad_w, 0, pad_h), value=pad_value)


def bilinear_resample_pos_embed(old_pos: torch.Tensor, target_num_patches: int) -> torch.Tensor:
    """Bilinearly resample a learned ``(N_patches, dim)`` position-embedding
    table to a different patch count, preserving dtype.

    This is the standard recipe for adapting SigLIP / ViT position embeddings
    when running a published VLM at a different image resolution. Both the
    source and target patch counts must be perfect squares (square grids).
    Computation is performed in fp32 (``F.interpolate`` rejects bf16 on CPU)
    and cast back to the input dtype before return.

    The transform is fully deterministic: two calls with bit-identical inputs
    return bit-identical outputs, with no RNG consumption. This matters for
    bootstrapping checkpoints whose downstream tests compare weights byte-
    for-byte against an independently-computed reference.

    Args:
        old_pos: Source position-embedding table, shape ``(N_old, dim)`` with
            ``N_old`` a perfect square.
        target_num_patches: Desired output ``N_new``, must be a perfect square.

    Returns:
        Resampled tensor of shape ``(target_num_patches, dim)`` and same dtype
        as ``old_pos``. Returns ``old_pos`` unchanged when ``N_old ==
        target_num_patches`` (no copy).

    Raises:
        AssertionError: if either patch count is not a perfect square.
    """
    old_n, dim = old_pos.shape
    if old_n == target_num_patches:
        return old_pos
    old_grid = int(old_n**0.5)
    new_grid = int(target_num_patches**0.5)
    assert old_grid * old_grid == old_n, f"non-square source grid: {old_n}"
    assert new_grid * new_grid == target_num_patches, f"non-square target grid: {target_num_patches}"
    grid = old_pos.reshape(old_grid, old_grid, dim).permute(2, 0, 1).unsqueeze(0).float()
    grid = torch.nn.functional.interpolate(
        grid, size=(new_grid, new_grid), mode="bilinear", align_corners=False
    )
    return grid.squeeze(0).permute(1, 2, 0).reshape(target_num_patches, dim).to(old_pos.dtype)
