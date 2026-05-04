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

The current resident is :func:`bilinear_resample_pos_embed`, which adapts
SigLIP / ViT-style learned position-embedding tables to a different patch
count when the policy runs the vision tower at a non-published resolution.

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
