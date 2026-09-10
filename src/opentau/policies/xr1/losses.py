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

"""Flow + frequency losses for ``xr1``.

Ported from ``xr1/mibot/models/VLA/XR1.py::compute_flow_loss``. Two halves:

* a **weighted masked MSE** on the velocity field, where the per-element weight comes from
  a no-grad rollout of the async action prefix (see :func:`prefix_rollout_weight`), is
  normalized to mean 1 over the unmasked slots and then clamped to ``[0.5, 5]``; and
* an **rFFT L1** term comparing the spectra of the predicted and target action sequences
  along the time axis, weighted by each sample's mean loss weight.

Two implementation notes that are easy to "fix" into a bug:

* ``torch.fft.rfft`` does not support bfloat16. The whole loss therefore runs in float32
  (the reference casts too), and the cast must stay -- removing it turns the frequency
  term into a hard crash on the real model.
* The reference's ``freq_excluded_dims = [17, 18, 19]`` is defined on its canonical 60-D
  action layout. RoboCasa365's flat-12 action has no columns there, so the shipped default
  for this policy is empty; the argument is kept so a differently-laid-out dataset can use
  it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce
from torch import Tensor

from opentau.policies.utils import PerSampleLoss, make_action_dim_mask


def build_flow_mask(
    *,
    batch_size: int,
    chunk_size: int,
    max_action_dim: int,
    device: torch.device,
    prefix_mask: Tensor | None = None,
    actions_is_pad: Tensor | None = None,
    real_action_dim: Tensor | None = None,
) -> Tensor:
    """``(B, chunk, max_action_dim)`` bool mask, built exactly as ``flow_matching_masked_mse`` does.

    AND-s the three OpenTau conditions: not a frozen async-prefix row, not a padded
    chunk row, and not a zero-pad action column. Sharing the construction (rather than
    re-deriving it) is what lets ``flow_and_freq_loss`` degenerate to
    ``flow_matching_masked_mse`` bit-for-bit at neutral settings.
    """
    if prefix_mask is None:
        prefix_mask = torch.zeros((batch_size, chunk_size), dtype=torch.bool, device=device)
    postfix_mask = rearrange(torch.logical_not(prefix_mask), "b c -> b c 1")
    if actions_is_pad is not None:
        postfix_mask = torch.logical_and(postfix_mask, rearrange(~actions_is_pad, "b c -> b c 1"))
    dim_mask = make_action_dim_mask(real_action_dim, max_action_dim, batch_size=batch_size, device=device)
    return postfix_mask & rearrange(dim_mask, "b d -> b 1 d")


def normalize_loss_weight(weight: Tensor, mask: Tensor, clamp: tuple[float, float]) -> Tensor:
    """Normalize ``weight`` to mean 1 over the masked slots, then clamp.

    Note the clamp applies to the **whole** tensor, not just the masked slots -- that is
    the reference's behaviour, and it matters because the unmasked slots' (clamped) values
    still feed the frequency term's per-sample ``weight.mean(dim=(1, 2))``.
    """
    with torch.no_grad():
        weight = weight.clone()
        if mask.any():
            weight[mask] = weight[mask] / weight[mask].mean()
        weight.clamp_(clamp[0], clamp[1])
    return weight


def flow_and_freq_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    weight: Tensor | None = None,
    freq_excluded_dims: tuple[int, ...] = (),
    weight_clamp: tuple[float, float] = (0.5, 5.0),
    return_per_sample: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, PerSampleLoss]:
    """Return ``(loss_mse, loss_freq)`` -- and optionally the per-sample MSE decomposition.

    Args:
        pred: Predicted velocity, ``(B, chunk, D)``.
        target: Target velocity ``action - noise``, same shape. Note the **sign**: XR-1
            regresses ``action - noise`` with an ascending flow, the opposite convention
            from pi05's ``noise - action``.
        mask: ``(B, chunk, D)`` bool from :func:`build_flow_mask`.
        weight: ``(B, chunk, D)`` per-element loss weight; ``None`` means all ones (the
            no-async-prefix case, which the reference also constructs as ``ones_like``).
        freq_excluded_dims: Action columns dropped from the frequency term.
        weight_clamp: ``(low, high)`` clamp applied after mean-1 normalization.
        return_per_sample: Also return a :class:`PerSampleLoss` for the MSE half, reduced
            over ``(chunk, D)`` so it is ``(B,)`` -- never ``(B * training_repeat,)``; the
            caller folds the repeat axis back in first.

    Returns:
        ``(loss_mse, loss_freq)``, or ``(loss_mse, loss_freq, per_sample)``.
    """
    pred = pred.float()
    target = target.float()
    weight = torch.ones_like(pred) if weight is None else weight.float()
    mask = mask.bool()

    if not torch.any(mask):
        zero = (pred.sum() + target.sum()) * 0.0
        if return_per_sample:
            bsz = pred.shape[0]
            empty = PerSampleLoss(
                sum=torch.zeros(bsz, device=pred.device), count=torch.zeros(bsz, device=pred.device)
            )
            return zero, zero, empty
        return zero, zero

    weight = normalize_loss_weight(weight, mask, weight_clamp)

    elementwise = F.mse_loss(pred, target, reduction="none") * weight
    loss_mse = elementwise[mask].mean()

    # --- frequency term -----------------------------------------------------------------
    # rfft is float32-only; `pred`/`target` are already cast above.
    freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
    # A sample contributes only when its LAST chunk row still has an active column --
    # i.e. the whole horizon is real, so its spectrum is meaningful.
    valid_batch = mask[:, -1].any(dim=1)
    if not torch.any(valid_batch):
        loss_freq = freq.sum() * 0.0
    else:
        freq_mask = mask[valid_batch, : freq.shape[1]].clone()
        dims = [dim for dim in freq_excluded_dims if dim < freq_mask.shape[-1]]
        if dims:
            freq_mask[:, :, dims] = False
        freq_weight = rearrange(reduce(weight, "b c d -> b", "mean"), "b -> b 1 1")
        selected = (freq * freq_weight)[valid_batch][freq_mask]
        loss_freq = selected.mean() if selected.numel() else freq.sum() * 0.0

    if not return_per_sample:
        return loss_mse, loss_freq

    masked_elementwise = elementwise * mask
    per_sample = PerSampleLoss(
        sum=reduce(masked_elementwise, "b c d -> b", "sum"),
        count=reduce(mask.float(), "b c d -> b", "sum"),
    )
    return loss_mse, loss_freq, per_sample


def fold_repeat_per_sample(per_sample: PerSampleLoss, batch_size: int, repeat: int) -> PerSampleLoss:
    """Collapse a ``(B * R,)`` per-sample loss back to ``(B,)``.

    The training forward evaluates ``training_repeat`` flow timesteps per sample by folding
    the repeat axis into the batch, so a naive per-sample decomposition is ``(B * R,)``.
    ``scripts/train.py`` gathers per-sample losses **alongside** ``(B,)`` provenance
    tensors in one ``gather_for_metrics`` call, so a longer loss vector does not error --
    it silently misattributes every row. The repeat is ``repeat_interleave``, so sample
    ``b``'s rows are contiguous.
    """
    if repeat == 1:
        return per_sample
    return PerSampleLoss(
        sum=reduce(per_sample.sum, "(b r) -> b", "sum", b=batch_size, r=repeat),
        count=reduce(per_sample.count, "(b r) -> b", "sum", b=batch_size, r=repeat),
    )
