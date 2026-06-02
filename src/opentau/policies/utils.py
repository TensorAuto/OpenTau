#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Utility functions for policy implementations in OpenTau.

This module provides helper functions for managing data queues, inspecting model
properties (device, dtype), determining output shapes, and logging model loading
information.
"""

import logging
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce
from torch import Tensor, nn


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
) -> dict[str, deque]:
    """Populates queues with batch data.

    If a queue is not full (e.g. at the start of an episode), it is filled by repeating
    the first observation. Otherwise, the latest observation is appended.

    Args:
        queues: A dictionary of deques to be populated.
        batch: A dictionary containing the data to add to the queues.
        exclude_keys: A list of keys to exclude from population. Defaults to None.

    Returns:
        dict[str, deque]: The updated dictionary of queues.
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note:
        Assumes that all parameters have the same device.

    Args:
        module: The PyTorch module to inspect.

    Returns:
        torch.device: The device of the module's parameters.
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note:
        Assumes that all parameters have the same dtype.

    Args:
        module: The PyTorch module to inspect.

    Returns:
        torch.dtype: The data type of the module's parameters.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module: A PyTorch module.
        input_shape: A tuple representing the input shape, e.g., (batch_size, channels, height, width).

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


@dataclass
class PerSampleLoss:
    """Per-sample decomposition of a masked loss, with the batch dim kept.

    ``sum`` and ``count`` are both ``(B,)`` and hold, for each sample, the
    summed unmasked loss and the number of unmasked slots that fed it. The
    masked *mean* for any group of samples is ``Σsum / Σcount`` — carrying the
    (numerator, denominator) pair rather than a per-sample mean is what lets a
    caller (e.g. the validation loop) regroup samples by provenance and recover
    an exact masked mean per group. Averaging per-sample means instead would
    double-normalize and weight a 1-slot sample the same as a 200-slot one.
    """

    sum: Tensor
    count: Tensor

    def __add__(self, other: "PerSampleLoss") -> "PerSampleLoss":
        # Pool several loss components (e.g. discrete-action CE + response CE)
        # into one (numerator, denominator); the pooled per-slot mean is then
        # Σ(sum_i) / Σ(count_i) over the pooled slots.
        return PerSampleLoss(sum=self.sum + other.sum, count=self.count + other.count)


def make_action_dim_mask(
    real_action_dim: Tensor | None,
    max_action_dim: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Per-sample bool mask over action dims; True for real dims, False for zero-pad.

    Heterogeneous datasets are zero-padded to ``max_action_dim`` along the last
    action axis to keep batches rectangular, but the flow-matching MSE on the
    velocity field should only score real dims for each sample. This helper
    builds the per-dim mask that callers AND into their existing per-timestep
    mask before reducing.

    Args:
        real_action_dim: Optional ``(B,)`` long tensor of the real (pre-pad)
            action dimensionality for each sample (the batch key emitted by
            ``LeRobotDataset._to_standard_data_format``). When ``None``, the
            returned mask is all-True so the dim-mask AND in the caller's
            reduction is a no-op (pi0 additionally harmonized its `.mean()`
            to `sum / mask.sum()` in this PR — see the PR body's "pi0
            loss-magnitude shift" note; the dim-mask itself is still a
            no-op when ``real_action_dim`` is None).
        max_action_dim: The padded action dim (last-axis length of ``actions``).
        batch_size: Used to construct the all-True fallback shape; when
            ``real_action_dim`` is provided, must match
            ``real_action_dim.shape[0]``.
        device: Output device.

    Returns:
        ``(batch_size, max_action_dim)`` bool tensor.
    """
    if real_action_dim is None:
        return torch.ones((batch_size, max_action_dim), dtype=torch.bool, device=device)
    if real_action_dim.shape != (batch_size,):
        # Catch caller drift (e.g. a sliced `real_action_dim` passed with the
        # original `batch_size`) at the helper boundary — silent shape
        # mismatches propagate into broadcast errors deep in the loss reduction.
        raise ValueError(
            f"real_action_dim.shape {tuple(real_action_dim.shape)} does not match "
            f"batch_size={batch_size}; expected ({batch_size},)"
        )
    arange = torch.arange(max_action_dim, device=device)
    return rearrange(arange, "d -> 1 d") < rearrange(real_action_dim.to(device=device), "b -> b 1")


def flow_matching_masked_mse(
    u_t: Tensor,
    v_t: Tensor,
    *,
    max_action_dim: int,
    prefix_mask: Tensor | None = None,
    actions_is_pad: Tensor | None = None,
    real_action_dim: Tensor | None = None,
    return_per_sample: bool = False,
) -> Tensor | tuple[Tensor, PerSampleLoss]:
    """Masked MSE for flow-matching velocity-field training.

    Shared across pi05, pi05_mem, pi06, pi07 (low_level), and pi07_paligemma
    (low_level). Builds a `(B, chunk_size, max_action_dim)` mask that AND-s
    together up to three conditions and reduces ``F.mse_loss(u_t, v_t)``
    over the unmasked slots:

      1. **Frozen-prefix (RTI delay):** ``~prefix_mask`` — False where the
         model isn't asked to predict (the action prefix is the actually
         executed action from a previous inference, frozen as ground truth).
         Pass ``None`` to disable (non-RTI policies); the helper builds an
         all-False prefix mask internally so every step is supervised.
      2. **Per-timestep chunk padding:** ``~actions_is_pad`` — False where
         the action chunk extends past episode end. Pass ``None`` to skip.
         Also covers VQA-style items (``actions_is_pad`` all-True ⇒ loss = 0).
      3. **Per-sample real action dim:** built from ``real_action_dim`` via
         :func:`make_action_dim_mask`. False on the zero-pad tail dims of
         each sample. Pass ``None`` to score all ``max_action_dim`` columns.

    Args:
        u_t: Target velocity field, shape ``(B, chunk_size, D)`` (D ≥ max_action_dim).
        v_t: Predicted velocity field, same shape as ``u_t``.
        max_action_dim: Number of leading action dims to score against; trailing
            dims are dropped before reduction. Keyword-only.
        prefix_mask: Optional bool ``(B, chunk_size)`` — True where the step is
            frozen (RTI delay). ``None`` ⇒ all-False (non-RTI behavior).
        actions_is_pad: Optional bool ``(B, chunk_size)`` — True where the
            action chunk is padded (no real action target). ``None`` ⇒ all-False.
        real_action_dim: Optional long ``(B,)`` — real (pre-pad) action dim per
            sample. ``None`` ⇒ all-True (every dim is real).
        return_per_sample: When True, additionally return a :class:`PerSampleLoss`
            holding the per-sample ``(Σ over masked slots, #masked slots)`` so the
            caller can regroup the loss by provenance. The scalar is computed
            exactly as in the default path (bit-identical), so toggling this flag
            never perturbs the training reduction.

    Returns:
        Scalar tensor (masked mean of ``(u_t - v_t)**2`` over the unmasked slots)
        when ``return_per_sample`` is False; otherwise ``(scalar, PerSampleLoss)``
        where the per-sample ``sum``/``count`` are over the same masked slots
        (so each sample's mean is ``sum / count``).
    """
    mse_loss = F.mse_loss(u_t, v_t, reduction="none")
    bsz, chunk_size = u_t.shape[:2]
    if prefix_mask is None:
        prefix_mask = torch.zeros((bsz, chunk_size), dtype=torch.bool, device=u_t.device)
    postfix_mask = rearrange(torch.logical_not(prefix_mask), "b c -> b c 1")
    if actions_is_pad is not None:
        in_episode_bound = rearrange(~actions_is_pad, "b c -> b c 1")
        postfix_mask = torch.logical_and(postfix_mask, in_episode_bound)
    mse_loss = mse_loss[:, :, :max_action_dim]
    dim_mask = make_action_dim_mask(real_action_dim, max_action_dim, batch_size=bsz, device=u_t.device)
    full_mask = postfix_mask & rearrange(dim_mask, "b d -> b 1 d")
    masked = mse_loss * full_mask
    scalar = masked.sum() / (full_mask.sum() + 1e-8)
    if not return_per_sample:
        return scalar
    per_sample = PerSampleLoss(
        sum=reduce(masked, "b c d -> b", "sum"),
        count=reduce(full_mask.float(), "b c d -> b", "sum"),
    )
    return scalar, per_sample


def ce_per_sample(masked_ce: Tensor, valid_mask: Tensor) -> PerSampleLoss:
    """Per-sample numerator/denominator for a masked token cross-entropy.

    Policies compute their CE as ``F.cross_entropy(..., reduction="none")``
    reshaped to ``(B, S)`` and zeroed at pad positions, then reduce it with
    ``.mean()`` to a scalar. This helper takes that same pad-zeroed ``(B, S)``
    tensor plus the per-token validity mask and returns the per-sample
    ``(Σ over valid tokens, #valid tokens)``, so a caller can pool CE per
    provenance group as ``Σsum / Σcount`` — the mean cross-entropy per valid
    token. Multiple CE components (e.g. discrete-action + response) pool by
    adding their :class:`PerSampleLoss` objects.

    Note this normalizes by *valid* token count, unlike the legacy scalar
    ``.mean()`` which divides by the full ``B * S`` (pad slots included); the
    per-group breakdown is therefore over valid tokens only.

    Args:
        masked_ce: ``(B, S)`` cross-entropy already zeroed at pad positions.
        valid_mask: ``(B, S)`` bool, True at non-pad (scored) tokens.

    Returns:
        ``PerSampleLoss`` whose ``sum`` and ``count`` are ``(B,)``.
    """
    return PerSampleLoss(
        sum=reduce(masked_ce, "b s -> b", "sum"),
        count=reduce(valid_mask.float(), "b s -> b", "sum"),
    )


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model.

    Args:
        missing_keys: Keys that were expected but not found.
        unexpected_keys: Keys that were found but not expected.
    """
    if missing_keys:
        # DO NOT UPDATE THIS MESSAGE WITHOUT UPDATING THE REGEX IN .gitlab/scripts/check_pi0_state_keys.py
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        # DO NOT UPDATE THIS MESSAGE WITHOUT UPDATING THE REGEX IN .gitlab/scripts/check_pi0_state_keys.py
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")
