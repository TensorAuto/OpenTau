#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Per-dim action mask for heterogeneous-DoF flow-matching MSE.

The padded tail dims of ``actions`` must not contribute to the velocity
loss when datasets in a mixture have different native action dims. These
tests exercise the helper and the per-policy MSE reduction directly,
with hand-crafted ``u_t`` / ``v_t`` tensors so the loss is predictable.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

from opentau.policies.pi06.modeling_pi06 import flow_matching_masked_mse
from opentau.policies.utils import make_action_dim_mask


def _reference_dense_mse(u_t: torch.Tensor, v_t: torch.Tensor, action_is_pad: torch.Tensor) -> torch.Tensor:
    """Pre-fix MSE: sum-of-squares masked only by timestep, divided by the
    count of unmasked ``(b, c, d)`` slots (the historical sum/denom pattern)."""
    mse = F.mse_loss(u_t, v_t, reduction="none")
    timestep_mask = rearrange(~action_is_pad, "b c -> b c 1")
    full = timestep_mask.expand(-1, -1, mse.shape[-1])
    return (mse * full).sum() / (full.sum() + 1e-8)


def test_helper_shape_and_values():
    action_dim = torch.tensor([7, 14], dtype=torch.long)
    mask = make_action_dim_mask(action_dim, max_action_dim=32, batch_size=2, device=torch.device("cpu"))
    assert mask.shape == (2, 32)
    assert mask.dtype == torch.bool
    # row 0: first 7 columns True, rest False
    assert mask[0, :7].all() and not mask[0, 7:].any()
    # row 1: first 14 columns True, rest False
    assert mask[1, :14].all() and not mask[1, 14:].any()


def test_helper_none_returns_all_true():
    mask = make_action_dim_mask(None, max_action_dim=32, batch_size=4, device=torch.device("cpu"))
    assert mask.shape == (4, 32)
    assert mask.all()


def test_helper_device_placement_when_action_dim_on_cpu():
    """The helper must move ``action_dim`` to the requested device."""
    action_dim = torch.tensor([3, 5], dtype=torch.long, device="cpu")
    mask = make_action_dim_mask(action_dim, max_action_dim=8, batch_size=2, device=torch.device("cpu"))
    assert mask.device.type == "cpu"


def test_flow_matching_masked_mse_excludes_padded_dims():
    """A sample's loss must only count its first ``action_dim`` columns."""
    torch.manual_seed(0)
    bsz, chunk, dim = 2, 4, 8
    u_t = torch.randn(bsz, chunk, dim)
    v_t = torch.randn(bsz, chunk, dim)
    prefix_mask = torch.zeros(bsz, chunk, dtype=torch.bool)
    actions_is_pad = torch.zeros(bsz, chunk, dtype=torch.bool)
    action_dim = torch.tensor([3, 5], dtype=torch.long)

    loss = flow_matching_masked_mse(
        u_t, v_t, prefix_mask, actions_is_pad, max_action_dim=dim, action_dim=action_dim
    )

    # Manually compute the expected loss over the unmasked slots only.
    per_elem = (u_t - v_t).pow(2)
    expected_num = per_elem[0, :, :3].sum() + per_elem[1, :, :5].sum()
    expected_denom = chunk * 3 + chunk * 5
    expected = expected_num / (expected_denom + 1e-8)
    torch.testing.assert_close(loss, expected)


def test_flow_matching_masked_mse_none_action_dim_matches_old_behavior():
    """``action_dim=None`` must reproduce the pre-fix sum/denom result."""
    torch.manual_seed(42)
    bsz, chunk, dim = 3, 5, 6
    u_t = torch.randn(bsz, chunk, dim)
    v_t = torch.randn(bsz, chunk, dim)
    prefix_mask = torch.zeros(bsz, chunk, dtype=torch.bool)
    actions_is_pad = torch.zeros(bsz, chunk, dtype=torch.bool)

    new = flow_matching_masked_mse(u_t, v_t, prefix_mask, actions_is_pad, max_action_dim=dim, action_dim=None)
    old = _reference_dense_mse(u_t, v_t, actions_is_pad)
    torch.testing.assert_close(new, old)


def test_flow_matching_masked_mse_action_dim_at_max_matches_old_behavior():
    """When every sample's ``action_dim == max_action_dim`` the dim mask is
    a no-op and the loss must equal the pre-fix computation bit-exactly."""
    torch.manual_seed(7)
    bsz, chunk, dim = 4, 6, 10
    u_t = torch.randn(bsz, chunk, dim)
    v_t = torch.randn(bsz, chunk, dim)
    prefix_mask = torch.zeros(bsz, chunk, dtype=torch.bool)
    actions_is_pad = torch.zeros(bsz, chunk, dtype=torch.bool)
    action_dim = torch.full((bsz,), dim, dtype=torch.long)

    new = flow_matching_masked_mse(
        u_t, v_t, prefix_mask, actions_is_pad, max_action_dim=dim, action_dim=action_dim
    )
    old = _reference_dense_mse(u_t, v_t, actions_is_pad)
    torch.testing.assert_close(new, old)


def test_flow_matching_masked_mse_combines_with_timestep_pad():
    """A slot must be excluded if its timestep OR its dim is padded."""
    bsz, chunk, dim = 2, 4, 6
    # u_t - v_t = 1 everywhere → per-element MSE = 1.
    u_t = torch.ones(bsz, chunk, dim)
    v_t = torch.zeros(bsz, chunk, dim)
    prefix_mask = torch.zeros(bsz, chunk, dtype=torch.bool)
    # Pad the last two timesteps of sample 0.
    actions_is_pad = torch.zeros(bsz, chunk, dtype=torch.bool)
    actions_is_pad[0, -2:] = True
    # action_dim = [4, 2]
    action_dim = torch.tensor([4, 2], dtype=torch.long)

    loss = flow_matching_masked_mse(
        u_t, v_t, prefix_mask, actions_is_pad, max_action_dim=dim, action_dim=action_dim
    )

    # Sample 0: 2 real timesteps * 4 real dims = 8 unmasked slots, each contributing 1.
    # Sample 1: 4 real timesteps * 2 real dims = 8 unmasked slots, each contributing 1.
    expected_num = 8 + 8
    expected_denom = 8 + 8
    expected = torch.tensor(expected_num / (expected_denom + 1e-8))
    torch.testing.assert_close(loss, expected)


def test_action_dim_zero_yields_zero_loss():
    """A sample with ``action_dim=0`` (e.g. VQA-style item with no real action)
    contributes nothing to the loss, and behaves as if fully masked."""
    bsz, chunk, dim = 2, 3, 4
    u_t = torch.ones(bsz, chunk, dim)
    v_t = torch.zeros(bsz, chunk, dim)
    prefix_mask = torch.zeros(bsz, chunk, dtype=torch.bool)
    actions_is_pad = torch.zeros(bsz, chunk, dtype=torch.bool)
    action_dim = torch.tensor([0, 0], dtype=torch.long)

    loss = flow_matching_masked_mse(
        u_t, v_t, prefix_mask, actions_is_pad, max_action_dim=dim, action_dim=action_dim
    )
    # Numerator and denominator both 0 → loss = 0 / (0 + eps) ≈ 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_helper_is_total_when_action_dim_exceeds_max():
    """``make_action_dim_mask`` clamps via the strict-less comparison, so
    requesting ``action_dim > max_action_dim`` simply yields an all-True
    mask (same as ``None``). The hard boundary check lives in the dataset
    standardization path (see ``tests/datasets/test_action_dim.py``); here
    we just confirm the helper is total and doesn't raise."""
    action_dim = torch.tensor([99], dtype=torch.long)
    mask = make_action_dim_mask(action_dim, max_action_dim=4, batch_size=1, device=torch.device("cpu"))
    assert mask.shape == (1, 4)
    assert mask.all()
