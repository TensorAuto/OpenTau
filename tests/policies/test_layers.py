#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :class:`opentau.policies.layers.PerGroupLinear`."""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from opentau.policies.layers import PerGroupLinear


def test_single_group_bit_identical_to_nn_linear():
    """num_groups==1 with copied weights matches nn.Linear bit-for-bit."""
    torch.manual_seed(0)
    lin = nn.Linear(8, 5)
    pgl = PerGroupLinear(8, 5, num_groups=1)
    with torch.no_grad():
        pgl.weight.copy_(lin.weight[None])
        pgl.bias.copy_(lin.bias[None])
    x = torch.randn(4, 3, 8)
    assert torch.equal(pgl(x), lin(x))  # group_index=None -> fast path
    assert torch.equal(pgl(x, torch.zeros(4, dtype=torch.long)), lin(x))


def test_multi_group_per_sample_routing_matches_reference():
    """Each sample's output uses the row picked by its group_index."""
    torch.manual_seed(0)
    g, b, t, i, o = 3, 5, 2, 8, 6
    pgl = PerGroupLinear(i, o, num_groups=g)
    x = torch.randn(b, t, i)
    idx = torch.tensor([0, 2, 1, 2, 0], dtype=torch.long)
    out = pgl(x, idx)
    ref = torch.stack([F.linear(x[k], pgl.weight[idx[k]], pgl.bias[idx[k]]) for k in range(b)])
    torch.testing.assert_close(out, ref)


def test_group_index_none_routes_to_group_zero():
    torch.manual_seed(0)
    pgl = PerGroupLinear(4, 4, num_groups=3)
    x = torch.randn(2, 4)
    ref = torch.stack([F.linear(x[k], pgl.weight[0], pgl.bias[0]) for k in range(2)])
    torch.testing.assert_close(pgl(x), ref)


def test_bias_broadcasts_over_extra_middle_dims():
    pgl = PerGroupLinear(3, 2, num_groups=2)
    x = torch.randn(4, 5, 7, 3)
    idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    out = pgl(x, idx)
    assert out.shape == (4, 5, 7, 2)
    ref = torch.stack([F.linear(x[k], pgl.weight[idx[k]], pgl.bias[idx[k]]) for k in range(4)])
    torch.testing.assert_close(out, ref)


def test_no_bias_variant():
    pgl = PerGroupLinear(3, 2, num_groups=2, bias=False)
    assert pgl.bias is None
    x = torch.randn(2, 3)
    idx = torch.tensor([0, 1], dtype=torch.long)
    out = pgl(x, idx)
    ref = torch.stack([F.linear(x[k], pgl.weight[idx[k]], None) for k in range(2)])
    torch.testing.assert_close(out, ref)


def test_state_dict_keys_match_nn_linear():
    """Keys are `weight`/`bias` (just one rank higher) so the legacy promotion
    shim that prepends a leading axis applies."""
    pgl = PerGroupLinear(3, 2, num_groups=2)
    assert set(pgl.state_dict()) == {"weight", "bias"}
    assert pgl.weight.shape == (2, 2, 3)
    assert pgl.bias.shape == (2, 2)


def test_legacy_nn_linear_promotes_into_single_group():
    """An nn.Linear state_dict loads into a 1-group PerGroupLinear after the
    unsqueeze(0) promotion, and forwards identically."""
    torch.manual_seed(0)
    lin = nn.Linear(8, 5)
    promoted = {k: v.unsqueeze(0) for k, v in lin.state_dict().items()}
    pgl = PerGroupLinear(8, 5, num_groups=1)
    missing, unexpected = pgl.load_state_dict(promoted, strict=True)
    assert not missing and not unexpected
    x = torch.randn(3, 8)
    torch.testing.assert_close(pgl(x), lin(x))


def test_num_groups_must_be_positive():
    with pytest.raises(ValueError, match="num_groups"):
        PerGroupLinear(3, 2, num_groups=0)


def test_construction_is_seed_deterministic():
    torch.manual_seed(123)
    a = PerGroupLinear(6, 4, num_groups=3)
    torch.manual_seed(123)
    b = PerGroupLinear(6, 4, num_groups=3)
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.bias, b.bias)


def test_init_matches_nn_linear_bounds_per_row():
    """Every group is kaiming-uniform like nn.Linear: |w| <= 1/sqrt(fan_in)."""
    pgl = PerGroupLinear(16, 8, num_groups=4)
    bound = 1.0 / math.sqrt(16)
    assert pgl.weight.abs().max().item() <= bound + 1e-6
