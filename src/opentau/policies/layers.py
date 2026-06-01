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

"""Reusable parametric layers shared across VLA policies.

:class:`PerGroupLinear` is a drop-in replacement for ``nn.Linear`` that keeps an
independent ``(weight, bias)`` per group and selects one row per sample with a
``(B,)`` long index — the same per-sample group axis the stacked
Normalize/Unnormalize stat buffers use (see :mod:`opentau.policies.normalize`).
Its state_dict key names match ``nn.Linear`` exactly (``<name>.weight`` /
``<name>.bias``) but carry a leading group axis, so the legacy single-group
promotion shim (``unsqueeze(0)``-style, see
``PreTrainedPolicy._promote_legacy_norm_buffers_in_state_dict``) applies to it.
"""

import math

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class PerGroupLinear(nn.Module):
    """``nn.Linear`` with one independent ``(weight, bias)`` per group.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        num_groups: Number of independent linear maps. ``1`` makes this
            numerically identical to a plain ``nn.Linear`` — the forward takes
            an ``F.linear`` fast path with the un-cast parameters, so its dtype
            / autocast behavior matches a plain ``nn.Linear`` bit-for-bit.
        bias: If ``True`` (default), adds a per-group learnable bias.

    Shape:
        - weight: ``(num_groups, out_features, in_features)``
        - bias: ``(num_groups, out_features)``

        These are an ``nn.Linear`` parameter with a leading group axis, which is
        why the state_dict keys stay identical to ``nn.Linear`` (just one rank
        higher). A legacy ``nn.Linear`` checkpoint promotes into a
        ``num_groups=1`` ``PerGroupLinear`` by prepending that axis.

    Forward:
        ``forward(x, group_index=None)`` where ``x`` is ``(B, *, in_features)``
        and ``group_index`` is a ``(B,)`` long tensor in ``[0, num_groups)``.
        ``group_index=None`` routes every sample to group ``0`` (the
        single-group default; used by direct-call unit tests and any caller
        that has not threaded an index).
    """

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True):
        super().__init__()
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}.")
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.empty(num_groups, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every group exactly like ``nn.Linear.reset_parameters``.

        Per-row kaiming-uniform weight (``a=sqrt(5)``) + fan-in-scaled uniform
        bias, so a freshly built per-group module matches ``nn.Linear``'s init
        distribution on each row.
        """
        for g in range(self.num_groups):
            nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[g])
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(self.bias[g], -bound, bound)

    def forward(self, x: Tensor, group_index: Tensor | None = None) -> Tensor:
        if self.num_groups == 1:
            # Bit-identical to a plain nn.Linear: same op, same un-cast params,
            # so the autocast / dtype promotion matches the pre-flag baseline.
            bias = None if self.bias is None else self.bias[0]
            return F.linear(x, self.weight[0], bias)

        if group_index is None:
            group_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Per-sample gather of the (out, in) map. `index_select` on a (B,) index
        # is the same op normalize._gather_and_broadcast uses for the stacked
        # stat buffers — ONNX/dynamo-traceable and collective-safe under FSDP
        # (fixed shape regardless of which groups appear in the local batch).
        # Cast to x's dtype so the matmul runs at the same precision the
        # autocast baseline would (and works without an active autocast too).
        weight = self.weight.index_select(0, group_index).to(dtype=x.dtype)  # (B, out, in)
        out = einops.einsum(x, weight, "b ... i, b o i -> b ... o")
        if self.bias is not None:
            bias = self.bias.index_select(0, group_index).to(dtype=x.dtype)  # (B, out)
            # Broadcast the per-sample bias across x's interior dims. Computed-
            # rank reshape (not an einops pattern) because the number of
            # interior dims is data-dependent — same rationale as
            # normalize._gather_and_broadcast.
            extra = out.ndim - bias.ndim
            bias = bias.reshape(bias.shape[0], *((1,) * extra), bias.shape[-1])
            out = out + bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_groups={self.num_groups}, bias={self.bias is not None}"
        )
