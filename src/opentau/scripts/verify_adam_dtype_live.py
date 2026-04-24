#!/usr/bin/env python
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

"""Print Adam optimizer-state dtypes under a live accelerate configuration.

This script replicates the exact prefix of ``train.py`` (build policy,
conditionally cast to bf16, construct AdamW, call ``accelerator.prepare``)
but on a minimal synthetic ``nn.Linear`` instead of pi05. It is meant to
be launched via ``accelerate launch`` so we can A/B the DDP and DeepSpeed
configs referenced in issue #181.

Usage:
    accelerate launch --config_file <cfg> \
        src/opentau/scripts/verify_adam_dtype_live.py

After ``accelerator.prepare``, one real backward + optimizer step is
taken, then ``optimizer.state`` is walked and
``(name, param.dtype, exp_avg.dtype, exp_avg_sq.dtype)`` is printed for
each parameter. The distributed type and accelerate mixed-precision
setting are printed up front so the reviewer can tell which branch of
the fix is exercised.
"""

from __future__ import annotations

import logging

import accelerate
import torch
from accelerate.optimizer import AcceleratedOptimizer
from torch import nn


def build_model(in_features: int = 1024, out_features: int = 1024) -> nn.Module:
    """Construct a minimal stand-in for the real policy.

    Args:
        in_features: Width of the linear layer's input.
        out_features: Width of the linear layer's output.

    Returns:
        A two-parameter ``nn.Linear`` model. Kept intentionally small so
        the script fits on CPU/single-GPU smoke tests.
    """
    return nn.Linear(in_features, out_features)


def cast_if_deepspeed(model: nn.Module, accelerator: accelerate.Accelerator) -> None:
    """Mirror the ``train.py`` bf16 cast policy from issue #181's fix.

    Args:
        model: The module to (possibly) cast in-place to bf16.
        accelerator: The already-constructed accelerator; its
            ``distributed_type`` selects the branch.
    """
    if accelerator.distributed_type == accelerate.DistributedType.DEEPSPEED:
        model.to(torch.bfloat16)


def step_once(model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Run a single backward + optimizer step with a dummy loss.

    Args:
        model: The prepared model (possibly an accelerate wrapper).
        optimizer: The prepared optimizer (an ``AcceleratedOptimizer``).
        device: Device used to allocate the synthetic input batch.
    """
    x = torch.randn(4, 1024, device=device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def print_state_dtypes(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Walk ``optimizer.state`` and print dtype info for every parameter.

    Args:
        model: The prepared model; used to recover per-parameter names.
        optimizer: The prepared optimizer. If it is an
            ``AcceleratedOptimizer`` it is unwrapped to reach the real
            torch optimizer whose ``state`` carries ``exp_avg`` and
            ``exp_avg_sq``.
    """
    inner = optimizer.optimizer if isinstance(optimizer, AcceleratedOptimizer) else optimizer
    # Build a reverse lookup: id(param) -> name. Works for both the DDP
    # wrapper and the raw module.
    underlying = model.module if hasattr(model, "module") else model
    id_to_name = {id(p): n for n, p in underlying.named_parameters()}

    print(f"{'name':<30} {'param':<12} {'exp_avg':<12} {'exp_avg_sq':<12}")
    for param, state in inner.state.items():
        name = id_to_name.get(id(param), "<unknown>")
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        exp_avg_dtype = exp_avg.dtype if exp_avg is not None else None
        exp_avg_sq_dtype = exp_avg_sq.dtype if exp_avg_sq is not None else None
        print(f"{name:<30} {str(param.dtype):<12} {str(exp_avg_dtype):<12} {str(exp_avg_sq_dtype):<12}")


def main() -> None:
    """Entry point: prepare a minimal model + optimizer and dump dtypes."""
    logging.basicConfig(level=logging.INFO)
    accelerator = accelerate.Accelerator()

    print(f"distributed_type = {accelerator.distributed_type}")
    print(f"mixed_precision  = {accelerator.state.mixed_precision}")

    model = build_model()
    cast_if_deepspeed(model, accelerator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model, optimizer = accelerator.prepare(model, optimizer)
    step_once(model, optimizer, accelerator.device)
    print_state_dtypes(model, optimizer)


if __name__ == "__main__":
    main()
