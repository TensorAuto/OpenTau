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

"""Show that bf16 Adam state harms convergence on a synthetic problem.

The dtype report from ``verify_adam_dtype_pure.py`` and
``verify_adam_dtype_live.py`` only proves the precision mismatch exists;
this script demonstrates that it actually harms convergence.

We fit two instances of AdamW on the same linear regression
(``y = X w_true + noise``, fixed seed) for 2000 steps:

1. ``bf16_params``: parameters live in bf16, so Adam state is bf16.
2. ``fp32_params_autocast``: parameters live in fp32, forward/backward is
   wrapped in ``torch.autocast(bfloat16)``. Adam state stays fp32, which
   mirrors the post-fix DDP path in ``train.py`` (issue #181, PR #176).

Loss is logged every 100 steps for both and the final loss ratio is
printed. The bf16-params path plateaus well above the fp32 path.
"""

from __future__ import annotations

import torch
from torch import nn

SEED = 1234
NUM_STEPS = 2000
LOG_EVERY = 100
FEATURES = 512
BATCH = 256


def make_regression_problem(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed-seed synthetic linear regression dataset.

    Args:
        device: Device on which to allocate the generated tensors.

    Returns:
        A ``(inputs, targets)`` pair where ``inputs`` is
        ``(BATCH, FEATURES)`` and ``targets`` is ``(BATCH,)``.
        Ground-truth weights are fixed across calls.
    """
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    inputs = torch.randn(BATCH, FEATURES, generator=generator).to(device)
    w_true = torch.randn(FEATURES, generator=generator).to(device)
    noise = 0.01 * torch.randn(BATCH, generator=generator).to(device)
    targets = inputs @ w_true + noise
    return inputs, targets


def _linear_fp32(features: int, device: torch.device) -> nn.Linear:
    model = nn.Linear(features, 1, bias=False)
    nn.init.zeros_(model.weight)
    return model.to(device=device, dtype=torch.float32)


def train_bf16_params(inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> list[float]:
    """Train with bf16 parameters (bf16 Adam state) and log per-step loss.

    Args:
        inputs: Input features of shape ``(BATCH, FEATURES)``.
        targets: Targets of shape ``(BATCH,)``.
        device: Device used for all tensors.

    Returns:
        A list of losses recorded every ``LOG_EVERY`` steps. The last
        entry is the final training loss.
    """
    torch.manual_seed(SEED)
    model = _linear_fp32(FEATURES, device).to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs_bf16 = inputs.to(dtype=torch.bfloat16)
    targets_bf16 = targets.to(dtype=torch.bfloat16)

    losses: list[float] = []
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        pred = model(inputs_bf16).squeeze(-1)
        loss = (pred - targets_bf16).pow(2).mean()
        loss.backward()
        optimizer.step()
        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            losses.append(float(loss.detach().to(dtype=torch.float32).item()))
            print(f"[bf16 params]    step={step:5d} loss={losses[-1]:.6f}")
    return losses


def train_fp32_params_autocast(
    inputs: torch.Tensor, targets: torch.Tensor, device: torch.device
) -> list[float]:
    """Train fp32 params with bf16 autocast; Adam state stays fp32.

    Args:
        inputs: Input features of shape ``(BATCH, FEATURES)``.
        targets: Targets of shape ``(BATCH,)``.
        device: Device used for all tensors.

    Returns:
        Losses sampled every ``LOG_EVERY`` steps; last entry is the
        final training loss. This path mirrors the post-fix DDP config.
    """
    torch.manual_seed(SEED)
    model = _linear_fp32(FEATURES, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = model(inputs).squeeze(-1)
            loss = (pred - targets).pow(2).mean()
        loss.backward()
        optimizer.step()
        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            losses.append(float(loss.detach().to(dtype=torch.float32).item()))
            print(f"[fp32+autocast]  step={step:5d} loss={losses[-1]:.6f}")
    return losses


def main() -> None:
    """Run both training paths and print the final loss ratio."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    inputs, targets = make_regression_problem(device)

    bf16_losses = train_bf16_params(inputs, targets, device)
    fp32_losses = train_fp32_params_autocast(inputs, targets, device)

    bf16_final = bf16_losses[-1]
    fp32_final = fp32_losses[-1]
    ratio = bf16_final / max(fp32_final, 1e-12)
    print(f"final bf16 loss  = {bf16_final:.6f}")
    print(f"final fp32 loss  = {fp32_final:.6f}")
    print(f"bf16 / fp32 ratio = {ratio:.2f}")


if __name__ == "__main__":
    main()
