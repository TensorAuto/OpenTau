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

"""Verify PyTorch's AdamW allocates optimizer state in the parameter dtype.

This is the minimal (no accelerate, no GPU) reproduction of the precision
bug reported in issue #181. When ``torch.optim.AdamW`` is constructed over
bf16 parameters, its ``exp_avg`` and ``exp_avg_sq`` state tensors are
initialised via ``torch.zeros_like(p)`` and therefore inherit bf16. The
Adam update then runs in bf16 with ``eps=1e-8``, which silently corrupts
training over many steps (see issue #181, PR #176).

The script exits 0 when the bug is reproduced (i.e. state is bf16, the
current PyTorch behaviour), 1 otherwise (e.g. a future PyTorch change).
"""

from __future__ import annotations

import sys

import torch


def run_adam_step(dtype: torch.dtype) -> torch.Tensor:
    """Construct AdamW over a single parameter and take one optimizer step.

    Args:
        dtype: The dtype used for the parameter tensor. Adam state inherits
            this dtype in the current PyTorch implementation.

    Returns:
        The dtype of the ``exp_avg`` state tensor after the step.
    """
    param = torch.zeros(4, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    # Arbitrary non-trivial loss to generate gradients.
    loss = (param * 2.0 + 1.0).sum()
    loss.backward()
    optimizer.step()
    state = optimizer.state[param]
    return state["exp_avg"].dtype


def main() -> int:
    """Run the dtype check and print a short diagnostic report.

    Returns:
        0 if the bf16 bug is reproduced (current PyTorch behaviour), 1
        otherwise. The caller's shell can forward this exit code.
    """
    bf16_state_dtype = run_adam_step(torch.bfloat16)
    fp32_state_dtype = run_adam_step(torch.float32)

    print(f"bf16 param -> exp_avg dtype: {bf16_state_dtype}")
    print(f"fp32 param -> exp_avg dtype: {fp32_state_dtype}")

    # The "bug" here is a fact about PyTorch, not an OpenTau defect: state
    # inherits the parameter dtype. The fix lives in train.py (do not cast
    # the policy to bf16 under DDP).
    bug_reproduced = bf16_state_dtype == torch.bfloat16 and fp32_state_dtype == torch.float32
    if bug_reproduced:
        print("OK: reproduced expected behaviour (bf16 params -> bf16 Adam state).")
        return 0
    print("FAIL: PyTorch AdamW state dtype handling differs from issue #181 assumption.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
