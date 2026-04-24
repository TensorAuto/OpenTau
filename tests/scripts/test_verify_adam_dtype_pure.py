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

"""CPU regression guard for the bf16 Adam-state assumption in issue #181.

The production fix is in ``src/opentau/scripts/train.py``; this test
simply pins the upstream PyTorch behaviour that motivates it so we get a
signal if ``torch.optim.AdamW`` ever starts up-casting its state.
"""

from __future__ import annotations

import torch

from opentau.scripts.verify_adam_dtype_pure import main, run_adam_step


def test_bf16_param_yields_bf16_state():
    """Adam state inherits the parameter dtype for bf16 params."""
    assert run_adam_step(torch.bfloat16) == torch.bfloat16


def test_fp32_param_yields_fp32_state():
    """Adam state inherits the parameter dtype for fp32 params."""
    assert run_adam_step(torch.float32) == torch.float32


def test_script_main_exits_zero():
    """The ``verify_adam_dtype_pure`` script exits 0 on current PyTorch."""
    assert main() == 0
