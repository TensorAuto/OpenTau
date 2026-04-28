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
import pytest
import torch

from opentau.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from opentau.optim.optimizers import (
    AdamConfig,
    AdamWConfig,
    SGDConfig,
    load_optimizer_state,
    save_optimizer_state,
)


@pytest.mark.parametrize(
    "config_cls, expected_class",
    [
        (AdamConfig, torch.optim.Adam),
        (AdamWConfig, torch.optim.AdamW),
        (SGDConfig, torch.optim.SGD),
    ],
)
def test_optimizer_build(config_cls, expected_class, model_params):
    config = config_cls()
    optimizer = config.build(model_params)
    assert isinstance(optimizer, expected_class)
    assert optimizer.defaults["lr"] == config.lr


def test_save_optimizer_state(optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    assert (tmp_path / OPTIMIZER_STATE).is_file()
    assert (tmp_path / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_and_load_optimizer_state(model_params, optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    loaded_optimizer = AdamConfig().build(model_params)
    loaded_optimizer = load_optimizer_state(loaded_optimizer, tmp_path)

    torch.testing.assert_close(optimizer.state_dict(), loaded_optimizer.state_dict())


@pytest.mark.parametrize("config_cls", [AdamConfig, AdamWConfig])
def test_fused_falls_back_to_foreach_on_cpu_only_host(config_cls, model_params, monkeypatch):
    """``fused=True`` must fall back to ``foreach`` when CUDA is unavailable.

    PyTorch raises at AdamW construction time if ``fused=True`` is
    requested without a CUDA (or MPS) device. The fallback in ``build()``
    is the only thing that keeps CPU-only tests (including this one) and
    single-process debug runs from breaking when the default flips to
    True.
    """
    # Force the CPU-only branch regardless of what this host actually has,
    # so the test passes identically on CI runners with or without GPUs.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = config_cls(fused=True)
    optimizer = config.build(model_params)

    # After the fallback, fused must be False (or absent / None) in the
    # optimizer's defaults — otherwise the first .step() would raise.
    assert optimizer.defaults.get("fused") in (False, None), (
        f"fused should have been forced to False on CPU-only host, got {optimizer.defaults.get('fused')!r}"
    )
