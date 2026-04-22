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

from __future__ import annotations

import logging
from types import SimpleNamespace

import accelerate

from opentau.scripts.train import _sync_deepspeed_gradient_accumulation_steps


def _make_accelerator(distributed_type, ds_grad_acc, is_main_process=True):
    plugin = SimpleNamespace(
        hf_ds_config=SimpleNamespace(config={"gradient_accumulation_steps": ds_grad_acc}),
        gradient_accumulation_steps=ds_grad_acc,
    )
    return SimpleNamespace(
        distributed_type=distributed_type,
        is_main_process=is_main_process,
        deepspeed_plugin=plugin,
    )


def _make_cfg(grad_acc):
    return SimpleNamespace(gradient_accumulation_steps=grad_acc)


def test_non_deepspeed_is_noop(caplog):
    accelerator = _make_accelerator(accelerate.DistributedType.MULTI_GPU, ds_grad_acc=2)
    cfg = _make_cfg(grad_acc=4)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 2
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 2
    assert not caplog.records


def test_deepspeed_matching_value_no_warning(caplog):
    accelerator = _make_accelerator(accelerate.DistributedType.DEEPSPEED, ds_grad_acc=2)
    cfg = _make_cfg(grad_acc=2)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 2
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_deepspeed_mismatch_overrides_and_warns_on_main(caplog):
    accelerator = _make_accelerator(accelerate.DistributedType.DEEPSPEED, ds_grad_acc=2, is_main_process=True)
    cfg = _make_cfg(grad_acc=4)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 4
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 4
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2" in message
    assert "4" in message


def test_deepspeed_mismatch_non_main_overrides_without_warning(caplog):
    accelerator = _make_accelerator(
        accelerate.DistributedType.DEEPSPEED, ds_grad_acc=2, is_main_process=False
    )
    cfg = _make_cfg(grad_acc=4)

    with caplog.at_level(logging.WARNING):
        _sync_deepspeed_gradient_accumulation_steps(accelerator, cfg)

    assert accelerator.deepspeed_plugin.hf_ds_config.config["gradient_accumulation_steps"] == 4
    assert accelerator.deepspeed_plugin.gradient_accumulation_steps == 4
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
