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

"""Unit tests for pure helpers in train.

The actual ``train()`` entrypoint wraps a full training loop and is
exercised end-to-end by ``.github/workflows/regression_test.yml``;
this file only covers the small parsing helpers that benefit from
isolated coverage.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import accelerate
import pytest

from opentau.scripts.train import (
    _find_unused_params_from_env,
    _sync_deepspeed_gradient_accumulation_steps,
)


class TestFindUnusedParamsFromEnv:
    """``FIND_UNUSED_PARAMS`` env var → bool."""

    def test_default_is_true_when_unset(self, monkeypatch):
        monkeypatch.delenv("FIND_UNUSED_PARAMS", raising=False)
        assert _find_unused_params_from_env() is True

    def test_explicit_true(self, monkeypatch):
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "true")
        assert _find_unused_params_from_env() is True

    def test_explicit_false(self, monkeypatch):
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "false")
        assert _find_unused_params_from_env() is False

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "TRUE")
        assert _find_unused_params_from_env() is True

        monkeypatch.setenv("FIND_UNUSED_PARAMS", "False")
        assert _find_unused_params_from_env() is False

        monkeypatch.setenv("FIND_UNUSED_PARAMS", "FALSE")
        assert _find_unused_params_from_env() is False

    def test_empty_string_is_treated_as_false(self, monkeypatch):
        # Empty string != "true" so we default to False. Matches the
        # strict "anything-non-true is false" semantics of the inline
        # expression we extracted; flipping to False on empty is the
        # safe choice because a misspelled env var should not silently
        # preserve the old default behavior.
        monkeypatch.setenv("FIND_UNUSED_PARAMS", "")
        assert _find_unused_params_from_env() is False

    def test_unknown_values_parse_as_false(self, monkeypatch):
        # "yes" / "1" / anything other than a case-insensitive "true"
        # counts as False. Keeps the parser strict.
        for value in ("yes", "1", "enabled", "Y", "on"):
            monkeypatch.setenv("FIND_UNUSED_PARAMS", value)
            assert _find_unused_params_from_env() is False, f"Expected {value!r} to parse as False, got True"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
