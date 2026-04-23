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

import pytest

from opentau.scripts.train import _find_unused_params_from_env


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
