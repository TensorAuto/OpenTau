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

import pytest

from opentau.scripts.annotate_mistakes import _find_response_runs, _parse_success_response


class TestFindResponseRuns:
    def test_empty(self):
        assert _find_response_runs([]) == []

    def test_single_run(self):
        assert _find_response_runs(["a", "a", "a"]) == [(0, 2, "a")]

    def test_multiple_runs(self):
        assert _find_response_runs(["a", "a", "b", "b", "b", "c"]) == [
            (0, 1, "a"),
            (2, 4, "b"),
            (5, 5, "c"),
        ]

    def test_skips_empty_string_run(self):
        # Falsy responses (e.g. "") form runs but are not emitted.
        assert _find_response_runs(["", "", "a", "a"]) == [(2, 3, "a")]

    def test_skips_none_run(self):
        # None responses are also dropped — not emitted as a run.
        assert _find_response_runs([None, None, "a"]) == [(2, 2, "a")]

    def test_alternating_with_empties(self):
        assert _find_response_runs(["a", "", "a"]) == [(0, 0, "a"), (2, 2, "a")]

    def test_all_empty(self):
        assert _find_response_runs(["", "", ""]) == []

    def test_singleton(self):
        assert _find_response_runs(["a"]) == [(0, 0, "a")]

    def test_singleton_empty(self):
        assert _find_response_runs([""]) == []


class TestParseSuccessResponse:
    def test_plain_true(self):
        assert _parse_success_response('{"success": true, "reason": "ok"}') is True

    def test_plain_false(self):
        assert _parse_success_response('{"success": false, "reason": "nope"}') is False

    def test_strips_json_fence(self):
        text = '```json\n{"success": true, "reason": "ok"}\n```'
        assert _parse_success_response(text) is True

    def test_strips_bare_fence(self):
        text = '```\n{"success": false, "reason": "x"}\n```'
        assert _parse_success_response(text) is False

    def test_strips_surrounding_whitespace(self):
        assert _parse_success_response('   \n{"success": true}\n  ') is True

    def test_rejects_string_false(self):
        # Stringy "false" is truthy in Python — must NOT be coerced silently.
        with pytest.raises(ValueError, match="Expected boolean 'success' value"):
            _parse_success_response('{"success": "false", "reason": "x"}')

    def test_rejects_string_true(self):
        with pytest.raises(ValueError, match="Expected boolean 'success' value"):
            _parse_success_response('{"success": "true", "reason": "x"}')

    def test_rejects_int(self):
        # bool is a subclass of int in Python — make sure we don't accept ints either.
        with pytest.raises(ValueError, match="Expected boolean 'success' value"):
            _parse_success_response('{"success": 1, "reason": "x"}')

    def test_rejects_missing_key(self):
        with pytest.raises(ValueError, match="Expected JSON object with 'success' key"):
            _parse_success_response('{"reason": "x"}')

    def test_rejects_non_object(self):
        with pytest.raises(ValueError, match="Expected JSON object with 'success' key"):
            _parse_success_response("[true]")

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError):
            _parse_success_response("not json at all")
