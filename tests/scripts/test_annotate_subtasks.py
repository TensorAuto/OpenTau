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

from opentau.scripts.annotate_subtasks import _coerce_subtasks, _parse_json_response


class TestParseJsonResponse:
    def test_plain_json_array(self):
        assert _parse_json_response('[{"time": 0.0, "subtask": "a"}]') == [{"time": 0.0, "subtask": "a"}]

    def test_strips_json_fence(self):
        text = '```json\n[{"time": 0.0, "subtask": "a"}]\n```'
        assert _parse_json_response(text) == [{"time": 0.0, "subtask": "a"}]

    def test_strips_bare_fence(self):
        text = '```\n[{"time": 0.0, "subtask": "a"}]\n```'
        assert _parse_json_response(text) == [{"time": 0.0, "subtask": "a"}]

    def test_strips_surrounding_whitespace(self):
        assert _parse_json_response("   \n[]\n  ") == []

    def test_rejects_non_array(self):
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_json_response('{"time": 0.0, "subtask": "a"}')

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError):
            _parse_json_response("not json at all")


class TestCoerceSubtasks:
    def test_passes_well_formed_entries(self):
        out = _coerce_subtasks([{"time": 0.0, "subtask": "a"}, {"time": 1.5, "subtask": "b"}])
        assert out == [{"time": 0.0, "subtask": "a"}, {"time": 1.5, "subtask": "b"}]

    def test_drops_non_dict_entries(self):
        out = _coerce_subtasks(
            [
                "junk",
                ["array", "junk"],
                None,
                {"time": 0.0, "subtask": "ok"},
            ]
        )
        assert out == [{"time": 0.0, "subtask": "ok"}]

    def test_drops_missing_keys(self):
        out = _coerce_subtasks(
            [
                {"time": 0.0},
                {"subtask": "a"},
                {"time": 0.0, "subtask": "ok"},
            ]
        )
        assert out == [{"time": 0.0, "subtask": "ok"}]

    def test_drops_uncoercible_time(self):
        out = _coerce_subtasks(
            [
                {"time": "not_a_number", "subtask": "a"},
                {"time": 0.0, "subtask": "ok"},
            ]
        )
        assert out == [{"time": 0.0, "subtask": "ok"}]

    def test_coerces_int_time_to_float(self):
        out = _coerce_subtasks([{"time": 0, "subtask": "ok"}])
        assert out == [{"time": 0.0, "subtask": "ok"}]
        assert isinstance(out[0]["time"], float)

    def test_raises_when_no_entries_survive(self):
        with pytest.raises(ValueError, match="no valid subtask entries"):
            _coerce_subtasks([{"time": "x"}, "junk", None], raw_text="garbage")

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="no valid subtask entries"):
            _coerce_subtasks([])

    def test_backfills_zero_when_first_entry_is_late(self):
        out = _coerce_subtasks(
            [
                {"time": 2.0, "subtask": "first"},
                {"time": 4.0, "subtask": "second"},
            ]
        )
        assert out == [
            {"time": 0.0, "subtask": "first"},
            {"time": 2.0, "subtask": "first"},
            {"time": 4.0, "subtask": "second"},
        ]

    def test_no_backfill_when_first_entry_is_zero(self):
        entries = [{"time": 0.0, "subtask": "first"}, {"time": 1.0, "subtask": "second"}]
        out = _coerce_subtasks(entries)
        assert out == entries
