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

from unittest.mock import MagicMock, patch

import anthropic
import pytest
from google import genai

from opentau.scripts import annotate_subtasks
from opentau.scripts.annotate_subtasks import (
    _annotate_subtasks,
    _coerce_subtasks,
    _is_gemini_model,
    _parse_json_response,
)


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

    def test_error_message_is_provider_agnostic(self):
        # _coerce_subtasks is reached from both Claude and Gemini paths, so its
        # error message must not single out a single provider.
        with pytest.raises(ValueError, match=r"^Model response had no valid subtask entries"):
            _coerce_subtasks([])


class TestIsGeminiModel:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-robotics-er-1.6-preview",
            "Gemini-Pro",
            "GEMINI-1.5",
            "robotics-er-1.6",
            "Robotics-ER-1.6-preview",
        ],
    )
    def test_routes_to_gemini(self, model):
        assert _is_gemini_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "Claude-Haiku-4-5",
            "gpt-4o",
            "",
        ],
    )
    def test_does_not_route_to_gemini(self, model):
        assert _is_gemini_model(model) is False


class TestAnnotateSubtasksDispatch:
    """Verify the dispatcher routes each model to the right backend with the right client.

    The runtime ``isinstance`` asserts in ``_annotate_subtasks`` would only catch
    a misrouted call when annotation actually runs — these tests pin the contract.
    """

    def _make_inputs(self):
        return {
            "task_description": "pick up the cup",
            "frames": [],
            "timestamps": [],
            "sample_fps": 1.0,
        }

    def test_claude_model_calls_claude_with_anthropic_client(self):
        anthropic_client = MagicMock(spec=anthropic.Anthropic)
        expected = [{"time": 0.0, "subtask": "ok"}]
        with (
            patch.object(annotate_subtasks, "_call_claude", return_value=expected) as call_claude,
            patch.object(annotate_subtasks, "_call_gemini") as call_gemini,
        ):
            out = _annotate_subtasks(anthropic_client, "claude-opus-4-7", **self._make_inputs())
        assert out == expected
        call_claude.assert_called_once()
        call_gemini.assert_not_called()
        # The dispatcher must hand the Anthropic client through unchanged.
        assert call_claude.call_args.args[0] is anthropic_client

    @pytest.mark.parametrize("model", ["gemini-robotics-er-1.6-preview", "robotics-er-1.6"])
    def test_gemini_model_calls_gemini_with_genai_client(self, model):
        gemini_client = MagicMock(spec=genai.Client)
        expected = [{"time": 0.0, "subtask": "ok"}]
        with (
            patch.object(annotate_subtasks, "_call_gemini", return_value=expected) as call_gemini,
            patch.object(annotate_subtasks, "_call_claude") as call_claude,
        ):
            out = _annotate_subtasks(gemini_client, model, **self._make_inputs())
        assert out == expected
        call_gemini.assert_called_once()
        call_claude.assert_not_called()
        assert call_gemini.call_args.args[0] is gemini_client

    def test_gemini_model_with_anthropic_client_trips_assert(self):
        # Mirror the failure mode the reviewer flagged: a future Vertex-style
        # ``models/gemini-...`` ID would currently fall through to the Claude
        # branch; verify the isinstance assert catches a client/model mismatch.
        anthropic_client = MagicMock(spec=anthropic.Anthropic)
        with pytest.raises(AssertionError):
            _annotate_subtasks(anthropic_client, "gemini-1.5-pro", **self._make_inputs())

    def test_claude_model_with_genai_client_trips_assert(self):
        gemini_client = MagicMock(spec=genai.Client)
        with pytest.raises(AssertionError):
            _annotate_subtasks(gemini_client, "claude-opus-4-7", **self._make_inputs())
