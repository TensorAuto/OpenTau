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

"""Unit tests for the Gemini Robotics-ER high-level planner (no network)."""

import json
from unittest.mock import MagicMock

import pytest
import torch
from google import genai

from opentau.planner.gemini_er_planner import (
    GeminiERPlanner,
    _parse_plan_response,
    _to_part,
)


class TestParsePlanResponse:
    def test_clean_json(self):
        subtask, memory = _parse_plan_response(
            '{"memory": "picked up the block", "subtask": "place the block in the bin"}',
            prev_memory="old",
        )
        assert subtask == "place the block in the bin"
        assert memory == "picked up the block"

    def test_fenced_json(self):
        text = '```json\n{"memory": "m", "subtask": "s"}\n```'
        assert _parse_plan_response(text, prev_memory="") == ("s", "m")

    def test_missing_memory_falls_back_to_previous(self):
        subtask, memory = _parse_plan_response('{"subtask": "s"}', prev_memory="kept")
        assert subtask == "s"
        assert memory == "kept"

    def test_empty_memory_falls_back_to_previous(self):
        _, memory = _parse_plan_response('{"memory": "  ", "subtask": "s"}', prev_memory="kept")
        assert memory == "kept"

    def test_missing_subtask_raises(self):
        with pytest.raises(ValueError, match="no 'subtask'"):
            _parse_plan_response('{"memory": "m"}', prev_memory="")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_plan_response("not json at all", prev_memory="")

    def test_non_object_raises(self):
        with pytest.raises(ValueError, match="not a JSON object"):
            _parse_plan_response('["a", "b"]', prev_memory="")


class TestToPart:
    def test_raw_bytes_pass_through(self):
        part = _to_part(b"\xff\xd8fakejpeg")
        assert part.inline_data.data == b"\xff\xd8fakejpeg"
        assert part.inline_data.mime_type == "image/jpeg"

    def test_bytes_with_encoding(self):
        part = _to_part((b"fakepng", "png"))
        assert part.inline_data.data == b"fakepng"
        assert part.inline_data.mime_type == "image/png"

    def test_tensor_chw_and_batched(self):
        for shape in [(3, 8, 8), (1, 3, 8, 8)]:
            part = _to_part(torch.rand(*shape))
            assert part.inline_data.mime_type == "image/jpeg"
            assert part.inline_data.data[:2] == b"\xff\xd8"  # JPEG magic

    def test_bad_tensor_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _to_part(torch.rand(8, 8))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            _to_part(123)


def _make_planner(response_text: str) -> tuple[GeminiERPlanner, MagicMock]:
    client = MagicMock(spec=genai.Client)
    response = MagicMock()
    response.text = response_text
    client.models.generate_content.return_value = response
    planner = GeminiERPlanner(client=client)
    return planner, client


class TestGeminiERPlanner:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No Gemini API key"):
            GeminiERPlanner()

    def test_unknown_prompt_key_raises(self):
        with pytest.raises(ValueError, match="not found"):
            GeminiERPlanner(client=MagicMock(spec=genai.Client), system_prompt_key="nope")

    def test_build_user_prompt_substitution(self):
        planner, _ = _make_planner("{}")
        prompt = planner._build_user_prompt("stack the cups", "did step one", [0.123456, 1.0])
        assert "stack the cups" in prompt
        assert "did step one" in prompt
        assert "[0.1235, 1.0]" in prompt

    def test_build_user_prompt_empty_memory_and_state(self):
        planner, _ = _make_planner("{}")
        prompt = planner._build_user_prompt("t", "", None)
        assert "(empty — first call)" in prompt
        assert "(not provided)" in prompt

    def test_plan_end_to_end(self):
        raw = json.dumps({"memory": "saw two blocks", "subtask": "pick up the red block"})
        planner, client = _make_planner(raw)

        result = planner.plan(
            task="put the blocks in the bin",
            images=[b"\xff\xd8fakejpeg", (b"fakepng", "png")],
            state=[0.0, 1.0],
            memory="",
        )

        assert result.subtask == "pick up the red block"
        assert result.memory == "saw two blocks"
        assert result.raw_text == raw
        assert result.latency_s >= 0

        _, kwargs = client.models.generate_content.call_args
        assert kwargs["model"] == GeminiERPlanner.DEFAULT_MODEL
        assert kwargs["config"].response_mime_type == "application/json"
        assert kwargs["config"].temperature == 0.0
        # 1 text part + 2 image parts
        assert len(kwargs["contents"]) == 3
        assert "put the blocks in the bin" in kwargs["contents"][0]

    def test_plan_empty_response_raises(self):
        planner, _ = _make_planner("")
        with pytest.raises(ValueError, match="no text"):
            planner.plan(task="t", images=[b"img"], memory="")

    def test_schema_rejected_falls_back_once(self):
        client = MagicMock(spec=genai.Client)
        good = MagicMock()
        good.text = '{"memory": "m", "subtask": "s"}'
        schema_error = genai.errors.APIError(400, {"error": {"message": "response_schema is not supported"}})
        client.models.generate_content.side_effect = [schema_error, good, good]

        planner = GeminiERPlanner(client=client)
        result = planner.plan(task="t", images=[b"img"], memory="")
        assert result.subtask == "s"
        assert planner._schema_supported is False

        # Subsequent calls go straight to the schemaless path: one more call, no error.
        planner.plan(task="t", images=[b"img"], memory="m")
        assert client.models.generate_content.call_count == 3

    def test_non_schema_api_error_propagates(self):
        client = MagicMock(spec=genai.Client)
        client.models.generate_content.side_effect = genai.errors.APIError(
            500, {"error": {"message": "internal failure"}}
        )
        planner = GeminiERPlanner(client=client)
        with pytest.raises(genai.errors.APIError):
            planner.plan(task="t", images=[b"img"], memory="")
