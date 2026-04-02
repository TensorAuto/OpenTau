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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from opentau.scripts.pi_mem_data_generator import _build_subtask_log, _enrich_list, main

SAMPLE_DATA = [
    {"time": 12.0, "subtask": "pick up cup", "success": True, "prompt": "Pick up a cup."},
    {"time": 20, "subtask": "pick up bottle", "success": False, "prompt": "Pick up a cup."},
    {"time": 28, "subtask": "pick up bottle", "success": True, "prompt": "Pick up a cup."},
]


class TestBuildSubtaskLog:
    def test_single_entry(self):
        log = _build_subtask_log(SAMPLE_DATA, up_to=0)
        assert log == "  1. [SUCCESS] (t=12.0s) pick up cup"

    def test_cumulative_entries(self):
        log = _build_subtask_log(SAMPLE_DATA, up_to=2)
        lines = log.split("\n")
        assert len(lines) == 3
        assert "[SUCCESS]" in lines[0]
        assert "[FAILED]" in lines[1]
        assert "[SUCCESS]" in lines[2]

    def test_missing_success_shows_unknown(self):
        data = [{"subtask": "test", "time": 1}]
        log = _build_subtask_log(data, up_to=0)
        assert "[UNKNOWN]" in log

    def test_missing_subtask_shows_unknown(self):
        data = [{"success": True, "time": 5}]
        log = _build_subtask_log(data, up_to=0)
        assert "unknown" in log

    def test_missing_time_omits_timestamp(self):
        data = [{"subtask": "reset", "success": True}]
        log = _build_subtask_log(data, up_to=0)
        assert "(t=" not in log
        assert "reset" in log


def _make_mock_client(replies: list[str]) -> MagicMock:
    """Build a mock OpenAI client that returns ``replies`` in order."""
    client = MagicMock()
    responses = []
    for text in replies:
        msg = SimpleNamespace(content=text)
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        responses.append(resp)
    client.chat.completions.create.side_effect = responses
    return client


class TestEnrichList:
    def test_basic_enrichment(self):
        data = [dict(d) for d in SAMPLE_DATA]
        client = _make_mock_client(["mem0", "mem1", "mem2"])
        _enrich_list(
            data,
            client=client,
            model="test-model",
            system_prompt="sys",
            output_key="memory",
            skip_existing=False,
            delay_s=0,
        )
        assert data[0]["memory"] == "mem0"
        assert data[1]["memory"] == "mem1"
        assert data[2]["memory"] == "mem2"
        assert client.chat.completions.create.call_count == 3

    def test_skip_existing(self):
        data = [dict(d) for d in SAMPLE_DATA]
        data[0]["memory"] = "already-set"
        client = _make_mock_client(["mem1", "mem2"])
        _enrich_list(
            data,
            client=client,
            model="m",
            system_prompt="s",
            output_key="memory",
            skip_existing=True,
            delay_s=0,
        )
        assert data[0]["memory"] == "already-set"
        assert data[1]["memory"] == "mem1"
        assert client.chat.completions.create.call_count == 2

    def test_previous_memory_propagation(self):
        """The reply from step N is passed as prev_memory in step N+1's prompt."""
        data = [dict(d) for d in SAMPLE_DATA[:2]]
        client = _make_mock_client(["first-summary", "second-summary"])
        _enrich_list(
            data,
            client=client,
            model="m",
            system_prompt="s",
            output_key="memory",
            skip_existing=False,
            delay_s=0,
        )
        second_call_kwargs = client.chat.completions.create.call_args_list[1]
        user_msg = second_call_kwargs.kwargs["messages"][-1]["content"]
        assert "first-summary" in user_msg

    def test_non_dict_item_raises(self):
        with pytest.raises(ValueError, match="must be an object"):
            _enrich_list(
                ["not-a-dict"],
                client=MagicMock(),
                model="m",
                system_prompt="s",
                output_key="memory",
                skip_existing=False,
                delay_s=0,
            )


class TestMainMissingApiKey:
    def test_exits_with_error_when_key_missing(self, tmp_path, monkeypatch):
        input_file = tmp_path / "data.json"
        input_file.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("opentau.scripts.pi_mem_data_generator._load_env_file"):
            rc = main([str(input_file)])
        assert rc == 1


class TestMainAtomicWrite:
    def test_write_is_atomic(self, tmp_path, monkeypatch):
        """After a successful run the file is updated and no .tmp files remain."""
        input_file = tmp_path / "data.json"
        input_file.write_text(json.dumps(SAMPLE_DATA), encoding="utf-8")

        monkeypatch.setenv("OPENAI_API_KEY", "test")  # gitleaks:allow
        mock_client = _make_mock_client(["m0", "m1", "m2"])

        with (
            patch("opentau.scripts.pi_mem_data_generator._load_env_file"),
            patch("opentau.scripts.pi_mem_data_generator.OpenAI", return_value=mock_client),
        ):
            rc = main([str(input_file)])

        assert rc == 0
        result = json.loads(input_file.read_text(encoding="utf-8"))
        assert result[0]["memory"] == "m0"
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []
