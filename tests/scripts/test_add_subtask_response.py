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

from opentau.scripts.add_subtask_response import _build_response_array


class TestBuildResponseArray:
    def test_empty_subtasks_returns_empty_strings(self):
        assert _build_response_array([], fps=30, episode_length=5, episode_index=0) == [""] * 5

    def test_zero_length_episode(self):
        assert (
            _build_response_array([{"time": 0.0, "subtask": "a"}], fps=30, episode_length=0, episode_index=0)
            == []
        )

    def test_single_subtask_covers_whole_episode(self):
        subtasks = [{"time": 0.0, "subtask": "pick up cup"}]
        out = _build_response_array(subtasks, fps=10, episode_length=5, episode_index=0)
        assert out == ["pick up cup"] * 5

    def test_two_subtasks_split_at_boundary(self):
        subtasks = [
            {"time": 0.0, "subtask": "a"},
            {"time": 0.3, "subtask": "b"},
        ]
        # fps=10, so boundary is frame 3.
        out = _build_response_array(subtasks, fps=10, episode_length=6, episode_index=0)
        assert out == ["a", "a", "a", "b", "b", "b"]

    def test_gap_before_first_subtask_gets_empty_string(self):
        subtasks = [{"time": 0.2, "subtask": "a"}]
        # fps=10 → starts at frame 2.
        out = _build_response_array(subtasks, fps=10, episode_length=5, episode_index=0)
        assert out == ["", "", "a", "a", "a"]

    def test_last_subtask_extends_to_episode_end(self):
        subtasks = [
            {"time": 0.0, "subtask": "a"},
            {"time": 0.2, "subtask": "b"},
        ]
        out = _build_response_array(subtasks, fps=10, episode_length=5, episode_index=0)
        assert out == ["a", "a", "b", "b", "b"]

    def test_unsorted_input_is_sorted_by_time(self):
        subtasks = [
            {"time": 0.3, "subtask": "b"},
            {"time": 0.0, "subtask": "a"},
        ]
        out = _build_response_array(subtasks, fps=10, episode_length=6, episode_index=0)
        assert out == ["a", "a", "a", "b", "b", "b"]

    def test_subtask_beyond_episode_length_is_skipped(self, caplog):
        subtasks = [
            {"time": 0.0, "subtask": "a"},
            {"time": 10.0, "subtask": "too_late"},
        ]
        with caplog.at_level("WARNING"):
            out = _build_response_array(subtasks, fps=10, episode_length=5, episode_index=7)
        assert out == ["a"] * 5
        assert "too_late" in caplog.text
        assert "Episode 7" in caplog.text

    def test_boundary_clamped_to_episode_length(self):
        subtasks = [
            {"time": 0.0, "subtask": "a"},
            {"time": 0.9, "subtask": "b"},
        ]
        # fps=10 → b would start at frame 9, but episode is length 5, so b is skipped.
        out = _build_response_array(subtasks, fps=10, episode_length=5, episode_index=0)
        assert out == ["a"] * 5

    def test_rounding_handles_floating_point_drift(self):
        # 2.5 * 30 = 74.99999... in fp64; int() would truncate to 74.
        subtasks = [
            {"time": 0.0, "subtask": "a"},
            {"time": 2.5, "subtask": "b"},
        ]
        out = _build_response_array(subtasks, fps=30, episode_length=150, episode_index=0)
        assert out[74] == "a"
        assert out[75] == "b"

    @pytest.mark.parametrize(
        "subtasks,fps,length,expected",
        [
            # Simultaneous times: later in list wins only for the zero-width gap.
            (
                [
                    {"time": 0.0, "subtask": "a"},
                    {"time": 0.0, "subtask": "b"},
                ],
                10,
                3,
                ["b", "b", "b"],
            ),
        ],
    )
    def test_parametrized_edge_cases(self, subtasks, fps, length, expected):
        assert _build_response_array(subtasks, fps=fps, episode_length=length, episode_index=0) == expected
