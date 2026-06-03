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

from pathlib import Path
from unittest.mock import Mock

import pytest

from opentau.envs.configs import RoboCasaEnv
from opentau.scripts.eval import collect_grid_summary_videos, eval_policy_all


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_collect_grid_summary_videos_strips_rank_and_excludes_clips(tmp_path):
    _touch(tmp_path / "libero_10_3_rank0" / "grid_summary.mp4")
    _touch(tmp_path / "libero_10_3_rank0" / "eval_episode_0.mp4")  # must be excluded
    _touch(tmp_path / "libero_goal_1_rank2" / "grid_summary.mp4")

    out = collect_grid_summary_videos(tmp_path)

    assert [name for name, _ in out] == ["libero_10_3", "libero_goal_1"]
    assert all(p.endswith("grid_summary.mp4") for _, p in out)


def test_collect_grid_summary_videos_missing_dir(tmp_path):
    assert collect_grid_summary_videos(tmp_path / "does_not_exist") == []


def test_eval_policy_all_rejects_recording_root_for_non_libero_env():
    """eval.recording_root drives the LIBERO-only dataset recorder, so it must fail
    fast (before any rollout) for a non-LIBERO env rather than silently mislabel the
    recorded dataset."""
    cfg = Mock()
    cfg.eval.recording_root = "/tmp/robocasa-rollout"
    cfg.env = RoboCasaEnv()  # not a LIBERO env

    with pytest.raises(NotImplementedError, match="recording_root"):
        eval_policy_all({}, policy=Mock(), n_episodes=1, cfg=cfg)
