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

import imageio.v2 as imageio
import numpy as np
import pytest

from opentau.envs.configs import RoboCasaEnv
from opentau.scripts.eval import collect_grid_summary_videos, create_grid_summary_video, eval_policy_all


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


def test_eval_policy_all_forwards_render_cap_and_grid_size_to_eval_policy(monkeypatch):
    """The grid-summary controls must reach the leaf evaluator. eval_main exposes
    `cfg.eval.max_episodes_rendered` / `cfg.eval.grid_size` by forwarding them through
    eval_policy_all -> run_one -> eval_one -> eval_policy. Regression guard: eval_main
    used to hardcode the render cap (10) and never threaded grid_size, so both
    EvalConfig fields were silently ignored and grids only ever tiled 10 rollouts."""
    captured: dict = {}

    def fake_eval_policy(**kwargs):
        # Stub the real rollout: capture the render controls and return an empty
        # per-episode result so the (real) run_one/eval_one wrappers still run.
        captured.update(kwargs)
        return {"per_episode": []}

    # Patch the module global by dotted path so eval_one's call-time lookup of
    # `eval_policy` resolves to the stub (keeps a single `from`-import style).
    monkeypatch.setattr("opentau.scripts.eval.eval_policy", fake_eval_policy)

    cfg = Mock()
    cfg.eval.recording_root = None  # skip the LIBERO-only recorder guard + block

    eval_policy_all(
        {"group": {0: object()}},  # one fake task; the env is never touched (eval_policy is stubbed)
        policy=Mock(),
        n_episodes=1,
        cfg=cfg,
        max_episodes_rendered=7,
        grid_size=(2, 3),
        videos_dir=None,
        max_parallel_tasks=1,
    )

    assert captured["max_episodes_rendered"] == 7
    assert captured["grid_size"] == (2, 3)


def test_create_grid_summary_video_streams_into_expected_grid(tmp_path):
    """The grid builder streams each clip frame-by-frame (bounded memory) and tiles
    them into a (grid_rows*H, grid_cols*W, 3) video, holding a clip's last frame
    once it ends. Two clips of different lengths exercise both paths."""
    h, w = 16, 16
    paths = []
    for k, n_frames in enumerate((3, 5)):  # different lengths -> hold-last-frame path
        clip = tmp_path / f"clip_{k}.mp4"
        frames = [np.full((h, w, 3), 40 * (k + 1), np.uint8) for _ in range(n_frames)]
        imageio.mimsave(str(clip), frames, fps=10)
        paths.append(str(clip))

    out = tmp_path / "grid_summary.mp4"
    create_grid_summary_video(paths, [False, True], str(out), fps=10, highlight_duration=0.2)

    assert out.exists()
    reader = imageio.get_reader(str(out))
    try:
        first_frame = reader.get_data(0)
        n_out = reader.count_frames()
    finally:
        reader.close()
    # 2 clips -> auto grid rows=ceil(sqrt(2))=2, cols=1 -> tiles stacked vertically.
    assert first_frame.shape == (2 * h, 1 * w, 3)
    # at least the longest clip's frames (5); + highlight frames (int(0.2*10)=2).
    assert n_out >= 5


def test_create_grid_summary_video_no_valid_videos_is_noop(tmp_path):
    """Missing input clips are skipped; with none left it writes nothing."""
    out = tmp_path / "grid_summary.mp4"
    create_grid_summary_video([str(tmp_path / "missing.mp4")], [True], str(out), fps=10)
    assert not out.exists()
