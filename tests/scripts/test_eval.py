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

import os
from pathlib import Path
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np
import pytest

from opentau.configs.default import EvalConfig
from opentau.envs.configs import RoboCasaEnv
from opentau.scripts.eval import (
    _cleanup_episode_clips,
    collect_grid_summary_videos,
    create_grid_summary_video,
    eval_policy_all,
)


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


def _make_clip(path: Path, h: int, w: int, n_frames: int, seed: int, fps: int = 10) -> None:
    """Write a deterministic high-entropy clip so CRF/size differences are visible.

    Random noise is a near-worst case for H.264, which makes the CRF size-drop
    assertion a conservative floor (real sim renders compress much better).
    """
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n_frames)]
    imageio.mimsave(str(path), frames, fps=fps)


def _count_frames(path: Path) -> int:
    reader = imageio.get_reader(str(path))
    try:
        return reader.count_frames()
    finally:
        reader.close()


def test_create_grid_summary_video_crf_shrinks_file(tmp_path):
    """A higher CRF yields a smaller grid file for the same input — this is the
    wandb-upload storage knob. crf=0 (near-lossless) must be larger than crf=40."""
    clips = []
    for k in range(2):
        c = tmp_path / f"clip_{k}.mp4"
        _make_clip(c, 64, 64, 20, seed=k)
        clips.append(str(c))

    low_crf = tmp_path / "low_crf.mp4"
    high_crf = tmp_path / "high_crf.mp4"
    create_grid_summary_video(clips, [True, False], str(low_crf), fps=10, highlight_duration=0.0, crf=0)
    create_grid_summary_video(clips, [True, False], str(high_crf), fps=10, highlight_duration=0.0, crf=40)

    assert low_crf.exists() and high_crf.exists()
    assert os.path.getsize(high_crf) < os.path.getsize(low_crf)


def test_create_grid_summary_video_frame_stride_subsamples(tmp_path):
    """frame_stride=k writes only every k-th composed frame, shrinking the body
    ~linearly. highlight_duration=0 removes the constant tail to count cleanly."""
    clips = []
    for k in range(2):
        c = tmp_path / f"clip_{k}.mp4"
        _make_clip(c, 32, 32, 10, seed=10 + k)
        clips.append(str(c))

    out1 = tmp_path / "stride1.mp4"
    out2 = tmp_path / "stride2.mp4"
    create_grid_summary_video(clips, [True, True], str(out1), fps=10, highlight_duration=0.0, frame_stride=1)
    create_grid_summary_video(clips, [True, True], str(out2), fps=10, highlight_duration=0.0, frame_stride=2)

    n1, n2 = _count_frames(out1), _count_frames(out2)
    assert n2 < n1  # striding wrote strictly fewer frames
    # 10 source timesteps -> stride 2 keeps indices 0,2,4,6,8 (== 5); allow +/-1
    # for any encoder frame-count quirk.
    assert (n1 // 2) - 1 <= n2 <= (n1 // 2) + 1


def test_create_grid_summary_video_default_params_keep_shape(tmp_path):
    """Codec/pixel-format regression guard: explicit libx264/yuv420p must not
    change the composed grid dimensions (4 clips -> 2x2 grid)."""
    clips = []
    for k in range(4):
        c = tmp_path / f"clip_{k}.mp4"
        _make_clip(c, 64, 64, 6, seed=20 + k)
        clips.append(str(c))

    out = tmp_path / "grid_summary.mp4"
    create_grid_summary_video(clips, [True, False, True, False], str(out), fps=10, highlight_duration=0.0)

    assert out.exists()
    reader = imageio.get_reader(str(out))
    try:
        shape = reader.get_data(0).shape
    finally:
        reader.close()
    assert shape == (2 * 64, 2 * 64, 3)


def test_cleanup_episode_clips_deletes_unless_kept(tmp_path):
    """Per-episode clips are removed after the grid is built (keep=False) and
    retained when keep=True; a missing path is tolerated."""
    paths = []
    for k in range(3):
        p = tmp_path / f"eval_episode_{k}.mp4"
        p.write_bytes(b"x")
        paths.append(str(p))
    paths.append(str(tmp_path / "already_gone.mp4"))  # exercises the missing_ok path

    _cleanup_episode_clips(paths, keep=True)
    assert all(Path(p).exists() for p in paths[:3])

    _cleanup_episode_clips(paths, keep=False)
    assert all(not Path(p).exists() for p in paths)


def test_eval_config_rejects_out_of_range_video_params():
    """__post_init__ guards the new grid-encoding knobs: CRF in [0, 51], stride
    >= 1, and a known x264 preset; valid bounds construct fine."""
    # Valid bounds construct without error.
    EvalConfig(video_crf=0, video_frame_stride=1, video_preset="slow")
    EvalConfig(video_crf=51, video_frame_stride=10, video_preset="ultrafast")

    with pytest.raises(ValueError, match="video_crf"):
        EvalConfig(video_crf=52)
    with pytest.raises(ValueError, match="video_crf"):
        EvalConfig(video_crf=-1)
    with pytest.raises(ValueError, match="video_frame_stride"):
        EvalConfig(video_frame_stride=0)
    with pytest.raises(ValueError, match="video_preset"):
        EvalConfig(video_preset="turbo")
