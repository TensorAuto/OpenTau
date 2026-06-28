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
import types
from pathlib import Path
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from accelerate import DistributedType

from opentau.configs.default import EvalConfig
from opentau.envs.configs import RoboCasaEnv
from opentau.scripts.eval import (
    _cleanup_episode_clips,
    _eval_uses_sharded_params,
    _rank_seed_offset,
    _resolve_eval_seed,
    collect_grid_summary_videos,
    create_grid_summary_video,
    eval_main,
    eval_policy,
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


class TestRankSeedOffset:
    """`_rank_seed_offset` controls whether ranks evaluate the same or distinct scenes."""

    def test_default_is_zero_so_all_ranks_seed_identically(self):
        # decorrelate=False -> every rank gets offset 0, regardless of rank index,
        # so the eval is reproducible across world sizes / node counts.
        for r in range(16):
            assert _rank_seed_offset(process_index=r, decorrelate=False, per_rank_span=12) == 0

    def test_decorrelated_blocks_tile_without_gap_or_overlap(self):
        # Each rank's seed block is [r*span, (r+1)*span); since stride == span the
        # blocks partition [0, W*span) exactly — no collision, no wasted seeds,
        # and no magic constant.
        n_batches, num_envs, world_size = 3, 4, 5
        span = n_batches * num_envs
        blocks = [
            set(
                range(
                    _rank_seed_offset(process_index=r, decorrelate=True, per_rank_span=span),
                    _rank_seed_offset(process_index=r, decorrelate=True, per_rank_span=span) + span,
                )
            )
            for r in range(world_size)
        ]
        union = set().union(*blocks)
        assert len(union) == world_size * span  # pairwise disjoint (no overlap)
        assert union == set(range(world_size * span))  # contiguous (no gap)

    def test_offset_is_collision_free_even_beyond_10000_episodes(self):
        # The old `* 10000` stride aliased once a rank exceeded 10000 episodes;
        # the span-based stride never does, because the stride IS the span.
        span = 12000  # a rank running >10000 episodes
        assert _rank_seed_offset(process_index=0, decorrelate=True, per_rank_span=span) == 0
        assert _rank_seed_offset(process_index=1, decorrelate=True, per_rank_span=span) == span
        # rank 0's last seed (span-1) is strictly below rank 1's first seed (span).
        assert span - 1 < _rank_seed_offset(process_index=1, decorrelate=True, per_rank_span=span)


class TestResolveEvalSeed:
    """`eval.seed` overrides the top-level `cfg.seed` for env scene seeding only."""

    @staticmethod
    def _cfg(*, top_seed, eval_seed):
        cfg = Mock()
        cfg.seed = top_seed
        cfg.eval.seed = eval_seed
        return cfg

    def test_falls_back_to_top_level_seed_when_eval_seed_is_none(self):
        assert _resolve_eval_seed(self._cfg(top_seed=1000, eval_seed=None)) == 1000

    def test_eval_seed_takes_precedence_when_set(self):
        assert _resolve_eval_seed(self._cfg(top_seed=1000, eval_seed=7)) == 7

    def test_eval_seed_zero_is_an_explicit_seed_not_unset(self):
        # 0 is a valid seed and must take precedence — not be treated as falsy/unset.
        assert _resolve_eval_seed(self._cfg(top_seed=1000, eval_seed=0)) == 0

    def test_returns_none_only_when_both_are_none(self):
        assert _resolve_eval_seed(self._cfg(top_seed=None, eval_seed=None)) is None

    def test_eval_config_default_seed_is_none(self):
        # The field exists and defaults to None (fall back to cfg.seed).
        assert EvalConfig().seed is None
        assert EvalConfig(seed=42).seed == 42


class TestEvalUsesShardedParams:
    """`_eval_uses_sharded_params` gates the AR-decode eval guard (FSDP / ZeRO-3).

    Calling the helper directly with a mock accelerator also guards against the
    decorator-placement regression: `@parser.wrap()` must sit on `eval_main`, not
    on this helper (a wrapped helper would not return a plain bool here).
    """

    @staticmethod
    def _acc(dist_type, zero_stage=0):
        acc = Mock()
        acc.distributed_type = dist_type
        acc.deepspeed_plugin.hf_ds_config.config = {"zero_optimization": {"stage": zero_stage}}
        return acc

    def test_fsdp_is_sharded(self):
        assert _eval_uses_sharded_params(self._acc(DistributedType.FSDP)) is True

    def test_deepspeed_zero3_is_sharded(self):
        assert _eval_uses_sharded_params(self._acc(DistributedType.DEEPSPEED, zero_stage=3)) is True

    def test_deepspeed_zero2_is_not_sharded(self):
        assert _eval_uses_sharded_params(self._acc(DistributedType.DEEPSPEED, zero_stage=2)) is False

    def test_multi_gpu_is_not_sharded(self):
        assert _eval_uses_sharded_params(self._acc(DistributedType.MULTI_GPU)) is False

    def test_eval_main_is_parser_wrapped(self):
        # Regression: inserting the helper above eval_main must not steal its
        # @parser.wrap() decorator (that breaks the opentau-eval entry point).
        assert getattr(eval_main, "__wrapped__", None) is not None


# --------------------------------------------------------------------------- #
# Shared light-weight harness for driving eval_policy with a stubbed rollout.
# --------------------------------------------------------------------------- #


class _FakeEnv:
    """Minimal stand-in for a gym VectorEnv.

    ``eval_policy`` only reads ``env.num_envs`` when ``rollout`` is stubbed and
    ``max_episodes_rendered == 0`` (no rendering -> no isinstance / metadata
    access on the env).
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs


def _make_eval_cfg(
    *,
    seed_list=None,
    goal_frames_dir=None,
    eval_use_discrete_actions=False,
    task="unknown",
):
    """Build the smallest ``cfg`` ``eval_policy`` reads with rollout stubbed:
    ``cfg.eval.{seed_list,goal_frames_dir,decorrelate_rank_seeds}``,
    ``cfg.env.task`` and ``cfg.policy.eval_use_discrete_actions``."""
    return types.SimpleNamespace(
        eval=types.SimpleNamespace(
            seed_list=seed_list,
            goal_frames_dir=goal_frames_dir,
            decorrelate_rank_seeds=False,
        ),
        env=types.SimpleNamespace(task=task),
        policy=types.SimpleNamespace(eval_use_discrete_actions=eval_use_discrete_actions),
    )


def _make_rollout_data(
    *,
    num_envs: int,
    n_steps: int = 3,
    successes=None,
    last_frames=None,
):
    """A minimal valid ``rollout_data`` dict for ``eval_policy``.

    ``eval_policy`` reads ``done`` / ``reward`` / ``success`` (each
    ``(num_envs, n_steps)``) and, when capture is on, ``last_frames``.
    ``done`` is all-True so ``argmax`` -> step 0 (every episode "ends"
    immediately); the success mask then reduces over the kept step.
    """
    if successes is None:
        successes = [False] * num_envs
    success = torch.zeros(num_envs, n_steps, dtype=torch.bool)
    for i, s in enumerate(successes):
        if s:
            success[i, 0] = True
    data = {
        "done": torch.ones(num_envs, n_steps, dtype=torch.bool),
        "reward": torch.zeros(num_envs, n_steps, dtype=torch.float32),
        "success": success,
    }
    if last_frames is not None:
        data["last_frames"] = last_frames
    return data


class TestEvalPolicySeedListParse:
    """Group B: sparse ``cfg.eval.seed_list`` parsing + final-batch right-pad
    in ``eval_policy`` (eval.py around lines 386-465).

    The seed string is split on ``,`` with whitespace/empty entries dropped;
    ``n_episodes`` becomes the parsed length; each batch slices
    ``num_envs`` seeds and the final SHORT batch is right-padded by repeating
    the last seed. A full batch is left untouched."""

    @staticmethod
    def _run_and_record_seeds(monkeypatch, *, seed_list, num_envs):
        """Run eval_policy with a stubbed rollout that records the per-batch
        ``seeds`` it is called with; returns the list of recorded seed lists."""
        recorded: list = []

        def fake_rollout(**kwargs):
            recorded.append(kwargs["seeds"])
            return _make_rollout_data(num_envs=num_envs)

        monkeypatch.setattr("opentau.scripts.eval.rollout", fake_rollout)

        cfg = _make_eval_cfg(seed_list=seed_list)
        env = _FakeEnv(num_envs=num_envs)
        # n_episodes is overridden by len(parsed seed_list); pass a wrong value
        # to prove the override happens.
        info = eval_policy(env, policy=Mock(), n_episodes=999, cfg=cfg)
        return recorded, info

    def test_full_batch_untouched_and_whitespace_dropped(self, monkeypatch):
        # "705, ,712" -> parsed [705, 712] (the " " entry dropped); num_envs=2
        # -> a single full batch, untouched (no padding).
        recorded, info = self._run_and_record_seeds(monkeypatch, seed_list="705, ,712", num_envs=2)
        assert len(recorded) == 1, "expected exactly one batch (n_episodes==2, num_envs==2)"
        assert recorded[0] == [705, 712]
        # The whitespace entry was dropped (no 0 / no extra seed appeared).
        assert all(s in (705, 712) for s in recorded[0])
        # n_episodes was overridden to the parsed length (2 -> 2 episodes scored).
        assert len(info["per_episode"]) == 2

    def test_short_final_batch_right_padded_with_last_seed(self, monkeypatch):
        # "705,708,712" -> [705,708,712]; num_envs=2 -> batch0 [705,708] (full),
        # batch1 [712] right-padded to [712,712].
        recorded, _info = self._run_and_record_seeds(monkeypatch, seed_list="705,708,712", num_envs=2)
        assert len(recorded) == 2
        assert recorded[0] == [705, 708]  # full batch, untouched
        assert recorded[1] == [712, 712]  # last seed repeated to fill num_envs


class TestEvalPolicyGoalFrameHarvest:
    """Group C: goal-frame harvest writer in ``eval_policy`` (eval.py around
    lines 524-551, manifest write 658-688) round-tripping into the reader
    ``RoboCasaGoalFrameSubgoalGenerator``.

    Pins: only SUCCESSFUL episodes are harvested; the PNG filename scheme is
    ``{task}__seed{seed}__{decoder}__{cam}__rank{rank}.png``; ``decoder`` is
    ``"discrete"`` iff ``cfg.policy.eval_use_discrete_actions`` else ``"flow"``;
    the manifest header is ``[task,seed,decoder,success,camera0,camera1,...]``
    derived from the cameras present; and the reader indexes exactly the
    harvested seed."""

    @staticmethod
    def _two_env_frames():
        """last_frames for a 2-env batch: env0 has 2-camera frames, env1 None."""
        h, w = 8, 8
        env0 = {
            "camera0": np.full((h, w, 3), 11, dtype=np.uint8),
            "camera1": np.full((h, w, 3), 22, dtype=np.uint8),
        }
        return [env0, None]

    def _run_harvest(self, monkeypatch, tmp_path, *, eval_use_discrete_actions, task):
        """Drive eval_policy once with capture on; env0 succeeds, env1 fails."""
        num_envs = 2
        frames = self._two_env_frames()

        def fake_rollout(**kwargs):
            assert kwargs["capture_last_frames"] is True
            return _make_rollout_data(
                num_envs=num_envs,
                successes=[True, False],  # only env0 (seed 705) succeeds
                last_frames=frames,
            )

        monkeypatch.setattr("opentau.scripts.eval.rollout", fake_rollout)

        cfg = _make_eval_cfg(
            seed_list="705,706",  # env0 -> seed 705 (success), env1 -> seed 706 (fail)
            goal_frames_dir=tmp_path,
            eval_use_discrete_actions=eval_use_discrete_actions,
            task=task,
        )
        env = _FakeEnv(num_envs=num_envs)
        info = eval_policy(env, policy=Mock(), n_episodes=2, cfg=cfg)
        return info

    def test_writer_round_trips_into_reader(self, monkeypatch, tmp_path):
        from opentau.envs.subgoal import RoboCasaGoalFrameSubgoalGenerator

        task = "CloseFridge"
        info = self._run_harvest(monkeypatch, tmp_path, eval_use_discrete_actions=False, task=task)

        # (a) Exactly one successful episode harvested -> one manifest row.
        assert info["goal_frames_saved"] == 1

        # (b) Only the successful env's PNG files exist, with the flow decoder + scheme.
        decoder = "flow"
        for cam in ("camera0", "camera1"):
            expected = tmp_path / f"{task}__seed705__{decoder}__{cam}__rank0.png"
            assert expected.is_file(), f"missing harvested PNG {expected.name}"
        # The failed env (seed 706) is NOT harvested.
        assert not list(tmp_path.glob(f"{task}__seed706__*"))

        # (c) Manifest header matches the dynamic [task,seed,decoder,success,camera*].
        manifest = tmp_path / "manifest.csv"
        assert manifest.is_file()
        import csv

        with open(manifest, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)
        assert header == ["task", "seed", "decoder", "success", "camera0", "camera1"]
        assert len(rows) == 1
        assert rows[0]["task"] == task
        assert rows[0]["seed"] == "705"
        assert rows[0]["decoder"] == "flow"
        assert rows[0]["success"] == "1"

        # (d) Reader round-trip: indexes exactly the harvested seed.
        gen = RoboCasaGoalFrameSubgoalGenerator([tmp_path], task=task, num_cams=2)
        assert gen.num_scenes == 1
        assert 705 in gen._index
        assert 706 not in gen._index
        # Both cameras were indexed for that scene.
        assert sorted(gen._index[705].keys()) == [0, 1]

    def test_decoder_column_reflects_discrete_flag(self, monkeypatch, tmp_path):
        # With eval_use_discrete_actions=True the decoder column / filename token
        # is "discrete" (vs "flow" in the round-trip test above).
        task = "OpenDoor"
        info = self._run_harvest(monkeypatch, tmp_path, eval_use_discrete_actions=True, task=task)
        assert info["goal_frames_saved"] == 1
        for cam in ("camera0", "camera1"):
            expected = tmp_path / f"{task}__seed705__discrete__{cam}__rank0.png"
            assert expected.is_file(), f"missing harvested PNG {expected.name}"
        assert not list(tmp_path.glob(f"{task}__seed705__flow__*"))

        import csv

        with open(tmp_path / "manifest.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["decoder"] == "discrete"
