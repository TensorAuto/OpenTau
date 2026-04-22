#!/usr/bin/env python

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
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import (
    DEFAULT_VIDEO_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    write_json,
    write_stats,
)
from opentau.scripts.segment_lerobot_dataset import (
    FFMPEG_MAX_WORKERS_ENV_VAR,
    _ffmpeg_trim_max_workers,
    _load_segments_by_episode,
    segment_dataset,
)


def _extract_image_path(cell: Any) -> str | None:
    """Extract image path from a parquet image cell.

    Args:
        cell: A parquet image value (string path or dict with `path`/`bytes`).

    Returns:
        Image path string if present, otherwise None.
    """
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        path = cell.get("path")
        if isinstance(path, str) and path:
            return path
    return None


def test_segment_lerobot_v21_dataset(tmp_path: Path, empty_lerobot_dataset_factory: Any) -> None:
    """Validate baseline segmentation behavior for v2.1 input.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "source_dataset"
    output_root = tmp_path / "segmented_dataset"

    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)

    for i in range(10):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i) + 0.5], dtype=np.float32),
                "task": "pick object",
            }
        )
    dataset.save_episode()

    segment_dataset(
        input_root=input_root,
        output_root=output_root,
        segments_by_episode={0: [(2, 5), (5, 10)]},
    )

    info = load_info(output_root)
    episodes = load_episodes(output_root)
    episodes_stats = load_episodes_stats(output_root)
    output_meta = LeRobotDatasetMetadata(repo_id=output_root.name, root=output_root)

    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 8
    assert info["total_videos"] == 0
    assert info["splits"] == {"train": "0:2"}

    assert episodes[0]["length"] == 3
    assert episodes[1]["length"] == 5
    assert len(episodes_stats) == 2
    assert int(episodes_stats[0]["state"]["count"][0]) == 3
    assert int(episodes_stats[1]["state"]["count"][0]) == 5

    ep0_table = pq.read_table(output_root / output_meta.get_data_file_path(0))
    ep1_table = pq.read_table(output_root / output_meta.get_data_file_path(1))
    ep0 = ep0_table.to_pydict()
    ep1 = ep1_table.to_pydict()

    assert ep0["frame_index"] == [0, 1, 2]
    assert ep0["episode_index"] == [0, 0, 0]
    assert ep0["index"] == [0, 1, 2]
    assert np.isclose(float(ep0["timestamp"][0]), 0.0)
    assert np.allclose(np.diff(np.asarray(ep0["timestamp"], dtype=np.float64)), 1.0 / dataset.fps)
    assert [float(x) for x in ep0["state"]] == [2.0, 3.0, 4.0]

    assert ep1["frame_index"] == [0, 1, 2, 3, 4]
    assert ep1["episode_index"] == [1, 1, 1, 1, 1]
    assert ep1["index"] == [3, 4, 5, 6, 7]
    assert np.isclose(float(ep1["timestamp"][0]), 0.0)
    assert np.allclose(np.diff(np.asarray(ep1["timestamp"], dtype=np.float64)), 1.0 / dataset.fps)
    assert [float(x) for x in ep1["state"]] == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_segment_lerobot_v20_input_outputs_v21(tmp_path: Path, empty_lerobot_dataset_factory: Any) -> None:
    """Ensure v2.0 input is accepted and output stays v2.1.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "source_v20_dataset"
    output_root = tmp_path / "segmented_from_v20"

    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(8):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i) + 1.0], dtype=np.float32),
                "task": "stack blocks",
            }
        )
    dataset.save_episode()

    # Convert source metadata to a v2.0-style dataset:
    # - set codebase_version to 2.0
    # - write legacy global stats.json used by v2.0 loaders
    source_info = load_info(input_root)
    source_info["codebase_version"] = "v2.0"
    write_json(source_info, input_root / "meta" / "info.json")
    write_stats(dataset.meta.stats, input_root)

    segment_dataset(
        input_root=input_root,
        output_root=output_root,
        segments_by_episode={0: [(0, 3), (3, 8)]},
    )

    out_info = load_info(output_root)
    out_episodes = load_episodes(output_root)
    out_stats = load_episodes_stats(output_root)

    assert out_info["codebase_version"] == "v2.1"
    assert out_info["total_episodes"] == 2
    assert out_info["total_frames"] == 8
    assert out_episodes[0]["length"] == 3
    assert out_episodes[1]["length"] == 5
    assert len(out_stats) == 2


def test_segment_lerobot_non_consecutive_and_overlapping_ranges(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """Cover non-consecutive and overlapping segment ranges.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "source_edge_case_dataset"
    output_root = tmp_path / "segmented_edge_case_dataset"

    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(25):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i) + 10.0], dtype=np.float32),
                "task": "edge-case task",
            }
        )
    dataset.save_episode()

    # Non-consecutive + overlapping segments:
    # - (0, 10) and (5, 15) overlap on [5..9]
    # - (18, 23) is non-consecutive with both
    segment_dataset(
        input_root=input_root,
        output_root=output_root,
        segments_by_episode={0: [(0, 10), (18, 23), (5, 15)]},
    )

    info = load_info(output_root)
    episodes = load_episodes(output_root)
    output_meta = LeRobotDatasetMetadata(repo_id=output_root.name, root=output_root)

    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 25
    assert [episodes[i]["length"] for i in sorted(episodes)] == [10, 5, 10]

    ep0 = pq.read_table(output_root / output_meta.get_data_file_path(0)).to_pydict()
    ep1 = pq.read_table(output_root / output_meta.get_data_file_path(1)).to_pydict()
    ep2 = pq.read_table(output_root / output_meta.get_data_file_path(2)).to_pydict()

    # Local frame indexing resets per output episode.
    assert ep0["frame_index"] == list(range(10))
    assert ep1["frame_index"] == list(range(5))
    assert ep2["frame_index"] == list(range(10))

    # Global index remains contiguous across output episodes.
    assert ep0["index"] == list(range(0, 10))
    assert ep1["index"] == list(range(10, 15))
    assert ep2["index"] == list(range(15, 25))

    # Timestamps are rebased per output episode.
    assert np.isclose(float(ep0["timestamp"][0]), 0.0)
    assert np.isclose(float(ep1["timestamp"][0]), 0.0)
    assert np.isclose(float(ep2["timestamp"][0]), 0.0)

    # Data slices match requested source windows.
    assert [float(x) for x in ep0["state"]] == [float(i) for i in range(0, 10)]
    assert [float(x) for x in ep1["state"]] == [float(i) for i in range(18, 23)]
    assert [float(x) for x in ep2["state"]] == [float(i) for i in range(5, 15)]


def test_segment_lerobot_copies_image_files_for_segments(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """Ensure segmented datasets copy and rewrite image file references.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "source_image_dataset"
    output_root = tmp_path / "segmented_image_dataset"
    image_key = "observation.images.camera"

    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
        image_key: {"dtype": "image", "shape": (3, 8, 8), "names": ["channel", "height", "width"]},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(8):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i) + 1.0], dtype=np.float32),
                "observation.images.camera": np.full((8, 8, 3), i / 8.0, dtype=np.float32),
                "task": "image task",
            }
        )
    dataset.save_episode()

    segment_dataset(
        input_root=input_root,
        output_root=output_root,
        segments_by_episode={0: [(1, 4), (4, 8)]},
    )

    out_meta = LeRobotDatasetMetadata(repo_id=output_root.name, root=output_root)
    ep0 = pq.read_table(output_root / out_meta.get_data_file_path(0)).to_pydict()
    ep1 = pq.read_table(output_root / out_meta.get_data_file_path(1)).to_pydict()

    ep0_paths = [_extract_image_path(cell) for cell in ep0[image_key]]
    ep1_paths = [_extract_image_path(cell) for cell in ep1[image_key]]
    assert all(path is not None for path in ep0_paths)
    assert all(path is not None for path in ep1_paths)
    for cell in ep0[image_key] + ep1[image_key]:
        if isinstance(cell, dict):
            assert cell.get("bytes") in (None, b"")

    for frame_idx, path in enumerate(ep0_paths):
        assert path is not None
        expected = output_root / f"images/{image_key}/episode_000000/frame_{frame_idx:06d}.png"
        assert Path(path) == expected
        assert expected.is_file()

    for frame_idx, path in enumerate(ep1_paths):
        assert path is not None
        expected = output_root / f"images/{image_key}/episode_000001/frame_{frame_idx:06d}.png"
        assert Path(path) == expected
        assert expected.is_file()


def test_load_segments_by_episode_from_json(tmp_path: Path) -> None:
    """Validate JSON segmentation-plan parsing.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    plan_path = tmp_path / "segments.json"
    plan_path.write_text('{"0": [[0, 3], [5, 8]], "2": [[10, 12]]}')
    parsed = _load_segments_by_episode(plan_path)
    assert parsed == {0: [(0, 3), (5, 8)], 2: [(10, 12)]}


def test_segment_lerobot_json_plan_with_two_source_episodes(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """Ensure JSON plan can segment multiple source episodes in one run.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "source_two_episode_dataset"
    output_root = tmp_path / "segmented_two_episode_dataset"

    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)

    # Episode 0
    for i in range(6):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i) + 100.0], dtype=np.float32),
                "task": "ep0",
            }
        )
    dataset.save_episode()

    # Episode 1
    for i in range(6):
        dataset.add_frame(
            {
                "state": np.array([float(i + 10)], dtype=np.float32),
                "actions": np.array([float(i) + 200.0], dtype=np.float32),
                "task": "ep1",
            }
        )
    dataset.save_episode()

    plan_path = tmp_path / "segments_two_episodes.json"
    plan_path.write_text('{"0": [[1, 4]], "1": [[2, 6]]}')
    segments_by_episode = _load_segments_by_episode(plan_path)
    segment_dataset(
        input_root=input_root,
        output_root=output_root,
        segments_by_episode=segments_by_episode,
    )

    info = load_info(output_root)
    episodes = load_episodes(output_root)
    out_meta = LeRobotDatasetMetadata(repo_id=output_root.name, root=output_root)

    assert info["total_episodes"] == 2
    assert [episodes[i]["length"] for i in sorted(episodes)] == [3, 4]

    ep0 = pq.read_table(output_root / out_meta.get_data_file_path(0)).to_pydict()
    ep1 = pq.read_table(output_root / out_meta.get_data_file_path(1)).to_pydict()
    assert [float(x) for x in ep0["state"]] == [1.0, 2.0, 3.0]
    assert [float(x) for x in ep1["state"]] == [12.0, 13.0, 14.0, 15.0]


def test_segment_lerobot_rejects_invalid_segment_bounds(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """Ensure segment bounds are validated in segment_dataset().

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "source_invalid_segment_dataset"
    output_root = tmp_path / "segmented_invalid_segment_dataset"
    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(5):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i)], dtype=np.float32),
                "task": "invalid-bounds",
            }
        )
    dataset.save_episode()

    with pytest.raises(ValueError, match="Expected 0 <= start < end <= source_length"):
        segment_dataset(
            input_root=input_root,
            output_root=output_root,
            segments_by_episode={0: [(-1, 2)]},
        )


def test_trim_video_segment_uses_frame_range_filter(tmp_path: Path) -> None:
    """Ensure ffmpeg trim command uses frame-range segmentation.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    src.write_bytes(b"fake")

    with (
        patch("opentau.scripts.segment_lerobot_dataset.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("opentau.scripts.segment_lerobot_dataset.subprocess.run") as run_mock,
    ):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stderr = ""

        from opentau.scripts.segment_lerobot_dataset import _trim_video_segment

        _trim_video_segment(src, dst, 5, 15)

    assert run_mock.call_count == 1
    cmd = run_mock.call_args.args[0]
    assert "ffmpeg" in cmd[0]
    assert "-vf" in cmd
    vf_expr = cmd[cmd.index("-vf") + 1]
    assert "trim=start_frame=5:end_frame=15" in vf_expr


@pytest.mark.parametrize("valid", ["1", "8", "64", " 16 ", "+4"])
def test_ffmpeg_trim_max_workers_env_var_valid(monkeypatch: pytest.MonkeyPatch, valid: str) -> None:
    """Valid integer env values in [1, 64] are used verbatim.

    Leading ``+`` signs are accepted (regression guard for the pre-review
    ``isdigit()`` check, which rejected them silently).

    Args:
        monkeypatch: pytest fixture for environment manipulation.
        valid: A string representation of a valid worker count.
    """
    monkeypatch.setenv(FFMPEG_MAX_WORKERS_ENV_VAR, valid)
    assert _ffmpeg_trim_max_workers() == int(valid.strip())


@pytest.mark.parametrize("bad", ["0", "-1", "65", "100", "abc", "", "  ", "3.14"])
def test_ffmpeg_trim_max_workers_env_var_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    """Invalid env values are rejected with a warning and ignored.

    Args:
        monkeypatch: pytest fixture for environment manipulation.
        caplog: pytest fixture capturing emitted log records.
        bad: A string representation of a rejected value.
    """
    monkeypatch.setenv(FFMPEG_MAX_WORKERS_ENV_VAR, bad)
    monkeypatch.setattr("opentau.scripts.segment_lerobot_dataset.os.cpu_count", lambda: 4)
    with caplog.at_level("WARNING", logger="opentau.scripts.segment_lerobot_dataset"):
        result = _ffmpeg_trim_max_workers()
    # CPU-based default: min(16, (4)*2) == 8
    assert result == 8
    # Empty / whitespace values are indistinguishable from unset and should
    # not emit a warning. Every other rejected value must log one.
    if bad.strip() != "":
        assert any(FFMPEG_MAX_WORKERS_ENV_VAR in rec.message for rec in caplog.records)


def test_ffmpeg_trim_max_workers_env_var_unset_uses_cpu_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is unset, the CPU-based default is returned.

    Args:
        monkeypatch: pytest fixture for environment manipulation.
    """
    monkeypatch.delenv(FFMPEG_MAX_WORKERS_ENV_VAR, raising=False)
    monkeypatch.setattr("opentau.scripts.segment_lerobot_dataset.os.cpu_count", lambda: 6)
    # min(16, 6*2) == 12
    assert _ffmpeg_trim_max_workers() == 12


def test_ffmpeg_trim_max_workers_env_var_caps_at_16(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU-based default is capped at 16 even on large machines.

    Args:
        monkeypatch: pytest fixture for environment manipulation.
    """
    monkeypatch.delenv(FFMPEG_MAX_WORKERS_ENV_VAR, raising=False)
    monkeypatch.setattr("opentau.scripts.segment_lerobot_dataset.os.cpu_count", lambda: 128)
    assert _ffmpeg_trim_max_workers() == 16


def test_ffmpeg_trim_max_workers_handles_none_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """``os.cpu_count()`` returning ``None`` must not crash the helper.

    Args:
        monkeypatch: pytest fixture for environment manipulation.
    """
    monkeypatch.delenv(FFMPEG_MAX_WORKERS_ENV_VAR, raising=False)
    monkeypatch.setattr("opentau.scripts.segment_lerobot_dataset.os.cpu_count", lambda: None)
    # Fallback cpu=4 -> min(16, 8) == 8
    assert _ffmpeg_trim_max_workers() == 8


@pytest.mark.parametrize("override", [1, 8, 64])
def test_ffmpeg_trim_max_workers_override_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, override: int
) -> None:
    """An explicit override wins over the env var and the CPU default.

    Args:
        monkeypatch: pytest fixture for environment manipulation.
        override: The integer override under test.
    """
    monkeypatch.setenv(FFMPEG_MAX_WORKERS_ENV_VAR, "32")
    assert _ffmpeg_trim_max_workers(override) == override


@pytest.mark.parametrize("bad", [0, -1, 65, 100, True, 3.14, "4"])
def test_ffmpeg_trim_max_workers_rejects_invalid_override(bad: Any) -> None:
    """Invalid overrides must raise ValueError, not silently fall back.

    Args:
        bad: A value that must be rejected by the override path.
    """
    with pytest.raises(ValueError, match=r"--ffmpeg-max-workers"):
        _ffmpeg_trim_max_workers(bad)


def _inject_fake_video_feature(input_root: Path, video_key: str) -> Path:
    """Declare a fake video feature on an image dataset and materialize the file.

    Registers a ``dtype=video`` entry under ``meta/info.json`` and creates a
    zero-byte stub file at the canonical video path so
    :func:`segment_dataset` will enqueue an ffmpeg job for it (we mock
    ``_trim_video_segment`` in tests, so the stub never has to be playable).

    Args:
        input_root: Root of the source dataset to mutate in place.
        video_key: Feature key for the fake video stream.

    Returns:
        Absolute path to the created stub video file.
    """
    info = load_info(input_root)
    info["features"][video_key] = {
        "dtype": "video",
        "shape": [3, 8, 8],
        "names": ["channel", "height", "width"],
        "info": {
            "video.fps": info.get("fps", 30),
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    if info.get("video_path") is None:
        info["video_path"] = DEFAULT_VIDEO_PATH
    write_json(info, input_root / "meta" / "info.json")

    rel_path = DEFAULT_VIDEO_PATH.format(episode_chunk=0, video_key=video_key, episode_index=0)
    src_video = input_root / rel_path
    src_video.parent.mkdir(parents=True, exist_ok=True)
    src_video.write_bytes(b"fake-video-stub")
    return src_video


def test_segment_dataset_dispatches_ffmpeg_jobs_in_parallel(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """Assert segment_dataset submits one trim per (segment, video_key) and
    propagates worker results via fut.result().

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "src_with_fake_video"
    output_root = tmp_path / "dst_with_fake_video"
    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(8):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i)], dtype=np.float32),
                "task": "parallel-dispatch",
            }
        )
    dataset.save_episode()

    video_key = "observation.images.fake_cam"
    _inject_fake_video_feature(input_root, video_key)

    with patch("opentau.scripts.segment_lerobot_dataset._trim_video_segment") as trim_mock:
        segment_dataset(
            input_root=input_root,
            output_root=output_root,
            segments_by_episode={0: [(0, 3), (3, 8)]},
        )

    # 2 segments × 1 fake video key = 2 trim calls.
    assert trim_mock.call_count == 2
    frame_ranges = sorted((call.args[2], call.args[3]) for call in trim_mock.call_args_list)
    assert frame_ranges == [(0, 3), (3, 8)]
    for call in trim_mock.call_args_list:
        src_path, dst_path, *_ = call.args
        assert Path(src_path).is_file()
        # Parent directory for the destination must be pre-created by the main
        # loop so workers never race on mkdir.
        assert Path(dst_path).parent.is_dir()


def test_segment_dataset_propagates_worker_exceptions(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """Exceptions raised inside ffmpeg workers must surface via fut.result().

    Regression guard for accidentally dropping ``fut.result()`` in the
    parallel-dispatch loop.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "src_worker_error"
    output_root = tmp_path / "dst_worker_error"
    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(6):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i)], dtype=np.float32),
                "task": "worker-error",
            }
        )
    dataset.save_episode()

    _inject_fake_video_feature(input_root, "observation.images.fake_cam")

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("synthetic ffmpeg failure")

    with (
        patch("opentau.scripts.segment_lerobot_dataset._trim_video_segment", side_effect=_boom),
        pytest.raises(RuntimeError, match="synthetic ffmpeg failure"),
    ):
        segment_dataset(
            input_root=input_root,
            output_root=output_root,
            segments_by_episode={0: [(0, 2), (2, 6)]},
        )


def test_segment_dataset_accepts_ffmpeg_max_workers_override(
    tmp_path: Path, empty_lerobot_dataset_factory: Any
) -> None:
    """The ``ffmpeg_max_workers`` kwarg must reach ``ThreadPoolExecutor``.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        empty_lerobot_dataset_factory: Fixture that creates a writable dataset.
    """
    input_root = tmp_path / "src_pool_size"
    output_root = tmp_path / "dst_pool_size"
    features = {
        "state": {"dtype": "float32", "shape": (1,), "names": None},
        "actions": {"dtype": "float32", "shape": (1,), "names": None},
    }
    dataset = empty_lerobot_dataset_factory(root=input_root, features=features, use_videos=False)
    for i in range(4):
        dataset.add_frame(
            {
                "state": np.array([float(i)], dtype=np.float32),
                "actions": np.array([float(i)], dtype=np.float32),
                "task": "pool-size",
            }
        )
    dataset.save_episode()

    _inject_fake_video_feature(input_root, "observation.images.fake_cam")

    from concurrent.futures import ThreadPoolExecutor as _RealPool

    observed_max_workers: list[int] = []

    class _SpyPool(_RealPool):
        def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
            observed_max_workers.append(int(max_workers) if max_workers is not None else -1)
            super().__init__(max_workers, *args, **kwargs)

    with (
        patch("opentau.scripts.segment_lerobot_dataset.ThreadPoolExecutor", _SpyPool),
        patch("opentau.scripts.segment_lerobot_dataset._trim_video_segment"),
    ):
        segment_dataset(
            input_root=input_root,
            output_root=output_root,
            segments_by_episode={0: [(0, 2), (2, 4)]},
            ffmpeg_max_workers=3,
        )

    assert observed_max_workers == [3]
