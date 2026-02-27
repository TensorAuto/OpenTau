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

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import load_episodes, load_episodes_stats, load_info, write_json, write_stats
from opentau.scripts.segment_lerobot_dataset import segment_dataset


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
        episode_id=0,
        segments=[(2, 5), (5, 10)],
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
        episode_id=0,
        segments=[(0, 3), (3, 8)],
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
        episode_id=0,
        segments=[(0, 10), (18, 23), (5, 15)],
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
        episode_id=0,
        segments=[(1, 4), (4, 8)],
    )

    out_meta = LeRobotDatasetMetadata(repo_id=output_root.name, root=output_root)
    ep0 = pq.read_table(output_root / out_meta.get_data_file_path(0)).to_pydict()
    ep1 = pq.read_table(output_root / out_meta.get_data_file_path(1)).to_pydict()

    ep0_paths = [_extract_image_path(cell) for cell in ep0[image_key]]
    ep1_paths = [_extract_image_path(cell) for cell in ep1[image_key]]
    assert all(path is not None for path in ep0_paths)
    assert all(path is not None for path in ep1_paths)

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
