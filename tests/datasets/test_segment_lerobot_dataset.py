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

import numpy as np
import pyarrow.parquet as pq

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import load_episodes, load_episodes_stats, load_info, write_json, write_stats
from opentau.scripts.segment_lerobot_dataset import segment_dataset


def test_segment_lerobot_v21_dataset(tmp_path, empty_lerobot_dataset_factory):
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
    assert [float(x) for x in ep0["state"]] == [2.0, 3.0, 4.0]

    assert ep1["frame_index"] == [0, 1, 2, 3, 4]
    assert ep1["episode_index"] == [1, 1, 1, 1, 1]
    assert ep1["index"] == [3, 4, 5, 6, 7]
    assert [float(x) for x in ep1["state"]] == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_segment_lerobot_v20_input_outputs_v21(tmp_path, empty_lerobot_dataset_factory):
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


def test_segment_lerobot_non_consecutive_and_overlapping_ranges(tmp_path, empty_lerobot_dataset_factory):
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

    # Data slices match requested source windows.
    assert [float(x) for x in ep0["state"]] == [float(i) for i in range(0, 10)]
    assert [float(x) for x in ep1["state"]] == [float(i) for i in range(18, 23)]
    assert [float(x) for x in ep2["state"]] == [float(i) for i in range(5, 15)]
