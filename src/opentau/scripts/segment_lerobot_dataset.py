#!/usr/bin/env python
#
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
"""Create a segmented LeRobot v2.1 dataset from a source episode.

This script builds a brand-new dataset where each output episode corresponds to one
`[start, end)` frame segment from a selected source episode.

Accepted input formats: LeRobot v2.0 and v2.1.
Output format: always LeRobot v2.1.

Example:
    python segment_lerobot_dataset.py ./input_dataset ./output_dataset \
        --episode-id 0 \
        --segment 0:100 \
        --segment 120:220
"""

import argparse
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from opentau.datasets.compute_stats import compute_episode_stats
from opentau.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from opentau.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    write_episode_stats,
    write_json,
)


def _parse_segment(text: str) -> tuple[int, int]:
    """Parse one CLI segment token.

    Args:
        text: Segment string formatted as ``START:END``.

    Returns:
        A 2-tuple ``(start, end)`` where ``start >= 0`` and ``end > start``.

    Raises:
        argparse.ArgumentTypeError: If the value is malformed or out of range.
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid segment '{text}'. Expected START:END with integer values.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid segment '{text}'. START and END must be integers."
        ) from exc
    if start < 0 or end < 0:
        raise argparse.ArgumentTypeError(f"Invalid segment '{text}'. START and END must be non-negative.")
    if end <= start:
        raise argparse.ArgumentTypeError(
            f"Invalid segment '{text}'. END must be strictly greater than START."
        )
    return start, end


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset segmentation.

    Returns:
        Parsed CLI namespace containing input/output roots, source episode id,
        and the list of frame-range segments.
    """
    parser = argparse.ArgumentParser(
        description="Create a segmented LeRobot v2.1 dataset from one source episode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_root", type=Path, help="Path to source LeRobot dataset root.")
    parser.add_argument("output_root", type=Path, help="Path to output dataset root (must not exist).")
    parser.add_argument(
        "--episode-id",
        type=int,
        default=0,
        help="Source episode index to segment.",
    )
    parser.add_argument(
        "--segment",
        type=_parse_segment,
        action="append",
        required=True,
        help="Segment as START:END in frame index domain. Repeat this argument for multiple segments.",
    )
    return parser.parse_args()


def _to_numpy_for_stats(column: pa.ChunkedArray) -> np.ndarray:
    """Convert an Arrow chunked column to a NumPy array for stats.

    Args:
        column: Arrow chunked column extracted from an episode parquet table.

    Returns:
        NumPy representation of the column values.
    """
    # For list/fixed-size-list numeric columns, `to_pylist` keeps the nested shape,
    # and `np.asarray` reconstructs the expected dense ndarray.
    return np.asarray(column.to_pylist())


def segment_dataset(
    input_root: Path,
    output_root: Path,
    episode_id: int,
    segments: list[tuple[int, int]],
) -> None:
    """Create a new segmented dataset from a source episode.

    Args:
        input_root: Source LeRobot dataset directory (v2.0 or v2.1).
        output_root: Destination directory for the new dataset. Must not exist.
        episode_id: Source episode index to slice.
        segments: List of ``(start, end)`` frame ranges in ``[start, end)`` form.

    Raises:
        ValueError: If inputs are invalid, source files are missing, or segment
            ranges are out of bounds.
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    if not input_root.is_dir():
        raise ValueError(f"Input dataset root does not exist: {input_root}")
    if output_root.exists():
        raise ValueError(f"Output dataset root already exists: {output_root}")
    if not segments:
        raise ValueError("At least one segment must be provided.")

    source_meta = LeRobotDatasetMetadata(repo_id=input_root.name, root=input_root)
    source_version = str(source_meta._version)
    if not (source_version.startswith("2.0") or source_version.startswith("2.1")):
        raise ValueError(
            "Only LeRobot dataset format v2.0 and v2.1 are supported as input by this script. "
            f"Found codebase_version={source_version}."
        )
    if episode_id not in source_meta.episodes:
        raise ValueError(f"Episode {episode_id} not found in source dataset.")

    source_episode = source_meta.episodes[episode_id]
    source_length = int(source_episode["length"])
    for start, end in segments:
        if end > source_length:
            raise ValueError(
                f"Segment ({start}, {end}) is out of bounds for source episode length {source_length}."
            )

    source_parquet_path = input_root / source_meta.get_data_file_path(episode_id)
    if not source_parquet_path.is_file():
        raise ValueError(f"Missing source parquet file for episode {episode_id}: {source_parquet_path}")
    source_table = pq.read_table(source_parquet_path)
    if source_table.num_rows != source_length:
        raise ValueError(
            f"Source metadata length ({source_length}) does not match parquet row count ({source_table.num_rows})."
        )

    output_root.mkdir(parents=True, exist_ok=False)
    chunks_size = int(source_meta.chunks_size)
    global_index_offset = 0
    total_frames = 0
    output_episodes: list[dict] = []

    # Write tasks as-is so existing task_index values remain valid.
    for task_index, task in sorted(source_meta.tasks.items(), key=lambda x: x[0]):
        append_jsonlines({"task_index": task_index, "task": task}, output_root / TASKS_PATH)

    source_episode_stats = source_meta.episodes_stats.get(cast(Any, episode_id), {})
    visual_keys = [k for k, ft in source_meta.features.items() if ft["dtype"] in ["image", "video"]]

    for output_episode_index, (start, end) in enumerate(segments):
        seg_len = end - start
        seg_table = source_table.slice(start, seg_len)

        replacement_arrays = {
            "episode_index": pa.array(np.full(seg_len, output_episode_index, dtype=np.int64)),
            "frame_index": pa.array(np.arange(seg_len, dtype=np.int64)),
            "index": pa.array(np.arange(global_index_offset, global_index_offset + seg_len, dtype=np.int64)),
        }
        for key, arr in replacement_arrays.items():
            col_idx = seg_table.schema.get_field_index(key)
            if col_idx >= 0:
                seg_table = seg_table.set_column(col_idx, key, arr)

        episode_chunk = output_episode_index // chunks_size
        output_parquet_path = output_root / source_meta.data_path.format(
            episode_chunk=episode_chunk,
            episode_index=output_episode_index,
        )
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(seg_table, output_parquet_path)

        # Build tasks list for this segment from task_index values present in rows.
        task_indices = seg_table.column("task_index").to_pylist()
        seen_task_indices = set()
        episode_tasks = []
        for task_index in task_indices:
            if task_index in seen_task_indices:
                continue
            seen_task_indices.add(task_index)
            episode_tasks.append(source_meta.tasks[int(task_index)])

        output_episodes.append(
            {
                "episode_index": output_episode_index,
                "tasks": episode_tasks,
                "length": seg_len,
            }
        )

        stats_features = {
            key: feature
            for key, feature in source_meta.features.items()
            if feature["dtype"] not in ["image", "video", "string"] and key in seg_table.column_names
        }
        stats_data: dict[str, list[str] | np.ndarray] = {
            key: _to_numpy_for_stats(seg_table.column(key)) for key in stats_features
        }
        episode_stats = compute_episode_stats(stats_data, stats_features)

        # Keep visual keys in stats for downstream compatibility.
        for key in visual_keys:
            if key in source_episode_stats:
                source_key_stats = cast(dict[str, np.ndarray], source_episode_stats[key])
                copied = {metric: np.array(val, copy=True) for metric, val in source_key_stats.items()}
                if "count" in copied:
                    copied["count"] = np.array([seg_len], dtype=np.int64)
                episode_stats[key] = copied

        write_episode_stats(output_episode_index, episode_stats, output_root)

        global_index_offset += seg_len
        total_frames += seg_len

    # Copy source video episode for each requested segment episode, if any.
    video_path_template = source_meta.video_path
    if source_meta.video_keys and video_path_template is None:
        raise ValueError("Source dataset declares video keys but has no video_path template in metadata.")
    video_path_template_str = cast(str, video_path_template) if video_path_template is not None else ""

    for video_key in source_meta.video_keys:
        src_video_path = input_root / source_meta.get_video_file_path(episode_id, video_key)
        if not src_video_path.is_file():
            raise ValueError(f"Missing source video for key '{video_key}': {src_video_path}")
        for output_episode_index in range(len(segments)):
            episode_chunk = output_episode_index // chunks_size
            dst_video_path = output_root / video_path_template_str.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=output_episode_index,
            )
            dst_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video_path, dst_video_path)

    for episode in output_episodes:
        append_jsonlines(episode, output_root / EPISODES_PATH)

    total_episodes = len(segments)
    info = deepcopy(source_meta.info)
    info["codebase_version"] = CODEBASE_VERSION
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = int(math.ceil(total_episodes / chunks_size)) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(source_meta.video_keys)
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_json(info, output_root / "meta" / "info.json")

    # Ensure expected meta files exist and are explicit outputs.
    _ = output_root / EPISODES_STATS_PATH


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    segment_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        episode_id=args.episode_id,
        segments=args.segment,
    )


if __name__ == "__main__":
    main()
