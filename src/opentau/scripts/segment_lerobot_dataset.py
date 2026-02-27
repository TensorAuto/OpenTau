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
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from opentau.datasets.compute_stats import compute_episode_stats
from opentau.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from opentau.datasets.utils import (
    DEFAULT_IMAGE_PATH,
    EPISODES_PATH,
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


def _trim_video_segment(src_video_path: Path, dst_video_path: Path, start_frame: int, end_frame: int) -> None:
    """Trim a source video to the requested frame interval.

    Args:
        src_video_path: Source episode video path.
        dst_video_path: Output path for the trimmed segment video.
        start_frame: Inclusive start frame index.
        end_frame: Exclusive end frame index.

    Raises:
        RuntimeError: If ffmpeg is unavailable or the trim command fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim segmented videos but was not found in PATH.")

    # Trim by exact frame indices and reset timeline to start at zero.
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_video_path),
        "-vf",
        vf,
        "-an",
        str(dst_video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to trim video segment {start_frame}:{end_frame} from '{src_video_path}'. "
            f"ffmpeg stderr: {result.stderr.strip()}"
        )


def _copy_segment_images_and_rewrite_column(
    image_cells: list[Any],
    input_root: Path,
    output_root: Path,
    image_key: str,
    output_episode_index: int,
    source_episode_index: int,
    source_segment_start: int,
) -> list[Any]:
    """Copy image files for a segment and rewrite per-row image references.

    Args:
        image_cells: Image column values from the sliced source table.
        input_root: Source dataset root path.
        output_root: Output dataset root path.
        image_key: Feature key for this image stream.
        output_episode_index: Output episode index receiving this segment.
        source_episode_index: Source episode index for image path fallback.
        source_segment_start: Start frame index of this segment in source episode.

    Returns:
        New image column values with updated file paths for copied images.

    Raises:
        FileNotFoundError: If a referenced source image file does not exist.
    """
    rewritten_cells: list[Any] = []
    for frame_index, cell in enumerate(image_cells):
        rel_dst = DEFAULT_IMAGE_PATH.format(
            image_key=image_key,
            episode_index=output_episode_index,
            frame_index=frame_index,
        )
        dst_path = output_root / rel_dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(cell, dict):
            image_bytes = cell.get("bytes")
            if isinstance(image_bytes, (bytes, bytearray)) and len(image_bytes) > 0:
                dst_path.write_bytes(bytes(image_bytes))
                new_cell = dict(cell)
                new_cell["path"] = str(dst_path)
                rewritten_cells.append(new_cell)
                continue

        src_path: Path | None = None
        if isinstance(cell, str):
            src_path = Path(cell)
        elif isinstance(cell, dict):
            path_val = cell.get("path")
            if isinstance(path_val, str) and path_val:
                src_path = Path(path_val)

        # Embedded-image rows may not require copying when path is empty.
        if src_path is None:
            rewritten_cells.append(cell)
            continue

        if not src_path.is_absolute():
            src_path = input_root / src_path
        if not src_path.is_file():
            # Fallback to canonical image location under input root.
            source_frame_index = source_segment_start + frame_index
            src_path = input_root / DEFAULT_IMAGE_PATH.format(
                image_key=image_key,
                episode_index=source_episode_index,
                frame_index=source_frame_index,
            )
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source image for key '{image_key}': {src_path}")

        shutil.copy2(src_path, dst_path)

        if isinstance(cell, str):
            rewritten_cells.append(str(dst_path))
        else:
            new_cell = dict(cell)
            new_cell["path"] = str(dst_path)
            rewritten_cells.append(new_cell)

    return rewritten_cells


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

    Notes:
        For visual features (``dtype`` in ``{"image", "video"}``), per-episode
        statistics (``min``, ``max``, ``mean``, ``std``) are inherited from the
        source episode statistics and only the ``count`` is updated to the segment
        length. They are not recomputed from the segmented visual data.

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

        # Recompute timestamps from local frame_index to avoid subtraction drift.
        if "timestamp" in seg_table.column_names:
            ts_idx = seg_table.schema.get_field_index("timestamp")
            recomputed_ts = np.arange(seg_len, dtype=np.float64) / float(source_meta.fps)
            seg_table = seg_table.set_column(
                ts_idx,
                "timestamp",
                pa.array(recomputed_ts, type=seg_table.schema.field("timestamp").type),
            )

        # For image-based datasets, copy only the segment frames and rewrite image references.
        image_keys = [k for k, ft in source_meta.features.items() if ft["dtype"] == "image"]
        for image_key in image_keys:
            if image_key not in seg_table.column_names:
                continue
            col_idx = seg_table.schema.get_field_index(image_key)
            image_cells = seg_table.column(image_key).to_pylist()
            rewritten = _copy_segment_images_and_rewrite_column(
                image_cells=image_cells,
                input_root=input_root,
                output_root=output_root,
                image_key=image_key,
                output_episode_index=output_episode_index,
                source_episode_index=episode_id,
                source_segment_start=start,
            )
            seg_table = seg_table.set_column(
                col_idx, image_key, pa.array(rewritten, type=seg_table.schema.field(image_key).type)
            )

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
        for output_episode_index, (start, end) in enumerate(segments):
            episode_chunk = output_episode_index // chunks_size
            dst_video_path = output_root / video_path_template_str.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=output_episode_index,
            )
            dst_video_path.parent.mkdir(parents=True, exist_ok=True)
            _trim_video_segment(src_video_path, dst_video_path, start, end)

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
