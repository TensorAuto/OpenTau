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
"""Add a ``response`` column to LeRobot dataset parquet files from per-episode subtask JSONs.

For every episode the script reads the subtask JSON (located via the
``subtask_path`` template in ``meta/info.json``), converts the time-based
subtask boundaries to frame indices using the dataset FPS, and writes the
active subtask string into a ``response`` column in the episode parquet.

If a subtask JSON is missing for an episode a warning is emitted and the
``response`` column is filled with empty strings.

Example::

    python src/opentau/scripts/add_subtask_response.py \
        --config_path configs/examples/add_subtask_response.json
"""

import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from opentau.configs import parser
from opentau.configs.default import DatasetMixtureConfig
from opentau.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    load_episodes,
    load_info,
    write_info,
)
from opentau.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _get_parquet_path(root: Path, data_path_template: str, ep_index: int, chunks_size: int) -> Path:
    ep_chunk = ep_index // chunks_size
    return root / data_path_template.format(episode_chunk=ep_chunk, episode_index=ep_index)


def _build_response_array(
    subtasks: list[dict[str, float | str]],
    fps: int,
    episode_length: int,
    episode_index: int,
) -> list[str]:
    """Map time-based subtask entries to per-frame response strings.

    Each frame between subtask *i*'s start frame and subtask *i+1*'s start
    frame receives subtask *i*'s string.  The last subtask extends to the
    end of the episode.  Frames before the first subtask's start frame
    (if > 0) receive an empty string.
    """
    responses = [""] * episode_length
    if not subtasks:
        return responses

    subtasks = sorted(subtasks, key=lambda s: s["time"])

    for i, entry in enumerate(subtasks):
        start_frame = round(entry["time"] * fps)
        if start_frame >= episode_length:
            logger.warning(
                "Episode %d: subtask '%s' starts at frame %d which is beyond episode length %d; skipping.",
                episode_index,
                entry["subtask"],
                start_frame,
                episode_length,
            )
            continue

        if i + 1 < len(subtasks):
            end_frame = min(round(subtasks[i + 1]["time"] * fps), episode_length)
        else:
            end_frame = episode_length

        if end_frame > start_frame:
            responses[start_frame:end_frame] = [entry["subtask"]] * (end_frame - start_frame)

    return responses


def _process_dataset(root: Path) -> None:
    """Add ``response`` column to every episode parquet in one dataset."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")

    info = load_info(root)
    fps: int = info["fps"]
    data_path_template: str = info["data_path"]
    chunks_size: int = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    subtask_path_template: str | None = info.get("subtask_path")
    if subtask_path_template is None:
        raise ValueError(
            f"info.json at {root} does not contain a 'subtask_path' field. "
            "Cannot locate per-episode subtask JSON files."
        )

    episodes = load_episodes(root)
    if not episodes:
        logger.warning("No episodes found in %s — nothing to do.", root)
        return

    logger.info(
        "Processing %d episodes in %s (fps=%d, subtask_path=%s)",
        len(episodes),
        root,
        fps,
        subtask_path_template,
    )

    for ep_index in tqdm(sorted(episodes), desc=str(root.name), unit="ep"):
        ep_info = episodes[ep_index]
        ep_length: int = ep_info["length"]
        parquet_path = _get_parquet_path(root, data_path_template, ep_index, chunks_size)

        if not parquet_path.is_file():
            logger.warning("Episode %d: parquet file not found at %s; skipping.", ep_index, parquet_path)
            continue

        subtask_file = root / subtask_path_template.format(episode_index=ep_index)

        if subtask_file.is_file():
            with open(subtask_file) as f:
                subtasks: list[dict[str, float | str]] = json.load(f)
            responses = _build_response_array(subtasks, fps, ep_length, ep_index)
        else:
            logger.warning(
                "Episode %d: subtask file not found at %s; filling response with empty strings.",
                ep_index,
                subtask_file,
            )
            responses = [""] * ep_length

        table = pq.read_table(parquet_path)
        if table.num_rows != ep_length:
            logger.warning(
                "Episode %d: parquet has %d rows but episodes.jsonl reports length %d. Using parquet row count.",
                ep_index,
                table.num_rows,
                ep_length,
            )
            if len(responses) < table.num_rows:
                responses.extend([""] * (table.num_rows - len(responses)))
            else:
                responses = responses[: table.num_rows]

        response_col = pa.array(responses, type=pa.string())

        if "response" in table.column_names:
            logger.warning(
                "Episode %d: overwriting existing 'response' column in %s.", ep_index, parquet_path
            )
            col_idx = table.schema.get_field_index("response")
            table = table.set_column(col_idx, "response", response_col)
        else:
            table = table.append_column("response", response_col)

        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        try:
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, parquet_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    if "response" not in info.get("features", {}):
        info.setdefault("features", {})["response"] = {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        }
        write_info(info, root)
        logger.info("Added 'response' feature to info.json")

    logger.info("Done processing %s", root)


@parser.wrap()
def add_subtask_response(cfg: DatasetMixtureConfig) -> None:
    """Add subtask response column to all datasets in the mixture config."""
    init_logging()

    if not cfg.datasets:
        raise ValueError("No datasets specified in the config.")

    for ds_cfg in cfg.datasets:
        if ds_cfg.root is None:
            raise ValueError(
                f"Dataset '{ds_cfg.repo_id}' has no 'root' specified. "
                "A local dataset root path is required for this script."
            )
        _process_dataset(Path(ds_cfg.root))


if __name__ == "__main__":
    add_subtask_response()
