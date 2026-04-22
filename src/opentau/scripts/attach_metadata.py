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
"""Attach segment-level metadata to a LeRobot v2.1 dataset (copy only).

Reads an annotations JSON describing, for every episode in the source
dataset, the per-segment subtask text, success flag, and an episode-level
quality score. Produces a copy of the dataset with:

  - ``data/chunk-XXX/episode_XXXXXX.parquet`` gains three columns:
      * ``response`` (string)      — the current segment's ``subtask`` text.
      * ``memory``   (string)      — a cumulative memory summary (generated
        via ``pi_mem_data_generator._enrich_list``; placeholder under
        ``--skip-memory``).
      * ``mistake``  (int64, 0/1)  — ``not segment.success`` for the
        current segment.
  - ``meta/info.json`` gains feature entries for the three new columns.
  - ``meta/episodes.jsonl`` gains ``quality`` (int 1-5) and ``segments``
    (list of segment start frame indices) per episode.

Stats files (``meta/episodes_stats.jsonl``, ``meta/stats.json``) are left
untouched — the new columns carry no normalization stats.

The script is copy-only: ``--copy-to`` is required. The source dataset is
never modified.

Input JSON schema::

    [
      {
        "episode_id": 0,
        "quality": 3,
        "segments": [
          {"start": 0,  "subtask": "...", "success": false},
          {"start": 50, "subtask": "...", "success": true}
        ]
      },
      ...
    ]

Validation is strict and fail-fast: a mismatch between the set of
``episode_id``s and the dataset's ``episode_index`` set, a non-integer
``start`` (frame index, not time), a first segment that does not start at 0,
non-monotonic starts, ``quality`` outside [1, 5], or a non-bool ``success``
all raise ``ValueError`` before anything is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import (
    EPISODES_PATH,
    INFO_PATH,
    load_jsonlines,
    write_info,
    write_jsonlines,
)
from opentau.scripts.pi_mem_data_generator import (
    SYSTEM_PROMPT,
    _enrich_list,
    _load_env_file,
    _normalize_openai_api_key,
)

RESPONSE_COL = "response"
MEMORY_COL = "memory"
MISTAKE_COL = "mistake"
NEW_COLUMNS = (RESPONSE_COL, MEMORY_COL, MISTAKE_COL)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy a LeRobot v2.1 dataset and attach per-segment response/memory/mistake "
            "parquet columns plus episode-level quality/segments in episodes.jsonl."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the source LeRobot v2.1 dataset root.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to the annotations JSON (see module docstring for schema).",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        required=True,
        help="Destination dataset root. Must not exist unless --overwrite is given.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --copy-to destination to be clobbered if it exists.",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help=(
            "Skip OpenAI calls; fill the memory column with a per-frame placeholder "
            "of the form f'memory{frame_index}'. Used by tests and offline runs."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model used when memory generation is enabled.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between OpenAI calls (rate limiting).",
    )
    return parser.parse_args()


def _copy_dataset_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    """Copy ``src`` to ``dst``, refusing to clobber unless ``overwrite``.

    Args:
        src: Source dataset root.
        dst: Destination dataset root.
        overwrite: If true, an existing destination is removed first.

    Raises:
        FileExistsError: If ``dst`` exists and ``overwrite`` is False.
    """
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {dst}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store"))


def _validate_episode_entry(entry: Any, idx: int) -> None:
    """Raise ``ValueError`` if a top-level episode annotation is malformed."""
    if not isinstance(entry, dict):
        raise ValueError(f"Annotation entry #{idx} must be an object, got {type(entry).__name__}.")
    for required in ("episode_id", "quality", "segments"):
        if required not in entry:
            raise ValueError(f"Annotation entry #{idx} missing required key '{required}'.")
    if not isinstance(entry["episode_id"], int) or isinstance(entry["episode_id"], bool):
        raise ValueError(
            f"Annotation entry #{idx} episode_id must be int, got {type(entry['episode_id']).__name__}."
        )


def _validate_quality(quality: Any, ep_id: int) -> None:
    """Raise ``ValueError`` unless ``quality`` is an int in ``[1, 5]``."""
    if isinstance(quality, bool) or not isinstance(quality, int):
        raise ValueError(f"Episode {ep_id} quality must be int, got {type(quality).__name__}.")
    if quality not in (1, 2, 3, 4, 5):
        raise ValueError(f"Episode {ep_id} quality must be in 1..5, got {quality}.")


def _validate_segments(segments: Any, ep_id: int, ep_length: int) -> None:
    """Raise ``ValueError`` if a segment list fails any structural check."""
    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError(f"Episode {ep_id} must have a non-empty 'segments' list.")
    prev_start = -1
    for k, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"Episode {ep_id} segment {k} must be an object.")
        for required in ("start", "subtask", "success"):
            if required not in seg:
                raise ValueError(f"Episode {ep_id} segment {k} missing required key '{required}'.")
        start = seg["start"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(
                f"Episode {ep_id} segment {k} start must be int (frame index, not a time), "
                f"got {type(start).__name__}."
            )
        if k == 0 and start != 0:
            raise ValueError(f"Episode {ep_id} first segment must start at 0, got {start}.")
        if start <= prev_start:
            raise ValueError(
                f"Episode {ep_id} segment starts must strictly increase; "
                f"segment {k} start={start} <= previous start={prev_start}."
            )
        if start >= ep_length:
            raise ValueError(f"Episode {ep_id} segment {k} start={start} >= episode length {ep_length}.")
        if not isinstance(seg["subtask"], str):
            raise ValueError(f"Episode {ep_id} segment {k} subtask must be a string.")
        if not isinstance(seg["success"], bool):
            raise ValueError(f"Episode {ep_id} segment {k} success must be bool.")
        prev_start = start


def _load_and_validate_annotations(
    annotations_path: Path,
    meta: LeRobotDatasetMetadata,
) -> dict[int, dict[str, Any]]:
    """Load the annotations JSON and verify it covers the dataset exactly.

    Args:
        annotations_path: Path to the annotations JSON file.
        meta: Metadata of the source dataset (for episode set and lengths).

    Returns:
        Mapping ``{episode_id: {"quality": int, "segments": [dict, ...]}}``.

    Raises:
        ValueError: On any schema, coverage, or monotonicity violation.
    """
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of episode annotations.")

    ann_by_ep: dict[int, dict[str, Any]] = {}
    for i, entry in enumerate(data):
        _validate_episode_entry(entry, i)
        ep_id = entry["episode_id"]
        if ep_id in ann_by_ep:
            raise ValueError(f"Duplicate episode_id {ep_id} in annotations.")
        ann_by_ep[ep_id] = {"quality": entry["quality"], "segments": entry["segments"]}

    dataset_ids = set(meta.episodes.keys())
    json_ids = set(ann_by_ep.keys())
    missing = sorted(dataset_ids - json_ids)
    extra = sorted(json_ids - dataset_ids)
    if missing:
        raise ValueError(
            f"Missing annotations for episodes: {missing[:20]}{'...' if len(missing) > 20 else ''}."
        )
    if extra:
        raise ValueError(
            f"Unknown episodes in JSON (not in dataset): {extra[:20]}{'...' if len(extra) > 20 else ''}."
        )

    for ep_id, payload in ann_by_ep.items():
        _validate_quality(payload["quality"], ep_id)
        _validate_segments(payload["segments"], ep_id, meta.episodes[ep_id]["length"])

    return ann_by_ep


def _build_per_frame_columns(
    segments: list[dict[str, Any]],
    ep_length: int,
    *,
    skip_memory: bool,
) -> tuple[list[str], list[str], list[int]]:
    """Expand a segment list into per-frame (response, memory, mistake) columns.

    Args:
        segments: Ordered list of segment dicts (validated). Each segment dict
            must carry ``start`` (int frame index), ``subtask`` (str),
            ``success`` (bool), and — when ``skip_memory`` is False —
            ``memory`` (str, populated by ``_generate_memories`` upstream).
        ep_length: Total frame count for the episode.
        skip_memory: When True, the memory column is filled with
            ``f"memory{frame_index}"`` instead of the generated memory.

    Returns:
        Tuple of three per-frame lists of length ``ep_length``.
    """
    response_col: list[str] = [""] * ep_length
    memory_col: list[str] = [""] * ep_length
    mistake_col: list[int] = [0] * ep_length

    starts = [seg["start"] for seg in segments]
    ends = starts[1:] + [ep_length]
    for seg, start, end in zip(segments, starts, ends, strict=True):
        subtask = seg["subtask"]
        mistake = 0 if seg["success"] else 1
        memory_text = "" if skip_memory else str(seg.get("memory", ""))
        for f in range(start, end):
            response_col[f] = subtask
            memory_col[f] = f"memory{f}" if skip_memory else memory_text
            mistake_col[f] = mistake
    return response_col, memory_col, mistake_col


def _generate_memories(
    ann_by_ep: dict[int, dict[str, Any]],
    meta: LeRobotDatasetMetadata,
    *,
    model: str,
    delay_s: float,
) -> None:
    """Run ``_enrich_list`` per-episode to attach a ``memory`` field to every segment.

    Uses the episode's first task string as the prompt seed. The OpenAI client
    is created here so failures surface before any file mutation.

    Args:
        ann_by_ep: Mapping produced by :func:`_load_and_validate_annotations`.
            Mutated in place: each segment gains a ``memory`` string.
        meta: Source dataset metadata (used to resolve task strings).
        model: OpenAI chat completion model name.
        delay_s: Seconds to sleep between OpenAI calls.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is missing or the API fails.
    """
    from openai import OpenAI  # Imported lazily so --skip-memory never needs the SDK.

    _load_env_file()
    api_key = _normalize_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Put it in a .env file at the repo root or export it, "
            "or re-run with --skip-memory."
        )
    client = OpenAI(api_key=api_key)

    for ep_id in tqdm(sorted(ann_by_ep.keys()), desc="Generating memories", unit="episode"):
        segments = ann_by_ep[ep_id]["segments"]
        ep_tasks = meta.episodes[ep_id].get("tasks") or []
        task_prompt = ep_tasks[0] if ep_tasks else ""
        enrich_payload: list[dict[str, Any]] = [
            {"subtask": seg["subtask"], "success": seg["success"], "prompt": task_prompt} for seg in segments
        ]
        _enrich_list(
            enrich_payload,
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            output_key="memory",
            skip_existing=False,
            delay_s=delay_s,
        )
        for seg, enriched in zip(segments, enrich_payload, strict=True):
            seg["memory"] = enriched["memory"]


def _rewrite_episode_parquet(
    parquet_path: Path,
    *,
    response: list[str],
    memory: list[str],
    mistake: list[int],
) -> None:
    """Append ``response``/``memory``/``mistake`` columns to a single parquet file.

    Refuses to overwrite columns that already exist (idempotency is enforced at
    the dataset level; reaching this function with existing columns means the
    idempotency check was skipped).

    Args:
        parquet_path: Parquet file in the copied tree (modified in place inside
            that copy).
        response: Per-frame response strings (length = parquet row count).
        memory: Per-frame memory strings (length = parquet row count).
        mistake: Per-frame 0/1 values (length = parquet row count).

    Raises:
        ValueError: If any length mismatches the parquet row count.
    """
    table = pq.read_table(parquet_path)
    num_rows = table.num_rows
    if len(response) != num_rows or len(memory) != num_rows or len(mistake) != num_rows:
        raise ValueError(
            f"Column length mismatch for {parquet_path}: rows={num_rows}, "
            f"response={len(response)}, memory={len(memory)}, mistake={len(mistake)}."
        )

    new_columns = {
        RESPONSE_COL: pa.array(response, type=pa.string()),
        MEMORY_COL: pa.array(memory, type=pa.string()),
        MISTAKE_COL: pa.array(mistake, type=pa.int64()),
    }
    existing = set(table.column_names)
    for name, array in new_columns.items():
        if name in existing:
            raise ValueError(f"Parquet {parquet_path} already has column '{name}'.")
        table = table.append_column(name, array)

    # Update embedded HuggingFace schema metadata so the datasets library picks
    # up the new columns with the right types.
    schema_meta = dict(table.schema.metadata or {})
    hf_key = b"huggingface"
    if hf_key in schema_meta:
        hf_info = json.loads(schema_meta[hf_key].decode())
        features = hf_info.setdefault("info", {}).setdefault("features", {})
        features[RESPONSE_COL] = {"dtype": "string", "_type": "Value"}
        features[MEMORY_COL] = {"dtype": "string", "_type": "Value"}
        features[MISTAKE_COL] = {"dtype": "int64", "_type": "Value"}
        schema_meta[hf_key] = json.dumps(hf_info).encode()
        table = table.replace_schema_metadata(schema_meta)

    pq.write_table(table, parquet_path)


def _update_info_json_features(root: Path) -> None:
    """Add feature entries for the three new parquet columns to ``meta/info.json``.

    Idempotent: existing entries are left alone.
    """
    info_path = root / INFO_PATH
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    new_entries = {
        RESPONSE_COL: {"dtype": "string", "shape": [1], "names": None},
        MEMORY_COL: {"dtype": "string", "shape": [1], "names": None},
        MISTAKE_COL: {"dtype": "int64", "shape": [1], "names": None},
    }
    for name, spec in new_entries.items():
        if name not in features:
            features[name] = spec
    write_info(info, root)


def _update_episodes_jsonl(root: Path, ann_by_ep: dict[int, dict[str, Any]]) -> None:
    """Rewrite ``meta/episodes.jsonl`` adding ``quality`` and ``segments`` per episode.

    Preserves all pre-existing fields (``episode_index``, ``tasks``,
    ``length``, ...). The ``segments`` field stores only start frame indices;
    ends are implicit (next start, or episode length for the last).
    """
    records = load_jsonlines(root / EPISODES_PATH)
    for rec in records:
        ep_id = int(rec["episode_index"])
        if ep_id not in ann_by_ep:
            raise ValueError(f"episodes.jsonl contains episode {ep_id} not in annotations.")
        payload = ann_by_ep[ep_id]
        rec["quality"] = int(payload["quality"])
        rec["segments"] = [int(seg["start"]) for seg in payload["segments"]]
    write_jsonlines(records, root / EPISODES_PATH)


def attach_metadata(
    dataset_path: Path,
    annotations_path: Path,
    *,
    copy_to: Path,
    overwrite: bool,
    skip_memory: bool,
    model: str,
    delay_s: float,
) -> None:
    """Copy a dataset and attach segment annotations.

    See module docstring for the end-to-end semantics.

    Args:
        dataset_path: Source LeRobot v2.1 dataset root.
        annotations_path: Annotations JSON path.
        copy_to: Destination dataset root (must not exist unless ``overwrite``).
        overwrite: Permit clobbering of ``copy_to``.
        skip_memory: Skip OpenAI calls; fill memory with placeholder text.
        model: OpenAI chat completion model name.
        delay_s: Seconds to sleep between OpenAI calls.

    Raises:
        ValueError: On annotation validation failure or column mismatch.
        FileExistsError: If ``copy_to`` exists and ``overwrite`` is False.
    """
    src = dataset_path.resolve()
    if not src.is_dir():
        raise ValueError(f"Input dataset root does not exist: {src}")
    if not (src / INFO_PATH).is_file():
        raise ValueError(f"Input is not a LeRobot dataset (missing {INFO_PATH}): {src}")

    src_meta = LeRobotDatasetMetadata(repo_id=src.name, root=src)
    if all(col in src_meta.features for col in NEW_COLUMNS):
        print(f"Dataset at {src} already has {list(NEW_COLUMNS)} in meta.features; skipping (no copy made).")
        return

    ann_by_ep = _load_and_validate_annotations(annotations_path, src_meta)

    # Generate all memories up front so OpenAI failures never leave the copy
    # half-modified. Under --skip-memory we skip this step entirely.
    if not skip_memory:
        _generate_memories(ann_by_ep, src_meta, model=model, delay_s=delay_s)

    dst = copy_to.resolve()
    print(f"Copying dataset: {src} -> {dst}")
    _copy_dataset_tree(src, dst, overwrite=overwrite)

    episode_indices = sorted(src_meta.episodes.keys())
    for ep_id in tqdm(episode_indices, desc="Rewriting parquet", unit="episode"):
        ep_length = int(src_meta.episodes[ep_id]["length"])
        response_col, memory_col, mistake_col = _build_per_frame_columns(
            ann_by_ep[ep_id]["segments"], ep_length, skip_memory=skip_memory
        )
        rel = src_meta.get_data_file_path(ep_id)
        _rewrite_episode_parquet(
            dst / rel,
            response=response_col,
            memory=memory_col,
            mistake=mistake_col,
        )

    _update_info_json_features(dst)
    _update_episodes_jsonl(dst, ann_by_ep)

    total_frames = int(sum(src_meta.episodes[ep]["length"] for ep in episode_indices))
    print(
        "Done. "
        f"Episodes: {len(episode_indices)}. "
        f"Frames: {total_frames}. "
        f"Memory source: {'placeholder' if skip_memory else model}."
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    attach_metadata(
        dataset_path=args.dataset,
        annotations_path=args.annotations,
        copy_to=args.copy_to,
        overwrite=args.overwrite,
        skip_memory=args.skip_memory,
        model=args.model,
        delay_s=args.delay,
    )


if __name__ == "__main__":
    main()
