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
"""Annotate each episode in a dataset mixture with subtask labels using Claude.

For every episode the script:

1. Samples ``--sample-fps`` frames per second from the first available video
   track (default 1 fps, giving a 30-50x reduction from typical 30-50 fps
   recordings).  If the resulting count exceeds ``--max-frames``, the sample
   rate is reduced proportionally so that at most ``--max-frames`` images are
   sent per episode.
2. Resizes each sampled frame to ``--target-width`` pixels wide (JPEG,
   default 640 px) to reduce image token cost.
3. Sends all sampled frames together with their timestamps to
   ``claude-opus-4-7`` and asks it to identify subtask transition times.
4. Saves the returned ``[{"time": float, "subtask": str}, ...]`` list as a
   per-episode JSON at ``{root}/subtasks/episode_{episode_index:06d}.json``.
5. Updates ``meta/info.json`` with a ``subtask_path`` field (required by
   ``add_subtask_response.py``).
6. Optionally expands the subtask boundaries into a per-frame ``response``
   column in each episode parquet (``--write-response-column``, on by
   default).

Episodes whose subtask JSON already exists are skipped — making the script
fully resumable after a crash.

The config file is read with plain JSON; it accepts both the full training
config format (``{"dataset_mixture": {"datasets": [...]}}``) and the simpler
``add_subtask_response`` format (``{"datasets": [...]}``) so that
``train_mixture_snippet.json`` can be passed directly.

Datasets with a ``root`` field are used directly.  Hub-only datasets (no
``root``) are downloaded to ``--hub-cache-dir`` (default:
``~/.cache/huggingface/opentau_subtasks``) via
``huggingface_hub.snapshot_download`` before processing; the downloaded
directory is then treated identically to a local dataset.

Example::

    python src/opentau/scripts/annotate_subtasks.py \\
        --config-path configs/examples/train_mixture_config.json

    # Low-cost dry run: 1 episode per dataset, do not update parquet
    python src/opentau/scripts/annotate_subtasks.py \\
        --config-path configs/examples/train_mixture_config.json \\
        --max-episodes-per-dataset 1 \\
        --no-write-response-column

    # Override download cache location
    python src/opentau/scripts/annotate_subtasks.py \\
        --config-path configs/examples/train_mixture_config.json \\
        --hub-cache-dir /data/hf_cache
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

import anthropic
import av
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from opentau.datasets.utils import DEFAULT_CHUNK_SIZE, load_episodes, load_info, write_info
from opentau.scripts.add_subtask_response import _build_response_array, _get_parquet_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a robot manipulation expert. "
    "You will be shown a sequence of video frames sampled at approximately 1 frame per second, "
    "each labelled with its timestamp in seconds. "
    "Your job is to identify every distinct phase of the robot's task and return ONLY a JSON "
    "array — no markdown fences, no prose, no explanation."
)

_USER_TEMPLATE = """\
Task: {task}

The frames below are from a single robot episode, sampled at roughly 1 fps. \
Identify each phase where the robot's action meaningfully changes.

Rules:
- First entry must be {{"time": 0.0, "subtask": "..."}}.
- Only add a new entry when there is a clear change in the robot's behaviour.
- Use timestamps taken directly from the list above (do not interpolate).
- Keep descriptions concise (5–15 words).
- Typical manipulation episodes have 2–6 phases.

Return ONLY a valid JSON array. Example:
[
  {{"time": 0.0, "subtask": "reaching toward the red cup"}},
  {{"time": 3.0, "subtask": "grasping the cup"}},
  {{"time": 5.0, "subtask": "lifting cup and placing on tray"}}
]"""


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------


def _resize(img: Image.Image, target_width: int) -> Image.Image:
    w, h = img.size
    if w == target_width:
        return img
    new_h = max(1, int(h * target_width / w))
    return img.resize((target_width, new_h), Image.LANCZOS)


def _to_b64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _extract_sampled_frames(
    video_path: Path,
    sample_fps: float,
    max_frames: int,
    target_width: int,
) -> tuple[list[Image.Image], list[float]]:
    """Decode the video and return (images, timestamps) at ≤ sample_fps.

    Decodes sequentially (no seeking) to avoid keyframe alignment issues.
    The stride is increased if the episode would produce more than max_frames
    samples, so at most max_frames images are ever returned.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        video_fps = float(stream.average_rate)
        total_frames = stream.frames or 0

        if total_frames == 0 and stream.duration and stream.time_base:
            total_frames = int(float(stream.duration * stream.time_base) * video_fps)

        n_at_rate = max(1, int(total_frames / max(video_fps, 1.0) * sample_fps))
        if n_at_rate > max_frames:
            stride = max(1, round(total_frames / max_frames))
        else:
            stride = max(1, round(video_fps / sample_fps))

        sampled_imgs: list[Image.Image] = []
        sampled_ts: list[float] = []

        for i, frame in enumerate(container.decode(video=0)):
            if i % stride == 0:
                ts = float(frame.pts * stream.time_base) if frame.pts is not None else i / max(video_fps, 1.0)
                img = _resize(frame.to_image(), target_width)
                sampled_imgs.append(img)
                sampled_ts.append(round(ts, 3))
                if len(sampled_imgs) >= max_frames:
                    break

    return sampled_imgs, sampled_ts


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array from the model response, tolerating markdown fences."""
    text = text.strip()
    # Strip ``` ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text.strip())
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed)}")
    return parsed


def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    task_description: str,
    frames: list[Image.Image],
    timestamps: list[float],
) -> list[dict]:
    """Send sampled frames to Claude and parse the subtask boundary JSON."""
    content: list[dict] = []

    # Interleave timestamp label → image for each sampled frame
    for ts, img in zip(timestamps, frames, strict=False):
        content.append({"type": "text", "text": f"t={ts:.1f}s:"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _to_b64_jpeg(img),
                },
            }
        )

    content.append({"type": "text", "text": _USER_TEMPLATE.format(task=task_description)})

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = response.content[0].text
    subtasks = _parse_json_response(raw_text)

    # Ensure first entry is at time 0.0
    if not subtasks or subtasks[0].get("time", -1) != 0.0:
        subtasks.insert(0, {"time": 0.0, "subtask": subtasks[0]["subtask"] if subtasks else task_description})
    return subtasks


# ---------------------------------------------------------------------------
# Per-episode annotation
# ---------------------------------------------------------------------------


def _find_video_key(info: dict) -> str | None:
    """Return the first feature key with dtype=='video'."""
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video":
            return key
    return None


def _subtask_path(root: Path, template: str, ep_index: int) -> Path:
    return root / template.format(episode_index=ep_index)


def _annotate_episode(
    client: anthropic.Anthropic,
    model: str,
    root: Path,
    info: dict,
    ep_index: int,
    ep_info: dict,
    subtask_tmpl: str,
    sample_fps: float,
    max_frames: int,
    target_width: int,
) -> bool:
    """Annotate one episode.  Returns True if the episode was processed."""
    out_path = _subtask_path(root, subtask_tmpl, ep_index)
    if out_path.exists():
        logger.debug("Episode %d: subtask file exists, skipping.", ep_index)
        return False

    # Locate the video file
    video_key = _find_video_key(info)
    if video_key is None:
        logger.warning("Episode %d: no video feature found in info.json; skipping.", ep_index)
        return False

    video_tmpl: str = info["video_path"]
    chunks_size: int = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    ep_chunk = ep_index // chunks_size
    video_path = root / video_tmpl.format(episode_chunk=ep_chunk, video_key=video_key, episode_index=ep_index)

    if not video_path.is_file():
        logger.warning("Episode %d: video file not found at %s; skipping.", ep_index, video_path)
        return False

    # Get the task description (use first task associated with this episode)
    tasks = ep_info.get("tasks", [])
    task_description = tasks[0] if tasks else "robot manipulation task"

    frames, timestamps = _extract_sampled_frames(video_path, sample_fps, max_frames, target_width)
    if not frames:
        logger.warning("Episode %d: no frames extracted from %s; skipping.", ep_index, video_path)
        return False

    subtasks = _call_claude(client, model, task_description, frames, timestamps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(subtasks, f, indent=2)

    logger.debug("Episode %d: wrote %d subtask entries to %s.", ep_index, len(subtasks), out_path)
    return True


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _update_parquet_response(root: Path, info: dict, ep_index: int, subtask_tmpl: str) -> None:
    """Expand subtask boundaries into a ``response`` column in the episode parquet."""
    fps: int = info["fps"]
    data_tmpl: str = info["data_path"]
    chunks_size: int = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    subtask_file = _subtask_path(root, subtask_tmpl, ep_index)
    if not subtask_file.exists():
        logger.warning("Episode %d: subtask file missing, skipping parquet update.", ep_index)
        return

    with open(subtask_file) as f:
        subtasks = json.load(f)

    parquet_path = _get_parquet_path(root, data_tmpl, ep_index, chunks_size)
    if not parquet_path.is_file():
        logger.warning("Episode %d: parquet not found at %s; skipping.", ep_index, parquet_path)
        return

    table = pq.read_table(parquet_path)
    ep_length = table.num_rows
    responses = _build_response_array(subtasks, fps, ep_length, ep_index)
    response_col = pa.array(responses, type=pa.string())

    if "response" in table.column_names:
        col_idx = table.schema.get_field_index("response")
        table = table.set_column(col_idx, "response", response_col)
    else:
        table = table.append_column("response", response_col)

    tmp = parquet_path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, parquet_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _process_dataset(
    client: anthropic.Anthropic,
    root: Path,
    model: str,
    sample_fps: float,
    max_frames: int,
    target_width: int,
    subtask_tmpl: str,
    write_response_column: bool,
    max_episodes: int | None,
) -> None:
    root = root.resolve()
    if not root.is_dir():
        logger.warning("Dataset root does not exist: %s — skipping.", root)
        return

    info = load_info(root)
    episodes = load_episodes(root)
    if not episodes:
        logger.warning("No episodes in %s — skipping.", root)
        return

    # Register the subtask_path in info.json if not already present
    if "subtask_path" not in info:
        info["subtask_path"] = subtask_tmpl
        write_info(info, root)
        logger.info("Set subtask_path='%s' in %s/meta/info.json", subtask_tmpl, root.name)

    ep_indices = sorted(episodes.keys())
    if max_episodes is not None:
        ep_indices = ep_indices[:max_episodes]

    n_annotated = 0
    n_skipped = 0

    for ep_index in tqdm(ep_indices, desc=root.name, unit="ep"):
        ep_info = episodes[ep_index]
        try:
            processed = _annotate_episode(
                client=client,
                model=model,
                root=root,
                info=info,
                ep_index=ep_index,
                ep_info=ep_info,
                subtask_tmpl=subtask_tmpl,
                sample_fps=sample_fps,
                max_frames=max_frames,
                target_width=target_width,
            )
        except Exception:
            logger.exception("Episode %d: annotation failed.", ep_index)
            n_skipped += 1
            continue

        if processed:
            n_annotated += 1
        else:
            n_skipped += 1

    logger.info("%s: annotated %d episodes, skipped %d.", root.name, n_annotated, n_skipped)

    if write_response_column:
        # Update info.json features if response column is new
        if "response" not in info.get("features", {}):
            info.setdefault("features", {})["response"] = {
                "dtype": "string",
                "shape": (1,),
                "names": None,
            }
            write_info(info, root)
            logger.info("Added 'response' feature to %s/meta/info.json", root.name)

        for ep_index in tqdm(ep_indices, desc=f"{root.name} (parquet)", unit="ep"):
            try:
                _update_parquet_response(root, info, ep_index, subtask_tmpl)
            except Exception:
                logger.exception("Episode %d: parquet update failed.", ep_index)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_datasets_from_config(config_path: Path) -> list[dict]:
    """Parse mixture config JSON, tolerating both top-level formats.

    Supports:
    - ``{"dataset_mixture": {"datasets": [...]}}``  (train_mixture_snippet.json style)
    - ``{"datasets": [...]}``                        (add_subtask_response.json style)
    """
    with open(config_path) as f:
        raw = json.load(f)

    if "dataset_mixture" in raw:
        datasets = raw["dataset_mixture"].get("datasets", [])
    elif "datasets" in raw:
        datasets = raw["datasets"]
    else:
        raise ValueError(
            f"Config {config_path} must contain either a top-level 'datasets' key or "
            "'dataset_mixture.datasets'."
        )
    return datasets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to the dataset mixture config JSON.",
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Frames per second to sample from each episode video (default: 1.0).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Maximum sampled frames to send per episode (controls API cost). "
        "If sampling at --sample-fps would exceed this, the rate is reduced "
        "proportionally (default: 40).",
    )
    p.add_argument(
        "--target-width",
        type=int,
        default=640,
        help="Resize each frame to this width (px) before encoding as JPEG (default: 640).",
    )
    p.add_argument(
        "--subtask-path-template",
        default="subtasks/episode_{episode_index:06d}.json",
        help="Template for per-episode subtask JSON paths, relative to the dataset root "
        "(default: subtasks/episode_{episode_index:06d}.json).",
    )
    p.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Anthropic model ID to use (default: claude-opus-4-7).",
    )
    p.add_argument(
        "--write-response-column",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After annotating, expand subtask boundaries into a 'response' column in "
        "each episode parquet (default: enabled).",
    )
    p.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Process at most this many episodes per dataset (useful for dry runs).",
    )
    p.add_argument(
        "--hub-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "opentau_subtasks",
        help="Directory for caching Hub dataset downloads (default: ~/.cache/huggingface/opentau_subtasks).",
    )
    return p.parse_args(argv)


def _resolve_root(ds_cfg: dict, hub_cache_dir: Path) -> Path:
    """Return the local root for a dataset, downloading from the Hub if needed."""
    if ds_cfg.get("root"):
        return Path(ds_cfg["root"])

    repo_id: str | None = ds_cfg.get("repo_id")
    if not repo_id:
        raise ValueError(f"Dataset config has neither 'root' nor 'repo_id': {ds_cfg}")

    revision: str | None = ds_cfg.get("revision")
    local_dir = hub_cache_dir / repo_id.replace("/", "--")
    logger.info(
        "Hub dataset '%s': downloading to %s (revision=%s) …",
        repo_id,
        local_dir,
        revision or "latest",
    )
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = _parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set in the environment.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    datasets = _load_datasets_from_config(args.config_path)

    if not datasets:
        logger.error("No datasets found in %s.", args.config_path)
        sys.exit(1)

    for ds_cfg in datasets:
        label = ds_cfg.get("repo_id") or ds_cfg.get("root", "<unknown>")
        try:
            root = _resolve_root(ds_cfg, args.hub_cache_dir)
        except Exception:
            logger.exception("Could not resolve root for dataset '%s'; skipping.", label)
            continue

        logger.info("Processing dataset: %s at %s", label, root)
        _process_dataset(
            client=client,
            root=root,
            model=args.model,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            target_width=args.target_width,
            subtask_tmpl=args.subtask_path_template,
            write_response_column=args.write_response_column,
            max_episodes=args.max_episodes_per_dataset,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
