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
"""Annotate each episode in a dataset mixture with a per-frame ``mistake`` column.

The subtask labels are read directly from the per-frame ``response`` column
already present in each episode parquet (written by ``annotate_subtasks.py``).
Each contiguous run of identical ``response`` values is treated as a single
subtask segment. For every segment:

1. The last frame of the segment is extracted from the dataset's ``camera0``
   video (resolved with the same lookup chain as ``annotate_subtasks.py``:
   inline ``data_features_name_mapping``, then
   ``DATA_FEATURES_NAME_MAPPING``, then the first ``dtype=='video'`` feature).
2. The frame is downsampled to ``--target-size`` × ``--target-size``
   (default 448) only when its shorter side already exceeds the target,
   then JPEG-encoded.
3. The frame and the segment's subtask string are sent to the configured
   VLM, which is asked to return ``{"success": bool, "reason": str}``.
4. If the VLM reports failure, every parquet row in the segment is set to
   ``mistake=1``; on success (or any parse / API failure) the rows are
   left at ``mistake=0``.

Episodes whose parquet already contains a ``mistake`` column are skipped
(making the script fully resumable). Episodes whose parquet has no
``response`` column are skipped with a warning. The ``mistake`` feature is
registered in ``meta/info.json`` as ``{"dtype": "int64", "shape": (1,),
"names": None}`` the first time it is added to a dataset.

Defaults to Google ``gemini-robotics-er-1.6-preview`` (the same Gemini ER
model used in ``annotate_subtasks.py``); Anthropic Claude is also supported
via ``--model``.

The config file is read with plain JSON; it accepts both the full training
config format (``{"dataset_mixture": {"datasets": [...]}}``) and the simpler
``add_subtask_response`` format (``{"datasets": [...]}``).

Example::

    python src/opentau/scripts/annotate_mistakes.py \\
        --config-path configs/examples/annotate_mistakes_example.json

    # Dry run: 1 episode per dataset
    python src/opentau/scripts/annotate_mistakes.py \\
        --config-path configs/examples/annotate_mistakes_example.json \\
        --max-episodes-per-dataset 1

A minimal mixture config is checked in at
``configs/examples/annotate_mistakes_example.json`` (Hub ``revision`` pinned
to ``v2.1`` since this script has only been tested against v2.1 datasets).
"""

from __future__ import annotations

import argparse
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
from google import genai
from google.genai import types as genai_types
from PIL import Image
from tqdm import tqdm

from opentau.datasets.utils import DEFAULT_CHUNK_SIZE, load_episodes, load_info, write_info
from opentau.scripts.add_subtask_response import _get_parquet_path
from opentau.scripts.annotate_subtasks import (
    _is_gemini_model,
    _load_datasets_from_config,
    _resize_and_center_crop,
    _resolve_camera0_video_key,
    _resolve_root,
    _to_b64_jpeg,
    _to_jpeg_bytes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a robot manipulation expert. "
    "You will be shown the final frame of a robot attempting a single subtask. "
    "Decide whether the robot completed the subtask successfully at that frame. "
    "Return ONLY a JSON object — no markdown fences, no prose."
)

_USER_TEMPLATE = """\
Task: {task}
Subtask: {subtask}

The image is the final frame of the robot's attempt at the subtask above. \
Was the subtask completed successfully?

Return ONLY a valid JSON object with this exact shape:
{{"success": true, "reason": "..."}}
or
{{"success": false, "reason": "..."}}"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_success_response(text: str) -> bool:
    """Parse a ``{"success": bool, ...}`` JSON object from the model response.

    Tolerates ``` ... ``` fences. Raises ``ValueError`` / ``json.JSONDecodeError``
    on any malformed payload — including a non-bool ``success`` value (e.g. the
    string ``"false"``, which would otherwise coerce to ``True``). The caller
    treats those as a mistake=0 default, so failing closed here is safer than
    flipping the verdict.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict) or "success" not in parsed:
        raise ValueError(f"Expected JSON object with 'success' key, got: {parsed!r}")
    success = parsed["success"]
    if not isinstance(success, bool):
        raise ValueError(f"Expected boolean 'success' value, got {type(success).__name__}: {success!r}")
    return success


# ---------------------------------------------------------------------------
# VLM calls (single image)
# ---------------------------------------------------------------------------


def _call_claude_single(
    client: anthropic.Anthropic,
    model: str,
    task: str,
    subtask: str,
    frame: Image.Image,
) -> str:
    # max_tokens is required by the Anthropic Messages API; the response is a
    # one-line ``{"success": bool, "reason": str}`` JSON object, so 1024 is a
    # comfortable model-agnostic cap that fits inside the smallest Anthropic
    # output ceiling (claude-3.5 family, 8192) without underutilizing larger
    # models.
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _to_b64_jpeg(frame),
                        },
                    },
                    {"type": "text", "text": _USER_TEMPLATE.format(task=task, subtask=subtask)},
                ],
            }
        ],
    )
    text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise ValueError(f"Claude response had no text blocks (stop_reason={response.stop_reason}).")
    return text_blocks[0].text


def _call_gemini_single(
    client: genai.Client,
    model: str,
    task: str,
    subtask: str,
    frame: Image.Image,
) -> str:
    # Gemini ER thinking is left at the model default. The earlier
    # ``thinking_budget=0`` workaround (introduced as a fix for output
    # truncation) was retired once we confirmed empirically that the default
    # output budget comfortably fits both internal reasoning and the
    # one-line JSON verdict for the success/failure prompt.
    response = client.models.generate_content(
        model=model,
        contents=[
            genai_types.Part.from_bytes(data=_to_jpeg_bytes(frame), mime_type="image/jpeg"),
            _USER_TEMPLATE.format(task=task, subtask=subtask),
        ],
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
    )
    raw_text = (response.text or "").strip()
    if not raw_text:
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        raise ValueError(f"Gemini response had no text (finish_reason={finish_reason}).")
    return raw_text


def _query_subtask_success(
    client: anthropic.Anthropic | genai.Client,
    model: str,
    task: str,
    subtask: str,
    frame: Image.Image,
) -> tuple[bool, bool]:
    """Ask the VLM whether the subtask was completed.

    Returns ``(success, ok)`` where ``success`` is the verdict (``True`` →
    ``mistake=0``, ``False`` → ``mistake=1``) and ``ok`` is ``False`` if the
    API call or response parse failed. Failures default to ``success=True``
    (no mistake) but the caller is expected to count ``not ok`` separately so
    a quiet API outage doesn't masquerade as "no mistakes found".
    """
    try:
        if _is_gemini_model(model):
            assert isinstance(client, genai.Client)
            raw = _call_gemini_single(client, model, task, subtask, frame)
        else:
            assert isinstance(client, anthropic.Anthropic)
            raw = _call_claude_single(client, model, task, subtask, frame)
        return _parse_success_response(raw), True
    except Exception as exc:
        logger.warning(
            "VLM query failed for subtask %r (%s); defaulting to success (mistake=0).",
            subtask,
            exc,
        )
        return True, False


# ---------------------------------------------------------------------------
# Subtask runs from the parquet response column
# ---------------------------------------------------------------------------


def _find_response_runs(responses: list[str | None]) -> list[tuple[int, int, str]]:
    """Return ``(start_idx, last_idx, subtask_string)`` for every contiguous non-empty run."""
    runs: list[tuple[int, int, str]] = []
    if not responses:
        return runs
    start = 0
    for i in range(1, len(responses)):
        if responses[i] != responses[start]:
            if responses[start]:
                runs.append((start, i - 1, str(responses[start])))
            start = i
    if responses[start]:
        runs.append((start, len(responses) - 1, str(responses[start])))
    return runs


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------


def _extract_frames_at_indices(
    video_path: Path,
    indices: list[int],
    target_size: int,
) -> dict[int, Image.Image]:
    """Decode the video once, returning ``{index: PIL.Image}`` for the requested indices.

    Indices are 0-based decode order, matching parquet row order for LeRobot v2.1
    datasets. Frames whose shorter side exceeds ``target_size`` are downsampled
    and center-cropped; smaller frames pass through unchanged (no upsampling).
    """
    if not indices:
        return {}
    wanted = set(indices)
    out: dict[int, Image.Image] = {}
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = _resize_and_center_crop(frame.to_image(), target_size)
                if len(out) == len(wanted):
                    break
    return out


# ---------------------------------------------------------------------------
# Per-episode annotation
# ---------------------------------------------------------------------------


def _annotate_episode(
    client: anthropic.Anthropic | genai.Client,
    model: str,
    root: Path,
    info: dict,
    video_key: str,
    ep_index: int,
    ep_info: dict,
    target_size: int,
) -> tuple[bool, int, int]:
    """Annotate one episode.

    Returns ``(processed, n_api_failures, n_missing_frames)``. ``processed`` is
    ``True`` iff the parquet was rewritten. ``n_api_failures`` counts segments
    where the VLM call/parse failed (defaulted to ``mistake=0``).
    ``n_missing_frames`` counts segments whose last frame could not be decoded
    from the video (also defaulted to ``mistake=0``).
    """
    data_tmpl: str = info["data_path"]
    chunks_size: int = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    parquet_path = _get_parquet_path(root, data_tmpl, ep_index, chunks_size)

    if not parquet_path.is_file():
        logger.warning("Episode %d: parquet not found at %s; skipping.", ep_index, parquet_path)
        return False, 0, 0

    schema_names = pq.read_metadata(parquet_path).schema.names
    if "mistake" in schema_names:
        logger.debug("Episode %d: 'mistake' column already present, skipping.", ep_index)
        return False, 0, 0
    if "response" not in schema_names:
        logger.warning(
            "Episode %d: parquet has no 'response' column (run annotate_subtasks.py first); skipping.",
            ep_index,
        )
        return False, 0, 0

    table = pq.read_table(parquet_path)
    responses = table.column("response").to_pylist()
    runs = _find_response_runs(responses)
    if not runs:
        logger.warning(
            "Episode %d: 'response' column has no non-empty subtask labels; skipping.",
            ep_index,
        )
        return False, 0, 0

    video_tmpl: str = info["video_path"]
    ep_chunk = ep_index // chunks_size
    video_path = root / video_tmpl.format(episode_chunk=ep_chunk, video_key=video_key, episode_index=ep_index)
    if not video_path.is_file():
        logger.warning("Episode %d: video file not found at %s; skipping.", ep_index, video_path)
        return False, 0, 0

    last_indices = [last_idx for _, last_idx, _ in runs]
    frames = _extract_frames_at_indices(video_path, last_indices, target_size)
    missing = [i for i in last_indices if i not in frames]
    n_missing_frames = len(missing)
    if missing:
        logger.warning(
            "Episode %d: could not extract frames at indices %s; those segments default to mistake=0.",
            ep_index,
            missing,
        )

    tasks = ep_info.get("tasks", [])
    task_description = tasks[0] if tasks else "robot manipulation task"

    mistakes = [0] * len(responses)
    n_failures = 0
    n_api_failures = 0
    for start_idx, last_idx, subtask in runs:
        frame = frames.get(last_idx)
        if frame is None:
            continue
        success, ok = _query_subtask_success(client, model, task_description, subtask, frame)
        if not ok:
            n_api_failures += 1
        if not success:
            n_failures += 1
            for j in range(start_idx, last_idx + 1):
                mistakes[j] = 1

    mistake_col = pa.array(mistakes, type=pa.int64())
    table = table.append_column("mistake", mistake_col)

    tmp = parquet_path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, parquet_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    logger.debug(
        "Episode %d: %d/%d subtask segments flagged as failures (%d/%d rows mistake=1).",
        ep_index,
        n_failures,
        len(runs),
        sum(mistakes),
        len(mistakes),
    )
    return True, n_api_failures, n_missing_frames


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _process_dataset(
    client: anthropic.Anthropic | genai.Client,
    root: Path,
    ds_cfg: dict,
    model: str,
    target_size: int,
    max_episodes: int | None,
) -> None:
    root = root.resolve()
    if not root.is_dir():
        logger.warning("Dataset root does not exist: %s — skipping.", root)
        return

    info = load_info(root)
    codebase_version = info.get("codebase_version")
    if codebase_version != "v2.1":
        logger.warning(
            "Dataset %s has codebase_version=%r; this script has only been tested against v2.1. "
            "For Hub datasets upgraded past v2.1, pin to the v2.1 tag via the 'revision' field.",
            root.name,
            codebase_version,
        )
    episodes = load_episodes(root)
    if not episodes:
        logger.warning("No episodes in %s — skipping.", root)
        return

    video_key = _resolve_camera0_video_key(info, ds_cfg)
    if video_key is None:
        logger.warning("Dataset %s: no usable camera0 video feature; skipping.", root.name)
        return

    mistake_feature_registered = "mistake" in info.get("features", {})

    ep_indices = sorted(episodes.keys())
    if max_episodes is not None:
        ep_indices = ep_indices[:max_episodes]

    n_annotated = 0
    n_skipped = 0
    n_api_failures_total = 0
    n_missing_frames_total = 0
    for ep_index in tqdm(ep_indices, desc=root.name, unit="ep"):
        ep_info = episodes[ep_index]
        try:
            processed, n_api_failures, n_missing_frames = _annotate_episode(
                client=client,
                model=model,
                root=root,
                info=info,
                video_key=video_key,
                ep_index=ep_index,
                ep_info=ep_info,
                target_size=target_size,
            )
        except Exception:
            logger.exception("Episode %d: annotation failed.", ep_index)
            n_skipped += 1
            continue

        if processed:
            n_annotated += 1
            n_api_failures_total += n_api_failures
            n_missing_frames_total += n_missing_frames
            # Defer registering the 'mistake' feature in info.json until after
            # the first parquet has actually been rewritten — otherwise a
            # mid-dataset crash leaves info.json advertising a column that
            # exists in only some parquets.
            if not mistake_feature_registered:
                info.setdefault("features", {})["mistake"] = {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": None,
                }
                write_info(info, root)
                logger.info("Added 'mistake' feature to %s/meta/info.json", root.name)
                mistake_feature_registered = True
        else:
            n_skipped += 1

    logger.info(
        "%s: annotated %d episodes, skipped %d, %d VLM call failures, %d missing frames "
        "(both default to mistake=0).",
        root.name,
        n_annotated,
        n_skipped,
        n_api_failures_total,
        n_missing_frames_total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Annotate every episode in a dataset mixture with a per-frame 'mistake' "
            "column using a VLM. Subtask boundaries are read from the per-frame "
            "'response' column written by annotate_subtasks.py; the last frame of "
            "each contiguous run is sent to the VLM to judge subtask success."
        ),
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to the dataset mixture config JSON.",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=448,
        help="Downsample each frame whose shorter side exceeds this many pixels so its "
        "shorter side equals target_size, then center-crop to a target_size × target_size "
        "square before encoding as JPEG. Frames already at or below target_size pass "
        "through unchanged — this only downsamples, it never upsamples (default: 448).",
    )
    p.add_argument(
        "--model",
        default="gemini-robotics-er-1.6-preview",
        help=(
            "Model ID to use. Defaults to 'gemini-robotics-er-1.6-preview' (Google), "
            "matching annotate_subtasks.py's Gemini support. Anthropic models "
            "(e.g. 'claude-opus-4-7') route through ANTHROPIC_API_KEY; model IDs "
            "starting with 'gemini' or 'robotics-er' route through GEMINI_API_KEY "
            "via google-genai."
        ),
    )
    p.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Process at most this many episodes per dataset (useful for dry runs).",
    )
    p.add_argument(
        "--max-api-retries",
        type=int,
        default=8,
        help=(
            "Number of automatic retries for the Anthropic SDK on 429/5xx responses; "
            "the SDK applies exponential backoff between attempts (default: 8). "
            "Ignored when using a Gemini model — google-genai's retry policy is "
            "configured via its own client options."
        ),
    )
    p.add_argument(
        "--hub-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "opentau_subtasks",
        help=(
            "Directory for caching Hub dataset downloads (default: "
            "~/.cache/huggingface/opentau_subtasks). The default deliberately "
            "matches annotate_subtasks.py so this script reuses datasets "
            "already downloaded by the prior step rather than re-downloading "
            "them — pass the same value here as you used there if you "
            "overrode it."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = _parse_args(argv)

    client: anthropic.Anthropic | genai.Client
    if _is_gemini_model(args.model):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in the environment.")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY is not set in the environment.")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key, max_retries=args.max_api_retries)

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
            ds_cfg=ds_cfg,
            model=args.model,
            target_size=args.target_size,
            max_episodes=args.max_episodes_per_dataset,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
