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

"""Iterate over a JSON array of task segments and generate cumulative memory summaries.

Top-level JSON must be a **list of objects**. For each object the script builds a
cumulative subtask log (subtask names and ``success`` outcomes up to that point),
then calls OpenAI to produce a running plain-text memory summary. The summary is
written back to each object under the ``memory`` key (configurable via
``--output-key``).

Examples::

    export OPENAI_API_KEY=sk-...
    python -m opentau.scripts.pi_mem_data_generator task_segments.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


def _resolve_dotenv_path() -> Path | None:
    """First ``.env`` file found walking up from this script (repo root typically)."""
    script = Path(__file__).resolve()
    for d in script.parents:
        candidate = d / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_env_file() -> Path | None:
    """Load ``.env`` walking up from this script; ``override=True`` so ``.env`` beats shell.

    Returns the path to the loaded ``.env`` if found in the walk, else ``None``.
    """
    path = _resolve_dotenv_path()
    if path is not None:
        if load_dotenv is not None:
            load_dotenv(path, override=True)
        logger.debug("Loaded environment file %s", path)
        return path
    if load_dotenv is not None:
        load_dotenv(override=True)
    return None


def _normalize_openai_api_key() -> str | None:
    raw = os.environ.get("OPENAI_API_KEY")
    if raw is None:
        return None
    key = raw.strip().strip('"').strip("'")
    if not key:
        return None
    os.environ["OPENAI_API_KEY"] = key
    return key


SYSTEM_PROMPT = """\
You are the memory module of a robotic manipulation system. You receive a log \
of subtasks that have ALREADY been executed and must produce a compact \
plain-text summary.

Critical rules:
- ONLY mention actions that appear in the log below. If an action is not in \
the log, it has NOT happened — do NOT mention it, do NOT infer it, do NOT \
speculate about it. You have zero knowledge beyond the log entries provided.
- Write simple, plain sentences. No bullet points, no numbered lists, \
no labels, no markdown, no structured formatting.
- If the same action failed earlier but succeeded later in the log, just \
mention the success. Drop the resolved failure.
- If a failure is the last entry for that action in the log, mention it.
- Merge completed actions into short phrases where possible.
- Omit timestamps.
- Keep it under 50 words.\
"""

USER_PROMPT_TEMPLATE = """\
Previous memory: {prev_memory}

Task prompt: {task_prompt}

Here is the complete log of actions executed so far:
{subtask_log}

Write a plain-text summary covering ONLY the actions listed above. \
Do not mention or infer any action that is not in this log.\
"""


def _call_openai(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str | None,
    user_content: str,
) -> str:
    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    resp = client.chat.completions.create(model=model, messages=messages)
    choice = resp.choices[0].message
    text = choice.content
    if not text:
        raise RuntimeError("OpenAI returned empty message content")
    return text.strip()


def _build_subtask_log(data: list[dict[str, Any]], up_to: int) -> str:
    """Format subtasks 0..up_to (inclusive) as a numbered list for the prompt."""
    lines: list[str] = []
    for idx in range(up_to + 1):
        item = data[idx]
        subtask = item.get("subtask", "unknown")
        success = item.get("success")
        outcome = "SUCCESS" if success else "FAILED" if success is False else "UNKNOWN"
        t = item.get("time")
        time_str = f" (t={t}s)" if t is not None else ""
        lines.append(f"  {idx + 1}. [{outcome}]{time_str} {subtask}")
    return "\n".join(lines)


def _enrich_list(
    data: list[Any],
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    output_key: str,
    skip_existing: bool,
    delay_s: float,
) -> None:
    """For each item, build a cumulative subtask log and request a memory summary."""
    prev_memory = "(none)"
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} must be an object, got {type(item).__name__}")
        if skip_existing and output_key in item and item[output_key]:
            logger.info("Skipping index %d (existing %s)", i, output_key)
            prev_memory = item[output_key]
            continue

        subtask_log = _build_subtask_log(data, up_to=i)
        task_prompt = item.get("prompt", "(not provided)")
        user_content = USER_PROMPT_TEMPLATE.format(
            prev_memory=prev_memory,
            task_prompt=task_prompt,
            subtask_log=subtask_log,
        )
        logger.info("Calling API for item %d (subtask: %s)", i, item.get("subtask"))
        prev_memory = _call_openai(
            client, model=model, system_prompt=system_prompt, user_content=user_content
        )
        item[output_key] = prev_memory
        if delay_s > 0:
            time.sleep(delay_s)


def main(argv: list[str] | None = None) -> int:
    _load_env_file()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "json_path",
        type=Path,
        help="Path to JSON file to read and update in place",
    )
    p.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system message for the chat completion.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Chat model (default: gpt-4o-mini or OPENAI_MODEL env).",
    )
    p.add_argument(
        "--output-key",
        type=str,
        default="memory",
        help="Key for per-field replies: a dict mapping each source field name to model text.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not call API if output-key already set (non-empty).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (rate limiting).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    api_key = _normalize_openai_api_key()
    if not api_key:
        logger.error("Missing OPENAI_API_KEY. Put it in repo .env or export it.")
        return 1

    path = args.json_path.resolve()
    if not path.is_file():
        logger.error("Not a file: %s", path)
        return 1

    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON: %s", e)
        return 1

    client = OpenAI(api_key=api_key)
    if args.verbose:
        tail = api_key[-4:] if len(api_key) >= 4 else "****"
        logger.debug("OpenAI client using API key ending in …%s (length %d)", tail, len(api_key))

    try:
        if not isinstance(data, list):
            logger.error("Top-level JSON must be a list of objects")
            return 1
        system = args.system_prompt if args.system_prompt else SYSTEM_PROMPT
        _enrich_list(
            data,
            client=client,
            model=args.model,
            system_prompt=system,
            output_key=args.output_key,
            skip_existing=args.skip_existing,
            delay_s=args.delay,
        )
    except Exception as e:
        logger.exception("OpenAI or processing failed: %s", e)
        return 1

    # Atomic os.replace (on Unix) to avoid partial JSON updates.
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
