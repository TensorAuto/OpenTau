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

"""Fit an ``accel`` failure-detection threshold from an eval run's ``eval_info.json``.

The score itself is free (:mod:`opentau.policies.accel`); the *detector* is not, and this
script is where that cost is paid — offline, once, out of the serving path. It reads the
per-episode ``accel`` streams an eval run recorded, fits a one-sided CUSUM threshold by
split conformal prediction on the **successful** rollouts, and writes a calibration sidecar
that :mod:`opentau.utils.accel_detector` can then apply at runtime.

Only successful rollouts are used, which is the method's main practical appeal — no failure
labels are needed. The real cost is episode count: the split-conformal threshold is an order
statistic of per-episode CUSUM peaks, so ``alpha = 0.1`` needs at least 9 successes and the
paper's operating point uses 50. At a policy with 20-60% success that is 80-250 episodes per
task, several times a default ``eval.n_episodes = 16`` run.

Producing the input::

    OPENTAU_ACCEL_PREFIX=auto opentau-eval \\
        --accelerate-config configs/examples/accelerate_ddp_config.yaml \\
        --config_path=configs/examples/pi05_libero_eval_config.json \\
        --eval.n_episodes=60 --env.max_parallel_tasks=1

Fitting from it::

    python src/opentau/scripts/calibrate_accel.py \\
        outputs/eval/.../eval_info.json --alpha 0.1 --out outputs/accel

``--per-task`` fits one threshold per task instead of one pooled threshold. Prefer it when
you have the episodes: task difficulty shifts the score's location, and a threshold pooled
across easy and hard tasks under-detects on the hard ones. Run without it to see how many
successes each task actually has.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from opentau.policies.accel import AccelProvenance
from opentau.utils.accel_detector import (
    DEFAULT_ALPHA,
    DEFAULT_SLACK_C,
    calibrate,
    conformal_rank,
    evaluate,
)
from opentau.utils.utils import init_logging

logger = logging.getLogger(__name__)


def load_eval_info(path: Path) -> list[dict]:
    """Read the per-task records an eval run wrote.

    Args:
        path: An ``eval_info.json``, or a directory containing one.

    Returns:
        The ``per_task`` list, each entry carrying ``task_group``/``task_id``/``metrics``.

    Raises:
        FileNotFoundError: When no ``eval_info.json`` is at ``path``.
        ValueError: When the file has no ``per_task`` records.
    """
    if path.is_dir():
        path = path / "eval_info.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval_info.json at {path}.")
    payload = json.loads(path.read_text())
    per_task = payload.get("per_task")
    if not per_task:
        raise ValueError(f"{path} has no 'per_task' records; is it an eval output?")
    return per_task


def split_by_outcome(task: dict) -> tuple[list[list[float]], list[list[float]], list[bool]]:
    """Split one task's episodes into successful and failing ``accel`` streams.

    ``eval.py`` attaches ``accels`` index-aligned with ``successes``, so the pairing is
    positional. A task whose lengths disagree is dropped rather than paired up — a stream
    labelled with another episode's outcome would poison the calibration invisibly.

    Args:
        task: One ``per_task`` record.

    Returns:
        ``(successful_streams, failing_streams, successes)``; all empty when the task
        recorded no accel or the two lists disagree in length.
    """
    metrics = task.get("metrics", {})
    accels = metrics.get("accels") or []
    successes = metrics.get("successes") or []
    if not accels:
        return [], [], []
    if len(accels) != len(successes):
        logger.warning(
            "%s/%s: %d accel stream(s) for %d success flag(s); skipping this task rather than "
            "pairing a stream with the wrong episode.",
            task.get("task_group"),
            task.get("task_id"),
            len(accels),
            len(successes),
        )
        return [], [], []
    good = [stream for stream, ok in zip(accels, successes, strict=True) if ok and stream]
    bad = [stream for stream, ok in zip(accels, successes, strict=True) if not ok and stream]
    return good, bad, list(successes)


def _resolve_provenance(tasks: list[dict]) -> AccelProvenance | None:
    """Return the shared provenance across ``tasks``, warning when they disagree.

    A single calibration covering streams produced under different configurations would be
    meaningless, so a disagreement drops the provenance entirely rather than picking one —
    that leaves the calibration unlabelled, and an unlabelled calibration skips the
    comparability check at apply time instead of passing it on false pretences.
    """
    seen = {json.dumps(t["metrics"]["accel_provenance"], sort_keys=True) for t in tasks}
    if not seen:
        # No task recorded one. Leaving the calibration unlabelled is the honest outcome:
        # `detect` then skips the comparability check rather than passing it falsely.
        logger.warning("No task recorded an accel provenance; the calibration will be unlabelled.")
        return None
    if len(seen) > 1:
        logger.warning(
            "Tasks disagree on accel provenance (%d distinct); the fitted calibration will be "
            "left unlabelled. Fit per-task instead.",
            len(seen),
        )
        return None
    return AccelProvenance.from_dict(json.loads(seen.pop()))


def fit(
    tasks: list[dict],
    *,
    alpha: float,
    slack_c: float,
    label: str,
) -> dict | None:
    """Fit and score one calibration over a set of tasks.

    Args:
        tasks: ``per_task`` records to pool.
        alpha: Target false-alarm rate.
        slack_c: CUSUM slack in units of the calibration standard deviation.
        label: Human-readable name for logging and the output filename.

    Returns:
        A dict with the calibration and its held-in evaluation, or ``None`` when there were
        too few successful episodes to certify ``alpha``.
    """
    good: list[list[float]] = []
    streams: list[list[float]] = []
    successes: list[bool] = []
    for task in tasks:
        task_good, _, task_successes = split_by_outcome(task)
        if not task_successes:
            continue
        good.extend(task_good)
        streams.extend(task["metrics"]["accels"])
        successes.extend(task_successes)

    if not good:
        logger.warning("%s: no successful episodes with an accel stream; skipping.", label)
        return None

    try:
        minimum = conformal_rank(len(good), alpha)
    except ValueError as exc:
        logger.warning("%s: %s", label, exc)
        return None

    provenance = _resolve_provenance([t for t in tasks if t.get("metrics", {}).get("accel_provenance")])
    calibration = calibrate(good, alpha=alpha, slack_c=slack_c, provenance=provenance)
    metrics = evaluate(streams, successes, calibration)

    logger.info(
        "%s: fitted on %d successful episode(s) (order statistic %d), eta=%.6g; held-in "
        "TPR=%.3f false-alarm=%.3f median lead=%.1f chunks over %d failing / %d successful",
        label,
        len(good),
        minimum,
        calibration.eta,
        metrics["true_positive_rate"],
        metrics["false_alarm_rate"],
        metrics["median_lead"],
        int(metrics["n_failed"]),
        int(metrics["n_successful"]),
    )
    return {"calibration": calibration, "metrics": metrics, "num_successful": len(good)}


def main() -> None:
    """Entry point."""
    init_logging()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eval_info", type=Path, help="eval_info.json, or a directory containing one")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="target false-alarm rate")
    parser.add_argument("--slack-c", type=float, default=DEFAULT_SLACK_C, help="CUSUM slack, in sigmas")
    parser.add_argument("--out", type=Path, default=Path("outputs/accel"), help="output directory")
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="fit one threshold per task instead of one pooled threshold",
    )
    args = parser.parse_args()

    per_task = load_eval_info(args.eval_info)
    with_accel = [t for t in per_task if t.get("metrics", {}).get("accels")]
    if not with_accel:
        raise SystemExit(
            f"No task in {args.eval_info} recorded accel streams. Re-run the eval with "
            "OPENTAU_ACCEL_PREFIX=auto (and env.max_parallel_tasks=1)."
        )
    logger.info("%d of %d task(s) carry accel streams.", len(with_accel), len(per_task))

    args.out.mkdir(parents=True, exist_ok=True)
    groups: list[tuple[str, list[dict]]]
    if args.per_task:
        groups = [(f"{t['task_group']}__{t['task_id']}", [t]) for t in with_accel]
    else:
        groups = [("pooled", with_accel)]

    written = 0
    summary: dict[str, dict] = {}
    for label, tasks in groups:
        result = fit(tasks, alpha=args.alpha, slack_c=args.slack_c, label=label)
        if result is None:
            continue
        path = result["calibration"].save(args.out / f"accel_calibration__{label}.json")
        summary[label] = {
            "path": str(path),
            "num_successful": result["num_successful"],
            "eta": result["calibration"].eta,
            **result["metrics"],
        }
        written += 1

    if not written:
        minimum = math.ceil(1.0 / args.alpha) - 1
        raise SystemExit(
            f"No calibration could be fitted at alpha={args.alpha}: every group had fewer than "
            f"the {minimum} successful episodes a finite split-conformal threshold needs at that "
            "rate. Collect more successful rollouts, drop --per-task to pool them, or raise --alpha."
        )

    summary_path = args.out / "accel_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %d calibration(s) and %s", written, summary_path)
    logger.warning(
        "Held-in TPR/FPR above are measured on the SAME episodes the threshold was fitted "
        "from, so they are optimistic. The conformal guarantee is on the false-alarm rate "
        "out of sample; measure the true-positive rate on a disjoint set of rollouts."
    )


if __name__ == "__main__":
    main()
