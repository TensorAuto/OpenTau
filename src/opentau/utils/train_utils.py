#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities for training checkpoint management and state persistence.

This module provides functions for saving and loading training checkpoints,
managing checkpoint directories, and pruning old checkpoints.
"""

import logging
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from opentau.configs.train import TrainPipelineConfig
from opentau.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    RUNNING_BEST_STATE,
    TRAINING_STEP,
)
from opentau.datasets.utils import load_json, write_json
from opentau.utils.random_utils import load_rng_state, save_rng_state

if TYPE_CHECKING:
    import accelerate


def log_output_dir(out_dir):
    """Log the output directory path with colored formatting.

    Args:
        out_dir: Output directory path to log.
    """
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def get_step_identifier(step: int, total_steps: int) -> str:
    """Generate a zero-padded step identifier string.

    Args:
        step: Current step number.
        total_steps: Total number of steps (used to determine padding width).

    Returns:
        Zero-padded string representation of the step number.
    """
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    """Save the current training step number to a file.

    Args:
        step: Current training step number.
        save_dir: Directory where the step file will be saved.
    """
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    """Load the training step number from a file.

    Args:
        save_dir: Directory containing the step file.

    Returns:
        Training step number.
    """
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Update the symlink pointing to the last checkpoint.

    Args:
        checkpoint_dir: Path to the checkpoint directory to link.

    Returns:
        Path to the symlink that was created or updated.
    """
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
) -> None:
    """Save training checkpoint including config and RNG state.

    Note: accelerate saves the model and training run. This method saves all
    other auxiliary objects such as configs and RNG state.

    Args:
        checkpoint_dir: Directory where the checkpoint will be saved.
        step: Current training step number.
        cfg: The training config used for this run.
    """
    cfg.save_pretrained(checkpoint_dir)
    save_training_step(step, checkpoint_dir)
    save_rng_state(checkpoint_dir)


def find_missing_rng_state_ranks(checkpoint_dir: Path, world_size: int) -> list[int]:
    """Return the process indices whose ``random_states_<i>.pkl`` is absent from a checkpoint.

    Accelerate writes exactly one ``random_states_<process_index>.pkl`` per process on
    every backend, so a complete checkpoint contains ``world_size`` of them. A non-empty
    result means some rank saved into a different directory or not at all (e.g. a divergent
    ``output_dir`` across ranks), leaving the checkpoint unresumable.

    Args:
        checkpoint_dir: Directory accelerate saved the per-rank RNG files into.
        world_size: Expected number of processes (``accelerator.num_processes``).

    Returns:
        Sorted list of missing process indices; empty when the checkpoint is complete.
    """
    present: set[int] = set()
    for path in Path(checkpoint_dir).glob("random_states_*.pkl"):
        index = path.name[len("random_states_") : -len(".pkl")]
        if index.isdigit():
            present.add(int(index))
    return sorted(set(range(world_size)) - present)


def reseed_new_ranks_on_resume(
    checkpoint_dir: Path,
    accelerator: "accelerate.Accelerator",
    seed: int | None,
) -> None:
    """Re-seed ranks that have no per-rank RNG file in the checkpoint.

    Accelerate's ``load_state`` reads ``random_states_<process_index>.pkl`` per
    rank and silently logs ``"Could not load random states"`` at INFO level when
    the file is missing. When resuming on more GPUs than the checkpoint was
    saved with, the new ranks are left with whatever default RNG state the
    process happened to start with -- correlated across ranks and
    irreproducible. This helper detects that case and assigns each new rank a
    deterministic, decorrelated seed via ``set_seed``.

    Surviving ranks (whose files loaded successfully) are not touched.

    Args:
        checkpoint_dir: Directory holding the saved ``random_states_*.pkl`` files.
        accelerator: The current Accelerator (provides ``num_processes`` and
            ``process_index``).
        seed: Base training seed (``cfg.seed``). If ``None``, no re-seeding is
            performed and a warning is emitted instead.
    """
    from opentau.utils.random_utils import set_seed

    saved = len(list(checkpoint_dir.glob("random_states_*.pkl")))
    current = accelerator.num_processes
    if saved == current:
        return
    if saved == 0:
        if accelerator.is_main_process:
            logging.warning(
                "No random_states_*.pkl files found in %s; skipping per-rank RNG reseed. "
                "This may indicate a corrupt or partial checkpoint, or a checkpoint produced "
                "by a future Accelerate layout.",
                checkpoint_dir,
            )
        return
    if saved < current:
        if accelerator.is_main_process:
            logging.warning(
                "Resuming on more processes (%d) than the checkpoint was saved with (%d). "
                "Ranks %d..%d had no per-rank RNG file; re-seeding them deterministically "
                "from cfg.seed. Note: global batch size has changed by a factor of %.3fx; "
                "consider scaling cfg.gradient_accumulation_steps to compensate.",
                current,
                saved,
                saved,
                current - 1,
                current / saved,
            )
        if accelerator.process_index >= saved:
            if seed is None:
                logging.warning(
                    "cfg.seed is None; rank %d will retain its default startup RNG (not reproducible).",
                    accelerator.process_index,
                )
            else:
                set_seed(seed, accelerator=accelerator)
    else:
        if accelerator.is_main_process:
            logging.info(
                "Resuming on fewer processes (%d) than the checkpoint was saved with (%d). "
                "Per-rank RNG files for ranks %d..%d are ignored. Note: global batch size "
                "has changed by a factor of %.3fx; consider scaling "
                "cfg.gradient_accumulation_steps to compensate.",
                current,
                saved,
                current,
                saved - 1,
                current / saved,
            )


def load_training_state(checkpoint_dir: Path) -> tuple[int, Optimizer, LRScheduler | None]:
    """Load training state including step, optimizer, scheduler, and RNG state.

    This is used to resume a training run. Note: optimizer and scheduler states
    are loaded by accelerate, not by this function.

    Args:
        checkpoint_dir: The checkpoint directory. Should contain a 'training_state' dir.

    Returns:
        Tuple containing the training step number. Note: optimizer and scheduler
        are loaded separately by accelerate.

    Raises:
        NotADirectoryError: If checkpoint_dir doesn't exist or is not a directory.
    """
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)

    load_rng_state(checkpoint_dir)
    step = load_training_step(checkpoint_dir)

    return step


def prune_old_checkpoints(
    latest_checkpoint_path: str,
    protected_paths: set[Path] | None = None,
) -> None:
    """Delete all checkpoint directories except the specified one (and any protected ones).

    Recursively deletes all checkpoint directories in a parent folder except
    for the specified one. This function is designed to clean up old model
    checkpoints, preserving only the most recent one. It includes safety checks
    to ensure it only deletes directories and handles potential filesystem errors.

    Args:
        latest_checkpoint_path: The full path to the checkpoint directory
            that should be kept.
        protected_paths: Additional checkpoint directories that must NOT be deleted
            (e.g. running-best checkpoints). Compared by resolved path. Defaults to None.
    """
    try:
        latest_checkpoint = Path(latest_checkpoint_path).resolve()
        parent_dir = latest_checkpoint.parent
        protected_resolved = {Path(p).resolve() for p in protected_paths} if protected_paths else set()

        if not parent_dir.is_dir():
            logging.error(f"Parent directory '{parent_dir.resolve()}' does not exist. Aborting cleanup.")
            return

        if not latest_checkpoint.is_dir():
            logging.warning(
                f"Checkpoint '{latest_checkpoint.resolve()}' is not a valid directory. Aborting cleanup."
            )
            return

        logging.info(
            f"Starting cleanup in '{parent_dir.resolve()}'. Keeping checkpoint: '{latest_checkpoint.name}'"
        )

        # Iterate and delete other directories
        for item in parent_dir.iterdir():
            # Skip the checkpoint we want to keep, any protected (e.g. running-best) dirs, and files
            item_resolved = item.resolve()
            if item_resolved == latest_checkpoint.resolve() or item_resolved in protected_resolved:
                continue
            if not item.is_dir():
                continue

            try:
                logging.info(f"Deleting old checkpoint directory: {item.name}")
                shutil.rmtree(item)
                logging.info(f"Successfully deleted {item.name}")
            except OSError as e:
                logging.error(f"Failed to delete '{item.name}'. Error: {e}")

    except Exception as e:
        logging.critical(f"An unexpected error occurred during checkpoint pruning setup: {e}")


def running_best_is_improvement(score: float | None, best: float, higher_is_better: bool) -> bool:
    """Return whether ``score`` strictly improves on ``best``.

    Strict comparison (``>`` / ``<``): a value equal to the current best does not count as an
    improvement, so a plateaued metric does not thrash the running-best pool. ``None`` and ``NaN``
    scores never count as an improvement (e.g. an eval that produced no episodes reports NaN).

    Args:
        score: The new metric value (may be None or NaN).
        best: The current best metric value.
        higher_is_better: True to maximize (e.g. success rate), False to minimize (e.g. loss).

    Returns:
        True iff ``score`` is finite and strictly better than ``best``.
    """
    if score is None:
        return False
    try:
        score = float(score)
    except (TypeError, ValueError):
        return False
    if math.isnan(score):
        return False
    return score > best if higher_is_better else score < best


class RunningBestTracker:
    """Rank-0 bookkeeping for running-best checkpoints.

    Tracks the best metric value seen so far and the pool of step numbers whose checkpoints are
    currently retained as running bests (most-recent last). Every method is pure Python /
    filesystem-only -- no distributed collectives -- so it runs on the main process and is unit
    testable on CPU. The actual (collective) ``accelerator.save_state`` is performed by the caller.

    Each pool entry stores the step and whether a regular ``save_freq`` checkpoint was written at
    that step (captured at registration time). On eviction, a regular checkpoint's directory is
    left to the normal retention logic; only running-best-only directories are deleted.
    """

    def __init__(
        self,
        output_dir: Path,
        total_steps: int,
        higher_is_better: bool,
        max_count: int,
        best: float | None = None,
        steps: list[dict] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.total_steps = total_steps
        self.higher_is_better = higher_is_better
        self.max_count = max_count
        self.best = best if best is not None else (float("-inf") if higher_is_better else float("inf"))
        # Each entry: {"step": int, "is_regular": bool}, ordered most-recent last.
        self.steps: list[dict] = list(steps) if steps else []

    def step_dir(self, step: int) -> Path:
        """Return the checkpoint directory for a given step."""
        return get_step_checkpoint_dir(self.output_dir, self.total_steps, step)

    def is_improvement(self, score: float | None) -> bool:
        """Return whether ``score`` strictly improves on the current best (no mutation)."""
        return running_best_is_improvement(score, self.best, self.higher_is_better)

    def register(self, step: int, score: float, is_regular: bool) -> list[Path]:
        """Record a new best at ``step`` and return the checkpoint directories to delete.

        Updates the high-water mark, adds ``step`` to the pool (de-duplicating, moving it to
        most-recent), then evicts the oldest entries beyond ``max_count``. An evicted step's
        directory is scheduled for deletion only if it was not a regular ``save_freq`` checkpoint
        and the directory still exists.

        Args:
            step: The training step that achieved the new best.
            score: The (finite) metric value at this step; becomes the new high-water mark.
            is_regular: Whether a regular ``save_freq`` checkpoint was also written at this step.

        Returns:
            List of running-best-only checkpoint directories the caller should ``rmtree``.
        """
        self.best = float(score)
        # De-duplicate (defensive; steps are monotonic in practice) and append as most-recent.
        self.steps = [e for e in self.steps if e["step"] != step]
        self.steps.append({"step": step, "is_regular": bool(is_regular)})

        to_delete: list[Path] = []
        while len(self.steps) > self.max_count:
            evicted = self.steps.pop(0)
            if not evicted["is_regular"]:
                d = self.step_dir(evicted["step"])
                if d.exists():
                    to_delete.append(d)
        return to_delete

    def protected_dirs(self) -> set[Path]:
        """Return the set of checkpoint directories currently retained as running bests."""
        return {self.step_dir(e["step"]) for e in self.steps}

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (high-water mark + pool)."""
        return {
            "higher_is_better": self.higher_is_better,
            "best": self.best,
            "steps": self.steps,
        }


def save_running_best_state(tracker: RunningBestTracker, metric: str) -> None:
    """Persist the running-best state next to the checkpoint directories.

    Written as a plain file in ``<output_dir>/checkpoints/`` (not inside a step dir), so it
    survives ``prune_old_checkpoints`` (which only deletes directories).

    Args:
        tracker: The running-best tracker to persist (provides ``output_dir``).
        metric: The resolved driving metric name (stored for provenance/debugging).
    """
    checkpoints_dir = tracker.output_dir / CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    state = {"metric": metric, **tracker.to_dict()}
    write_json(state, checkpoints_dir / RUNNING_BEST_STATE)


def load_running_best_state(
    output_dir: Path,
    total_steps: int,
    higher_is_better: bool,
    max_count: int,
) -> RunningBestTracker:
    """Load a running-best tracker, self-healing against pruned directories.

    If the state file is absent (a fresh run, or a resume of a run that predates the feature),
    a fresh tracker is returned. Pool entries whose checkpoint directory no longer exists are
    dropped (a regular-coincident best may have been pruned later by the normal retention logic),
    but the high-water mark is preserved regardless.

    If the persisted optimization direction disagrees with ``higher_is_better`` (a resume that
    flipped ``running_best_metric`` between maximize and minimize), the persisted high-water mark
    is meaningless under the new direction, so it is discarded (reset to ``-inf``/``+inf``) while
    the existing pool is kept; a warning is emitted.

    Args:
        output_dir: The run output directory (its ``checkpoints/`` subdir holds the state file).
        total_steps: Total training steps (for step-dir zero-padding).
        higher_is_better: True to maximize the metric, False to minimize.
        max_count: Maximum number of running bests to keep.

    Returns:
        A ``RunningBestTracker`` restored from disk, or a fresh one if no state exists.
    """
    state_path = Path(output_dir) / CHECKPOINTS_DIR / RUNNING_BEST_STATE
    if not state_path.exists():
        return RunningBestTracker(output_dir, total_steps, higher_is_better, max_count)

    state = load_json(state_path)
    # A resume that flips the metric direction makes the persisted best uncomparable; drop it.
    best = state.get("best")
    if state.get("higher_is_better") is not None and state["higher_is_better"] != higher_is_better:
        logging.warning(
            "running_best.json was saved with higher_is_better=%s but the current run uses %s; "
            "discarding the persisted high-water mark (the pool is kept).",
            state["higher_is_better"],
            higher_is_better,
        )
        best = None
    tracker = RunningBestTracker(
        output_dir,
        total_steps,
        higher_is_better,
        max_count,
        best=best,
        steps=state.get("steps", []),
    )
    # Self-heal: drop pool entries whose checkpoint directory is gone.
    tracker.steps = [e for e in tracker.steps if tracker.step_dir(e["step"]).exists()]
    return tracker
