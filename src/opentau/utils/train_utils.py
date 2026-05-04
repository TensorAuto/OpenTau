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


def prune_old_checkpoints(latest_checkpoint_path: str) -> None:
    """Delete all checkpoint directories except the specified one.

    Recursively deletes all checkpoint directories in a parent folder except
    for the specified one. This function is designed to clean up old model
    checkpoints, preserving only the most recent one. It includes safety checks
    to ensure it only deletes directories and handles potential filesystem errors.

    Args:
        latest_checkpoint_path: The full path to the checkpoint directory
            that should be kept.
    """
    try:
        latest_checkpoint = Path(latest_checkpoint_path).resolve()
        parent_dir = latest_checkpoint.parent

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
            # Skip the checkpoint we want to keep and any files
            if item.resolve() == latest_checkpoint.resolve() or not item.is_dir():
                continue

            try:
                logging.info(f"Deleting old checkpoint directory: {item.name}")
                shutil.rmtree(item)
                logging.info(f"Successfully deleted {item.name}")
            except OSError as e:
                logging.error(f"Failed to delete '{item.name}'. Error: {e}")

    except Exception as e:
        logging.critical(f"An unexpected error occurred during checkpoint pruning setup: {e}")
