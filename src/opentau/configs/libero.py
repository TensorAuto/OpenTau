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
"""LIBERO environment configuration module.

This module provides configuration classes for LIBERO benchmark evaluation,
which is a benchmark suite for learning manipulation tasks. It extends the
base training pipeline configuration with LIBERO-specific evaluation parameters.
"""

import os
from dataclasses import dataclass

from libero.libero import benchmark, get_libero_path

from opentau.configs.train import TrainPipelineConfig
from opentau.utils.monkey_patch import torch_load_patch

LIBERO_BENCHMARK_DICT = benchmark.get_benchmark_dict()


@dataclass
class LiberoEnvConfig:
    """Configuration for LIBERO environment evaluation.

    LIBERO is a benchmark suite for learning manipulation tasks. This configuration
    specifies which task suite and task to run, along with evaluation parameters.

    Args:
        suite: Task suite to run. Must be 'spatial', 'object', 'goal', or '100'.
        id: Index of the task in the suite to run.
        max_steps: Maximum number of steps to run for each task. Defaults to 1000.
        chunk_usage: Number of actions to perform in each chunk before getting a
            new observation. If None, will be set from the training config's
            `action_chunk`. Defaults to None.
        n_simulations: Number of simulations to run for each task. Defaults to 100.
        video_dir: Directory to save videos of the task execution. Defaults to None.

    Raises:
        ValueError: If the suite name is invalid or if the task id is out of range
            for the specified suite.
    """

    suite: str  # Task suite to run. Must be 'spatial', 'object', 'goal', or '100'.
    id: int  # index of the task in the suite to run.
    max_steps: int = 1000  # maximum number of steps to run for each task.
    chunk_usage: int | None = (
        None  # number of actions to perform in each chunk before getting a new observation.
    )
    n_simulations: int = 100  # number of simulations to run for each task.
    video_dir: str = None  # directory to save videos of the task execution.

    def __post_init__(self):
        """Validate LIBERO configuration and initialize task-specific attributes."""
        torch_load_patch()
        suite = f"libero_{self.suite}".lower()
        if suite not in LIBERO_BENCHMARK_DICT:
            raise ValueError(
                f"Invalid suites: '{self.suite}'. "
                f"Available suites are: {[k.replace('libero_', '') for k in LIBERO_BENCHMARK_DICT]}"
            )
        suite = LIBERO_BENCHMARK_DICT[suite]()
        try:
            task = suite.get_task(self.id)
        except IndexError as e:
            raise ValueError(
                f"Invalid task id: {self.id} for suite: {self.suite}. "
                f"Available ids must be from 0 to {len(suite.tasks) - 1}."
            ) from e

        self.bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.init_states = suite.get_task_init_states(self.id)
        self.task = task


@dataclass
class TrainConfigWithLiberoEval(TrainPipelineConfig):
    """Training configuration extended with LIBERO evaluation settings.

    This configuration extends the base training pipeline configuration with
    LIBERO-specific evaluation parameters.

    Args:
        libero: Configuration for LIBERO environment evaluation. Must be provided.
            Defaults to None.

    Raises:
        ValueError: If `libero` is None or if `chunk_usage` is not within valid
            range (1 to action_chunk).
    """

    libero: LiberoEnvConfig = None

    def __post_init__(self):
        """Validate LIBERO configuration and set default chunk_usage if needed."""
        super().__post_init__()
        if self.libero is None:
            raise ValueError("Libero config must be provided.")
        if self.libero.chunk_usage is None:
            self.libero.chunk_usage = self.action_chunk
        assert 1 <= self.libero.chunk_usage <= self.action_chunk, (
            f"Chunk usage must be between 1 and {self.action_chunk=}, got {self.libero.chunk_usage=}."
        )
