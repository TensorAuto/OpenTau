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
"""RoboCasa environment configuration module.

Mirrors :mod:`opentau.configs.libero` for the RoboCasa kitchen benchmark — extends the
base training pipeline configuration with RoboCasa-specific evaluation parameters used by
``scripts/robocasa_eval/eval.py`` and the bundled-in evaluation hook in
``scripts/train.py``.
"""

from dataclasses import dataclass

from opentau.configs.train import TrainPipelineConfig


@dataclass
class RoboCasaEvalConfig:
    """Configuration for RoboCasa environment evaluation.

    Args:
        task: Comma-separated RoboCasa task class names (e.g. ``"PnPCounterToCab"``).
            Each name becomes one logical evaluation group, vectorised across rollouts.
        max_steps: Hard cap on policy steps per episode (in addition to ``_check_success``).
            Defaults to 1500.
        chunk_usage: Number of actions to execute per policy call before re-querying.
            If ``None``, defaults to the training config's ``action_chunk``.
        n_simulations: Number of rollouts to run per task. Defaults to 50.
        video_dir: Optional output directory for rollout videos.
        action_dim: Effective action dim sent to RoboCasa (the policy output is
            truncated or zero-padded to this size). Defaults to 16 (PI 0.5 action head).
        seed_base: Starting seed for rollout 0; rollout ``i`` uses ``seed_base + i``.
        camera_height: Off-screen camera height (pixels). Defaults to 256.
        camera_width: Off-screen camera width (pixels). Defaults to 256.
        split: Dataset split passed to ``robocasa.utils.env_utils.create_env``.

    Raises:
        ValueError: If ``task`` is empty.
    """

    task: str = "PnPCounterToCab"
    max_steps: int = 1500
    chunk_usage: int | None = None
    n_simulations: int = 50
    video_dir: str | None = None
    action_dim: int = 16
    seed_base: int = 0
    camera_height: int = 256
    camera_width: int = 256
    split: str | None = "all"

    def __post_init__(self):
        task_names = [s.strip() for s in str(self.task).split(",") if s.strip()]
        if not task_names:
            raise ValueError("RoboCasaEvalConfig.task must contain at least one task class name.")
        self.task_names = task_names


@dataclass
class TrainConfigWithRoboCasaEval(TrainPipelineConfig):
    """Training configuration extended with RoboCasa evaluation settings.

    Args:
        robocasa: Configuration for RoboCasa environment evaluation. Must be provided.

    Raises:
        ValueError: If ``robocasa`` is None or ``chunk_usage`` is outside ``[1, action_chunk]``.
    """

    robocasa: RoboCasaEvalConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.robocasa is None:
            raise ValueError("RoboCasa config must be provided.")
        if self.robocasa.chunk_usage is None:
            self.robocasa.chunk_usage = self.action_chunk
        if not 1 <= self.robocasa.chunk_usage <= self.action_chunk:
            raise ValueError(
                f"Chunk usage must be between 1 and {self.action_chunk=}, got {self.robocasa.chunk_usage=}."
            )
