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
"""Deployment configuration classes for inference servers.

This module provides configuration classes for deploying trained models
as inference servers, including gRPC server settings.
"""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the gRPC inference server.

    This class contains all configuration parameters needed to run a gRPC
    inference server for robot policy models.

    Args:
        port: Port number to serve on. Must be between 1 and 65535.
            Defaults to 50051.
        max_workers: Maximum number of gRPC worker threads for handling
            concurrent requests. Defaults to 4.
        max_send_message_length_mb: Maximum size of outgoing messages in
            megabytes. Defaults to 100.
        max_receive_message_length_mb: Maximum size of incoming messages in
            megabytes. Defaults to 100.

    Raises:
        ValueError: If port is not in valid range or max_workers is less than 1.

    Example:
        >>> config = ServerConfig(port=50051, max_workers=8)
        >>> config.port
        50051
    """

    port: int = 50051
    max_workers: int = 4
    max_send_message_length_mb: int = 100
    max_receive_message_length_mb: int = 100
    # Which training-time norm head to use for inference requests. Either:
    #   - set both `robot_type` and `control_mode` to address the head by
    #     `(robot_type, control_mode)` (preferred for multi-head checkpoints),
    #   - or set `dataset_repo_id` to a training-time dataset name (the
    #     policy maps it via its persisted `dataset_to_norm_index`;
    #     back-compat path that also works on legacy per-dataset checkpoints).
    # When all three are ``None`` (default), single-head policies fall back
    # to the `_resolve_dataset_index` zero-default; multi-head ones raise.
    # The robot_type / control_mode pair takes precedence over
    # `dataset_repo_id` when both are set.
    dataset_repo_id: str | None = None
    robot_type: str | None = None
    control_mode: str | None = None

    def __post_init__(self):
        """Validate server configuration parameters."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"`port` must be between 1 and 65535, got {self.port}.")
        if self.max_workers < 1:
            raise ValueError(f"`max_workers` must be at least 1, got {self.max_workers}.")
        if self.max_send_message_length_mb < 1:
            raise ValueError(
                f"`max_send_message_length_mb` must be at least 1, got {self.max_send_message_length_mb}."
            )
        if self.max_receive_message_length_mb < 1:
            raise ValueError(
                f"`max_receive_message_length_mb` must be at least 1, got {self.max_receive_message_length_mb}."
            )

    @property
    def max_send_message_length(self) -> int:
        """Get maximum send message length in bytes.

        Returns:
            Maximum send message length in bytes.
        """
        return self.max_send_message_length_mb * 1024 * 1024

    @property
    def max_receive_message_length(self) -> int:
        """Get maximum receive message length in bytes.

        Returns:
            Maximum receive message length in bytes.
        """
        return self.max_receive_message_length_mb * 1024 * 1024


@dataclass
class PlannerConfig:
    """Configuration for the high-level planner of the gRPC inference server.

    When ``enabled`` is False (the default), the inference server runs the VLA
    policy only. When enabled, a Gemini Robotics-ER planner runs
    asynchronously alongside the policy: it
    consumes the latest observation (images, state) plus the request prompt
    (treated as the overall task) and a memory-as-language string, and produces
    the subtask the VLA policy is conditioned on.

    Args:
        enabled: Whether to spin up the high-level planner. Defaults to False.
        model: Gemini model ID used for planning. Defaults to
            ``gemini-robotics-er-1.5-preview``.
        api_key_env: Environment variable holding the Gemini API key
            (``GOOGLE_API_KEY`` is also tried as a fallback). Defaults to
            ``GEMINI_API_KEY``.
        interval_s: Wall-clock seconds between planner calls. The planner runs
            on its own free-running background loop, decoupled from request
            arrival; the VLA path never blocks on replanning and reads
            whatever subtask is currently available. The loop skips a cycle
            when no new observation arrived since the last plan. Defaults
            to 5.0.
        first_plan_timeout_s: Maximum seconds a request blocks at the start of
            inference for a task (no subtask available yet) waiting for the
            initial subtask. On timeout the request falls back to the raw
            task prompt. Defaults to 30.0.
        max_output_tokens: Generation cap for the planner response.
            Defaults to 512.
        temperature: Sampling temperature for the planner. Defaults to 0.0.
        include_state: Whether to include the robot proprioceptive state in
            the planner prompt. Defaults to True.
        system_prompt_key: Key into ``planner/prompts.yaml`` for the system
            prompt template.
        user_prompt_key: Key into ``planner/prompts.yaml`` for the user
            prompt template.

    Raises:
        ValueError: If ``interval_s``, ``first_plan_timeout_s`` or
            ``max_output_tokens`` are out of range.
    """

    enabled: bool = False
    model: str = "gemini-robotics-er-1.5-preview"
    api_key_env: str = "GEMINI_API_KEY"
    interval_s: float = 5.0
    first_plan_timeout_s: float = 30.0
    max_output_tokens: int = 512
    temperature: float = 0.0
    include_state: bool = True
    system_prompt_key: str = "gemini_er_planner_system"
    user_prompt_key: str = "gemini_er_planner_user"

    def __post_init__(self):
        """Validate planner configuration parameters."""
        if self.interval_s <= 0:
            raise ValueError(f"`interval_s` must be positive, got {self.interval_s}.")
        if self.first_plan_timeout_s <= 0:
            raise ValueError(f"`first_plan_timeout_s` must be positive, got {self.first_plan_timeout_s}.")
        if self.max_output_tokens < 1:
            raise ValueError(f"`max_output_tokens` must be at least 1, got {self.max_output_tokens}.")
