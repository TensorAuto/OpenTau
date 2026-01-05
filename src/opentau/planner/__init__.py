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
"""High-level planning for robots using vision-language models.

This module provides high-level planning capabilities that convert natural
language task descriptions into low-level action plans using vision-language
models (VLMs). It supports both manipulation and navigation tasks, with
integration for both open-source and closed-source models.

The planner acts as a bridge between high-level language commands (e.g., "Pick
up the red block and place it on the table") and low-level action sequences
that can be executed by robot policies. It processes visual observations
(camera images) along with task descriptions to generate structured plans.

Key Features:

    - **Multi-model Support**: Works with both open-source models (CogVLM,
      SmolVLM variants) and closed-source models (GPT-4o via OpenAI API).
    - **Task-specific Planners**: Specialized planners for manipulation and
      navigation tasks with task-appropriate prompts and image processing.
    - **Conversation Memory**: Maintains conversation history for multi-turn
      planning and context-aware plan generation.
    - **Cost Tracking**: Automatic cost calculation for GPT-4o API usage.
    - **Prompt Library**: YAML-based prompt templates for different task types
      and scenarios.
    - **Image Processing**: Automatic conversion of camera tensors to base64
      format for API-based models.

Main Classes:

    - **BaseHighLevelPlanner**: Abstract base class defining the planner
      interface with inference and cost calculation methods.
    - **HighLevelPlanner**: Planner for manipulation tasks, supporting both
      GPT-4o and open-source vision-language models (CogVLM, SmolVLM variants).
    - **NavHighLevelPlanner**: Specialized planner for navigation tasks with
      support for processing multiple camera views.
    - **Memory**: Conversation history manager that stores and retrieves
      multi-turn conversations between user and LLM assistant.

Supported Models:

    - **Open-source**: CogVLM-Chat-HF, SmolVLM-256M-Instruct,
      SmolVLM-500M-Instruct, SmolVLM2-2.2B-Instruct
    - **Closed-source**: GPT-4o (via OpenAI API)

Modules:

    - **high_level_planner**: Core planner implementations for manipulation
      and navigation tasks.
    - **utils.memory**: Conversation memory management for maintaining context.
    - **utils.utils**: Utility functions for image encoding and prompt loading.

Example:
    Create a planner and generate a plan:

        >>> from opentau.planner import HighLevelPlanner, Memory
        >>> planner = HighLevelPlanner()
        >>> memory = Memory()
        >>> image_dict = {"camera0": camera_tensor}
        >>> task = "Pick up the red block and place it on the table"
        >>> plan = planner.inference(
        ...     image_dict=image_dict,
        ...     model_name="gpt4o",
        ...     task=task,
        ...     mem=memory
        ... )
"""

from .high_level_planner import HighLevelPlanner as HighLevelPlanner
from .high_level_planner import NavHighLevelPlanner as NavHighLevelPlanner
from .utils.memory import Memory as Memory
