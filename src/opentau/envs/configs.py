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

import abc
import logging
from copy import copy
from dataclasses import dataclass, field

import draccus

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.constants import ACTION, OBS_IMAGES, OBS_STATE
from opentau.utils.accelerate_utils import get_proc_accelerator


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    import_name: str = None
    make_id: str = None
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)
    max_parallel_tasks: int = 1
    disable_env_checker: bool = True

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_ids: list[int] | None = None
    fps: int = 30
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True
    camera_name_mapping: dict[str, str] | None = None
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,  # TODO: fix the name
            "pixels/agentview_image": f"{OBS_IMAGES}.image",  # TODO: fix the name
            "pixels/robot0_eye_in_hand_image": f"{OBS_IMAGES}.image2",  # TODO: fix the name
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
            self.features["pixels/agentview_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
            self.features["pixels/robot0_eye_in_hand_image"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(360, 360, 3)
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    @property
    def gym_kwargs(self) -> dict:
        suite_names = [s.strip() for s in str(self.task).split(",") if s.strip()]

        accelerator = get_proc_accelerator()

        task_ids: dict[str, list[int] | None]

        if accelerator is None:
            task_ids = {suite: copy(self.task_ids) for suite in suite_names}
            logging.info(f"[LIBERO environment] No accelerator found, using {task_ids=}.")
        else:
            from opentau.envs.libero import _get_suite

            task_ids = {
                suite: _get_suite(suite).tasks if self.task_ids is None else self.task_ids
                for suite in suite_names
            }
            logging.info(f"[LIBERO environment] Before distributing, using {task_ids=}.")
            task_ids = {
                suite: [
                    tsk
                    for idx, tsk in enumerate(tasks)
                    if idx % accelerator.num_processes == accelerator.process_index
                ]
                for suite, tasks in task_ids.items()
            }
            logging.info(
                f"[LIBERO environment] After distributing, using {task_ids=} on {accelerator.process_index=}."
            )

        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "task_ids": task_ids,
        }
