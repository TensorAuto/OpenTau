#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import torch
from einops import rearrange
from gymnasium import spaces

from lerobot.common.policies.pi0.modeling_pi0 import resize_with_pad
from lerobot.configs.train import TrainPipelineConfig


class Metaworld(gym.Env):
    """
    A custom gymnasium environment wrapper for Metaworld environments.

    This wrapper provides a standardized interface for Metaworld environments
    that can be registered with gymnasium and used with gym.make().
    """

    # Class-level metadata for gymnasium registration
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 80}
    TASK_TO_PROMPT = {
        "faucet-open-v3": "Rotate the faucet counter-clockwise.",
        "sweep-v3": "Sweep a puck off the table.",
        "assembly-v3": "Pick up a nut and place it onto a peg.",
        "faucet-close-v3": "Rotate the faucet clockwise.",
        "push-v3": "Push the puck to a goal.",
        "lever-pull-v3": "Pull a lever down 90 degrees.",
        "dial-turn-v3": "Rotate a dial 180 degrees.",
        "stick-push-v3": "Grasp a stick and push a box using the stick.",
        "coffee-button-v3": "Push a button on the coffee machine.",
        "handle-pull-side-v3": "Pull a handle up sideways.",
        "basketball-v3": "Dunk the basketball into the basket.",
        "stick-pull-v3": "Grasp a stick and pull a box with the stick.",
        "sweep-into-v3": "Sweep a puck into a hole.",
        "disassemble-v3": "Pick a nut out of the a peg.",
        "shelf-place-v3": "Pick and place a puck onto a shelf.",
        "coffee-push-v3": "Push a mug under a coffee machine.",
        "handle-press-side-v3": "Press a handle down sideways.",
        "hammer-v3": "Hammer a screw on the wall.",
        "plate-slide-v3": "Slide a plate into a cabinet.",
        "plate-slide-side-v3": "Slide a plate into a cabinet sideways.",
        "button-press-wall-v3": "Bypass a wall and press a button.",
        "handle-press-v3": "Press a handle down.",
        "handle-pull-v3": "Pull a handle up.",
        "soccer-v3": "Kick a soccer into the goal.",
        "plate-slide-back-side-v3": "Get a plate from the cabinet sideways.",
        "plate-slide-back-v3": "Get a plate from the cabinet.",
        "drawer-close-v3": "Push and close a drawer.",
        "button-press-topdown-v3": "Press a button from the top.",
        "reach-v3": "Reach a goal position.",
        "button-press-topdown-wall-v3": "Bypass a wall and press a button from the top.",
        "reach-wall-v3": "Bypass a wall and reach a goal.",
        "peg-insert-side-v3": "Insert a peg sideways.",
        "push-back-v3": "Pull a puck to a goal.",
        "push-wall-v3": "Bypass a wall and push a puck to a goal.",
        "pick-out-of-hole-v3": "Pick up a puck from a hole.",
        "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck.",
        "button-press-v3": "Press a button.",
        "pick-place-v3": "Pick and place a puck to a goal.",
        "coffee-pull-v3": "Pull a mug from a coffee machine.",
        "peg-unplug-side-v3": "Unplug a peg sideways.",
        "window-close-v3": "Push and close a window.",
        "window-open-v3": "Push and open a window.",
        "door-open-v3": "Open a door with a revolving joint.",
        "door-close-v3": "Close a door with a revolving joint.",
        "drawer-open-v3": "Open a drawer.",
        "hand-insert-v3": "Insert the gripper into a hole.",
        "box-close-v3": "Grasp the cover and close the box with it.",
        "door-lock-v3": "Lock the door by rotating the lock clockwise.",
        "door-unlock-v3": "Unlock the door by rotating the lock counter-clockwise.",
        "bin-picking-v3": "Grasp the puck from one bin and place it into another bin.",
    }
    CAMEL_TO_HYPHEN_TASK_NAME = {
        "SawyerNutAssemblyEnvV3": "assembly-v3",
        "SawyerBasketballEnvV3": "basketball-v3",
        "SawyerBinPickingEnvV3": "bin-picking-v3",
        "SawyerBoxCloseEnvV3": "box-close-v3",
        "SawyerButtonPressTopdownEnvV3": "button-press-topdown-v3",
        "SawyerButtonPressTopdownWallEnvV3": "button-press-topdown-wall-v3",
        "SawyerButtonPressEnvV3": "button-press-v3",
        "SawyerButtonPressWallEnvV3": "button-press-wall-v3",
        "SawyerCoffeeButtonEnvV3": "coffee-button-v3",
        "SawyerCoffeePullEnvV3": "coffee-pull-v3",
        "SawyerCoffeePushEnvV3": "coffee-push-v3",
        "SawyerDialTurnEnvV3": "dial-turn-v3",
        "SawyerNutDisassembleEnvV3": "disassemble-v3",
        "SawyerDoorCloseEnvV3": "door-close-v3",
        "SawyerDoorLockEnvV3": "door-lock-v3",
        "SawyerDoorUnlockEnvV3": "door-unlock-v3",
        "SawyerDoorEnvV3": "door-open-v3",
        "SawyerDrawerCloseEnvV3": "drawer-close-v3",
        "SawyerDrawerOpenEnvV3": "drawer-open-v3",
        "SawyerFaucetCloseEnvV3": "faucet-close-v3",
        "SawyerFaucetOpenEnvV3": "faucet-open-v3",
        "SawyerHammerEnvV3": "hammer-v3",
        "SawyerHandInsertEnvV3": "hand-insert-v3",
        "SawyerHandlePressSideEnvV3": "handle-press-side-v3",
        "SawyerHandlePressEnvV3": "handle-press-v3",
        "SawyerHandlePullSideEnvV3": "handle-pull-side-v3",
        "SawyerHandlePullEnvV3": "handle-pull-v3",
        "SawyerLeverPullEnvV3": "lever-pull-v3",
        "SawyerPegInsertionSideEnvV3": "peg-insert-side-v3",
        "SawyerPegUnplugSideEnvV3": "peg-unplug-side-v3",
        "SawyerPickOutOfHoleEnvV3": "pick-out-of-hole-v3",
        "SawyerPickPlaceEnvV3": "pick-place-v3",
        "SawyerPickPlaceWallEnvV3": "pick-place-wall-v3",
        "SawyerPlateSlideBackSideEnvV3": "plate-slide-back-side-v3",
        "SawyerPlateSlideBackEnvV3": "plate-slide-back-v3",
        "SawyerPlateSlideSideEnvV3": "plate-slide-side-v3",
        "SawyerPlateSlideEnvV3": "plate-slide-v3",
        "SawyerPushBackEnvV3": "push-back-v3",
        "SawyerPushEnvV3": "push-v3",
        "SawyerPushWallEnvV3": "push-wall-v3",
        "SawyerReachEnvV3": "reach-v3",
        "SawyerReachWallEnvV3": "reach-wall-v3",
        "SawyerShelfPlaceEnvV3": "shelf-place-v3",
        "SawyerSoccerEnvV3": "soccer-v3",
        "SawyerStickPullEnvV3": "stick-pull-v3",
        "SawyerStickPushEnvV3": "stick-push-v3",
        "SawyerSweepEnvV3": "sweep-v3",
        "SawyerSweepIntoGoalEnvV3": "sweep-into-v3",
        "SawyerWindowCloseEnvV3": "window-close-v3",
        "SawyerWindowOpenEnvV3": "window-open-v3",
    }

    def __init__(
        self,
        task: str = "Meta-World/MT1",
        env_name: str = "button-press-v3",
        render_mode: Optional[str] = "rgb_array",
        max_episode_steps: int = 200,
        camera_name: str = "corner",
        train_cfg: TrainPipelineConfig = None,
        **kwargs,
    ):
        """
        Initialize the LeRobot Metaworld wrapper.

        Args:
            task: The gymnasium task ID (e.g., "Meta-World/MT1")
            env_name: The Metaworld environment name (e.g., "button-press-v3") (Only used if already_vectorized is False)
            render_mode: Rendering mode for the environment
            max_episode_steps: Maximum number of steps per episode
            camera_name: Camera name for rendering
            train_cfg: Training Config used during training
            **kwargs: Additional arguments passed to the underlying environment
        """
        super().__init__()

        self.task = task
        self.env_name = env_name
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.camera_name = camera_name
        self._spec = None
        self.train_cfg = train_cfg

        # Create the underlying Metaworld environment
        self._env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            render_mode=render_mode,
            camera_name=camera_name,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        # Set up action and observation spaces
        self.action_space = self._env.action_space

        # Define proper observation space using Gymnasium Spaces API
        # The observation space should match the output of _to_standard_data_format
        if train_cfg is not None:
            # Define observation space based on the standardized data format
            self.observation_space = spaces.Dict(
                {
                    "camera0": spaces.Box(
                        low=0.0, high=1.0, shape=(3, *train_cfg.resolution), dtype=np.float32
                    ),
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(train_cfg.max_state_dim,), dtype=np.float32
                    ),
                    "prompt": spaces.Text(max_length=100, min_length=1),
                    "img_is_pad": spaces.Box(low=0, high=1, shape=(train_cfg.num_cams,), dtype=bool),
                    "action_is_pad": spaces.Box(low=0, high=1, shape=(train_cfg.action_chunk,), dtype=bool),
                }
            )
        else:
            # Fallback to original observation space if no train_cfg provided
            self.observation_space = self._env.observation_space

        # Set the spec from the underlying environment
        self.spec = getattr(self._env, "spec", None)

        # Track episode information
        self._episode_steps = 0
        self._episode_reward = 0.0

    def _to_standard_data_format(self, observation) -> dict[str, Any]:
        """
        Convert the observation to the standard data format for a single environment.
        """
        # Get rendered image (already flipped in render method)
        img = self.render()

        # Convert to numpy array and normalize
        camera0 = img.astype(np.float32) / 255.0
        camera0 = rearrange(camera0, "h w c -> 1 c h w")

        # Temporarily convert to torch for resize_with_pad, then back to numpy
        # resize_with_pad expects (b, c, h, w)
        camera0_torch = torch.from_numpy(camera0)
        camera0_torch = resize_with_pad(camera0_torch, *self.train_cfg.resolution, pad_value=0)
        camera0_torch = rearrange(camera0_torch, "1 c h w -> c h w")
        camera0 = camera0_torch.numpy()

        # Create padded state array
        state = np.zeros(self.train_cfg.max_state_dim, dtype=np.float32)
        state[:4] = observation[:4].astype(np.float32)

        return {
            "camera0": camera0,
            "state": state,
            "prompt": Metaworld.TASK_TO_PROMPT[self.env_name],
            "img_is_pad": np.zeros(self.train_cfg.num_cams, dtype=bool),
            "action_is_pad": np.zeros(self.train_cfg.action_chunk, dtype=bool),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for the environment
            options: Additional options for reset

        Returns:
            Tuple of (observation, info)
        """
        observation, info = self._env.reset(seed=seed, options=options)
        observation = self._to_standard_data_format(observation)

        # Reset episode tracking
        self._episode_steps = 0
        self._episode_reward = 0.0

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self._env.step(action)
        observation = self._to_standard_data_format(observation)

        # Update episode tracking
        self._episode_steps += 1
        self._episode_reward += reward

        # Add custom info
        info["episode_steps"] = self._episode_steps
        info["episode_reward"] = self._episode_reward

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            Rendered frame if render_mode is "rgb_array", None otherwise
        """
        img = self._env.render()
        if img is not None and self.camera_name == "corner":
            # Flip the image vertically due to Metaworld bug with "corner" camera
            img = cv2.flip(np.array(img), 0)
        return img

    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self._env, "close"):
            self._env.close()

    def seed(self, seed: Optional[int] = None):
        """
        Set the random seed for the environment.

        Args:
            seed: Random seed

        Returns:
            List of seeds used
        """
        if hasattr(self._env, "seed"):
            return self._env.seed(seed)
        return [seed]

    @property
    def unwrapped(self):
        """Return the underlying environment."""
        return self._env.unwrapped


def register_lerobot_metaworld_envs():
    """
    Register LeRobot Metaworld environments with gymnasium.

    This allows us to use the Metaworld environment with gym.make().
    """

    # Register the main wrapper
    gym.register(
        id="Metaworld",
        entry_point="lerobot.common.envs.metaworld:Metaworld",
    )


# Register environments when module is imported
try:
    register_lerobot_metaworld_envs()
except Exception:
    # If registration fails, it might be because environments are already registered
    # or the metaworld package is not available
    print("Metaworld environments are already registered or the metaworld package is not available")
    pass
