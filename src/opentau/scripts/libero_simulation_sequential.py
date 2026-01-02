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

import logging
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from torch.utils.data._utils.collate import default_collate

from opentau.configs import parser
from opentau.configs.libero import TrainConfigWithLiberoEval
from opentau.policies.factory import get_policy_class
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.utils.libero import LiberoObservationRecorder, libero2torch, summarize_libero_results
from opentau.utils.monkey_patch import gym_is_gymnasium_patch
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device, init_logging

LIBERO_ACTION_DIM = 7


def run_simulations(
    policy: PreTrainedPolicy, cfg: TrainConfigWithLiberoEval, device: str, dtype: torch.dtype
):
    gym_is_gymnasium_patch()
    # This import has to happen after the `gym_is_gymnasium_patch` is called,
    #   so we can't put it at the top of the file.
    from libero.libero.envs import OffScreenRenderEnv

    init_states = cfg.libero.init_states

    steps_taken = {}
    for sim_idx in range(1, cfg.libero.n_simulations + 1):
        # This environment provides interaction with the policy without rendering a UI.
        # To record videos, we use the `LiberoObservationRecorder` class and manually record frames.
        env = OffScreenRenderEnv(
            bddl_file_name=cfg.libero.bddl_file,
            camera_heights=cfg.resolution[0],
            camera_widths=cfg.resolution[1],
        )
        s0 = init_states[sim_idx % len(init_states)]
        env.seed(sim_idx)
        env.set_init_state(s0)

        video_root = cfg.libero.video_dir and (
            Path(cfg.libero.video_dir) / cfg.libero.suite / str(cfg.libero.id) / str(sim_idx)
        )
        camera_names = ["agentview_image", "robot0_eye_in_hand_image"]
        with LiberoObservationRecorder(video_root, camera_names=camera_names) as recorder:
            obs = env.reset()
            # Warm up the environment with a few no-op steps
            for _ in range(5):
                obs, *_ = env.step([0.0] * LIBERO_ACTION_DIM)
            recorder.record(obs)

            for step_idx in range(cfg.libero.max_steps):
                if step_idx % cfg.libero.chunk_usage == 0:
                    logging.debug(f"Resetting policy before step {step_idx + 1} for simulation {sim_idx}")
                    # Invalidate the cache and force the policy to recompute a new batch of actions
                    policy.reset()

                torch_input = libero2torch(obs, cfg, device, dtype)
                torch_input = default_collate([torch_input])
                action = policy.select_action(torch_input)
                action = action.flatten().numpy(force=True)[:LIBERO_ACTION_DIM]
                action[-1] = 2.0 * (action[-1] > 0) - 1.0  # gripper open/close should be -1 or 1
                obs, reward, done, info = env.step(action)
                recorder.record(obs)
                logging.debug(f"Step: {step_idx + 1}, Reward: {reward}, Done: {done}, Info: {info}")
                if done or reward > 0:
                    steps_taken[sim_idx] = step_idx + 1
                    break

            env.close()

    return steps_taken


@parser.wrap()
def main(cfg: TrainConfigWithLiberoEval):
    init_logging(level=logging.DEBUG if cfg.debug else logging.INFO)
    logging.info(pformat(asdict(cfg)))

    device = auto_torch_device()
    dtype = torch.bfloat16

    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    with torch.inference_mode():
        steps_taken = run_simulations(policy, cfg, device, dtype)

    results = [-1] * cfg.libero.n_simulations
    for sim_idx, step in steps_taken.items():
        results[sim_idx - 1] = step
    summary = summarize_libero_results(results)
    logging.info(str(summary))
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
