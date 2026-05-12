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
"""Evaluate a policy on RoboCasa kitchen environments.

Mirrors :mod:`opentau.scripts.eval` (LIBERO) but binds the env layer to RoboCasa via
:class:`opentau.envs.configs.RoboCasaEnv`. The rollout / metric machinery is shared with
``scripts/eval.py`` — this module only handles RoboCasa-specific config parsing, output
layout, and the fact that RoboCasa rollouts have no LIBERO-style dataset recorder.

The same entrypoint is also reusable from ``scripts/train.py`` for periodic in-loop
evaluation: ``cfg.env`` can be a ``RoboCasaEnv`` config and ``make_envs(cfg.env, ...)``
will produce robocasa vec envs that ``eval_policy_all`` consumes unchanged.

Run::

    opentau-eval \\
        --accelerate-config configs/examples/accelerate_ddp_config.yaml \\
        --config_path=configs/examples/pi05_robocasa_eval_config.json

or directly::

    accelerate launch src/opentau/scripts/robocasa_eval/eval.py \\
        --config_path=configs/examples/pi05_robocasa_eval_config.json
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datetime as dt
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from termcolor import colored

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.envs.configs import RoboCasaEnv
from opentau.envs.factory import make_envs
from opentau.envs.utils import close_envs
from opentau.policies.factory import make_policy
from opentau.scripts.eval import consolidate_eval_info, eval_policy_all
from opentau.utils.accelerate_utils import acc_print, set_proc_accelerator
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import init_logging, is_launched_with_accelerate


@parser.wrap()
def robocasa_eval_main(cfg: TrainPipelineConfig):
    """Run distributed RoboCasa evaluation for the policy described by ``cfg``."""
    accelerator = Accelerator()
    set_proc_accelerator(accelerator)

    init_logging(accelerator=accelerator)
    logging.info(pformat(asdict(cfg)))

    if not isinstance(cfg.env, RoboCasaEnv):
        raise ValueError(
            f"robocasa_eval_main expects cfg.env to be a RoboCasaEnv config "
            f"(set 'env.type'='robocasa' in the JSON), got {type(cfg.env).__name__}."
        )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    details = f"{cfg.env.type}-{cfg.env.task}-{cfg.eval.n_episodes}"
    now = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    eval_output_dir = Path(cfg.output_dir) / "post-training-eval" / f"{details}-{now}"

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {eval_output_dir}")

    logging.info("Making RoboCasa environment.")
    envs = make_envs(
        cfg.env,
        cfg,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy)
    policy.to(torch.bfloat16)
    policy = accelerator.prepare(policy)
    policy.eval()
    with (
        torch.no_grad(),
        torch.autocast(device_type=accelerator.device.type) if cfg.policy.use_amp else nullcontext(),
    ):
        eval_info = eval_policy_all(
            envs=envs,
            policy=policy,
            n_episodes=cfg.eval.n_episodes,
            cfg=cfg,
            max_episodes_rendered=cfg.eval.max_episodes_rendered,
            videos_dir=eval_output_dir / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            # RoboCasa has no LIBERO-style dataset recorder; episode data isn't useful
            # to keep around and inflates GPU memory, so leave it off.
            return_episode_data=False,
        )

        acc_print("Local Eval Info", eval_info)
        eval_info = gather_object([eval_info])

        if accelerator.is_main_process:
            eval_info = consolidate_eval_info(eval_info)
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_output_dir / "eval_info.json", "w") as f:
                json.dump(eval_info, f, indent=2)
            print("Overall Aggregated Metrics:")
            print(eval_info["overall"])
            for task_group, task_group_info in eval_info["per_group"].items():
                print(f"\nAggregated Metrics for {task_group}:")
                print(task_group_info)

    close_envs(envs)
    accelerator.end_training()

    logging.info("End of RoboCasa eval")


def main():
    robocasa_eval_main()


if __name__ == "__main__":
    if not is_launched_with_accelerate():
        raise Exception(
            "This script should be launched with accelerate. "
            "Please use `accelerate launch` to run this script."
        )
    main()
