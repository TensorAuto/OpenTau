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

#!/usr/bin/env python

import logging
from dataclasses import asdict
from pprint import pformat

import torch
from dotenv import load_dotenv

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.planner import HighLevelPlanner, Memory
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    init_logging,
)

load_dotenv()


@parser.wrap()
def inference_main(cfg: TrainPipelineConfig):
    """
    Reflects the whole pipeline from passing tasks to high level planner to generating actions from low level planner

    Args:
        cfg: configuration file. For example look at examples/dev_config.json
    """

    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device)
    policy.to(dtype=torch.bfloat16)
    policy.eval()

    hlp = HighLevelPlanner()
    mem = Memory(len=2)

    # compile the model if possible
    if hasattr(torch, "compile"):
        logging.info("Attempting to compile the policy with torch.compile()...")
        try:
            # Other options: "default", "max-autotune" (longer compile time)
            policy = torch.compile(policy)
            logging.info("Policy compiled successfully.")
        except Exception as e:
            logging.warning(f"torch.compile failed with error: {e}. Proceeding without compilation.")
    else:
        logging.warning(
            "torch.compile is not available. Requires PyTorch 2.0+. Proceeding without compilation."
        )

    # Always reset policy before episode to clear out action cache.
    policy.reset()

    for i in range(5):
        # create dummy observation for pi05
        camera_observations = {
            f"camera{i}": torch.zeros((1, 3, *cfg.resolution), dtype=torch.bfloat16, device=device)
            for i in range(cfg.num_cams)
        }

        task = "Pick up yellow lego block and put it in the bin"

        sub_task = hlp.inference(camera_observations, "", task, "gpt4o", mem).split("\n")[1].split('"')[1]

        if mem:
            mem.add_conversation("assistant", [{"type": "text", "text": sub_task}])

        logging.info(f"{sub_task}")
        observation = {
            **camera_observations,
            "state": torch.zeros((1, cfg.max_state_dim), dtype=torch.bfloat16, device=device),
            "prompt": [sub_task],
            "img_is_pad": torch.zeros((1, cfg.num_cams), dtype=torch.bool, device=device),
            "action_is_pad": torch.zeros((1, cfg.action_chunk), dtype=torch.bool, device=device),
        }

        with torch.inference_mode():
            for _ in range(1000):
                action = policy.select_action(observation)
                action = action.to("cpu").numpy()
                print(f"Output dummy action: {action}")

    logging.info("End of inference")


if __name__ == "__main__":
    init_logging()
    inference_main()
