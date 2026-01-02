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

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    create_dummy_observation,
    init_logging,
)


@parser.wrap()
def inference_main(cfg: TrainPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = auto_torch_device()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    policy = attempt_torch_compile(policy, device_hint=device)

    # Always reset policy before episode to clear out action cache.
    policy.reset()

    observation = create_dummy_observation(cfg, device, dtype=torch.bfloat16)

    print(observation.keys())

    with torch.inference_mode():
        for _ in range(1000):
            action = policy.select_action(observation)
            action = action.to("cpu", torch.float32).numpy()
            print(f"Output shape: {action.shape}")

    logging.info("End of inference")


if __name__ == "__main__":
    init_logging()
    inference_main()
