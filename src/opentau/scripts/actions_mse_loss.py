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

import logging
from dataclasses import asdict
from pprint import pformat

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    init_logging,
)


@parser.wrap()
def inference_main(cfg: TrainPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    # build lerobot dataset and dataloader
    datasets = make_dataset_mixture(cfg)

    # load trained or finetunned model. Change the batch size to 1 in the config

    device = auto_torch_device()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    policy_sample_actions = attempt_torch_compile(policy.sample_actions, device_hint=device)

    # Always reset policy before episode to clear out action cache.
    policy.reset()

    for dataset in datasets.datasets:
        robot_dof = dataset.meta.info["features"]["actions"]["shape"][0]
        assert cfg.max_action_dim >= robot_dof
        print(f"The batch size is {cfg.batch_size}")
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

        pred = []
        truth = []
        with torch.inference_mode():
            for batch in dataloader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = batch[key].to(device)
                action = policy_sample_actions(batch)
                predicted_action = action.to("cpu", torch.float32).numpy()
                pred.append(predicted_action[0, :, :robot_dof].squeeze(0))
                truth.append(batch["actions"][:, 0, :].squeeze(0)[:robot_dof].to(torch.float32).cpu().numpy())

        pred = np.stack(pred, axis=0)
        truth = np.stack(truth, axis=0)

        print(f"the mean squared error loss per dimension is {np.mean((pred - truth) ** 2, axis=0)}")

        print(f"the r2 score per dimension is {r2_score(pred, truth, multioutput='raw_values')}")
    logging.info("End of inference")


if __name__ == "__main__":
    init_logging()
    inference_main()
