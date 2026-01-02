#!/usr/bin/env python

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
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.policies.factory import make_policy, make_policy_config
from opentau.utils.random_utils import set_seed


def get_policy_stats(ds_repo_id: str, policy_name: str, policy_kwargs: dict):
    set_seed(1337)

    train_cfg = TrainPipelineConfig(
        # TODO(rcadene, aliberts): remove dataset download
        dataset_mixture=DatasetMixtureConfig(
            datasets=[DatasetConfig(repo_id=ds_repo_id, episodes=[0])],
            weights=[1.0],
            action_freq=30.0,
            image_resample_strategy="nearest",
            vector_resample_strategy="nearest",
        ),
        policy=make_policy_config(policy_name, **policy_kwargs),
        batch_size=8,
        use_policy_training_preset=True,
    )
    train_cfg.validate()  # Needed for auto-setting some parameters

    dataset = make_dataset_mixture(train_cfg)
    policy = make_policy(train_cfg.policy, ds_meta=dataset.meta)
    policy.to(torch.bfloat16)
    policy.cuda()
    policy.eval()

    dataloader = dataset.get_dataloader()

    with torch.no_grad():
        batch = next(iter(dataloader))
        # send all batch tensors to GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        losses = policy.forward(batch)

    policy.reset()

    # HACK: We reload a batch with no delta_indices as `select_action` won't expect a timestamps dimension
    # We simulate having an environment using a dataset by setting delta_indices to None and dropping tensors
    # indicating padding (those ending with "_is_pad")
    dataset.delta_indices = None
    actions_queue = train_cfg.policy.n_action_steps

    actions = {}
    for i in range(actions_queue):
        action = policy.select_action(batch).contiguous()
        actions[str(i)] = action

    return losses, actions


def save_policy_to_safetensors(output_dir: Path, ds_repo_id: str, policy_name: str, policy_kwargs: dict):
    if output_dir.exists():
        print(f"Overwrite existing safetensors in '{output_dir}':")
        print(f" - Validate with: `git add {output_dir}`")
        print(f" - Revert with: `git checkout -- {output_dir}`")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dict, actions = get_policy_stats(ds_repo_id, policy_name, policy_kwargs)
    print("Saving safetensors to", output_dir)
    save_file(output_dict, output_dir / "output_dict.safetensors")
    save_file(actions, output_dir / "actions.safetensors")


if __name__ == "__main__":
    artifacts_cfg = [
        ("lerobot/droid_100", "pi0", {"chunk_size": 50, "pretrained_path": "lerobot/pi0"}, "pretrained")
    ]
    if len(artifacts_cfg) == 0:
        raise RuntimeError("No policies were provided!")
    for ds_repo_id, policy, policy_kwargs, file_name_extra in artifacts_cfg:
        ds_name = ds_repo_id.split("/")[-1]
        output_dir = Path(__file__).parent / f"{ds_name}_{policy}_{file_name_extra}"
        output_dir = output_dir.resolve()
        save_policy_to_safetensors(output_dir, ds_repo_id, policy, policy_kwargs)
