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


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from opentau.configs.train import TrainPipelineConfig
from opentau.policies.pretrained import PreTrainedPolicy


def _trainable_params(params):
    """Keep only parameters that require grad, preserving param-group structure.

    ``get_optim_params()`` may return either a flat iterable of ``Parameter`` (what most
    policies do) or a list of param-group dicts, which is how a policy asks for per-group
    hyperparameters -- ``flux3_action`` uses it for the embodiment heads' 5x learning rate.

    The ``requires_grad`` filter has to reach *inside* the groups. Applying it to the dicts
    themselves raises ``AttributeError: 'dict' object has no attribute 'requires_grad'``,
    which silently made param groups unusable: a policy returning them could not be trained
    at all, and the per-group learning rates it asked for never took effect.

    Empty groups are kept rather than dropped, so group *indices* stay stable -- any config
    that refers to a group by position (a per-group learning-rate schedule, say) would
    otherwise be silently remapped when a group happened to be fully frozen.

    Args:
        params: A flat iterable of parameters, or a list of param-group dicts.

    Returns:
        The same shape that was passed in, filtered to trainable parameters.
    """
    materialized = list(params)
    if not materialized:
        return []
    if not isinstance(materialized[0], dict):
        if any(isinstance(entry, dict) for entry in materialized):
            raise TypeError(
                "get_optim_params() mixed bare parameters with param-group dicts; torch "
                "requires one or the other. Return a flat iterable of parameters, or a list "
                "where every entry is a group dict."
            )
        return [p for p in materialized if p.requires_grad]

    groups = []
    for index, group in enumerate(materialized):
        if not isinstance(group, dict):
            raise TypeError(
                f"get_optim_params() returned a param-group dict first but entry {index} is "
                f"{type(group).__name__}; torch requires every entry to be a group."
            )
        if "params" not in group:
            raise KeyError(
                f"param group {index} from get_optim_params() has no 'params' key; a group "
                "must name the parameters it applies its hyperparameters to."
            )
        groups.append({**group, "params": [p for p in group["params"] if p.requires_grad]})
    return groups


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    # When using `accelerate`, unused parameters that require grad can result in a RuntimeError("Expected to have
    #   finished reduction in the prior iteration before starting a new one.")
    optimizer = cfg.optimizer.build(_trainable_params(params))
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
