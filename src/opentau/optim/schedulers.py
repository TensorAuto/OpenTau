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
import abc
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from opentau.constants import SCHEDULER_STATE
from opentau.datasets.utils import write_json
from opentau.utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0"""

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        del num_training_steps

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, self.num_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.num_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < self.num_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("hold_warmup_constant_cooldown")
@dataclass
class HoldWarmupConstantCooldownSchedulerConfig(LRSchedulerConfig):
    """Black Forest Labs' FLUX 3 Action schedule: hold at zero, warm up, hold, cool down.

    Their published DROID recipe runs two different curves at once -- the trunk is held at
    zero for the first ``hold_steps`` updates while AdamW still accumulates moments, then
    warms up and stays flat; the freshly initialized embodiment heads skip the hold and
    warm up immediately, since they start from noise and nothing is being protected. Both
    then cool down linearly together over the same window.

    ``LambdaLR`` takes one lambda per parameter group, so the two shapes ride on the same
    scheduler: ``head_param_group_indices`` selects which of
    ``optimizer.param_groups`` get the head curve. That matters because the group *order*
    is the policy's, not this scheduler's -- naming the indices explicitly keeps a
    reordering from silently swapping the curves.

    Args:
        hold_steps: Updates the trunk is held at zero LR.
        cooldown_start: Update index where the linear cooldown begins.
        total_steps: Update index where the cooldown reaches ``final_lr_scale``.
        final_lr_scale: Multiplier at the end of cooldown.
        head_warmup_steps: Warmup duration for head groups, which have no hold.
        head_param_group_indices: Which optimizer param groups use the head curve.
    """

    #: Inherited from :class:`LRSchedulerConfig`. For this schedule it is the absolute
    #: update index at which trunk warmup *completes*, not a duration -- the trunk's
    #: warmup is preceded by ``hold_steps``, so a duration alone would not locate it.
    num_warmup_steps: int = 3_000
    hold_steps: int = 1_000
    cooldown_start: int = 25_000
    total_steps: int = 30_000
    final_lr_scale: float = 0.0
    head_warmup_steps: int = 1_000
    head_param_group_indices: tuple[int, ...] = (1,)

    def __post_init__(self):
        if not 0 <= self.hold_steps < self.num_warmup_steps <= self.cooldown_start < self.total_steps:
            raise ValueError(
                "expected 0 <= hold_steps < num_warmup_steps <= cooldown_start < total_steps, got "
                f"{self.hold_steps}, {self.num_warmup_steps}, {self.cooldown_start}, {self.total_steps}."
            )
        if not 0 < self.head_warmup_steps <= self.cooldown_start:
            # Warming past the cooldown start makes the head curve non-monotonic: it is
            # still ramping up while the cooldown is already scaling down, so the factor
            # rises to a peak below 1 and then cliff-drops the moment warmup ends.
            raise ValueError(
                "expected 0 < head_warmup_steps <= cooldown_start, got "
                f"{self.head_warmup_steps} and {self.cooldown_start}."
            )
        if not 0.0 <= self.final_lr_scale <= 1.0:
            raise ValueError(f"final_lr_scale must be in [0, 1], got {self.final_lr_scale}.")

    def _cooldown(self, step: int) -> float:
        if step < self.cooldown_start:
            return 1.0
        span = self.total_steps - self.cooldown_start
        frac = min(1.0, (step - self.cooldown_start) / span)
        return 1.0 + frac * (self.final_lr_scale - 1.0)

    def _trunk_factor(self, step: int) -> float:
        if step < self.hold_steps:
            return 0.0
        if step < self.num_warmup_steps:
            return (step - self.hold_steps) / (self.num_warmup_steps - self.hold_steps)
        return self._cooldown(step)

    def _head_factor(self, step: int) -> float:
        if step < self.head_warmup_steps:
            return step / self.head_warmup_steps
        return self._cooldown(step)

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        heads = set(self.head_param_group_indices)
        # Negative indices are a natural spelling for "the last group", but the lookup
        # below iterates ``range(len(param_groups))``, so a negative entry would match
        # nothing and hand *every* group the trunk curve -- silently freezing the heads.
        invalid = [i for i in heads if i < 0 or i >= len(optimizer.param_groups)]
        if invalid:
            raise ValueError(
                f"head_param_group_indices {sorted(invalid)} are out of range for the "
                f"optimizer's {len(optimizer.param_groups)} parameter group(s); indices must "
                "be non-negative and the policy's get_optim_params() decides the grouping."
            )
        if num_training_steps and num_training_steps > self.total_steps:
            logging.warning(
                "training for %d steps but the schedule ends at %d; every step past it runs "
                "at final_lr_scale=%s. Set total_steps to the run length.",
                num_training_steps,
                self.total_steps,
                self.final_lr_scale,
            )
        lambdas = [
            (self._head_factor if i in heads else self._trunk_factor)
            for i in range(len(optimizer.param_groups))
        ]
        return LambdaLR(optimizer, lambdas, -1)


@LRSchedulerConfig.register_subclass("constant")
@dataclass
class ConstantSchedulerConfig(LRSchedulerConfig):
    """Constant learning rate scheduler that doesn't change the learning rate over time"""

    num_warmup_steps: int = 0

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        del num_training_steps

        def lr_lambda(current_step):
            # Always return 1.0 to keep the learning rate constant
            return 1.0

        return LambdaLR(optimizer, lr_lambda, -1)


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
