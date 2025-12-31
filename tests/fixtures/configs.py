from dataclasses import dataclass

import pytest

from opentau.optim.optimizers import OptimizerConfig
from opentau.optim.schedulers import LRSchedulerConfig
from opentau.configs.policies import PreTrainedConfig


@dataclass
class ConcretePolicyConfig(PreTrainedConfig):
    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return OptimizerConfig()

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        pass


@pytest.fixture(scope="session")
def get_inherited_pretrainedconfig():
    return ConcretePolicyConfig
