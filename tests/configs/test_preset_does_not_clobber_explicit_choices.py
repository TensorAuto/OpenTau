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

"""``use_policy_training_preset`` fills gaps; it does not overrule the config.

The preset path is the only one that yields a policy's param groups, so overwriting an
explicitly chosen ``scheduler`` there made "per-group learning rates" and "a specific
schedule" mutually exclusive — and the config's choice vanished with no warning.
"""

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.transforms import ImageTransformsConfig
from opentau.optim.schedulers import ConstantSchedulerConfig, CosineDecayWithWarmupSchedulerConfig
from opentau.policies.pi05.configuration_pi05 import PI05Config


def _cfg(**kwargs):
    ds = DatasetConfig(
        repo_id="m",
        root="/tmp/m",
        image_transforms=ImageTransformsConfig(enable=False),
        episodes=[0],
        video_backend=None,
    )
    policy = PI05Config()
    return TrainPipelineConfig(
        dataset_mixture=DatasetMixtureConfig(datasets=[ds], weights=[1.0], action_freq=15.0),
        policy=policy,
        batch_size=2,
        action_chunk=policy.chunk_size,
        use_policy_training_preset=True,
        steps=100,
        **kwargs,
    )


def test_the_preset_fills_an_unset_scheduler():
    """Unchanged behaviour for every config that does not choose one."""
    cfg = _cfg()
    cfg.validate()
    assert isinstance(cfg.scheduler, CosineDecayWithWarmupSchedulerConfig)
    assert cfg.optimizer is not None


def test_an_explicit_scheduler_survives_the_preset_path():
    """The regression: it used to be silently replaced by the policy's preset."""
    chosen = ConstantSchedulerConfig(num_warmup_steps=7)
    cfg = _cfg(scheduler=chosen)
    cfg.validate()
    assert cfg.scheduler is chosen, "the config's own scheduler was discarded"
    assert cfg.scheduler.num_warmup_steps == 7
    # the optimizer was not chosen, so the preset still fills it
    assert cfg.optimizer is not None


def test_an_explicit_optimizer_survives_too():
    from opentau.optim.optimizers import AdamWConfig

    chosen = AdamWConfig(lr=1.234e-4)
    cfg = _cfg(optimizer=chosen)
    cfg.validate()
    assert cfg.optimizer is chosen
    assert cfg.optimizer.lr == 1.234e-4
    assert cfg.scheduler is not None, "the unchosen half must still come from the preset"
