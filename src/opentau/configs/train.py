# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import datetime as dt
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common.envs.configs import EnvConfig
from lerobot.common.optim import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.configs import parser
from lerobot.configs.default import DatasetMixtureConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig

TRAIN_CONFIG_NAME = "train_config.json"


# Somehow, calling `logging.warning()` sets the logger level to WARNING.
# We print directly to stderr instead.
def warn(*args, **kwargs):
    """A dummy warning function to avoid using the logging module."""
    print("WARNING:", *args, **kwargs, file=sys.stderr)


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset_mixture: DatasetMixtureConfig
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # parameters for the Standard Data Format
    resolution: tuple[int, int] = (224, 224)  # resolution of images (H, W) in data sample
    num_cams: int = 2  # number of cameras for the cloud VLM in each data sample
    action_expert_num_cams: int = 1  # number of cameras for the action decoder in each data sample
    max_state_dim: int = 32  # maximum dimension of the state vector
    max_action_dim: int = 32  # maximum dimension of the action vector
    action_chunk: int = 50  # size of action chunk
    frozen_actions: int = 0  # number of actions from the previous chunk to condition on
    loss_weighting: dict[str, float] = field(default_factory=lambda: {"MSE": 1, "CE": 1})
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    dataloader_batch_size: int | None = None
    # Prefetch factor for the dataloader.
    prefetch_factor: int | None = None
    steps: int = 100_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = False
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    wandb: WandBConfig = field(default_factory=WandBConfig)
    # Whether to set the logging level to DEBUG. By default, the logging level will be INFO.
    debug: bool = False
    # Enable anomaly detection for debugging NaN/Inf values (warning: large computational overhead)
    trace_nans: bool = False
    # optional environment and evaluation config for evaluation
    env: EnvConfig | None = None
    eval: EvalConfig | None = field(default_factory=EvalConfig)
    eval_freq: int = 0  # evaluate every eval_freq steps
    last_checkpoint_only: bool = True

    def __post_init__(self):
        self.checkpoint_path = None

        if self.dataloader_batch_size is None and self.batch_size is None:
            raise ValueError("At least one of `batch_size` and `dataloader_batch_size` should be set.")
        if self.batch_size is None:
            self.batch_size = self.dataloader_batch_size * self.gradient_accumulation_steps
        if self.dataloader_batch_size is None:
            if self.batch_size % self.gradient_accumulation_steps != 0:
                raise ValueError(
                    "`batch_size` must be divisible by `gradient_accumulation_steps` "
                    "when `dataloader_batch_size` is not set. "
                    f"Got {self.batch_size=}, {self.gradient_accumulation_steps=}."
                )
            self.dataloader_batch_size = self.batch_size // self.gradient_accumulation_steps
        if self.dataloader_batch_size * self.gradient_accumulation_steps != self.batch_size:
            raise ValueError(
                "`batch_size` must be equal to `dataloader_batch_size * gradient_accumulation_steps`. "
                f"Got {self.batch_size=}, {self.dataloader_batch_size=}, {self.gradient_accumulation_steps=}."
            )
        assert (
            self.batch_size >= 1 and self.gradient_accumulation_steps >= 1 and self.dataloader_batch_size >= 1
        )

        if self.policy:
            self.policy.max_state_dim = self.max_state_dim
            self.policy.max_action_state = self.max_action_dim
            self.policy.chunk_size = self.action_chunk
            self.policy.frozen_actions = self.frozen_actions
        if self.job_name:
            warn(
                "cfg.job_name is deprecated and ignored. Set cfg.wandb.project and/or cfg.wandb.name instead."
            )

    def validate(self):
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path

        if not self.job_name:
            self.job_name = f"{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy:
            self.policy.max_state_dim = self.max_state_dim
            self.policy.max_action_state = self.max_action_dim
            self.policy.chunk_size = self.action_chunk

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        cfg = draccus.parse(cls, config_file, args=cli_args)

        return cfg
