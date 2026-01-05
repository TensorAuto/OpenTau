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
"""Training pipeline configuration module.

This module provides the TrainPipelineConfig class which contains all configuration
parameters needed to run a training pipeline, including dataset settings, policy
configuration, training hyperparameters, and evaluation settings.
"""

import datetime as dt
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from opentau.configs import parser
from opentau.configs.default import DatasetMixtureConfig, EvalConfig, WandBConfig
from opentau.configs.policies import PreTrainedConfig
from opentau.envs.configs import EnvConfig
from opentau.optim import OptimizerConfig
from opentau.optim.schedulers import LRSchedulerConfig
from opentau.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


# Somehow, calling `logging.warning()` sets the logger level to WARNING.
# We print directly to stderr instead.
def warn(*args, **kwargs):
    """Print a warning message to stderr.

    This function is used instead of logging.warning() to avoid setting the logger
    level to WARNING.

    Args:
        *args: Variable length argument list to print.
        **kwargs: Arbitrary keyword arguments passed to print().
    """
    print("WARNING:", *args, **kwargs, file=sys.stderr)


@dataclass
class TrainPipelineConfig(HubMixin):
    """Configuration for the training pipeline.

    This class contains all configuration parameters needed to run a training
    pipeline, including dataset settings, policy configuration, training hyperparameters,
    and evaluation settings.

    Args:
        dataset_mixture: Configuration for the dataset mixture to use during training.
        policy: Configuration for the policy model. If None, must be set via CLI or
            from a pretrained checkpoint.
        output_dir: Directory where all run outputs will be saved. If another training
            session uses the same directory, its contents will be overwritten unless
            `resume` is set to True.
        job_name: Name identifier for the training job. If not provided, defaults to
            the policy type.
        resume: If True, resume a previous run. Requires `output_dir` to point to
            an existing run directory with at least one checkpoint. When resuming,
            the configuration from the checkpoint is used by default, regardless of
            command-line arguments.
        seed: Random seed used for training (model initialization, dataset shuffling)
            and for evaluation environments. Defaults to 1000.
        resolution: Resolution of images (height, width) in data samples. Defaults to (224, 224).
        num_cams: Number of cameras for the cloud VLM in each data sample. Defaults to 2.
        max_state_dim: Maximum dimension of the state vector. Defaults to 32.
        max_action_dim: Maximum dimension of the action vector. Defaults to 32.
        action_chunk: Size of action chunk. Defaults to 50.
        loss_weighting: Dictionary mapping loss type names to their weights.
            Defaults to {"MSE": 1, "CE": 1}.
        num_workers: Number of workers for the dataloader. Defaults to 4.
        batch_size: Total batch size for training. If None, calculated from
            `dataloader_batch_size * gradient_accumulation_steps`.
        gradient_accumulation_steps: Number of gradient accumulation steps.
            Defaults to 1.
        dataloader_batch_size: Batch size used by the dataloader. If None, calculated
            from `batch_size // gradient_accumulation_steps`.
        prefetch_factor: Prefetch factor for the dataloader. If None, uses default.
        steps: Total number of training steps. Defaults to 100,000.
        log_freq: Frequency of logging in training iterations. Defaults to 200.
        save_checkpoint: Whether to save checkpoints during training. Defaults to True.
        save_freq: Frequency of checkpoint saving in training iterations. Checkpoints
            are saved every `save_freq` steps and after the last training step.
            Defaults to 20,000.
        use_policy_training_preset: If True, use optimizer and scheduler presets from
            the policy configuration. Defaults to False.
        optimizer: Configuration for the optimizer. Required if
            `use_policy_training_preset` is False.
        scheduler: Configuration for the learning rate scheduler. Required if
            `use_policy_training_preset` is False.
        wandb: Configuration for Weights & Biases logging. Defaults to WandBConfig().
        debug: If True, set logging level to DEBUG. Defaults to False.
        trace_nans: Enable anomaly detection for debugging NaN/Inf values.
            Warning: causes large computational overhead. Defaults to False.
        env: Optional environment configuration for evaluation. Defaults to None.
        eval: Configuration for evaluation settings. Defaults to EvalConfig().
        eval_freq: Frequency of evaluation in training steps. If 0, evaluation
            is disabled. Defaults to 0.
        last_checkpoint_only: If True, only evaluate the last checkpoint.
            Defaults to True.
    """

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
    max_state_dim: int = 32  # maximum dimension of the state vector
    max_action_dim: int = 32  # maximum dimension of the action vector
    action_chunk: int = 50  # size of action chunk
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
        """Initialize post-creation attributes and validate batch size configuration."""
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
        if self.job_name:
            warn(
                "cfg.job_name is deprecated and ignored. Set cfg.wandb.project and/or cfg.wandb.name instead."
            )

    def validate(self):
        """Validate and finalize the training configuration.

        This method performs several validation and setup tasks:
        - Loads policy configuration from CLI arguments or pretrained path if specified
        - Sets up checkpoint paths for resuming training
        - Validates output directory and creates default if needed
        - Sets up optimizer and scheduler from presets if enabled
        - Updates policy configuration with training parameters

        Raises:
            ValueError: If required configurations are missing or invalid.
            FileExistsError: If output directory exists and resume is False.
            NotADirectoryError: If config_path for resuming doesn't exist locally.
        """
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
        """Get list of field names that support path-based loading.

        This enables the parser to load config from the policy using
        `--policy.path=local/dir`.

        Returns:
            List of field names that support path-based configuration loading.
        """
        return ["policy"]

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the configuration to a directory.

        Args:
            save_directory: Directory path where the configuration will be saved.
        """
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
        """Load a training configuration from a pretrained model or local path.

        Args:
            cls: The class to instantiate.
            pretrained_name_or_path: Can be either:

                - A string, the model id of a pretrained config hosted inside a model
                  repo on huggingface.co.
                - A path to a directory containing a configuration file saved using
                  the `_save_pretrained` method.
                - A path to a saved configuration JSON file.
            force_download: Whether to force (re-)downloading the config files and
                configuration from the HuggingFace Hub. Defaults to False.
            resume_download: Whether to resume downloading the config files.
                Defaults to None.
            proxies: Dictionary of proxies to use for requests. Defaults to None.
            token: The token to use as HTTP bearer authorization. If True, will use
                the token generated when running `huggingface-cli login`. Defaults to None.
            cache_dir: Path to a directory in which a downloaded pretrained model
                configuration should be cached. Defaults to None.
            local_files_only: Whether to only look at local files (i.e., do not try
                to download the config). Defaults to False.
            revision: The specific model version to use. It can be a branch name, a
                tag name, or a commit id. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parser.

        Returns:
            An instance of TrainPipelineConfig loaded from the specified path.

        Raises:
            FileNotFoundError: If the configuration file is not found on the
                HuggingFace Hub or in the local path.
        """
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
