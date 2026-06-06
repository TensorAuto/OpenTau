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
from opentau.configs.deployment import ServerConfig
from opentau.configs.policies import (
    PreTrainedConfig,
    load_resolved_config_dict,
    strip_deprecated_fields_from_json,
    warn_deprecated_latency_fields_from_dict,
    warn_removed_policy_fields_from_dict,
    write_stripped_config_to_tempfile,
)
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
        running_best_count: Number of "running best" checkpoints to keep, i.e. the
            checkpoint(s) that achieved the best metric so far. If 0, the feature is disabled
            (like `eval_freq`/`val_freq`). When >= 1, after an eval/validation step that hits a
            new best the current weights are saved (or, if that step already wrote a regular
            `save_freq` checkpoint, that directory is reused); older running-bests are evicted
            down to `running_best_count`, but one that coincided with a regular checkpoint is
            left to the normal retention logic. Running-bests are protected from
            `last_checkpoint_only` pruning, and work even when `save_checkpoint` is False (only
            the best-performing checkpoints are then written, so resume granularity is limited
            to running-best steps). Defaults to 0.
        running_best_metric: Which metric drives the running best. One of "auto", "eval_success"
            (maximize sim-eval success rate), or "val_loss" (minimize validation loss). "auto"
            resolves to "eval_success" when sim eval is configured (env set and eval_freq > 0),
            otherwise "val_loss". Only used when `running_best_count` >= 1. Defaults to "auto".
        server: Configuration for the gRPC inference server. Defaults to ServerConfig().
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
    val_freq: int = 0  # validate every val_freq steps, if 0, then a validation split is not created
    last_checkpoint_only: bool = True
    # Keep the best-performing checkpoint(s) ("running best") in addition to the regular ones.
    # 0 disables the feature (cf. eval_freq/val_freq); >= 1 keeps that many running bests.
    running_best_count: int = 0
    running_best_metric: str = "auto"  # "auto" | "eval_success" | "val_loss"
    # gRPC inference server configuration
    server: ServerConfig = field(default_factory=ServerConfig)

    def __post_init__(self):
        """Initialize post-creation attributes and validate batch size configuration."""
        self.checkpoint_path = None
        # Runtime-only resolution of ``running_best_metric`` ("auto" -> concrete metric),
        # set in ``validate()``. Deliberately not a dataclass field so it is never serialized
        # (a resumed "auto" run must stay "auto", not freeze to whatever it resolved to once).
        self.running_best_metric_resolved = None

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

            # The policy's ``n_obs_steps`` determines the T dimension its
            # encoder expects; the dataset_mixture's ``n_obs_history`` is
            # what the dataloader actually produces. They must agree.
            if self.dataset_mixture is not None:
                dm = self.dataset_mixture
                dm_n_obs = dm.n_obs_history if dm.n_obs_history is not None else 1
                if self.policy.n_obs_steps != dm_n_obs:
                    raise ValueError(
                        f"policy.n_obs_steps ({self.policy.n_obs_steps}) != "
                        f"dataset_mixture.n_obs_history ({dm.n_obs_history}; "
                        "treated as 1 when unset). Set dataset_mixture.n_obs_history "
                        "to match policy.n_obs_steps."
                    )

        self._validate_running_best()

    def _validate_running_best(self):
        """Validate the running-best checkpoint config and resolve the driving metric.

        The feature is enabled by ``running_best_count`` >= 1 (0 disables it, like
        ``eval_freq``/``val_freq``). Sets ``self.running_best_metric_resolved`` to a concrete
        metric ("eval_success" or "val_loss") when enabled. The ``running_best_metric`` /
        ``running_best_count`` bounds are checked unconditionally so a bad value is caught even
        when the feature is off.

        Raises:
            ValueError: If ``running_best_count`` < 0, ``running_best_metric`` is not one of the
                allowed values, or the feature is enabled without a usable metric source.
        """
        allowed = {"auto", "eval_success", "val_loss"}
        if self.running_best_metric not in allowed:
            raise ValueError(
                f"running_best_metric must be one of {sorted(allowed)}, got {self.running_best_metric!r}."
            )
        if self.running_best_count < 0:
            raise ValueError(f"running_best_count must be >= 0, got {self.running_best_count}.")

        if self.running_best_count == 0:
            return

        has_eval = self.env is not None and self.eval_freq > 0
        has_val = self.val_freq > 0
        if self.running_best_metric == "eval_success" and not has_eval:
            raise ValueError(
                "running_best_count >= 1 with running_best_metric='eval_success' requires a sim eval "
                "source: set env and eval_freq > 0."
            )
        if self.running_best_metric == "val_loss" and not has_val:
            raise ValueError(
                "running_best_count >= 1 with running_best_metric='val_loss' requires val_freq > 0."
            )
        if self.running_best_metric == "auto" and not (has_eval or has_val):
            raise ValueError(
                "running_best_count >= 1 but no metric source is configured. Set eval_freq > 0 "
                "with an env (for eval success rate), or val_freq > 0 (for validation loss)."
            )
        # Resolve "auto": prefer sim eval when available, else validation loss.
        if self.running_best_metric == "auto":
            self.running_best_metric_resolved = "eval_success" if has_eval else "val_loss"
        else:
            self.running_best_metric_resolved = self.running_best_metric

        if not self.save_checkpoint:
            warn(
                "running_best_count >= 1 but save_checkpoint is False; only running-best "
                "checkpoints will be saved, so resume granularity is limited to running-best steps."
            )

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
        config_path = save_directory / TRAIN_CONFIG_NAME
        with open(config_path, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)
        strip_deprecated_fields_from_json(config_path)

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

        if config_file is None:
            return draccus.parse(cls, config_file, args=cli_args)

        # Resolve $refs once and reuse — the warn helpers and the strip step
        # would otherwise each walk the full ref tree from disk.
        config_data = load_resolved_config_dict(config_file)
        warn_deprecated_latency_fields_from_dict(config_data, config_file)
        warn_removed_policy_fields_from_dict(config_data, config_file)
        # Strip deprecated/removed keys via a temp file rather than mutating the
        # source — `config_file` may be an HF cache symlink to a content-addressed
        # blob, where a rewrite would silently corrupt the cache.
        tmp_config = write_stripped_config_to_tempfile(config_data)
        try:
            return draccus.parse(cls, str(tmp_config), args=cli_args)
        finally:
            tmp_config.unlink(missing_ok=True)
