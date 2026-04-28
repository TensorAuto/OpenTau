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
import json
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import nullcontext
from pprint import pformat
from typing import Any

import accelerate
import torch
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils import DistributedDataParallelKwargs, gather_object
from termcolor import colored

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.utils import cycle
from opentau.envs.factory import make_envs
from opentau.envs.utils import close_envs
from opentau.optim.factory import make_optimizer_and_scheduler
from opentau.policies.factory import make_policy
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.scripts.eval import consolidate_eval_info, eval_policy_all
from opentau.utils.accelerate_utils import set_proc_accelerator
from opentau.utils.logging_utils import AverageMeter, MetricsTracker
from opentau.utils.random_utils import set_seed
from opentau.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    load_training_step,
    prune_old_checkpoints,
    save_checkpoint,
)
from opentau.utils.utils import (
    encode_accelerator_state_dict,
    format_big_number,
    init_logging,
    is_launched_with_accelerate,
)


def update_policy(
    train_config: TrainPipelineConfig,
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: AcceleratedOptimizer,
    grad_clip_norm: float,
    accelerator: accelerate.Accelerator,
    lr_scheduler: AcceleratedScheduler | None = None,
) -> tuple[MetricsTracker, dict]:
    policy.train()
    losses = policy.forward(batch)
    loss = (
        train_config.loss_weighting["MSE"] * losses["MSE"] + train_config.loss_weighting["CE"] * losses["CE"]
    )

    accelerator.backward(loss)
    accelerator.unscale_gradients(optimizer=optimizer)

    if accelerator.sync_gradients:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        if accelerator.is_main_process:
            train_metrics.grad_norm = grad_norm

    optimizer.step()
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # This calls `torch.distributed.all_gather_into_tensor` under the hood, which is not so efficient.
    # We don't actually want to broadcast the gathered tensors to all processes, but only to the main process.
    # Nonetheless, we still do this for correctness, safety, and simplicity.
    _first_loss_tensor = next(lt for lt in losses.values() if isinstance(lt, torch.Tensor))
    zero = torch.tensor(0.0, device=_first_loss_tensor.device, dtype=_first_loss_tensor.dtype)
    loss = accelerator.gather_for_metrics(loss).mean().item()
    mse_loss = accelerator.gather_for_metrics(losses["MSE"]).to(dtype=torch.float32).mean().item()
    ce_loss = accelerator.gather_for_metrics(losses["CE"]).to(dtype=torch.float32).mean().item()
    l1_loss = accelerator.gather_for_metrics(losses.get("L1", zero)).to(dtype=torch.float32).mean().item()
    accuracy = (
        accelerator.gather_for_metrics(losses.get("Accuracy", zero)).to(dtype=torch.float32).mean().item()
    )
    # This actually calls `.update` method of the `AverageMeter` class. This operation is not idempotent.
    # See MetricsTracker.__setattr__ for more details.
    # In other words, setting `train_metrics.loss = 1` and `train_metrics.loss = 2` consecutively results in
    #   an average of 1.5 when formatted as a string, not just 2.
    if accelerator.is_main_process:
        train_metrics.loss = loss
        train_metrics.mse_loss = mse_loss
        train_metrics.ce_loss = ce_loss
        train_metrics.l1_loss = l1_loss
        train_metrics.accuracy = accuracy
        train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics


_VAL_METRIC_KEYS: tuple[str, ...] = ("loss", "mse_loss", "ce_loss", "l1_loss", "accuracy")


def _mixture_weighted_aggregate(
    per_dataset_trackers: dict[str, MetricsTracker],
    name_to_weight: dict[str, float],
    metric_keys: tuple[str, ...] = _VAL_METRIC_KEYS,
) -> dict[str, float]:
    """Mixture-weighted average of per-dataset validation metrics.

    Weights are taken from ``name_to_weight`` and renormalized over only the
    names present in ``per_dataset_trackers`` (empty datasets are skipped
    upstream by ``WeightedDatasetMixture.get_per_dataset_dataloaders`` and so
    will be missing from the trackers). When the renormalization total is 0
    -- empty trackers, or all selected datasets have weight 0 -- every metric
    is returned as ``0.0``.

    Args:
        per_dataset_trackers: One ``MetricsTracker`` per non-empty validation
            dataset, keyed by dataset name.
        name_to_weight: Mapping from dataset name to its mixture weight (need
            not be normalized; need not be a strict subset/superset of the
            tracker keys, but must contain every tracker key).
        metric_keys: The metric attribute names to aggregate.

    Returns:
        Dict mapping each ``metric_keys`` entry to its weighted average.
    """
    weights = {name: name_to_weight[name] for name in per_dataset_trackers}
    total = sum(weights.values())
    if total <= 0:
        return dict.fromkeys(metric_keys, 0.0)

    per_dataset_dicts = {
        name: tracker.to_dict(use_avg=True) for name, tracker in per_dataset_trackers.items()
    }
    return {
        k: sum((w / total) * per_dataset_dicts[name][k] for name, w in weights.items()) for k in metric_keys
    }


def _find_unused_params_from_env() -> bool:
    """Parse the ``FIND_UNUSED_PARAMS`` env var into a bool.

    Under DDP this controls whether the reducer walks the autograd graph
    after each backward to discover parameters that did not receive a
    gradient. It is silently ignored under DeepSpeed. Default is True
    for safety (policies with config-gated heads can produce unused
    params); set ``FIND_UNUSED_PARAMS=false`` once a run has been
    audited with ``scripts/find_unused_params.py`` to reclaim the
    per-step graph-walk cost (~10-15% of step time on pi05).

    Returns:
        bool: True when the env var is unset or equals ``"true"``
        (case-insensitive); False for any other value.
    """
    return os.environ.get("FIND_UNUSED_PARAMS", "true").lower() == "true"


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    find_unused = _find_unused_params_from_env()
    accelerator_kwargs = {
        "step_scheduler_with_optimizer": False,
        "split_batches": False,  # split_batches == True is not working anyways
        "kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=find_unused)],
    }
    if cfg.wandb.enable:
        accelerator_kwargs["log_with"] = "wandb"
    if cfg.gradient_accumulation_steps > 1:
        accelerator_kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps

    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    init_logging(accelerator, level=logging.DEBUG if cfg.debug else logging.INFO)
    # Register accelerator globally for use in other modules, (e.g., detect current rank, etc.)
    set_proc_accelerator(accelerator)

    logging.info(pformat(cfg.to_dict()))

    if accelerator.is_main_process:
        accelerator_config = encode_accelerator_state_dict(accelerator.state.__dict__)
        logging.info(pformat(accelerator_config))

        # Ensure `gradient_accumulation_steps` is consistent between TrainPipelineConfig and DeepSpeedConfig
        if accelerator.distributed_type == accelerate.DistributedType.DEEPSPEED:
            deepspeed_config, deepspeed_key = accelerator.deepspeed_plugin.hf_ds_config.find_config_node(
                "gradient_accumulation_steps"
            )
            ds_grad_acc_steps = deepspeed_config.get(deepspeed_key, 1)
            if ds_grad_acc_steps != cfg.gradient_accumulation_steps:
                raise ValueError(
                    "The `gradient_accumulation_steps` in TrainPipelineConfig does not match the value "
                    f"specified in DeepSpeedConfig {cfg.gradient_accumulation_steps} != {ds_grad_acc_steps}. "  # nosec B608
                )

        if cfg.wandb.enable:
            step = load_training_step(cfg.checkpoint_path) if cfg.resume else None
            slurm_dict = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
            accelerator.init_trackers(
                cfg.wandb.project,
                config={**cfg.to_dict(), "accelerator": accelerator_config, "slurm": slurm_dict},
                init_kwargs={"wandb": cfg.wandb.to_wandb_kwargs(step=step)},
            )
            tracker = accelerator.get_tracker("wandb", unwrap=True)
            cfg.wandb.run_id = tracker.id
            logging.info(f"tracker initialized with wandb job id: {tracker.id}")

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Enable anomaly detection for debugging NaN/Inf values
    # (warning: large computational overhead)
    torch.autograd.set_detect_anomaly(cfg.trace_nans)
    if cfg.trace_nans:
        logging.warning("Anomaly detection is enabled. This may significantly slow down training.")
    else:
        logging.info("Anomaly detection is disabled.")

    logging.info("Creating dataset")
    if cfg.val_freq > 0:
        train_dataset, val_dataset = make_dataset_mixture(cfg)
    else:
        train_dataset = make_dataset_mixture(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    eval_envs = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_envs = make_envs(
            cfg.env, cfg, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
    policy.to(torch.bfloat16)
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if cfg.val_freq > 0:
        train_dataloader = train_dataset.get_dataloader()
        # One DataLoader per underlying val dataset so we can report per-dataset
        # validation losses. The aggregate is computed by averaging across all.
        per_dataset_val_dataloaders = val_dataset.get_per_dataset_dataloaders()
        val_names = list(per_dataset_val_dataloaders.keys())
        prepared = accelerator.prepare(
            policy,
            optimizer,
            train_dataloader,
            lr_scheduler,
            *per_dataset_val_dataloaders.values(),
        )
        policy, optimizer, train_dataloader, lr_scheduler = prepared[:4]
        per_dataset_val_dataloaders = dict(zip(val_names, prepared[4:], strict=True))
    else:
        train_dataloader = train_dataset.get_dataloader()
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
    train_dl_iter = cycle(train_dataloader)

    # Register the LR scheduler for checkpointing
    accelerator.register_for_checkpointing(lr_scheduler)

    if cfg.resume:
        # load accelerator state
        # This will load the model, optimizer, and lr_scheduler state
        accelerator.load_state(cfg.checkpoint_path)

        # all processes should load the step & rng states
        step = load_training_state(cfg.checkpoint_path)
        logging.info(f"Resuming training from checkpoint {cfg.checkpoint_path}")

    policy.train()

    # setup metrics tracker to average metrics over the logging interval
    train_metrics = {
        "loss": AverageMeter("total_loss", ":.3f"),
        "mse_loss": AverageMeter("mse_loss", ":.3f"),
        "ce_loss": AverageMeter("ce_loss", ":.3f"),
        "l1_loss": AverageMeter("l1_loss", ":.3f"),
        "accuracy": AverageMeter("accuracy", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "grad_norm": AverageMeter("grad_norm", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes,  # split_batches are not working
        train_metrics,
        initial_step=step,
    )

    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        for _ in range(cfg.gradient_accumulation_steps):
            with accelerator.accumulate(policy) if cfg.gradient_accumulation_steps > 1 else nullcontext():
                logging.debug(f"{step=}, {accelerator.sync_gradients=}")
                batch = next(train_dl_iter)

                train_tracker = update_policy(
                    cfg,
                    train_tracker,
                    policy,
                    batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = (step % cfg.save_freq == 0 or step == cfg.steps) and cfg.save_checkpoint
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_val_step = cfg.val_freq > 0 and step % cfg.val_freq == 0

        # Only `train_tracker` on the main process keeps useful statistics,
        #  because we guarded it with if accelerator.is_main_process in the `update_policy` function.
        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            log_dict = train_tracker.to_dict(use_avg=True)
            accelerator.log({"Training/Loss": log_dict["loss"]}, step=step)
            accelerator.log({"Training/MSE Loss": log_dict["mse_loss"]}, step=step)
            accelerator.log({"Training/CE Loss": log_dict["ce_loss"]}, step=step)
            accelerator.log({"Training/L1 Loss": log_dict["l1_loss"]}, step=step)
            accelerator.log({"Training/Accuracy": log_dict["accuracy"]}, step=step)
            accelerator.log({"Training/Learning Rate": log_dict["lr"]}, step=step)
            accelerator.log({"Training/Grad Norm": log_dict["grad_norm"]}, step=step)
            accelerator.log({"Training/Num Samples": log_dict["samples"]}, step=step)
            train_tracker.reset_averages()

        if is_saving_step:
            # TODO: investigate whether this barrier is needed
            accelerator.wait_for_everyone()
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

            # save the accelerator state
            # This will save the model, optimizer, and lr_scheduler state
            accelerator.save_state(checkpoint_dir)

            # save axillary objects such as configs, training step, and rng state
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                cfg.policy.pretrained_path = checkpoint_dir
                save_checkpoint(checkpoint_dir, step, cfg)
                if cfg.last_checkpoint_only:
                    prune_old_checkpoints(checkpoint_dir)

            accelerator.wait_for_everyone()

        if is_val_step:
            policy.eval()

            def _make_val_tracker(current_step: int = step) -> MetricsTracker:
                return MetricsTracker(
                    cfg.batch_size * accelerator.num_processes,
                    {
                        "loss": AverageMeter("val_total_loss", ":.3f"),
                        "mse_loss": AverageMeter("val_mse_loss", ":.3f"),
                        "ce_loss": AverageMeter("val_ce_loss", ":.3f"),
                        "l1_loss": AverageMeter("val_l1_loss", ":.3f"),
                        "accuracy": AverageMeter("val_accuracy", ":.3f"),
                    },
                    initial_step=current_step,
                )

            per_dataset_trackers: dict[str, MetricsTracker] = {
                name: _make_val_tracker() for name in per_dataset_val_dataloaders
            }

            logging.info(f"Validation at step {step}...")

            with torch.no_grad():
                for ds_name, ds_loader in per_dataset_val_dataloaders.items():
                    ds_tracker = per_dataset_trackers[ds_name]
                    for batch in ds_loader:
                        losses = policy.forward(batch)
                        loss = (
                            cfg.loss_weighting["MSE"] * losses["MSE"]
                            + cfg.loss_weighting["CE"] * losses["CE"]
                        )

                        # Gather and average metrics across processes
                        _first_loss_tensor = next(
                            lt for lt in losses.values() if isinstance(lt, torch.Tensor)
                        )
                        zero = torch.tensor(
                            0.0, device=_first_loss_tensor.device, dtype=_first_loss_tensor.dtype
                        )

                        loss = accelerator.gather_for_metrics(loss).mean().item()
                        mse_loss = (
                            accelerator.gather_for_metrics(losses["MSE"])
                            .to(dtype=torch.float32)
                            .mean()
                            .item()
                        )
                        ce_loss = (
                            accelerator.gather_for_metrics(losses["CE"]).to(dtype=torch.float32).mean().item()
                        )
                        l1_loss = (
                            accelerator.gather_for_metrics(losses.get("L1", zero))
                            .to(dtype=torch.float32)
                            .mean()
                            .item()
                        )
                        accuracy = (
                            accelerator.gather_for_metrics(losses.get("Accuracy", zero))
                            .to(dtype=torch.float32)
                            .mean()
                            .item()
                        )

                        if accelerator.is_main_process:
                            ds_tracker.loss = loss
                            ds_tracker.mse_loss = mse_loss
                            ds_tracker.ce_loss = ce_loss
                            ds_tracker.l1_loss = l1_loss
                            ds_tracker.accuracy = accuracy

            if accelerator.is_main_process:
                for ds_name, ds_tracker in per_dataset_trackers.items():
                    logging.info(f"Validation/{ds_name} {ds_tracker}")
                    ds_dict = ds_tracker.to_dict(use_avg=True)
                    accelerator.log({f"Validation/{ds_name}/Loss": ds_dict["loss"]}, step=step)
                    accelerator.log({f"Validation/{ds_name}/MSE Loss": ds_dict["mse_loss"]}, step=step)
                    accelerator.log({f"Validation/{ds_name}/CE Loss": ds_dict["ce_loss"]}, step=step)
                    accelerator.log({f"Validation/{ds_name}/L1 Loss": ds_dict["l1_loss"]}, step=step)
                    accelerator.log({f"Validation/{ds_name}/Accuracy": ds_dict["accuracy"]}, step=step)

                # Mixture-weighted aggregate across the per-dataset trackers, so the
                # overall scalar reflects the training mixture rather than being
                # implicitly dominated by whichever val subset has the most batches.
                name_to_weight = dict(
                    zip(val_dataset.dataset_names, val_dataset.dataset_weights, strict=True)
                )
                agg = _mixture_weighted_aggregate(per_dataset_trackers, name_to_weight)
                logging.info(f"Validation/aggregate {agg}")
                accelerator.log({"Validation/Loss": agg["loss"]}, step=step)
                accelerator.log({"Validation/MSE Loss": agg["mse_loss"]}, step=step)
                accelerator.log({"Validation/CE Loss": agg["ce_loss"]}, step=step)
                accelerator.log({"Validation/L1 Loss": agg["l1_loss"]}, step=step)
                accelerator.log({"Validation/Accuracy": agg["accuracy"]}, step=step)

            # This barrier is probably necessary to ensure
            # other processes wait for the main process to finish saving
            accelerator.wait_for_everyone()

        if is_eval_step and eval_envs:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=accelerator.device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy_all(
                    eval_envs,
                    policy,
                    cfg.eval.n_episodes,
                    cfg,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=cfg.eval.max_episodes_rendered,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )

            eval_info = gather_object([eval_info])  # gather across all accelerator processes
            if accelerator.is_main_process:
                eval_info = consolidate_eval_info(eval_info)
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_per_gpu_s": AverageMeter("eval_per_gpu_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    eval_metrics,
                    initial_step=step,
                )
                eval_tracker.eval_per_gpu_s = aggregated.get("eval_per_gpu_s", float("nan"))
                eval_tracker.avg_sum_reward = aggregated.get("avg_sum_reward", float("nan"))
                eval_tracker.pc_success = aggregated.get("pc_success", float("nan"))
                logging.info(eval_tracker)
                eval_dict = eval_tracker.to_dict(use_avg=True)
                accelerator.log({"Success Rate": eval_dict["pc_success"]}, step=step)
                accelerator.log({"Evaluation Time": eval_dict["eval_per_gpu_s"]}, step=step)
                for group, v in eval_info["per_group"].items():
                    accelerator.log({f"Success/{group}": v["pc_success"]}, step=step)

                # Save eval_info to the same directory as videos
                videos_dir = cfg.output_dir / "eval" / f"videos_step_{step_id}"
                with open(videos_dir / "eval_info.json", "w") as f:
                    json.dump(eval_info, f, indent=2)

            # This barrier is to ensure all processes finishes evaluation before the next training step
            # Some processes might be slower than others
            accelerator.wait_for_everyone()

    if cfg.eval_freq > 0 and eval_envs:
        close_envs(eval_envs)

    accelerator.end_training()
    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    if not is_launched_with_accelerate():
        raise Exception(
            "This script should be launched with accelerate. Please use `accelerate launch` to run this script."
        )

    train()
