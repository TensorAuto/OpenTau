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

    # accelerator.backward(loss)
    # accelerator.unscale_gradients(optimizer=optimizer)

    # if accelerator.sync_gradients:
    #     grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    #     if accelerator.is_main_process:
    #         train_metrics.grad_norm = grad_norm

    # optimizer.step()
    # optimizer.zero_grad()

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


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    accelerator_kwargs = {
        "step_scheduler_with_optimizer": False,
        "split_batches": False,  # split_batches == True is not working anyways
        "kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=True)],
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
    dataset = make_dataset_mixture(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    eval_envs = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_envs = make_envs(
            cfg.env, cfg, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
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

    dataloader = dataset.get_dataloader()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

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
                batch = next(dl_iter)

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

        # Only `train_tracker` on the main process keeps useful statistics,
        #  because we guarded it with if accelerator.is_main_process in the `update_policy` function.
        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            log_dict = train_tracker.to_dict(use_avg=True)
            accelerator.log({"Training Loss": log_dict["loss"]}, step=step)
            accelerator.log({"MSE Loss": log_dict["mse_loss"]}, step=step)
            accelerator.log({"CE Loss": log_dict["ce_loss"]}, step=step)
            accelerator.log({"L1 Loss": log_dict["l1_loss"]}, step=step)
            accelerator.log({"Accuracy": log_dict["accuracy"]}, step=step)
            accelerator.log({"Learning Rate": log_dict["lr"]}, step=step)
            accelerator.log({"Grad Norm": log_dict["grad_norm"]}, step=step)
            accelerator.log({"Num Samples": log_dict["samples"]}, step=step)
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

        if is_eval_step:
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
