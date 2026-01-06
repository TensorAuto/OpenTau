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
"""Evaluate a policy on an environment by running rollouts and computing metrics."""

import concurrent.futures as cf
import datetime as dt
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import TypedDict

import einops
import gymnasium as gym
import imageio
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from termcolor import colored
from torch import nn
from tqdm import trange

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.envs.factory import make_envs
from opentau.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from opentau.policies.factory import make_policy
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.utils.accelerate_utils import acc_print, get_proc_accelerator, set_proc_accelerator
from opentau.utils.io_utils import write_video
from opentau.utils.libero_dataset_recorder import aggregate_task_results, consolidate_task_result
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    init_logging,
    inside_slurm,
    is_launched_with_accelerate,
)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    cfg: TrainPipelineConfig,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    acc = get_proc_accelerator()
    if acc is not None and not isinstance(policy, PreTrainedPolicy):
        policy = acc.unwrap_model(policy)

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    successes = np.zeros((env.num_envs,), dtype=bool)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to OpenTau policy format.
        observation = preprocess_observation(observation, cfg=cfg)
        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        if return_observations:
            all_observations.append(deepcopy(observation))

        with torch.inference_mode():
            action = policy.select_action(observation)

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # Once a success, always a success.
        if "is_success" in info:
            successes = successes | info["is_success"].astype(bool)

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation, cfg=cfg)
        observation = add_envs_task(env, observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key, value0 in all_observations[0].items():
            if isinstance(value0, torch.Tensor):
                stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
            elif isinstance(value0, list):
                stacked_observations[key] = list(zip(*[obs[key] for obs in all_observations], strict=True))
            else:
                raise TypeError(
                    f"Unsupported observation type for key {key}: {type(value0)}. "
                    "Only `torch.Tensor` and `list` are supported for now."
                )
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    n_episodes: int,
    cfg: TrainPipelineConfig,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    grid_size: tuple[int, int] | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        cfg: The training config.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
        grid_size: The grid size to use for rendering concatenated rollouts.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    all_done_indices = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []
        rendered_successes: list[bool] = []

    if return_episode_data:
        episode_data: dict[str, list | torch.Tensor] | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            # HACK: to get different seeds per accelerator process when using distributed eval.
            acc = get_proc_accelerator()
            acc_offset = acc.process_index * 10000 if acc else 0
            seeds = range(
                start_seed + acc_offset + (batch_ix * env.num_envs),
                start_seed + acc_offset + ((batch_ix + 1) * env.num_envs),
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            cfg=cfg,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )
        if return_episode_data:
            if not episode_data:
                episode_data = deepcopy(rollout_data)
            else:
                for key, value in rollout_data.items():
                    if isinstance(value, torch.Tensor):
                        episode_data[key] = torch.cat([episode_data[key], value], dim=0)
                    elif isinstance(value, list):
                        episode_data[key].extend(value)

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        all_done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(all_done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index, success in zip(
                batch_stacked_frames,
                all_done_indices.flatten().tolist(),
                batch_successes.tolist(),
                strict=False,
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                rendered_successes.append(success)
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Create grid summary video if we have videos to render
    if max_episodes_rendered > 0 and len(video_paths) > 0:
        try:
            grid_summary_path = videos_dir / "grid_summary.mp4"
            create_grid_summary_video(
                video_paths=video_paths,
                success_statuses=rendered_successes,
                output_path=str(grid_summary_path),
                fps=env.unwrapped.metadata["render_fps"],
                highlight_duration=2.0,
                grid_size=grid_size,
            )
            logging.info(f"Grid summary video created: {grid_summary_path}")
        except Exception as e:
            logging.error(f"Failed to create grid summary video: {e}")

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
                "done_index": done_index,
            }
            for i, (sum_reward, max_reward, success, seed, done_index) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    all_done_indices[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def create_grid_summary_video(
    video_paths: list[str],
    success_statuses: list[bool],
    output_path: str,
    fps: float,
    highlight_duration: float = 1.0,
    grid_size: tuple[int, int] | None = None,
) -> None:
    """Create a grid summary video from individual episode videos.

    Args:
        video_paths: List of paths to individual video files
        success_statuses: List of boolean success statuses for each video
        output_path: Path where the summary video will be saved
        fps: Frames per second for the output video
        highlight_duration: Duration in seconds to show the highlighting at the end
        grid_size: Tuple of (rows, cols) for the grid. If None, will be auto-calculated as square grid.
    """
    if len(video_paths) != len(success_statuses):
        raise ValueError(
            f"Number of videos ({len(video_paths)}) must match number of success statuses ({len(success_statuses)})"
        )

    # Auto-calculate grid size if not provided
    if grid_size is None:
        # Calculate square grid size
        n_videos = len(video_paths)
        grid_rows = int(np.ceil(np.sqrt(n_videos)))
        grid_cols = int(np.ceil(n_videos / grid_rows))
        grid_size = (grid_rows, grid_cols)

    grid_rows, grid_cols = grid_size
    expected_videos = grid_rows * grid_cols

    if len(video_paths) > expected_videos:
        raise ValueError(
            f"Too many videos ({len(video_paths)}) for grid size {grid_size} (max {expected_videos})"
        )

    # Load all videos
    videos = []
    max_frames = 0
    for video_path in video_paths:
        if not Path(video_path).exists():
            logging.warning(f"Video file not found: {video_path}")
            continue
        video = imageio.mimread(video_path)
        videos.append(video)
        max_frames = max(max_frames, len(video))

    if not videos:
        logging.error("No valid videos found to create grid summary")
        return

    # Get dimensions from first video
    frame_height, frame_width = videos[0][0].shape[:2]
    grid_width = frame_width * grid_cols
    grid_height = frame_height * grid_rows

    # Create grid frames
    grid_frames = []

    for frame_idx in range(max_frames):
        # Create empty grid frame
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Fill grid with video frames
        for i, video in enumerate(videos):
            row = i // grid_cols
            col = i % grid_cols

            # Use last frame if video is shorter
            frame_to_use = min(frame_idx, len(video) - 1)
            frame = video[frame_to_use]

            # Ensure frame is RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                y_start = row * frame_height
                y_end = y_start + frame_height
                x_start = col * frame_width
                x_end = x_start + frame_width
                grid_frame[y_start:y_end, x_start:x_end] = frame

        grid_frames.append(grid_frame)

    # Add highlighting frames at the end
    highlight_frames = int(highlight_duration * fps)
    for _ in range(highlight_frames):
        # Create highlighted version of the last frame
        highlighted_frame = grid_frame.copy()

        for i, success in enumerate(success_statuses):
            row = i // grid_cols
            col = i % grid_cols

            y_start = row * frame_height
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width

            # Create colored overlay
            color = np.array([0, 255, 0]) if success else np.array([255, 0, 0])  # Green or Red
            overlay = np.full((frame_height, frame_width, 3), color, dtype=np.uint8)

            # Blend with original frame (50% opacity)
            highlighted_frame[y_start:y_end, x_start:x_end] = (
                0.5 * highlighted_frame[y_start:y_end, x_start:x_end] + 0.5 * overlay
            ).astype(np.uint8)

        grid_frames.append(highlighted_frame)

    # Save the grid video
    imageio.mimsave(output_path, grid_frames, fps=fps)
    logging.info(f"Grid summary video saved to: {output_path}")


@parser.wrap()
def eval_main(cfg: TrainPipelineConfig):
    accelerator = Accelerator()
    set_proc_accelerator(accelerator)

    init_logging(accelerator=accelerator)
    logging.info(pformat(asdict(cfg)))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    details = f"{cfg.env.type}-{cfg.env.task}-{cfg.eval.n_episodes}"
    now = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    eval_output_dir = Path(cfg.output_dir) / "post-training-eval" / f"{details}-{now}"

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {eval_output_dir}")

    logging.info("Making environment.")
    envs = make_envs(cfg.env, cfg, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(cfg=cfg.policy)
    policy.to(torch.bfloat16)
    policy = accelerator.prepare(policy)
    policy.eval()
    with (
        torch.no_grad(),
        torch.autocast(device_type=accelerator.device.type) if cfg.policy.use_amp else nullcontext(),
    ):
        eval_info = eval_policy_all(
            envs=envs,
            policy=policy,
            n_episodes=cfg.eval.n_episodes,
            cfg=cfg,
            max_episodes_rendered=10,
            videos_dir=eval_output_dir / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            return_episode_data=bool(cfg.eval.recording_root),
        )

        acc_print("Local Eval Info", eval_info)
        eval_info = gather_object([eval_info])

        if accelerator.is_main_process:
            eval_info = consolidate_eval_info(eval_info)
            with open(eval_output_dir / "eval_info.json", "w") as f:
                json.dump(eval_info, f, indent=2)
            print("Overall Aggregated Metrics:")
            print(eval_info["overall"])
            for task_group, task_group_info in eval_info["per_group"].items():
                print(f"\nAggregated Metrics for {task_group}:")
                print(task_group_info)

    # Close all vec envs
    close_envs(envs)
    accelerator.end_training()

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    n_episodes: int,
    cfg: TrainPipelineConfig,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> tuple[TaskMetrics, dict]:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        n_episodes=n_episodes,
        cfg=cfg,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    ), task_result


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    n_episodes: int,
    cfg: TrainPipelineConfig,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> tuple[str, int, TaskMetrics, dict]:
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        acc = get_proc_accelerator()
        if acc is None:
            task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        else:
            task_videos_dir = videos_dir / f"{task_group}_{task_id}_rank{acc.local_process_index}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics, task_result = eval_one(
        env,
        policy=policy,
        n_episodes=n_episodes,
        cfg=cfg,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics, task_result


# compute aggregated metrics helper (robust to lists/scalars)
def _agg_from_list(xs):
    if not xs:
        return float("nan")
    arr = np.array(list(xs), dtype=float)
    return float(np.nanmean(arr))


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    n_episodes: int,
    cfg: TrainPipelineConfig,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        n_episodes=n_episodes,
        cfg=cfg,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    task_results = []
    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics, tres = task_runner(task_group, task_id, env)
            task_results.append(tres)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics, tres = fut.result()
                task_results.append(tres)
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    if cfg.eval.recording_root is not None:
        acc = get_proc_accelerator()
        acc_rank = acc.local_process_index if acc else 0
        recording_dir = Path(cfg.eval.recording_root) / f"rank{acc_rank}"
        logging.info(f"Consolidating Libero dataset to {recording_dir}...")
        consolidate_task_result(
            aggregate_task_results(task_results),
            output_dir=recording_dir,
            allow_overwrite=True,
        )

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def consolidate_eval_info(eval_infos: list[dict]) -> dict:
    n_gpu_procs = len(eval_infos)
    per_tasks = [per_task for einfo in eval_infos for per_task in einfo["per_task"]]
    per_tasks.sort(key=lambda x: (x["task_group"], x["task_id"]))

    per_groups = {}
    for group in {t["task_group"] for t in per_tasks}:
        group_tasks = [t for t in per_tasks if t["task_group"] == group]
        per_groups[group] = {
            "avg_sum_reward": _agg_from_list(r for t in group_tasks for r in t["metrics"]["sum_rewards"]),
            "avg_max_reward": _agg_from_list(r for t in group_tasks for r in t["metrics"]["max_rewards"]),
            "pc_success": _agg_from_list(s for t in group_tasks for s in t["metrics"]["successes"]) * 100,
            "n_episodes": sum(1 for t in group_tasks for _ in t["metrics"]["successes"]),
            "video_paths": [p for t in group_tasks for p in t["metrics"].get("video_paths", [])],
        }

    total_time = sum(einfo["overall"]["eval_s"] for einfo in eval_infos if "overall" in einfo)
    n_episodes = sum(1 for t in per_tasks for _ in t["metrics"]["successes"])
    overall = {
        "avg_sum_reward": _agg_from_list(r for t in per_tasks for r in t["metrics"]["sum_rewards"]),
        "avg_max_reward": _agg_from_list(r for t in per_tasks for r in t["metrics"]["max_rewards"]),
        "pc_success": _agg_from_list(s for t in per_tasks for s in t["metrics"]["successes"]) * 100,
        "n_episodes": n_episodes,
        "video_paths": [p for t in per_tasks for p in t["metrics"].get("video_paths", [])],
        "eval_per_gpu_s": total_time / n_gpu_procs,
        "eval_ep_s": total_time / n_episodes,
    }
    return {
        "per_task": per_tasks,
        "per_group": per_groups,
        "overall": overall,
    }


def main():
    eval_main()


if __name__ == "__main__":
    if not is_launched_with_accelerate():
        raise Exception(
            "This script should be launched with accelerate. Please use `accelerate launch` to run this script."
        )
    main()
