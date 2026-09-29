# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Shared history packing for training and synchronous inference."""

from collections import deque
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..models.positional import batched_prc_action, batched_prc_vid, times_to_ids
from . import normalization, packing

if TYPE_CHECKING:
    from ..config import PolicyConfig


class ObservationHistory:
    """Record each control tick, including ticks served from the action queue."""

    def __init__(self, length: int):
        self.observations = deque(maxlen=length)
        self.commands = deque(maxlen=length)

    def append(self, batch: dict, config: "PolicyConfig", last_command: Tensor | None) -> dict:
        state = batch["state"]
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        if state.ndim != 2 or state.shape[-1] != config.action_dim:
            raise ValueError("select_action needs one measured state per tick; reset between episodes")
        current = {"state": state.detach().clone()}
        for key in config.camera_keys:
            image = batch[key]
            if image.ndim == 5 and image.shape[1] == 1:
                image = image[:, 0]
            if image.ndim != 4 or image.shape[0] != state.shape[0]:
                raise ValueError("select_action needs one image per camera and tick")
            current[key] = image.detach().clone()
        if self.observations:
            for key, value in current.items():
                previous = self.observations[-1][key]
                if previous.shape != value.shape or previous.device != value.device:
                    raise ValueError(
                        "observation shape/device changed; reset before starting another episode"
                    )
        command = state if last_command is None else last_command
        repeats = config.n_obs_steps if not self.observations else 1
        for _ in range(repeats):
            self.observations.append(current)
            self.commands.append(command.detach().clone())
        return {
            **batch,
            **{key: torch.stack([item[key] for item in self.observations], 1) for key in current},
            "command_history": torch.stack(list(self.commands), 1),
        }


def observation_window(batch: dict, config: "PolicyConfig") -> dict:
    """Explicit offline window, or a fresh episode padded with its first observation."""
    state = batch["state"]
    if state.ndim == 2 or (state.ndim == 3 and state.shape[1] == 1 and "command_history" not in batch):
        return ObservationHistory(config.n_obs_steps).append(batch, config, None)
    if state.ndim != 3 or state.shape[1:] != (config.n_obs_steps, config.action_dim):
        raise ValueError("prediction needs exactly n_obs_steps measured states")
    commands = batch.get("command_history")
    if commands is None or commands.shape != state.shape or commands.device != state.device:
        raise ValueError("offline history prediction needs matching absolute command_history")
    return batch


def pack_conditioning(vae, cameras: Tensor, states: Tensor, commands: Tensor, config: "PolicyConfig"):
    """One window: independent visual snapshots, normalized states and optional past commands."""
    if cameras.shape[1] != config.n_obs_steps:
        raise ValueError("history prediction needs exactly n_obs_steps camera observations")
    if cameras.dtype == torch.uint8:
        cameras = (cameras.cpu().float() / 255).to(states.device)
    canvas = packing.materialize_video(
        cameras, None, states.device, layout=config.camera_layout, canvas_hw=config.canvas_hw
    )
    indices = snapshot_indices(config)
    tokens, positions = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=states.device.type == "cuda"):
        for index in indices:
            latent = packing.encode_single_frame(vae, canvas[:, index], config.latent_hw, single_frame=True)
            token, ids = batched_prc_vid(
                latent, times_to_ids(torch.full((1, 1), index / config.video_position_fps))
            )
            tokens.append(token)
            positions.append(ids)
    action = pack_actions(conditioning_values(states, commands, config), None, config)
    return {
        "x_video_cond": torch.cat(tokens, 1),
        "x_video_cond_ids": torch.cat(positions, 1),
        **action,
    }


def snapshot_indices(config: "PolicyConfig") -> list[int]:
    return (
        [config.n_obs_steps - 1]
        if config.history_snapshots == 1
        else [
            round(i * (config.n_obs_steps - 1) / (config.history_snapshots - 1))
            for i in range(config.history_snapshots)
        ]
    )


def conditioning_values(states: Tensor, commands: Tensor, config: "PolicyConfig") -> Tensor:
    """Raw measured states and preceding absolute commands -> unscaled conditioning channels."""
    if states.ndim != 3 or states.shape[1:] != (config.n_obs_steps, config.action_dim):
        raise ValueError("history conditioning needs exactly n_obs_steps measured states")
    if commands.shape != states.shape:
        raise ValueError("history conditioning needs matching absolute command_history")
    values = normalization.normalize(states.float(), config.state_normalization, config.normalization_clip)
    if config.condition_on_past_actions:
        past = commands[:, 1:].float()
        if config.action_parameterization == "joint_delta":
            past = normalization.deltas(past, commands[:, 0].float(), config.absolute_action_dims)
        past = normalization.normalize(past, config.action_normalization, config.normalization_clip)
        past = torch.cat([torch.zeros_like(commands[:, :1]), past], 1)
        values = torch.cat([past, values], -1)
    return values


def pack_actions(values: Tensor, actions: Tensor | None, config: "PolicyConfig") -> dict[str, Tensor]:
    b = values.shape[0]
    times = (torch.arange(config.n_obs_steps).float() - (config.n_obs_steps - 1)) / config.fps
    cond, ids = batched_prc_action(
        values.transpose(1, 2),
        times_to_ids(times[None].expand(b, -1)),
        torch.full((b, 1), -1, dtype=torch.long),
    )
    result = {f"x_{config.action_modality}_cond": cond, f"x_{config.action_modality}_cond_ids": ids}
    if actions is not None:
        tok, pos = batched_prc_action(
            (actions * config.action_scale).transpose(1, 2),
            times_to_ids((torch.arange(config.chunk_size).float() / config.fps)[None].expand(b, -1)),
        )
        result.update(
            {f"x_{config.action_modality}": tok.to(torch.bfloat16), f"x_{config.action_modality}_ids": pos}
        )
    return result


@torch.no_grad()
def encode_windows(vae, videos: Tensor, config: "PolicyConfig") -> Tensor:
    """Encode each snapshot and the future independently, as in collab's task packer."""
    clips = [videos[:, :, i : i + 1] for i in snapshot_indices(config)]
    clips.append(videos[:, :, config.n_obs_steps :])
    # Per-window VAE calls preserve the native task path's numerical contract.
    latents = [
        torch.cat([vae.encode_task(v[None].to(torch.bfloat16)).clone() for v in clip]) for clip in clips
    ]
    return torch.cat(latents, 2)[..., : config.latent_hw[0], : config.latent_hw[1]]


def pack_video(latents: Tensor, config: "PolicyConfig") -> dict[str, Tensor]:
    b = latents.shape[0]
    times = torch.tensor(snapshot_indices(config)).float()[None].expand(b, -1) / config.video_position_fps
    cond, cond_ids = batched_prc_vid(latents[:, :, : config.history_snapshots], times_to_ids(times))
    future = latents[:, :, config.history_snapshots :]
    tok, ids = batched_prc_vid(
        future,
        packing.video_time_ids(
            future.shape[2],
            packing.latent_frames(config.n_obs_steps),
            b,
            fps=config.video_position_fps,
        ),
    )
    return {"x_video_cond": cond, "x_video_cond_ids": cond_ids, "x_video": tok, "x_video_ids": ids}
