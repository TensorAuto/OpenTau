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

"""Observation adapters for ``xr1``: raw state -> XR-1 state, and the history buffer.

Two conversions live here, both shared by the training and inference paths so they cannot
drift apart:

**State.** RoboCasa's ``agent_pos`` is 16-D
``[base_pos(3), base_quat(4), ee_pos_rel(3), ee_quat_rel(4), gripper_qpos(2)]``
(``envs/robocasa.py::_format_raw_obs``). Xiaomi-Robotics-1 trained on a different, 14-D
**EE-first** vector with the two quaternions converted to axis-angle:
``[ee_pos_rel(3), ee_rot_rel_aa(3), gripper_qpos(2), base_pos(3), base_rot_aa(3)]``, then
zero-padded to the DiT projector's 60 columns. Both the reordering and the conversion are
silent when wrong -- an axis-angle vector built from a re-ordered quaternion is still a
plausible-looking 3-vector.

**History.** The reference keeps a ``deque(maxlen=(T-1)*k + 1)`` and samples
``max(0, len(items) - 1 - (T-1-i)*k)``, which **clamps** to the earliest available frame at
the start of an episode. OpenTau's dataset does the same at episode boundaries, but the two
existing inference buffers (pi07 / pi05_mem) **zero-pad** instead -- so a policy ported by
copying them would see, for the first ``(T-1)*k`` steps of every episode, a distribution it
was never trained on. :func:`opentau.policies.utils.history_slot_indices` carries both
conventions behind a ``pad_mode`` argument; xr1 passes ``"clamp"``. (``_build_history_batch``
is a per-policy *method* on pi07 / pi05_mem, not a shared helper -- those callers still
inline the arithmetic this function factored out.)
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor

from opentau.policies.utils import history_slot_indices

#: Slices of the **env's** 16-D ``agent_pos`` (``envs/robocasa.py::_format_raw_obs``),
#: which flattens the RoboCasa key converter's dict **base-first**.
_ENV_LAYOUT = {
    "base_pos": slice(0, 3),
    "base_quat": slice(3, 7),
    "ee_pos_rel": slice(7, 10),
    "ee_quat_rel": slice(10, 14),
    "gripper_qpos": slice(14, 16),
}

#: Slices of the **RoboCasa365 LeRobot dataset's** 16-D ``observation.state``, which
#: flattens the same five fields **EE-first**. The two are *not* interchangeable, and the
#: mismatch is silent: both are 16 wide, both contain two unit quaternions, and an
#: axis-angle vector built from the wrong one is still a plausible 3-vector.
#:
#: Verified empirically against ``pepijn223/robocasa_pretrain_human300_v4`` rather than
#: assumed, using a signature that cannot be argued with: a mobile base rotates only about
#: z, so its quaternion is ``(0, 0, sin(t/2), cos(t/2))`` and its x/y components are
#: **exactly** zero. In the dataset that pattern sits at columns 10:14, and the general
#: (all-four-free) quaternion sits at 3:7 -- the opposite assignment from the env.
_DATASET_LAYOUT = {
    "ee_pos_rel": slice(0, 3),
    "ee_quat_rel": slice(3, 7),
    "base_pos": slice(7, 10),
    "base_quat": slice(10, 14),
    "gripper_qpos": slice(14, 16),
}

#: ``state_adapter`` value -> field slices.
STATE_LAYOUTS = {
    "robocasa_panda_omron": _ENV_LAYOUT,
    "robocasa365_dataset": _DATASET_LAYOUT,
}

#: Width of the adapted state before zero-padding to ``state_token_dim``.
XR1_STATE_DIM = 14

#: Below this the quaternion (or its vector part) is treated as degenerate and the
#: axis-angle result is exactly zero -- the reference's threshold, verbatim.
_DEGENERATE_EPS = 1e-12


def quat_to_axis_angle(quat: Tensor, quat_order: str = "xyzw") -> Tensor:
    """Batched quaternion -> axis-angle, matching the reference element for element.

    The reference (``eval_robocasa365/entry.py::quat_xyzw_to_axis_angle``) does, per
    quaternion, in float64: normalize; negate if ``w < 0`` (so the rotation is taken the
    short way round); ``angle = 2 * atan2(|xyz|, clip(w, -1, 1))``; return
    ``xyz / |xyz| * angle``; and return exact zeros when either ``|q|`` or ``|xyz|`` is
    below 1e-12.

    Args:
        quat: ``(..., 4)`` quaternions.
        quat_order: ``"xyzw"`` (robosuite / the reference) or ``"wxyz"``.

    Returns:
        ``(..., 3)`` axis-angle vectors, in the input dtype.

    Raises:
        ValueError: on an unknown ``quat_order`` or a trailing dimension that is not 4.
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected a trailing dimension of 4, got {tuple(quat.shape)}.")
    if quat_order == "xyzw":
        xyzw = quat
    elif quat_order == "wxyz":
        xyzw = torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)
    else:
        raise ValueError(f"quat_order must be 'xyzw' or 'wxyz', got '{quat_order}'.")

    out_dtype = quat.dtype
    q = xyzw.to(torch.float64)

    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    q_normalized = q / norm.clamp_min(_DEGENERATE_EPS)
    # Short-way-round canonicalization: q and -q are the same rotation, and the reference
    # always takes the w >= 0 representative.
    sign = torch.where(q_normalized[..., 3:4] < 0, -1.0, 1.0).to(q_normalized.dtype)
    q_normalized = q_normalized * sign

    xyz = q_normalized[..., :3]
    w = q_normalized[..., 3:4]
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.clamp(-1.0, 1.0))
    axis_angle = xyz / sin_half.clamp_min(_DEGENERATE_EPS) * angle

    degenerate = (norm < _DEGENERATE_EPS) | (sin_half < _DEGENERATE_EPS)
    axis_angle = torch.where(degenerate, torch.zeros_like(axis_angle), axis_angle)
    return axis_angle.to(out_dtype)


def robocasa_state_to_xr1_state(
    raw_state: Tensor, layout: str = "robocasa_panda_omron", quat_order: str = "xyzw"
) -> Tensor:
    """``(..., 16)`` RoboCasa state -> ``(..., 14)`` XR-1 state.

    Args:
        raw_state: The 16-D per-frame state.
        layout: ``"robocasa_panda_omron"`` for the **env's** base-first ``agent_pos``, or
            ``"robocasa365_dataset"`` for the **dataset's** EE-first ``observation.state``.
        quat_order: Quaternion component order within each 4-slice.

    Returns:
        ``[ee_pos_rel(3), ee_rot_aa(3), gripper_qpos(2), base_pos(3), base_rot_aa(3)]``.

    Raises:
        ValueError: on an unknown layout or a trailing dimension that is not 16.
    """
    if layout not in STATE_LAYOUTS:
        raise ValueError(f"Unknown state layout '{layout}'; expected one of {sorted(STATE_LAYOUTS)}.")
    if raw_state.shape[-1] != 16:
        raise ValueError(f"The '{layout}' state adapter expects a 16-D state; got {tuple(raw_state.shape)}.")
    fields = STATE_LAYOUTS[layout]
    return torch.cat(
        [
            raw_state[..., fields["ee_pos_rel"]],
            quat_to_axis_angle(raw_state[..., fields["ee_quat_rel"]], quat_order),
            raw_state[..., fields["gripper_qpos"]],
            raw_state[..., fields["base_pos"]],
            quat_to_axis_angle(raw_state[..., fields["base_quat"]], quat_order),
        ],
        dim=-1,
    )


def robocasa_agent_pos_to_xr1_state(agent_pos: Tensor, quat_order: str = "xyzw") -> Tensor:
    """The env-layout adapter. Thin alias kept because it names the common case."""
    return robocasa_state_to_xr1_state(agent_pos, "robocasa_panda_omron", quat_order)


def adapt_state(
    state: Tensor, *, state_adapter: str, state_token_dim: int, quat_order: str = "xyzw"
) -> Tensor:
    """Apply the configured adapter and zero-pad to ``state_token_dim``.

    Args:
        state: ``(B, T, D_raw)`` (or ``(B, D_raw)``) raw per-frame state.
        state_adapter: ``"robocasa_panda_omron"`` (env layout), ``"robocasa365_dataset"``
            (dataset layout) or ``"identity"``.
        state_token_dim: Width of each DiT state token (60 in the reference).
        quat_order: Quaternion component order in the raw state.

    Returns:
        The same leading shape with a trailing ``state_token_dim``.

    Raises:
        ValueError: on an unknown adapter, or a raw state wider than ``state_token_dim``.
    """
    if state_adapter in STATE_LAYOUTS:
        adapted = robocasa_state_to_xr1_state(state, state_adapter, quat_order)
    elif state_adapter == "identity":
        adapted = state
    else:
        raise ValueError(f"Unknown state_adapter '{state_adapter}'.")

    width = adapted.shape[-1]
    if width > state_token_dim:
        raise ValueError(f"Adapted state width ({width}) exceeds state_token_dim ({state_token_dim}).")
    if width < state_token_dim:
        adapted = torch.nn.functional.pad(adapted, (0, state_token_dim - width))
    return adapted


class XR1ObservationBuffer:
    """Per-rollout observation deques feeding the ``n_obs_steps``-frame history window.

    One instance belongs to one rollout batch. ``reset()`` records the batch size, and a
    later ``append`` with a different one raises rather than silently stacking frames from
    two different episodes -- the failure mode you get from sharing a policy across a
    thread pool, which is why ``env.max_parallel_tasks`` must stay 1 for xr1
    (``scripts/eval.py`` shares one policy object across its worker threads).
    """

    def __init__(self, n_obs_steps: int, history_interval: int, buffer_size: int):
        self.n_obs_steps = n_obs_steps
        self.history_interval = history_interval
        self.buffer_size = buffer_size
        self._state: list[Tensor] = []
        self._images: dict[str, list[Tensor]] = {}
        self._batch_size: int | None = None

    def reset(self) -> None:
        self._state = []
        self._images = {}
        self._batch_size = None

    def append(self, state: Tensor, images: dict[str, Tensor]) -> None:
        """Push one timestep: ``state`` ``(B, D)`` and each image ``(B, C, H, W)``."""
        bsize = state.shape[0]
        if self._batch_size is None:
            self._batch_size = bsize
        elif bsize != self._batch_size:
            raise ValueError(
                f"XR1ObservationBuffer was filled at batch size {self._batch_size} but received "
                f"{bsize}. Call `policy.reset()` between rollout batches; sharing one policy "
                "across concurrent tasks (env.max_parallel_tasks > 1) is not supported."
            )
        self._state.append(state)
        if len(self._state) > self.buffer_size:
            self._state.pop(0)
        for key, value in images.items():
            buf = self._images.setdefault(key, [])
            buf.append(value)
            if len(buf) > self.buffer_size:
                buf.pop(0)

    def indices(self) -> list[int]:
        """The buffer slots the current window reads, oldest first (clamped at the start)."""
        return history_indices(len(self._state), self.n_obs_steps, self.history_interval, pad_mode="clamp")

    def window(self) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """Return ``(state (B, T, D), {key: (B, T, C, H, W)}, is_pad (B, T))``.

        ``is_pad`` marks the slots that are *repeats* of the oldest real frame. Nothing in
        the xr1 forward consumes it (the reference has no such signal), but the dataset
        emits ``obs_history_is_pad`` and the training path needs the two sides to agree on
        what the field means.
        """
        if not self._state:
            raise RuntimeError("XR1ObservationBuffer.window() called before any append().")
        idx = self.indices()
        state = torch.stack([self._state[i] for i in idx], dim=1)
        images = {key: torch.stack([buf[i] for i in idx], dim=1) for key, buf in self._images.items()}
        # A slot is "padded" when clamping made it repeat the frame before it.
        pad_flags = [False] + [idx[i] == idx[i - 1] for i in range(1, len(idx))]
        is_pad = torch.tensor(pad_flags, dtype=torch.bool, device=state.device)
        return state, images, rearrange(is_pad, "t -> 1 t").expand(state.shape[0], -1)


def history_indices(
    buffer_length: int, n_obs_steps: int, interval: int, pad_mode: str = "clamp"
) -> list[int]:
    """xr1's view of :func:`opentau.policies.utils.history_slot_indices`, defaulting to clamp.

    The shared helper defaults to ``"zero"`` (the existing pi07 / pi05_mem convention);
    xr1 flips the default because the reference clamps, and a caller that forgets the
    argument here would silently reintroduce the divergence this module exists to avoid.
    """
    return history_slot_indices(buffer_length, n_obs_steps, interval, pad_mode=pad_mode)
