#!/usr/bin/env python

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

"""Denoising acceleration (``accel``) — a cost-free uncertainty proxy for flow-matching heads.

Implements the estimator of *The Geometry of Flow-Matching Uncertainty*
(arXiv:2607.27933). A maximally *certain* conditional-flow-matching field is an
affine-isotropic contraction toward a point mass, so its denoising trajectory is a
straight line traversed at constant velocity; any *bend* in that trajectory betrays
non-zero posterior covariance. Over the first ``p`` of ``T`` Euler steps,

.. math::

    \\mathrm{accel}_p = \\frac{p \\sum_{t=1}^{p-1} \\lVert v_t - v_{t-1} \\rVert}
                             {\\sum_{t=0}^{p-1} \\lVert v_t \\rVert}

where ``v_t`` is the velocity the sampler *already* evaluates to take its Euler step.
No extra network evaluations, no resampling, no training, no auxiliary probe — the
whole cost is two vector norms per denoise step.

Why a *prefix* and not the whole path: the field's ``1/(1-s)`` factor diverges at the
clean-action end, so the final Euler steps are dominated by discretization noise rather
than posterior information. That is a statement about *integration order*, not about
which end the time variable calls zero — OpenTau's samplers integrate noise -> data with
``time`` running 1 -> 0, so the prefix is always the first ``p`` loop iterations.
:func:`default_prefix` returns the paper's online-detector choice, ``T - 1``.

Two masks are mandatory for the number to mean anything on OpenTau checkpoints, and both
are silent when wrong:

* **Action-dim mask.** The samplers work in ``max_action_dim`` (32) while a real
  embodiment uses far fewer (7 for LIBERO). The padded tail is masked out of the
  training loss (:func:`opentau.policies.utils.make_action_dim_mask`), so those columns
  are *unsupervised network output* — pure noise landing in both the numerator and the
  denominator. ``real_action_dim`` is a dataset-side key and is absent at inference, so
  :func:`resolve_action_dim_mask` recovers the mask from the normalization buffers.
* **Row mask.** Frozen real-time-chunking prefix rows carry velocities for a state that
  is discarded, and only the executed window of the chunk is ever applied. Both are
  excluded by the samplers via :meth:`AccelMeter.set_row_mask`.

The score is only comparable within a fixed
``(task, embodiment, norm head, ACTION norm mode, delta-vs-absolute, T, p, dtype)``
tuple; :class:`AccelProvenance` records that tuple so a calibrated threshold can never
be applied across a mismatch.

See :mod:`opentau.utils.accel_detector` for the offline CUSUM + split-conformal detector
that consumes the per-chunk stream produced here.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from opentau.configs.types import NormalizationMode
from opentau.policies.normalize import stat_names_for_mode

logger = logging.getLogger(__name__)

# Denominator floor. Below this the ratio is not a curvature measurement — every velocity
# in the prefix was (near-)zero, which happens when the masks eliminate every scored
# element. Reporting 0.0 there would read as "maximally certain", the exact opposite of
# "no information", so the meter emits NaN instead.
DENOMINATOR_FLOOR = 1e-12

# A prefix of 1 yields an empty numerator sum, i.e. accel == 0 by construction regardless
# of the field. Same false-confidence trap as above, so it is rejected rather than clamped.
MIN_PREFIX = 2

# Environment variable that enables ``accel`` on an entry point with no config field for it.
# Accepts ``auto`` (the paper's default prefix) or an integer ``>= MIN_PREFIX``.
ACCEL_PREFIX_ENV = "OPENTAU_ACCEL_PREFIX"

# Collects the meters built by `make_meter` while a diagnostic is running, and doubles as
# the "record a per-step trace" switch (`None` = off). Non-`None` makes every prefix
# readable from a single denoise pass via `AccelMeter.value_at`. Diagnostics only.
#
# A `ContextVar` rather than a module global or a `policy.` attribute: the meter is built
# deep inside `sample_actions`, so both the switch and the hand-back have to reach it
# without widening any policy signature — but the gRPC server calls `sample_actions` from
# multiple threads, and a plain global would splice one thread's diagnostic run into
# another's serving traffic. A `ContextVar` is per-thread (and per-task) by construction.
_TRACE_SINK: contextvars.ContextVar[list[AccelMeter] | None] = contextvars.ContextVar(
    "accel_trace_sink", default=None
)


@contextlib.contextmanager
def record_traces() -> Iterator[list[AccelMeter]]:
    """Capture the meters built inside this context, with per-step traces enabled.

    The score at every prefix is recoverable from one denoise pass
    (:meth:`AccelMeter.value_at`), but the policies hand back only ``last_accel`` — the
    single configured prefix — so a prefix study needs the meter object itself. Yielding a
    sink is how it gets one without any policy exposing its internals.

    Off by default: the serving path never needs it, and a captured meter keeps its trace
    tensors alive. Drain the list between iterations on a long study.

    Yields:
        A list receiving every meter built in this context, in creation order. The previous
        setting is restored on exit, including on exception.
    """
    created: list[AccelMeter] = []
    token = _TRACE_SINK.set(created)
    try:
        yield created
    finally:
        _TRACE_SINK.reset(token)


def default_prefix(num_steps: int) -> int:
    """Return the paper's online-detector prefix for a ``num_steps``-step schedule.

    The paper reports its failure-detection results with the second-to-last prefix
    (``accel_-2``), i.e. ``p = T - 1``, having found that truncated prefixes correlate
    more strongly with posterior spread than the full path. Its *best*-prefix sweep peaks
    nearer ``p/T ~ 0.4-0.5``, so treat this as a sane default and not an optimum — for a
    short schedule (``pi06``/``pi07`` default to ``T = 5``) the two are far apart.

    Args:
        num_steps: Total Euler steps ``T`` in the denoise schedule.

    Returns:
        A prefix in ``[MIN_PREFIX, num_steps]``.

    Raises:
        ValueError: If ``num_steps`` is below :data:`MIN_PREFIX`.
    """
    if num_steps < MIN_PREFIX:
        raise ValueError(
            f"accel needs at least {MIN_PREFIX} denoise steps to form one velocity "
            f"difference, but num_steps={num_steps}."
        )
    return max(MIN_PREFIX, num_steps - 1)


def resolve_prefix(prefix: int | None, num_steps: int) -> int | None:
    """Validate and clamp a configured ``accel`` prefix against the denoise schedule.

    Args:
        prefix: Configured prefix, or ``None`` to leave ``accel`` disabled.
        num_steps: Total Euler steps ``T`` in the denoise schedule.

    Returns:
        ``None`` when disabled, else a prefix in ``[MIN_PREFIX, num_steps]``.

    Raises:
        ValueError: If ``prefix`` is below :data:`MIN_PREFIX`, or if the schedule is too
            short to form a velocity difference.
    """
    if prefix is None:
        return None
    if prefix < MIN_PREFIX:
        raise ValueError(
            f"accel prefix must be >= {MIN_PREFIX} (a shorter prefix has an empty "
            f"numerator and would report 0.0 for every field), got {prefix}."
        )
    if num_steps < MIN_PREFIX:
        raise ValueError(
            f"accel needs at least {MIN_PREFIX} denoise steps to form one velocity "
            f"difference, but num_steps={num_steps}."
        )
    return min(prefix, num_steps)


@dataclass(frozen=True)
class AccelProvenance:
    """The context that makes one ``accel`` value comparable to another.

    Every field here shifts the score's distribution, so a threshold calibrated under one
    tuple is meaningless under a different one. Persist this alongside the values and
    refuse to apply a calibration across a mismatch (:func:`assert_comparable`).

    Attributes:
        policy_type: ``config.type`` of the policy that produced the score.
        num_steps: Total Euler steps ``T``.
        prefix: Prefix ``p`` actually integrated over.
        chunk_size: Full predicted chunk length.
        n_action_steps: Executed window length (the scored rows).
        max_delay: Configured real-time-chunking depth.
        action_norm_mode: ``normalization_mapping["ACTION"]`` name.
        has_delta_action_map: Whether actions live in delta space.
        velocity_dtype: dtype the velocity projection ran in, as a string. bf16 puts a
            positive-biased noise floor under the score (see module docs of
            :mod:`opentau.utils.accel_detector`).
        num_scored_dims: Action dims that survived the dim mask, per norm head.
        dataset_index: Norm-head row index per sample, when resolvable.
    """

    policy_type: str
    num_steps: int
    prefix: int
    chunk_size: int
    n_action_steps: int
    max_delay: int
    action_norm_mode: str
    has_delta_action_map: bool
    velocity_dtype: str
    num_scored_dims: tuple[int, ...] = ()
    dataset_index: tuple[int, ...] = ()

    # Fields that must agree for two scores to be comparable. `dataset_index` and
    # `num_scored_dims` are deliberately excluded: they vary per sample within one run,
    # and comparability across norm heads is checked separately by the caller that knows
    # which head a calibration was fitted on.
    COMPARABLE_FIELDS = (
        "policy_type",
        "num_steps",
        "prefix",
        "chunk_size",
        "n_action_steps",
        "max_delay",
        "action_norm_mode",
        "has_delta_action_map",
        "velocity_dtype",
    )

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> AccelProvenance:
        """Rebuild from :meth:`to_dict` output, tolerating unknown future keys."""
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in payload.items() if k in known}
        for tuple_field in ("num_scored_dims", "dataset_index"):
            if tuple_field in kwargs and kwargs[tuple_field] is not None:
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)


def assert_comparable(fitted: AccelProvenance, observed: AccelProvenance) -> None:
    """Raise unless two provenances agree on every distribution-shifting field.

    Args:
        fitted: Provenance recorded when a threshold was calibrated.
        observed: Provenance of the stream the threshold is about to be applied to.

    Raises:
        ValueError: On the first disagreeing field, naming both values.
    """
    diffs = [
        f"{name}: calibrated on {getattr(fitted, name)!r}, observed {getattr(observed, name)!r}"
        for name in AccelProvenance.COMPARABLE_FIELDS
        if getattr(fitted, name) != getattr(observed, name)
    ]
    if diffs:
        raise ValueError(
            "accel calibration is not applicable to this stream — "
            + "; ".join(diffs)
            + ". Recalibrate under the deployment configuration."
        )


def _action_dim_signature(
    unnormalize: torch.nn.Module, action_key: str
) -> tuple[Tensor, NormalizationMode, float]:
    """Return the per-head, per-dim normalization *scale* of an action feature.

    One extractor feeds both consumers of these buffers —
    :func:`resolve_action_dim_mask` (which thresholds the scale to find padded dims) and
    :func:`action_dim_scale` (which divides by it to standardize a raw-space spread). They
    must read the same buffer, the same stat names, and the same live ``eps``; two
    hand-maintained copies would drift the moment a normalization mode is added.

    Args:
        unnormalize: The policy's ``Unnormalize`` module.
        action_key: Feature key to read stats for.

    Returns:
        ``(signature, norm_mode, eps)`` where ``signature`` is ``(num_heads, dim)``: ``std``
        under MEAN_STD, ``hi - lo`` under MIN_MAX / QUANTILE. Note this is the *range*, not
        the divisor — MIN_MAX maps ``[lo, hi]`` onto ``[-1, 1]``, so its divisor is half of
        this. :func:`action_dim_scale` applies that factor; the mask only needs the
        zero-vs-nonzero distinction, which the factor cannot change.

    Raises:
        ValueError: When the feature is missing, IDENTITY-normalized, or has no buffer.
    """
    norm_map = getattr(unnormalize, "norm_map", {})
    features = getattr(unnormalize, "features", {})
    feature = features.get(action_key)
    if feature is None:
        raise ValueError(
            f"accel: no feature {action_key!r} on the policy's Unnormalize; cannot derive "
            "the action-dim mask."
        )
    norm_mode = norm_map.get(feature.type, NormalizationMode.IDENTITY)
    if norm_mode is NormalizationMode.IDENTITY:
        raise ValueError(
            "accel: ACTION normalization is IDENTITY, so no stats buffer exists and the "
            "padded-action-dim mask cannot be derived. IDENTITY also leaves per-dim scale "
            "heterogeneous, which violates the estimator's standardized-dimension premise. "
            "Disable accel for this checkpoint or retrain/serve with MEAN_STD."
        )

    buffer = getattr(unnormalize, "buffer_" + action_key.replace(".", "_"), None)
    if buffer is None:
        raise ValueError(f"accel: Unnormalize has no stats buffer for {action_key!r}.")

    # Import locally: `_materialize` is a private FSDP2/DTensor shim in `normalize`, and a
    # module-level import would make this module's import graph depend on it by name.
    from opentau.policies.normalize import _materialize

    if norm_mode is NormalizationMode.MEAN_STD:
        signature = _materialize(buffer["std"]).abs()
    else:
        lo_name, hi_name = stat_names_for_mode(norm_mode)
        signature = (_materialize(buffer[hi_name]) - _materialize(buffer[lo_name])).abs()

    signature = signature.reshape(signature.shape[0], -1)
    return signature, norm_mode, float(unnormalize.eps)


def _fit_dim_width(per_head: Tensor, max_action_dim: int, pad_value: float | bool) -> Tensor:
    """Pad or truncate a ``(heads, dim)`` per-dim quantity to the sampler's action width."""
    width = per_head.shape[1]
    if width < max_action_dim:
        pad = torch.full(
            (per_head.shape[0], max_action_dim - width),
            pad_value,
            dtype=per_head.dtype,
            device=per_head.device,
        )
        return torch.cat([per_head, pad], dim=1)
    return per_head[:, :max_action_dim] if width > max_action_dim else per_head


def _gather_heads(per_head: Tensor, dataset_index: Tensor) -> Tensor:
    """Select one ``(dim,)`` row of a per-head quantity for each sample."""
    index = dataset_index.to(device=per_head.device, dtype=torch.long).reshape(-1)
    index = index.clamp_(0, per_head.shape[0] - 1)
    return per_head.index_select(0, index)


def action_dim_scale(
    unnormalize: torch.nn.Module,
    *,
    max_action_dim: int,
    dataset_index: Tensor,
    action_key: str = "actions",
) -> Tensor:
    """Return the per-sample, per-dim divisor that maps raw actions into normalized space.

    ``accel`` is computed on the sampler's *normalized* velocities, so anything meant to be
    compared against it — most importantly the resampling-based posterior spread that
    :mod:`opentau.scripts.diagnose_accel` uses to choose a prefix — has to be expressed in
    the same units. A spread measured on ``sample_actions`` output is in raw units, where a
    gripper in ``[0, 1]`` and a shoulder joint in radians contribute on wildly different
    scales; dividing by this vector removes that.

    Degenerate dims are returned as ``1.0`` rather than ``0.0`` so a caller can divide
    unconditionally. They carry no signal and are excluded by
    :func:`resolve_action_dim_mask` anyway, so the value is arbitrary — it only has to be
    finite and non-zero.

    Args:
        unnormalize: The policy's ``Unnormalize`` module.
        max_action_dim: Width to pad/truncate to. Padded dims get ``1.0``.
        dataset_index: ``(B,)`` long tensor of norm-head row indices.
        action_key: Feature key to read stats for.

    Returns:
        ``(B, max_action_dim)`` float32, strictly positive.
    """
    signature, norm_mode, eps = _action_dim_signature(unnormalize, action_key)
    # MIN_MAX / QUANTILE map `[lo, hi]` onto `[-1, 1]`, i.e. divide by half the range.
    scale = signature.to(torch.float32)
    if norm_mode is not NormalizationMode.MEAN_STD:
        scale = scale / 2.0
    scale = torch.where(torch.isfinite(scale) & (scale >= eps), scale, torch.ones_like(scale))
    return _gather_heads(_fit_dim_width(scale, max_action_dim, 1.0), dataset_index)


def resolve_action_dim_mask(
    unnormalize: torch.nn.Module,
    *,
    max_action_dim: int,
    dataset_index: Tensor,
    action_key: str = "actions",
) -> Tensor:
    """Recover which action dims carry real, supervised signal, from the norm buffers.

    ``real_action_dim`` (:mod:`opentau.datasets.lerobot_dataset`) is a dataset-side key
    and does not exist at inference, so the pad tail has to be reconstructed. A padded
    column is constant-zero across the dataset, so its ``std`` (MEAN_STD) or range
    (MIN_MAX / QUANTILE) is zero — the same degenerate-dim test
    ``Normalize._snapping_possible`` uses.

    This is a **heuristic, and it errs toward dropping signal**: a genuinely real but
    constant dim (a locked joint, a gripper axis pinned open, a DOF never commanded in
    that repo) is also zero-variance and gets masked out. That direction is the safe one
    for a curvature ratio — a constant dim contributes no curvature anyway — but do not
    describe the result as an exact reconstruction of ``real_action_dim``. When the
    training dataset is on hand, cross-check against it.

    The buffers are ``(num_datasets, action_dim)``, so the mask is resolved **per norm
    head** and gathered per sample; on a co-trained mixture two samples in one batch can
    legitimately score a different number of dims.

    Args:
        unnormalize: The policy's ``Unnormalize`` module (its ``eps`` and buffers are read
            live, never the module-level default — ``eps`` is ``config_version``-dependent).
        max_action_dim: Width of the sampler's action tensor. The mask is padded with
            ``False`` beyond the buffer width and truncated if the buffer is wider.
        dataset_index: ``(B,)`` long tensor of norm-head row indices.
        action_key: Feature key to read stats for.

    Returns:
        ``(B, max_action_dim)`` bool tensor; ``True`` where the dim carries signal.

    Raises:
        ValueError: When the action feature is IDENTITY-normalized (no buffer exists, so
            the mask is underivable) or has no buffer for another reason.
    """
    signature, norm_mode, eps = _action_dim_signature(unnormalize, action_key)
    # `(num_datasets, action_dim)` -> per-head bool over dims.
    head_mask = _fit_dim_width(torch.isfinite(signature) & (signature >= eps), max_action_dim, False)
    index = dataset_index.to(device=head_mask.device, dtype=torch.long).reshape(-1)
    index = index.clamp_(0, head_mask.shape[0] - 1)
    per_sample = _gather_heads(head_mask, dataset_index)

    if not bool(per_sample.any()):
        # Name the head that was actually selected and contrast it with the others. The
        # common cause is not a broken checkpoint but a *routing* one: a co-trained mixture
        # carries a placeholder head whose stats are all zero, and an observation with no
        # dataset provenance falls back to head 0 — which may be exactly that placeholder.
        # Reporting only "the mask is empty" sends the reader to audit their action stats
        # when the fix is to tag the observation with its norm head.
        per_head = [int(n) for n in head_mask.sum(dim=1).tolist()]
        selected = sorted({int(i) for i in index.tolist()})
        usable = [i for i, n in enumerate(per_head) if n > 0]
        raise ValueError(
            f"accel: every action dim of norm head(s) {selected} is degenerate in the "
            f"{norm_mode.name} stats, so the score would be computed over zero dimensions. "
            f"Scorable dims per head: {per_head}."
            + (
                f" Head(s) {usable} do have usable stats — if that is where this sample belongs, "
                "tag the observation with its `dataset_repo_id` (or `robot_type`/`control_mode`) "
                "so `_resolve_dataset_index` routes it there instead of falling back to head 0."
                if usable
                else " No head has usable action stats; this checkpoint cannot support accel."
            )
        )
    return per_sample


@dataclass
class AccelMeter:
    """Running ``accel`` accumulator for one flow-matching sample call.

    Threaded through a sampler's Euler loop as an optional argument rather than stashed on
    the ``nn.Module``: the samplers are ``torch.compile``d and ONNX-exported at several
    entry points, and writing to a module attribute inside the traced region is a Dynamo
    side-effect on an ``nn.Module`` — the construct most likely to break under
    ``fullgraph=True`` or ``dynamo=True`` export. A per-call object also cannot leak state
    between calls or across threads.

    Everything accumulates in float32 on-device with no ``.item()`` inside the loop, so no
    host synchronization is added. (The samplers' ``while`` condition already forces one
    per iteration; ``accel`` does not add a second.)

    Usage from a sampler::

        if accel is not None:
            accel.set_row_mask(row_mask)          # (B|1, chunk) bool, before the loop
        while ...:
            v_t = self.denoise_step(...)
            if accel is not None:
                accel.update(v_t)                 # inside the loop, before the Euler step
            x_t += dt * v_t

    Attributes:
        prefix: Number of leading Euler steps to integrate over.
        batch_size: Number of independent samples.
        device: Device the accumulators live on.
        dim_mask: Optional ``(B, action_dim)`` bool from :func:`resolve_action_dim_mask`.
    """

    prefix: int
    batch_size: int
    device: torch.device
    dim_mask: Tensor | None = None
    record_trace: bool = False

    _numerator: Tensor = field(init=False, repr=False)
    _denominator: Tensor = field(init=False, repr=False)
    _prev: Tensor | None = field(default=None, init=False, repr=False)
    _score_mask: Tensor | None = field(default=None, init=False, repr=False)
    _steps: int = field(default=0, init=False)
    _trace: list[tuple[Tensor, Tensor]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prefix < MIN_PREFIX:
            raise ValueError(f"accel prefix must be >= {MIN_PREFIX}, got {self.prefix}.")
        self._numerator = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self._denominator = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        if self.dim_mask is not None:
            # (B, dim) -> (B, 1, dim) so it broadcasts over the chunk axis.
            self._score_mask = rearrange(self.dim_mask.to(torch.float32), "b d -> b 1 d")

    def set_row_mask(self, row_mask: Tensor) -> None:
        """Restrict scoring to a subset of chunk rows.

        Call once before the Euler loop. Combines multiplicatively with ``dim_mask``.

        Args:
            row_mask: ``(B, chunk)`` or ``(1, chunk)`` bool/float; ``True`` scores the row.
                Exclude frozen real-time-chunking prefix rows (their velocities describe a
                state that is overwritten every iteration) and rows outside the executed
                window (they are never applied).
        """
        rows = rearrange(row_mask.to(torch.float32), "b c -> b c 1")
        self._score_mask = rows if self._score_mask is None else rows * self._score_mask

    def update(self, v_t: Tensor) -> None:
        """Accumulate one Euler step's velocity. A no-op once ``prefix`` steps are in.

        Args:
            v_t: ``(B, chunk, action_dim)`` velocity, any dtype (cast to float32 here).
        """
        if self._steps >= self.prefix:
            return
        v = v_t.to(torch.float32)
        if self._score_mask is not None:
            v = v * self._score_mask
        v = v.flatten(1)
        self._denominator = self._denominator + torch.linalg.vector_norm(v, dim=1)
        if self._prev is not None:
            self._numerator = self._numerator + torch.linalg.vector_norm(v - self._prev, dim=1)
        self._prev = v
        self._steps += 1
        if self.record_trace:
            self._trace.append((self._numerator, self._denominator))

    @property
    def steps(self) -> int:
        """Euler steps actually accumulated (``<= prefix``; short if the schedule was)."""
        return self._steps

    def value(self) -> Tensor:
        """Return the ``(B,)`` float32 ``accel`` score.

        Follows Algorithm 1's running form ``J / (S / (t+1))``, i.e. the multiplier is the
        number of velocities accumulated — which equals ``prefix`` whenever the schedule
        was long enough, and degrades gracefully when it was not.

        Returns:
            ``(B,)`` float32. NaN where the score is undefined: fewer than two velocities
            (empty numerator) or an all-but-zero denominator (every scored element masked
            out). NaN is deliberate — 0.0 would read as "maximally certain".
        """
        nan = torch.full_like(self._denominator, float("nan"))
        if self._steps < MIN_PREFIX:
            return nan
        scored = self._steps * self._numerator / self._denominator.clamp_min(DENOMINATOR_FLOOR)
        return torch.where(self._denominator > DENOMINATOR_FLOOR, scored, nan)

    def value_at(self, prefix: int) -> Tensor:
        """Return the ``(B,)`` score this meter *would* have reported at a shorter prefix.

        ``accel_p`` is a prefix statistic of one velocity sequence, so every prefix is
        recoverable from a single denoise pass — the numerator and denominator are running
        sums and :meth:`update` snapshots both. That turns a prefix sweep from ``T - 1``
        forward passes into one, which is what makes the resampling-based prefix study in
        :mod:`opentau.scripts.diagnose_accel` affordable.

        Requires ``record_trace=True``; the serving path leaves it off so the hot loop
        allocates nothing extra.

        Args:
            prefix: Prefix to evaluate, in ``[MIN_PREFIX, steps]``.

        Returns:
            ``(B,)`` float32, NaN under the same conditions as :meth:`value`.

        Raises:
            RuntimeError: If the meter was built without ``record_trace``.
            ValueError: If ``prefix`` is outside ``[MIN_PREFIX, steps]``.
        """
        if not self.record_trace:
            raise RuntimeError(
                "AccelMeter.value_at requires record_trace=True — build the meter inside "
                "`opentau.policies.accel.record_traces()`."
            )
        if not MIN_PREFIX <= prefix <= self._steps:
            raise ValueError(f"prefix must be in [{MIN_PREFIX}, {self._steps}] for this meter, got {prefix}.")
        numerator, denominator = self._trace[prefix - 1]
        nan = torch.full_like(denominator, float("nan"))
        scored = prefix * numerator / denominator.clamp_min(DENOMINATOR_FLOOR)
        return torch.where(denominator > DENOMINATOR_FLOOR, scored, nan)

    def prefix_values(self) -> dict[int, list[float]]:
        """Return ``{p: per-sample accel_p}`` for every prefix this meter can evaluate."""
        return {
            p: [float(x) for x in self.value_at(p).detach().to("cpu", torch.float32).tolist()]
            for p in range(MIN_PREFIX, self._steps + 1)
        }

    def to_list(self) -> list[float]:
        """Return the score as plain Python floats.

        Fully escapes ``torch.inference_mode()``: a tensor allocated inside inference mode
        stays an inference tensor even after ``.detach().clone()``, and any later in-place
        mutation of it (exactly what a CUSUM accumulator does) raises. Going through
        ``.tolist()`` yields ordinary floats with no such taint.
        """
        return [float(x) for x in self.value().detach().to("cpu", torch.float32).tolist()]


def make_meter(
    policy: torch.nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    dataset_index: Tensor | None = None,
    max_action_dim: int | None = None,
) -> AccelMeter | None:
    """Build an :class:`AccelMeter` for one sample call, or ``None`` when disabled.

    Reads ``policy.accel_prefix`` (``None`` disables) and derives the action-dim mask from
    ``policy.unnormalize_outputs``. Shared by every flow-matching policy wrapper so the
    enable knob, the mask derivation, and the IDENTITY refusal cannot drift between them.

    Args:
        policy: The policy wrapper (must expose ``config`` and ``unnormalize_outputs``).
        batch_size: Number of samples in the call.
        device: Device to accumulate on.
        dataset_index: ``(B,)`` norm-head row indices; a zeros row is assumed when absent.
        max_action_dim: Sampler action width; defaults to ``policy.config.max_action_dim``.

    Returns:
        A meter, or ``None`` when ``accel_prefix`` is unset.

    Raises:
        ValueError: When ``accel`` is enabled but the configuration cannot support it
            (IDENTITY action normalization, too-short schedule, bad prefix).
    """
    prefix = getattr(policy, "accel_prefix", None)
    if prefix is None:
        return None

    config = policy.config
    prefix = resolve_prefix(prefix, config.num_steps)
    if max_action_dim is None:
        max_action_dim = config.max_action_dim

    if dataset_index is None:
        dataset_index = torch.zeros(batch_size, dtype=torch.long, device=device)

    dim_mask = resolve_action_dim_mask(
        policy.unnormalize_outputs,
        max_action_dim=max_action_dim,
        dataset_index=dataset_index,
    )
    sink = _TRACE_SINK.get()
    meter = AccelMeter(
        prefix=prefix,
        batch_size=batch_size,
        device=device,
        dim_mask=dim_mask,
        record_trace=sink is not None,
    )
    if sink is not None:
        sink.append(meter)
    return meter


def build_provenance(
    policy: torch.nn.Module,
    meter: AccelMeter,
    *,
    dataset_index: Tensor | None = None,
    velocity_dtype: torch.dtype | None = None,
) -> AccelProvenance:
    """Record the context of a batch of ``accel`` scores.

    Args:
        policy: The policy wrapper that produced them.
        meter: The meter used, for the effective prefix and dim mask.
        dataset_index: ``(B,)`` norm-head row indices, if resolved.
        velocity_dtype: dtype the velocity projection ran in. Defaults to the dtype of the
            policy's parameters, which is what the projection inherits.

    Returns:
        A frozen :class:`AccelProvenance`.
    """
    config = policy.config
    norm_mode = config.normalization_mapping.get("ACTION")
    if velocity_dtype is None:
        velocity_dtype = next(policy.parameters()).dtype

    num_scored: tuple[int, ...] = ()
    if meter.dim_mask is not None:
        num_scored = tuple(int(n) for n in meter.dim_mask.sum(dim=-1).detach().cpu().tolist())

    index: tuple[int, ...] = ()
    if dataset_index is not None:
        index = tuple(int(i) for i in dataset_index.detach().cpu().reshape(-1).tolist())

    return AccelProvenance(
        policy_type=getattr(config, "type", type(config).__name__),
        num_steps=int(config.num_steps),
        prefix=int(meter.prefix),
        chunk_size=int(config.chunk_size),
        n_action_steps=int(config.n_action_steps),
        max_delay=int(getattr(config, "max_delay", 0)),
        action_norm_mode=getattr(norm_mode, "name", str(norm_mode)),
        has_delta_action_map=bool(getattr(config, "delta_action_state_map", None)),
        velocity_dtype=str(velocity_dtype).removeprefix("torch."),
        num_scored_dims=num_scored,
        dataset_index=index,
    )


def configure_accel(
    policy: torch.nn.Module,
    cfg: Any,
    *,
    override: int | str | None = None,
) -> int | None:
    """Enable the denoising-acceleration proxy on ``policy``, if anything asked for it.

    The single enable knob shared by every serving entry point (``inference.py``, the gRPC
    server, the RoboCasa WebSocket server), so the precedence order and the refusals cannot
    drift between them.

    **Off by default and never enabled implicitly.** When nothing requests it, this leaves
    ``policy.accel_prefix`` at ``None``, :func:`make_meter` returns ``None``, and every
    ``accel`` line inside the sampler stays dead — the traced graph is unchanged.

    Resolution order, first non-``None`` wins:

    1. ``override`` — an entry point's own flag.
    2. ``cfg.policy.accel_prefix`` — the config field on ``PreTrainedConfig``; read through
       ``getattr`` so a policy config predating it still resolves.
    3. ``$OPENTAU_ACCEL_PREFIX`` — for entry points driven without a config file.

    Call this *before* an entry point's warmup ``sample_actions`` calls: the warmup then
    compiles and autotunes the same graph real requests take rather than forcing a recompile
    on the first one, and a checkpoint that cannot support ``accel`` (IDENTITY-normalized
    actions, an all-degenerate dim mask) fails at startup instead of on the first request.

    Args:
        policy: The loaded policy wrapper.
        cfg: The parsed pipeline config; only a possible ``cfg.policy.accel_prefix`` and
            ``cfg.policy.type`` are read. Typed loosely to keep this module off the config
            package's import graph.
        override: Entry-point-level request that beats both config and environment.

    Returns:
        The resolved prefix, or ``None`` when ``accel`` stays disabled.

    Raises:
        ValueError: When ``accel`` is requested but this policy cannot produce it — its
            sampler is not wired for ``accel``, its config has no denoise schedule, or the
            requested prefix is invalid for that schedule.
    """
    requested: int | str | None = override
    if requested is None:
        requested = getattr(cfg.policy, "accel_prefix", None)
    if requested is None:
        requested = os.environ.get(ACCEL_PREFIX_ENV)
    if requested is None or (isinstance(requested, str) and not requested.strip()):
        return None

    policy_type = getattr(cfg.policy, "type", type(cfg.policy).__name__)
    # A policy family whose sampler has not been wired for accel would accept the attribute
    # and then never read it — the exact silent no-op an operator would read as "the score is
    # always missing". Refuse instead.
    if not hasattr(policy, "accel_prefix"):
        raise ValueError(
            f"accel was requested ({requested!r}) but policy type {policy_type!r} does not "
            "expose `accel_prefix`, i.e. its sampler is not wired for denoising acceleration."
        )
    num_steps = getattr(policy.config, "num_steps", None)
    if num_steps is None:
        raise ValueError(
            f"accel was requested ({requested!r}) but policy type {policy_type!r} has no "
            "`num_steps` denoise schedule to read velocities off."
        )

    text = str(requested).strip().lower()
    if text == "auto":
        prefix = default_prefix(num_steps)
    else:
        try:
            requested_int = int(text)
        except ValueError as exc:
            raise ValueError(
                f"accel prefix must be an integer >= {MIN_PREFIX} or the literal 'auto', got {requested!r}."
            ) from exc
        prefix = resolve_prefix(requested_int, num_steps)

    policy.accel_prefix = prefix
    logger.info("accel enabled: prefix=%d of num_steps=%d (requested %r)", prefix, num_steps, requested)
    return prefix


def executed_row_mask(
    *,
    prefix_mask: Tensor,
    delay: Tensor,
    chunk_size: int,
    n_action_steps: int,
    device: torch.device,
) -> Tensor:
    """Build the chunk-row mask a sampler should hand to :meth:`AccelMeter.set_row_mask`.

    Excludes two kinds of row, both of which would otherwise inject rows carrying no
    posterior information into both sums:

    * **Frozen real-time-chunking rows** (``prefix_mask``). Their state is overwritten with
      the committed action before every ``denoise_step`` and their conditioning time is
      pinned to 0, so the velocity there describes nothing. It is masked out of the
      training loss for the same reason.
    * **Rows outside the executed window.** ``select_action`` applies
      ``actions[delay : delay + n_action_steps]`` and re-plans, so later rows never reach
      the robot. Note this is ``[delay, delay + n_action_steps)``, *not* ``[0,
      n_action_steps)`` — the naive slice is wrong whenever ``delay > 0``.

    Args:
        prefix_mask: ``(B, chunk)`` or ``(1, chunk)`` bool; ``True`` marks a frozen row.
        delay: Scalar or ``(B,)`` long tensor of frozen rows.
        chunk_size: Full chunk length.
        n_action_steps: Executed window length.
        device: Device to build on.

    Returns:
        Bool tensor broadcastable to ``(B, chunk)``; ``True`` scores the row.
    """
    positions = rearrange(torch.arange(chunk_size, device=device), "c -> 1 c")
    horizon = delay.reshape(-1, 1).to(device) + n_action_steps if delay.ndim else delay + n_action_steps
    return (~prefix_mask) & (positions < horizon)
