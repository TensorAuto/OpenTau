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

"""Measure the ``accel`` noise floor and stochasticity of a checkpoint — run this FIRST.

``accel`` (:mod:`opentau.policies.accel`) is a ratio whose numerator is a difference of
nearly-equal velocity vectors, i.e. a catastrophic-cancellation regime. OpenTau serves
policies in bfloat16 and the velocity projection runs in that dtype before being widened
to float32, so there is a **positive-biased rounding floor beneath every score**. The bias
compresses the low-uncertainty end — exactly where the paper's "certain field => accel -> 0"
premise lives — so a checkpoint whose real signal sits near that floor is reporting
arithmetic, not geometry.

This script measures three things on the actual checkpoint, with no repo state changed:

1. **The dtype floor.** The same observation and the *same fixed noise* denoised in
   float32 and in the serving dtype. Any difference is pure rounding, because the field,
   the conditioning, and the noise draw are identical.
2. **The stochasticity.** Repeats of the same observation with freshly drawn noise.
   ``sample_actions`` takes no generator, so production scores are a random variable; the
   detector's calibration has to absorb this spread.
3. **The prefix study.** Which ``p`` makes ``accel_p`` track the *actual* posterior spread
   best. ``default_prefix`` returns ``T - 1`` because that is what the paper's online
   detector used; the same paper's sweep peaks nearer ``p/T ~ 0.4-0.5``, and on a ``T = 5``
   OpenTau schedule those are different answers. Settling it needs a reference measure of
   spread, which only resampling can provide: ``K`` independent denoises of one observation,
   whose disagreement *is* the posterior width. The figure of merit is then the rank
   correlation between ``accel_p`` and that reference, across observations.

   The mean-``accel_p``-per-``p`` column is also reported, but it cannot select a prefix and
   should not be read as if it could — numerator and denominator both accumulate over the
   prefix, so the curve rises with ``p`` whether or not the extra steps carry information.

The go/no-go it exists to answer: **is the between-observation spread of ``accel`` large
compared to the dtype floor?** If it is not, no threshold calibrated on this checkpoint is
measuring uncertainty, and the fix is to keep the action projection in float32 rather than
to tune the detector.

Run::

    python src/opentau/scripts/diagnose_accel.py \\
        --config_path=configs/examples/pi05_libero_eval_config.json \\
        --policy.path=TensorAuto/<run>@6000

Add ``--policy.device=cuda`` on a GPU box. Everything is inference-only; no training runs.

Three env vars tune cost, since ``@parser.wrap()`` parses only ``TrainPipelineConfig``
fields and none of these belong in a training config:

* ``OPENTAU_ACCEL_OBSERVATIONS`` (default 24) — the prefix study's sample size.
* ``OPENTAU_ACCEL_RESAMPLES`` (default 32) — ``K``. Set ``0`` to skip the study, which is
  the only genuinely expensive part (``observations * K`` denoise passes).
* ``OPENTAU_ACCEL_MEASURE_DTYPE_FLOOR`` (default 1) — set ``0`` to skip the float32 leg,
  which holds a second copy of the weights and is the first thing to fail on a GPU shared
  with other work. The floor does not change between runs, so measure it once.
"""

# NOTE: deliberately no `from __future__ import annotations` here. `parser.wrap` resolves the
# config class with `inspect.getfullargspec(fn).annotations[...]` (configs/parser.py:363), which
# returns the *string* "TrainPipelineConfig" under PEP 563 and then fails inside draccus with an
# unrelated-looking `TypeError: must be called with a dataclass type or instance`. No other
# `@parser.wrap()`-decorated script in the repo imports it either.

import contextlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, reduce

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.accel import (
    action_dim_scale,
    default_prefix,
    record_traces,
    resolve_action_dim_mask,
)
from opentau.policies.candidates import refuse_candidates
from opentau.policies.factory import make_policy
from opentau.policies.utils import to_dtype_preserving_siglip_float32
from opentau.utils.utils import init_logging

logger = logging.getLogger(__name__)

# Resamples per observation for the prefix study. `0` skips it, which is the right setting
# once a checkpoint's prefix is settled — it is the only expensive part of this script.
RESAMPLES_ENV = "OPENTAU_ACCEL_RESAMPLES"

# Observations drawn from the configured mixture. This is the prefix study's sample size —
# the rank correlation is taken across observations, so too few makes every rho unresolvable.
OBSERVATIONS_ENV = "OPENTAU_ACCEL_OBSERVATIONS"

# Set to `0` to skip the float32 leg of the dtype-floor measurement. It holds a second,
# wider copy of the weights, so it is the first thing to fail on a GPU shared with other
# work — and the floor it measures does not change between runs on one checkpoint.
DTYPE_FLOOR_ENV = "OPENTAU_ACCEL_MEASURE_DTYPE_FLOOR"


@dataclass
class PrefixStudy:
    """Which prefix ``p`` actually makes ``accel_p`` track posterior spread best.

    Attributes:
        num_resamples: ``K`` denoise passes per observation behind each spread estimate.
        num_observations: Points entering each rank correlation.
        divergences: Per-observation posterior spread (the reference), normalized units.
        accel_by_prefix: ``{p: per-observation mean accel_p}``.
        rho: ``{p: Spearman(accel_p, divergence)}``. The figure of merit.
        best_prefix: ``argmax_p rho[p]``, or ``None`` if no correlation was computable.
        default_prefix: What :func:`~opentau.policies.accel.default_prefix` would return.
        rho_at_default: ``rho[default_prefix]``, for the "is the default good enough?" call.
    """

    num_resamples: int
    num_observations: int
    divergences: list[float]
    accel_by_prefix: dict[int, list[float]]
    rho: dict[int, float]
    best_prefix: int | None
    default_prefix: int
    rho_at_default: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy."""
        return {
            "num_resamples": self.num_resamples,
            "num_observations": self.num_observations,
            "divergences": self.divergences,
            "accel_by_prefix": {str(k): v for k, v in self.accel_by_prefix.items()},
            "rho": {str(k): v for k, v in self.rho.items()},
            "best_prefix": self.best_prefix,
            "default_prefix": self.default_prefix,
            "rho_at_default": self.rho_at_default,
        }


@dataclass
class AccelDiagnosticReport:
    """Everything :func:`diagnose_accel` measured, JSON-serializable.

    Attributes:
        num_steps: The checkpoint's denoise schedule length ``T``.
        prefix: Prefix used for the headline numbers.
        num_scored_dims: Action dims that survived the dim mask.
        max_action_dim: Sampler action width, for contrast with the above.
        float32_scores: Per-observation ``accel`` under float32.
        serving_scores: Per-observation ``accel`` under the serving dtype.
        serving_dtype: The serving dtype, as a string.
        dtype_floor: Median absolute float32-vs-serving difference — the rounding floor.
        noise_spread: Std of repeated scores on one observation with redrawn noise.
        observation_spread: Std of scores across distinct observations (the signal).
        signal_to_floor: ``observation_spread / dtype_floor``. The go/no-go ratio.
        prefix_sweep: ``accel_p`` per prefix ``p``, averaged over observations. Descriptive
            only — see :class:`PrefixStudy` for the number that actually selects a prefix.
        prefix_study: The resampling-based prefix selection, when it was run.
        observation_source: Where the observations came from. Recorded because a run on
            synthetic frames measures the out-of-distribution regime, not deployment, and
            the two are not comparable.
    """

    num_steps: int
    prefix: int
    num_scored_dims: list[int]
    max_action_dim: int
    float32_scores: list[float]
    serving_scores: list[float]
    serving_dtype: str
    dtype_floor: float
    noise_spread: float
    observation_spread: float
    signal_to_floor: float
    prefix_sweep: dict[int, float] = field(default_factory=dict)
    prefix_study: PrefixStudy | None = None
    observation_source: str = "unspecified"

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy."""
        return {
            "num_steps": self.num_steps,
            "prefix": self.prefix,
            "num_scored_dims": self.num_scored_dims,
            "max_action_dim": self.max_action_dim,
            "float32_scores": self.float32_scores,
            "serving_scores": self.serving_scores,
            "serving_dtype": self.serving_dtype,
            "dtype_floor": self.dtype_floor,
            "noise_spread": self.noise_spread,
            "observation_spread": self.observation_spread,
            "signal_to_floor": self.signal_to_floor,
            "prefix_sweep": {str(k): v for k, v in self.prefix_sweep.items()},
            "prefix_study": self.prefix_study.to_dict() if self.prefix_study else None,
            "observation_source": self.observation_source,
        }


def _fixed_noise(policy, batch_size: int, device: torch.device, seed: int) -> torch.Tensor:
    """Draw a reproducible noise tensor of the sampler's action shape.

    ``sample_actions`` accepts ``noise=`` precisely so a caller can pin it; production
    always passes ``None`` and draws from the ungeneratored global RNG.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (batch_size, policy.config.chunk_size, policy.config.max_action_dim)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


@contextlib.contextmanager
def _forced_float32_embeddings(policy):
    """Make the pi0/pi05-family forward actually run in float32, not just its parameters.

    Those modeling modules route their embedding path through a module-level
    ``_preferred_dtype()`` that returns **bfloat16 unconditionally** (float32 only under ONNX
    export), so ``policy.to(torch.float32)`` alone produces bfloat16 activations meeting
    float32 weights and dies in the first attention projection. Patching the hook is the same
    mechanism the repo's own pi05 tests use to run this model on CPU in float32.

    That the hook exists at all is the finding this diagnostic is built around: the served
    numeric precision is fixed in code, not chosen by the caller's cast.

    There is more than one such hook per family — pi05 defines one in ``modeling_pi05`` (the
    image/state/action embeddings) and a second in ``paligemma_with_expert`` (the backbone's
    internal casts). Patching only the first still leaves bfloat16 activations arriving at
    float32 attention weights, so every hook under ``opentau.policies`` is swept.

    Args:
        policy: The loaded policy.

    Yields:
        ``True`` when at least one hook was found and patched, ``False`` when the family has
        none (in which case the caller is relying on the parameter cast alone).
    """
    patched = {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith("opentau.policies")
        and module is not None
        and callable(getattr(module, "_preferred_dtype", None))
    }
    if not patched:
        yield False
        return

    originals = {name: module._preferred_dtype for name, module in patched.items()}
    logger.info(
        "Forcing float32 activations via %d _preferred_dtype hook(s): %s", len(patched), sorted(patched)
    )
    for module in patched.values():
        module._preferred_dtype = lambda: torch.float32
    try:
        yield True
    finally:
        for name, module in patched.items():
            module._preferred_dtype = originals[name]


def _score_once(policy, batch: dict, noise: torch.Tensor | None) -> list[float]:
    """Run one ``sample_actions`` and return the per-sample ``accel``.

    The batch is re-cast to the policy's *current* parameter dtype on every call. This
    matters because the whole point of the dtype-floor measurement is to run the identical
    observation through the identical field twice at two precisions: the dataloader hands
    back bfloat16 frames, so the float32 leg would otherwise feed bfloat16 activations into
    float32 weights and die inside the first attention projection. Integer and boolean
    entries (masks, indices, pad flags) are left alone.
    """
    dtype = next(policy.parameters()).dtype
    batch = {
        key: (value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
        for key, value in batch.items()
    }
    if noise is not None:
        noise = noise.to(dtype)
    with torch.inference_mode():
        policy.sample_actions(batch, noise=noise)
    return list(policy.last_accel or [])


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation between two equal-length samples, NaN pairs dropped.

    Rank-based on purpose: the prefix study correlates ``accel_p`` against a resampled
    posterior spread, and the two are not on a common scale or even a common curve — only
    their *ordering* is claimed to agree. Any monotone redefinition of the spread (variance
    vs standard deviation, per-element vs summed) leaves this number unchanged, which is
    what keeps the conclusion from depending on an arbitrary choice of divergence.
    """
    from scipy.stats import spearmanr

    paired = [(x, y) for x, y in zip(xs, ys, strict=True) if np.isfinite(x) and np.isfinite(y)]
    if len(paired) < 3:
        return float("nan")
    a = [p[0] for p in paired]
    b = [p[1] for p in paired]
    # `spearmanr` warns and returns NaN on a constant input; say so ourselves instead.
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _posterior_spread(policy, chunks: list[torch.Tensor], dataset_index: torch.Tensor) -> list[float]:
    """Reduce ``K`` resampled action chunks to one posterior-spread scalar per sample.

    This is the **reference quantity** the prefix study scores ``accel_p`` against: the
    thing ``accel`` claims to be a cheap proxy for. It costs ``K`` full denoise passes, which
    is exactly why it is confined to this offline diagnostic.

    Two details make it commensurate with ``accel``, which is computed on *normalized*
    velocities over a masked subset of the chunk:

    * **Per-dim standardization.** ``sample_actions`` returns raw units, where a gripper in
      ``[0, 1]`` and a shoulder joint in radians would contribute to a spread on wildly
      different scales — and since the scale vector is fixed across observations, that
      skewed weighting survives the rank correlation rather than cancelling out.
      :func:`~opentau.policies.accel.action_dim_scale` undoes it.
    * **The same masks.** Padded action dims are unsupervised network output, and chunk rows
      past the executed window are never applied; both are excluded here exactly as the
      meter excludes them.

    A delta-action policy re-adds the chunk-start state before returning, but that offset is
    identical across the ``K`` resamples of one observation, so it cancels in the deviation
    and no delta-space inverse is needed here.

    Args:
        policy: The policy the chunks came from.
        chunks: ``K`` tensors of shape ``(B, chunk, dim)``, one per resample.
        dataset_index: ``(B,)`` norm-head row indices.

    Returns:
        ``B`` floats: the root-mean-square per-element standard deviation across resamples,
        in normalized units. ``sqrt(tr(Sigma) / n)`` of the scored sub-block.
    """
    stacked = torch.stack([c.to(torch.float32) for c in chunks], dim=0)  # (K, B, chunk, dim)
    width = stacked.shape[-1]
    scale = action_dim_scale(
        policy.unnormalize_outputs,
        max_action_dim=width,
        dataset_index=dataset_index,
    ).to(stacked.device)
    dim_mask = resolve_action_dim_mask(
        policy.unnormalize_outputs,
        max_action_dim=width,
        dataset_index=dataset_index,
    ).to(stacked.device)

    deviation = stacked.std(dim=0, unbiased=True) / rearrange(scale, "b d -> b 1 d")

    mask = rearrange(dim_mask.to(torch.float32), "b d -> b 1 d").expand_as(deviation).clone()
    executed = min(policy.config.n_action_steps, deviation.shape[1])
    mask[:, executed:, :] = 0.0

    total = reduce(mask, "b c d -> b", "sum")
    summed = reduce(deviation.pow(2) * mask, "b c d -> b", "sum")
    spread = torch.sqrt(summed / total.clamp_min(1.0))
    return [float(x) if n > 0 else float("nan") for x, n in zip(spread.tolist(), total.tolist(), strict=True)]


def resample_observation(
    policy, batch: dict, dataset_index: torch.Tensor, *, num_resamples: int, seed: int
) -> tuple[float, dict[int, float]]:
    """Denoise one observation ``K`` times and return its spread and per-prefix scores.

    Each resample uses an independent noise draw, so the ``K`` chunks are ``K`` samples from
    the policy's action posterior for this observation. Their spread is the ground truth;
    the ``accel_p`` values are the candidate proxies, averaged over the same ``K`` draws so
    the comparison is not decided by one lucky noise vector.

    Args:
        policy: A policy with ``accel_prefix`` already set to ``T`` (the full schedule), so
            every prefix is present in the trace.
        batch: One observation.
        dataset_index: ``(B,)`` norm-head row indices for it.
        num_resamples: ``K``.
        seed: Base seed; resample ``k`` uses ``seed + k``.

    Returns:
        ``(spread, {prefix: mean accel_p})``, both averaged over the batch axis.
    """
    device = next(policy.parameters()).device
    batch_size = len(batch["state"])
    chunks: list[torch.Tensor] = []
    per_prefix: dict[int, list[float]] = {}

    for k in range(num_resamples):
        noise = _fixed_noise(policy, batch_size, device, seed + k)
        with record_traces() as meters:
            dtype = next(policy.parameters()).dtype
            cast = {
                key: (val.to(dtype) if isinstance(val, torch.Tensor) and val.is_floating_point() else val)
                for key, val in batch.items()
            }
            with torch.inference_mode():
                chunks.append(policy.sample_actions(cast, noise=noise.to(dtype)).to(torch.float32))
        if not meters:
            raise RuntimeError(
                "accel produced no meter for this resample — `policy.accel_prefix` must be "
                "set before calling `resample_observation`."
            )
        for prefix, values in meters[-1].prefix_values().items():
            per_prefix.setdefault(prefix, []).extend(values)

    spreads = _posterior_spread(policy, chunks, dataset_index)
    finite_spreads = [s for s in spreads if np.isfinite(s)]

    def _mean(values: list[float]) -> float:
        finite = [v for v in values if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    return (
        float(np.mean(finite_spreads)) if finite_spreads else float("nan"),
        {prefix: _mean(values) for prefix, values in per_prefix.items()},
    )


def measure_prefix_quality(
    policy,
    observations: list[dict],
    *,
    num_resamples: int = 32,
    seed: int = 0,
) -> PrefixStudy:
    """Rank every candidate prefix by how well ``accel_p`` tracks the true posterior spread.

    This is the measurement that :func:`~opentau.policies.accel.default_prefix` is otherwise
    only *assuming*. ``default_prefix`` returns ``T - 1`` because that is what the paper's
    online detector used, on the paper's checkpoints and data — a reasonable prior, but the
    same paper's own sweep peaks nearer ``p/T ~ 0.4-0.5``, so on a short OpenTau schedule
    (``T = 5`` for pi06/pi07) the two answers are far apart and nothing in the repo had ever
    checked which one applies here.

    A mean-``accel_p``-versus-``p`` sweep cannot settle it: ``accel_p`` grows with ``p``
    almost by construction, since both sums accumulate over the prefix. Magnitude is not
    quality. The paper's actual criterion is the rank correlation between ``accel_p`` and a
    *reference* measure of posterior spread, and the reference has to be obtained the
    expensive way — by resampling (:func:`_posterior_spread`). That is affordable exactly
    once, offline, which is what this function is.

    Cost is ``len(observations) * num_resamples`` denoise passes; every prefix comes out of
    the same passes, so widening the prefix range is free.

    Args:
        policy: A loaded flow-matching policy. Its ``accel_prefix`` is set to ``T`` for the
            duration and restored afterwards.
        observations: Distinct observations. The correlation is taken *across* these, so
            they must span a range of genuine uncertainty for the result to mean anything —
            frames drawn at a stride through real episodes, not repeats of one scene.
        num_resamples: ``K``, the resamples per observation used to estimate the spread.
        seed: Base seed for the noise draws.

    Returns:
        The :class:`PrefixStudy`.

    Raises:
        ValueError: When fewer than three observations are supplied — a rank correlation
            over two points is either +1 or -1 regardless of the data.
    """
    if len(observations) < 3:
        raise ValueError(
            f"the prefix study correlates accel_p against posterior spread across "
            f"observations, and {len(observations)} of them cannot produce a meaningful rank "
            "correlation (any two points correlate perfectly). Supply at least 3; 8+ is better."
        )

    original = policy.accel_prefix
    divergences: list[float] = []
    accel_by_prefix: dict[int, list[float]] = {}
    try:
        # The full schedule, so the trace covers every candidate prefix in one pass.
        policy.accel_prefix = policy.config.num_steps
        for i, obs in enumerate(observations):
            dataset_index = policy._resolve_dataset_index(obs)
            spread, per_prefix = resample_observation(
                policy, obs, dataset_index, num_resamples=num_resamples, seed=seed + 1000 * i
            )
            divergences.append(spread)
            for prefix, value in per_prefix.items():
                accel_by_prefix.setdefault(prefix, []).append(value)
            logger.info(
                "observation %d/%d: posterior spread %.6g over %d resamples",
                i + 1,
                len(observations),
                spread,
                num_resamples,
            )
    finally:
        policy.accel_prefix = original

    rho = {prefix: _spearman(values, divergences) for prefix, values in accel_by_prefix.items()}
    finite_rho = {p: r for p, r in rho.items() if np.isfinite(r)}
    best = max(finite_rho, key=lambda p: finite_rho[p]) if finite_rho else None
    fallback = default_prefix(policy.config.num_steps)

    return PrefixStudy(
        num_resamples=num_resamples,
        num_observations=len(observations),
        divergences=divergences,
        accel_by_prefix={p: list(v) for p, v in accel_by_prefix.items()},
        rho=rho,
        best_prefix=best,
        default_prefix=fallback,
        rho_at_default=rho.get(fallback, float("nan")),
    )


def diagnose_accel(
    policy,
    observations: list[dict],
    *,
    num_noise_repeats: int = 8,
    num_resamples: int = 32,
    measure_dtype_floor: bool = True,
    seed: int = 0,
) -> AccelDiagnosticReport:
    """Measure the ``accel`` floor, spread, and prefix profile of a loaded policy.

    Args:
        policy: A loaded flow-matching policy exposing ``accel_prefix`` / ``last_accel``.
        observations: Distinct observation batches. At least two are needed for the
            between-observation spread that the go/no-go ratio compares against.
        num_noise_repeats: Repeats on the first observation with freshly drawn noise.
        num_resamples: ``K`` for the prefix study; ``0`` skips it.
        measure_dtype_floor: Run the float32 leg. Skipping it leaves ``dtype_floor`` and
            ``signal_to_floor`` NaN, and is the way to fit on a GPU shared with other work.
        seed: Seed for the pinned noise draw.

    Returns:
        The :class:`AccelDiagnosticReport`.

    Raises:
        ValueError: When fewer than two observations are supplied.
    """
    if len(observations) < 2:
        raise ValueError(
            "accel diagnosis needs at least two distinct observations — the go/no-go "
            "compares between-observation spread against the rounding floor, and one "
            "observation has no spread."
        )

    config = policy.config
    device = next(policy.parameters()).device
    serving_dtype = next(policy.parameters()).dtype
    policy.accel_prefix = default_prefix(config.num_steps)

    batch_size = len(observations[0]["state"])
    dataset_index = policy._resolve_dataset_index(observations[0])
    dim_mask = resolve_action_dim_mask(
        policy.unnormalize_outputs,
        max_action_dim=config.max_action_dim,
        dataset_index=dataset_index,
    )
    num_scored = [int(n) for n in dim_mask.sum(dim=-1).cpu().tolist()]
    logger.info(
        "accel scores %s of %d action dims (the rest are zero-variance in the norm stats "
        "and are treated as unsupervised padding)",
        num_scored,
        config.max_action_dim,
    )

    # (1) dtype floor: identical field, identical noise, only the arithmetic differs.
    noise = _fixed_noise(policy, batch_size, device, seed)
    serving_scores: list[float] = []
    for obs in observations:
        serving_scores.extend(_score_once(policy, obs, noise))

    float32_scores: list[float] = []
    if measure_dtype_floor:
        logger.info("Re-running in float32 to isolate the rounding floor...")
        with _forced_float32_embeddings(policy) as forced:
            if not forced:
                logger.warning(
                    "This policy family has no `_preferred_dtype` hook, so the float32 leg relies "
                    "on the parameter cast alone. If its forward pins an activation dtype the way "
                    "the pi0/pi05 family does, the measured floor will be wrong."
                )
            policy.to(dtype=torch.float32)
            for obs in observations:
                float32_scores.extend(_score_once(policy, obs, noise))

        # `to_dtype_preserving_siglip_float32` rather than a blanket cast: pi0/pi05-family
        # towers pin the SigLIP patch-embedding conv and position table to float32 for openpi
        # parity, and a blanket `.to(bfloat16)` would silently re-round them (CLAUDE.md rule 6).
        to_dtype_preserving_siglip_float32(policy, dtype=serving_dtype)
    else:
        # Holding a float32 copy of the weights roughly doubles resident memory, which is
        # what makes this the first leg to fail on a shared GPU. The floor is a fixed
        # property of (checkpoint, dtype) — it does not move between runs — so skipping it
        # to get the prefix study done is a reasonable trade, as long as the report does not
        # then present a fabricated ratio. It reports NaN.
        logger.warning(
            "Skipping the float32 leg: the dtype floor and signal/floor ratio will be NaN. "
            "Run once with it enabled to establish the floor for this checkpoint."
        )

    paired = [
        abs(a - b)
        for a, b in zip(float32_scores, serving_scores, strict=False)
        if np.isfinite(a) and np.isfinite(b)
    ]
    dtype_floor = float(np.median(paired)) if paired else float("nan")

    # (2) stochasticity: same observation, freshly drawn noise each time.
    repeats: list[float] = []
    for _ in range(num_noise_repeats):
        repeats.extend(_score_once(policy, observations[0], None))
    noise_spread = float(np.std([r for r in repeats if np.isfinite(r)]))

    finite_serving = [s for s in serving_scores if np.isfinite(s)]
    observation_spread = float(np.std(finite_serving)) if finite_serving else float("nan")

    # (3) where in the schedule the signal actually lives. One pass at the full schedule
    # yields every prefix, because accel_p is a prefix statistic of one velocity sequence.
    original_prefix = policy.accel_prefix
    try:
        policy.accel_prefix = config.num_steps
        with record_traces() as meters:
            _score_once(policy, observations[0], noise)
        prefix_sweep = {
            prefix: float(np.mean([v for v in values if np.isfinite(v)]))
            if any(np.isfinite(v) for v in values)
            else float("nan")
            for prefix, values in meters[-1].prefix_values().items()
        }
    finally:
        policy.accel_prefix = original_prefix

    # (4) which prefix actually correlates with posterior spread — the sweep above cannot
    # answer this, since accel_p rises with p regardless of how informative it is.
    prefix_study = None
    if num_resamples > 0 and len(observations) >= 3:
        logger.info(
            "Measuring prefix quality: %d observations x %d resamples = %d denoise passes...",
            len(observations),
            num_resamples,
            len(observations) * num_resamples,
        )
        prefix_study = measure_prefix_quality(policy, observations, num_resamples=num_resamples, seed=seed)

    return AccelDiagnosticReport(
        num_steps=int(config.num_steps),
        prefix=int(policy.accel_prefix),
        num_scored_dims=num_scored,
        max_action_dim=int(config.max_action_dim),
        float32_scores=[float(x) for x in float32_scores],
        serving_scores=[float(x) for x in serving_scores],
        serving_dtype=str(serving_dtype).removeprefix("torch."),
        dtype_floor=dtype_floor,
        noise_spread=noise_spread,
        observation_spread=observation_spread,
        signal_to_floor=_signal_to_floor(observation_spread, dtype_floor),
        prefix_sweep=prefix_sweep,
        prefix_study=prefix_study,
    )


def _signal_to_floor(observation_spread: float, dtype_floor: float) -> float:
    """Return the go/no-go ratio, or NaN when there is no floor to compare against.

    The distinction this exists to preserve: a floor that was **never measured** (the
    float32 leg was skipped, or no observation produced a finite pair) is not the same as a
    floor measured at zero. Collapsing them to ``inf`` — as a plain
    ``if dtype_floor and isfinite(...) else inf`` does, because NaN is truthy and not finite
    — makes an unmeasured run print the most reassuring verdict the script has.

    Args:
        observation_spread: Std of ``accel`` across observations (the signal).
        dtype_floor: Median float32-vs-serving difference, or NaN if unmeasured.

    Returns:
        The ratio; ``inf`` when the floor was measured as exactly zero (float32 and the
        serving dtype agreed on every observation); NaN when it was never measured.
    """
    if not np.isfinite(dtype_floor):
        return float("nan")
    if dtype_floor > 0.0:
        return observation_spread / dtype_floor
    return float("inf")


def format_report(report: AccelDiagnosticReport) -> str:
    """Render a human-readable verdict.

    The threshold below is a judgement call, not a theorem: at a signal-to-floor ratio near
    1 the score is dominated by bf16 rounding, and by ~10 the rounding is a minor
    perturbation. Between those, treat the number as suggestive and look at the raw scores.
    """
    ratio = report.signal_to_floor
    if np.isnan(ratio):
        # Must precede the thresholds: every `<` against NaN is False, so a NaN ratio would
        # otherwise fall through to the "OK" branch and claim the signal dominates a floor
        # that was never measured.
        verdict = (
            "NOT MEASURED — the float32 leg was skipped, so there is no rounding floor to "
            "compare the signal against and this run cannot answer the go/no-go. Re-run "
            "once with OPENTAU_ACCEL_MEASURE_DTYPE_FLOOR=1 (the floor is fixed per "
            "checkpoint, so once is enough)."
        )
    elif ratio < 3.0:
        verdict = (
            "STOP — the between-observation spread is within 3x of the pure-rounding floor, "
            "so accel here is mostly measuring bfloat16 arithmetic. Keep the action "
            "projection in float32 before calibrating any threshold."
        )
    elif ratio < 10.0:
        verdict = (
            "MARGINAL — real signal exceeds the rounding floor but not comfortably. Expect "
            "the certain end of the range to be compressed; prefer a float32 action "
            "projection if latency allows."
        )
    else:
        verdict = "OK — real signal dominates the rounding floor."

    best_prefix = None
    finite_sweep = {p: v for p, v in report.prefix_sweep.items() if np.isfinite(v)}
    if finite_sweep:
        best_prefix = max(finite_sweep, key=lambda p: finite_sweep[p])

    lines = [
        "",
        "=" * 78,
        "accel diagnostic",
        "=" * 78,
        f"  schedule            T = {report.num_steps}, prefix p = {report.prefix}",
        f"  scored action dims  {report.num_scored_dims} of {report.max_action_dim}",
        f"  serving dtype       {report.serving_dtype}",
        f"  observations        {report.observation_source}",
        "",
        f"  dtype floor         {report.dtype_floor:.6g}   (median |float32 - serving|, same noise)",
        f"  noise spread        {report.noise_spread:.6g}   (std over redrawn noise, one observation)",
        f"  observation spread  {report.observation_spread:.6g}   (std across observations = the signal)",
        f"  signal / floor      {report.signal_to_floor:.2f}",
        "",
        f"  {verdict}",
        "",
        "  prefix sweep (mean accel_p):",
    ]
    for p, value in sorted(report.prefix_sweep.items()):
        marker = "  <- max" if p == best_prefix else ""
        lines.append(f"    p={p:>3}/{report.num_steps}  {value:.6g}{marker}")
    lines.append("")
    lines.append(
        "  The largest accel_p is NOT the best prefix. Both of its sums accumulate over the "
        "prefix, so it rises with p almost regardless of how informative the score is; the "
        "column above is descriptive only. The selection criterion is below."
    )

    study = report.prefix_study
    if study is None:
        lines.append("")
        lines.append(
            "  No prefix study was run (needs >= 3 observations and num_resamples > 0), so "
            f"the prefix in use, p = {report.prefix}, is the paper's default rather than a "
            "measurement on this checkpoint."
        )
    else:
        lines.append("")
        lines.append(
            f"  prefix quality (Spearman rho vs resampled posterior spread, "
            f"K={study.num_resamples}, n={study.num_observations}):"
        )
        for p, value in sorted(study.rho.items()):
            marks = []
            if p == study.best_prefix:
                marks.append("<- best")
            if p == study.default_prefix:
                marks.append("(default_prefix)")
            suffix = ("  " + " ".join(marks)) if marks else ""
            shown = "   nan" if not np.isfinite(value) else f"{value:+.3f}"
            lines.append(f"    p={p:>3}/{report.num_steps}  rho = {shown}{suffix}")
        lines.append("")
        lines.extend(_prefix_verdict(study))

    lines.append("=" * 78)
    return "\n".join(lines)


def _prefix_verdict(study: PrefixStudy) -> list[str]:
    """Turn the measured rank correlations into an actionable recommendation."""
    if study.best_prefix is None:
        return [
            "  INCONCLUSIVE — no prefix produced a computable correlation. Usually every "
            "observation shared one spread value; draw observations from more episodes."
        ]

    best_rho = study.rho[study.best_prefix]
    if best_rho < 0.3:
        return [
            f"  WEAK — even the best prefix (p={study.best_prefix}) correlates only "
            f"rho={best_rho:+.3f} with the resampled posterior spread on this checkpoint. "
            "accel is a poor proxy here whatever prefix you pick, so a detector built on it "
            "will be working with little signal. Treat downstream detection numbers as a "
            "test of that, not as a tuning problem.",
        ]

    lines = [
        f"  Best prefix p={study.best_prefix} (rho={best_rho:+.3f}); "
        f"default_prefix would pick p={study.default_prefix} (rho={study.rho_at_default:+.3f})."
    ]
    gap = best_rho - study.rho_at_default
    if np.isfinite(gap) and gap > 0.1:
        lines.append(
            f"  The default gives up {gap:.3f} of rank correlation here. Set "
            f"`--policy.accel_prefix={study.best_prefix}` for this checkpoint, and calibrate "
            "any threshold under the same prefix (it is part of AccelProvenance)."
        )
    else:
        lines.append(
            "  The default is within noise of the best; no reason to override it on this checkpoint."
        )
    return lines


def _allocate_draws(sizes: list[int], count: int) -> list[int]:
    """Split ``count`` observations across datasets of the given sizes.

    Fair-share with carry-over: each dataset is offered an even slice of what is *still*
    outstanding, capped by what it actually holds, so a member smaller than its slice is
    made up by the ones after it instead of quietly lowering the total. The total is also a
    hard ceiling — asking for fewer observations than there are datasets must not round up
    to one-each, since every extra observation costs ``K`` denoise passes in the study.

    Args:
        sizes: Frame count per dataset. Empty datasets should be filtered out first.
        count: Total observations wanted.

    Returns:
        Per-dataset draw counts, summing to ``min(count, sum(sizes))``.
    """
    quotas = [0] * len(sizes)
    remaining = min(count, sum(sizes))
    while remaining > 0:
        # Re-passing is what makes the carry-over work in both directions. One pass can only
        # push a shortfall *forward*, so a mixture whose LAST member is the short one would
        # still come up short — the datasets with spare frames have already been offered
        # their slice by then.
        hungry = [i for i, size in enumerate(sizes) if quotas[i] < size]
        if not hungry:
            break
        progressed = False
        for position, index in enumerate(hungry):
            if remaining <= 0:
                break
            share = max(1, remaining // (len(hungry) - position))
            take = min(share, remaining, sizes[index] - quotas[index])
            if take > 0:
                quotas[index] += take
                remaining -= take
                progressed = True
        if not progressed:
            break
    return quotas


def dataset_observations(cfg: TrainPipelineConfig, device: torch.device, count: int) -> list[dict]:
    """Draw real observation batches from the configured dataset mixture.

    **Strongly preferred over synthetic frames.** The headline number is a ratio of the
    between-observation spread to the rounding floor, and the numerator is only meaningful
    if the observations sit on the manifold the policy was trained on. Random-noise images
    are out of distribution by construction, which the paper's own §2.4 predicts produces a
    chaotic, sharply-curved field — so a synthetic run measures the epistemic-OOD regime
    and reports a spread that says nothing about deployment.

    Frames are taken at a stride across the dataset rather than consecutively: adjacent
    frames of one episode share a scene and a posterior, so a contiguous window would
    understate the very spread being measured.

    Args:
        cfg: Pipeline config carrying a populated ``dataset_mixture``.
        device: Device to move the batches to.
        count: How many single-sample observations to draw.

    Returns:
        ``count`` batches, each ready to hand to ``policy.sample_actions``.

    Raises:
        ValueError: When the mixture yields no frames.
    """
    from torch.utils.data import DataLoader

    from opentau.datasets.factory import make_dataset_mixture

    mixture = make_dataset_mixture(cfg)
    datasets = [d for d in (getattr(mixture, "datasets", None) or [mixture]) if len(d)]
    if not datasets:
        raise ValueError("The configured dataset mixture is empty; cannot draw observations.")

    # Spread the draw across every configured dataset, not just the first. The prefix study
    # correlates accel against posterior spread *across* observations, so it is only as
    # informative as the range of scenes those observations cover; silently sampling one
    # member of a mixture would narrow that range without saying so.
    quotas = _allocate_draws([len(d) for d in datasets], count)
    observations: list[dict] = []
    for position, (dataset, wanted) in enumerate(zip(datasets, quotas, strict=True)):
        if wanted <= 0:
            continue
        total = len(dataset)
        stride = max(1, total // wanted)
        indices = [min(i * stride, total - 1) for i in range(wanted)]
        logger.info(
            "Drawing %d observation(s) from dataset %d (%d frames) at stride %d",
            wanted,
            position,
            total,
            stride,
        )
        for batch in DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1):
            observations.append(
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            )

    if len(observations) != count:
        # Never silent: `count` is the prefix study's sample size, i.e. the thing that decides
        # whether neighbouring prefixes are separable at all. A run that quietly went to
        # half the requested observations produces rank correlations nobody knows to distrust.
        logger.warning(
            "Requested %d observation(s) but the mixture could only supply %d (dataset sizes: "
            "%s). The prefix study's rank correlations are computed over the smaller sample.",
            count,
            len(observations),
            [len(d) for d in datasets],
        )
    return observations


def synthetic_observations(cfg: TrainPipelineConfig, device: torch.device, count: int) -> list[dict]:
    """Fallback observations built from noise, for when no dataset is reachable.

    See :func:`dataset_observations` for why this measures the wrong regime. Kept only so
    the script degrades to *something* rather than failing outright, and the report says
    loudly which mode produced it.
    """
    generator = torch.Generator(device="cpu").manual_seed(1234)
    observations = []
    for _ in range(count):
        obs = {
            "state": torch.randn(1, cfg.policy.max_state_dim, generator=generator).to(device),
            "prompt": ["pick up the object"],
        }
        for key in cfg.policy.image_features:
            obs[key] = torch.rand(1, 3, 224, 224, generator=generator).to(device)
        observations.append(obs)
    return observations


@parser.wrap()
def diagnose_main(cfg: TrainPipelineConfig):
    """Entry point: load the configured policy and report its ``accel`` characteristics.

    Args:
        cfg: Standard OpenTau train/eval config. ``cfg.policy`` selects the checkpoint;
            ``cfg.dataset_mixture`` (when populated) supplies real observations.
    """
    init_logging()
    cfg.validate()

    # This study pins the noise itself (`noise=` on every `sample_actions` call) so that the
    # observation spread it measures is not confounded by the draw. Best-of-N does the
    # opposite — it fans the draw out and then discards all but one — so the two cannot both
    # be in effect.
    refuse_candidates(
        cfg,
        reason="this diagnostic controls the noise draw to isolate the observation-driven "
        "spread of accel, and reads `policy.last_accel` positionally as one score per "
        "observation. Best-of-N would fan the draw out and publish one score per selected "
        "candidate, silently changing what every number in the report means. Run it with "
        "policy.n_candidates=1.",
    )

    logger.info("Loading policy %s from %s", cfg.policy.type, cfg.policy.pretrained_path)
    policy = make_policy(cfg=cfg.policy)
    policy.eval()

    device = torch.device(getattr(cfg.policy, "device", None) or "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    policy.to(device=device)
    to_dtype_preserving_siglip_float32(policy, dtype=dtype)

    # The rank correlation is taken over these, so `count` is the study's sample size: at 8
    # points Spearman needs |rho| >= 0.74 to clear p < 0.05, which is too blunt to separate
    # neighbouring prefixes. 24 is a reasonable floor; the cost is linear in it.
    count = int(os.environ.get(OBSERVATIONS_ENV, "24"))
    datasets = getattr(getattr(cfg, "dataset_mixture", None), "datasets", None)
    if datasets:
        observations = dataset_observations(cfg, device, count)
        source = f"{len(datasets)} configured dataset(s)"
    else:
        logger.warning(
            "No dataset_mixture configured — falling back to SYNTHETIC noise observations. "
            "Those are out-of-distribution by construction, so the measured observation "
            "spread reflects the epistemic/OOD regime rather than deployment. Point "
            "--config_path at a config carrying this checkpoint's training data instead."
        )
        observations = synthetic_observations(cfg, device, count)
        source = "SYNTHETIC noise (not deployment-representative)"

    # `TrainPipelineConfig` has no field for this and `@parser.wrap()` parses only config
    # fields, so the knob follows the same env-var route as `OPENTAU_ACCEL_PREFIX`.
    num_resamples = int(os.environ.get(RESAMPLES_ENV, "32"))
    measure_floor = os.environ.get(DTYPE_FLOOR_ENV, "1") not in ("0", "false", "False")

    report = diagnose_accel(
        policy,
        observations,
        num_resamples=num_resamples,
        measure_dtype_floor=measure_floor,
    )
    report.observation_source = source
    print(format_report(report))  # noqa: T201 — this script's entire purpose is this report

    out_dir = Path(cfg.output_dir or ".") / "accel_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    logger.info("Wrote %s", out_path)


def main():
    """Console entry point."""
    diagnose_main()


if __name__ == "__main__":
    main()
