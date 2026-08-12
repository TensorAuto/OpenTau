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
3. **The prefix sweep.** ``accel_p`` for every ``p`` in ``[2, T]``. The paper finds the
   best prefix mid-schedule and warns the last steps are discretization-dominated; this
   shows where that turn happens on your checkpoint rather than assuming the paper's.

The go/no-go it exists to answer: **is the between-observation spread of ``accel`` large
compared to the dtype floor?** If it is not, no threshold calibrated on this checkpoint is
measuring uncertainty, and the fix is to keep the action projection in float32 rather than
to tune the detector.

Run::

    python src/opentau/scripts/diagnose_accel.py \\
        --config_path=configs/examples/pi05_libero_eval_config.json \\
        --policy.path=TensorAuto/<run>@6000

Add ``--policy.device=cuda`` on a GPU box. Everything is inference-only; no training runs.
"""

# NOTE: deliberately no `from __future__ import annotations` here. `parser.wrap` resolves the
# config class with `inspect.getfullargspec(fn).annotations[...]` (configs/parser.py:363), which
# returns the *string* "TrainPipelineConfig" under PEP 563 and then fails inside draccus with an
# unrelated-looking `TypeError: must be called with a dataclass type or instance`. No other
# `@parser.wrap()`-decorated script in the repo imports it either.

import contextlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.accel import default_prefix, resolve_action_dim_mask
from opentau.policies.factory import make_policy
from opentau.policies.utils import to_dtype_preserving_siglip_float32
from opentau.utils.utils import init_logging

logger = logging.getLogger(__name__)


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
        prefix_sweep: ``accel_p`` per prefix ``p``, averaged over observations.
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


def _sweep_prefixes(policy, batch: dict, noise: torch.Tensor, dataset_index) -> dict[int, float]:
    """Return mean ``accel_p`` for every prefix ``p`` in ``[2, T]``.

    Runs the sampler once per prefix. That is ``T - 1`` forward passes and is the one
    genuinely expensive thing this script does — it exists to locate the schedule's
    information peak on *your* checkpoint rather than trusting the paper's, which was
    measured on different data.
    """
    original = policy.accel_prefix
    sweep: dict[int, float] = {}
    try:
        for p in range(2, policy.config.num_steps + 1):
            policy.accel_prefix = p
            scores = _score_once(policy, batch, noise)
            finite = [s for s in scores if np.isfinite(s)]
            sweep[p] = float(np.mean(finite)) if finite else float("nan")
    finally:
        policy.accel_prefix = original
    return sweep


def diagnose_accel(
    policy,
    observations: list[dict],
    *,
    num_noise_repeats: int = 8,
    seed: int = 0,
) -> AccelDiagnosticReport:
    """Measure the ``accel`` floor, spread, and prefix profile of a loaded policy.

    Args:
        policy: A loaded flow-matching policy exposing ``accel_prefix`` / ``last_accel``.
        observations: Distinct observation batches. At least two are needed for the
            between-observation spread that the go/no-go ratio compares against.
        num_noise_repeats: Repeats on the first observation with freshly drawn noise.
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

    logger.info("Re-running in float32 to isolate the rounding floor...")
    float32_scores: list[float] = []
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

    paired = [
        abs(a - b)
        for a, b in zip(float32_scores, serving_scores, strict=True)
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

    # (3) where in the schedule the signal actually lives.
    prefix_sweep = _sweep_prefixes(policy, observations[0], noise, dataset_index)

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
        signal_to_floor=(
            observation_spread / dtype_floor if dtype_floor and np.isfinite(dtype_floor) else float("inf")
        ),
        prefix_sweep=prefix_sweep,
    )


def format_report(report: AccelDiagnosticReport) -> str:
    """Render a human-readable verdict.

    The threshold below is a judgement call, not a theorem: at a signal-to-floor ratio near
    1 the score is dominated by bf16 rounding, and by ~10 the rounding is a minor
    perturbation. Between those, treat the number as suggestive and look at the raw scores.
    """
    ratio = report.signal_to_floor
    if ratio < 3.0:
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
        "  Note: the largest accel_p is not automatically the best prefix — the paper's "
        "1/(1-s) tail inflates the final steps with discretization noise rather than "
        "posterior information. Prefer a prefix where the sweep is rising, not its peak "
        "at p == T."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


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
    dataset = mixture.datasets[0] if hasattr(mixture, "datasets") else mixture
    total = len(dataset)
    if total == 0:
        raise ValueError("The configured dataset mixture is empty; cannot draw observations.")

    stride = max(1, total // count)
    indices = [min(i * stride, total - 1) for i in range(count)]
    logger.info("Drawing %d observation(s) from %d frames at stride %d: %s", count, total, stride, indices)

    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1)
    observations = []
    for batch in loader:
        observations.append(
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
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

    logger.info("Loading policy %s from %s", cfg.policy.type, cfg.policy.pretrained_path)
    policy = make_policy(cfg=cfg.policy)
    policy.eval()

    device = torch.device(getattr(cfg.policy, "device", None) or "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    policy.to(device=device)
    to_dtype_preserving_siglip_float32(policy, dtype=dtype)

    count = 8
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

    report = diagnose_accel(policy, observations)
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
