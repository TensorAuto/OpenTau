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

"""Offline CUSUM + split-conformal failure detector over a per-chunk ``accel`` stream.

Consumes what :mod:`opentau.policies.accel` produces during a rollout and turns it into a
runtime alarm, following *The Geometry of Flow-Matching Uncertainty* (arXiv:2607.27933)
§4.2 and Appendix A/E. Nothing here runs in the serving hot path — calibration is fitted
once against held-out **successful** rollouts and the resulting threshold is persisted
next to the checkpoint.

The statistic is a one-sided CUSUM over the per-chunk score stream :math:`z_t`::

    S_0 = 0
    S_t = max(0, S_{t-1} + z_t - mu_0 - k),      k = c * sigma

with an alarm at the first ``t`` where ``S_t > eta``.

**Why a CUSUM height rather than a time-indexed conformal band.** A time-indexed band has
to align calibration episodes on a common clock, which forces right-padding every
successful rollout with its terminal value up to the longest (timed-out, i.e. *failing*)
episode. Past the longest successful rollout the band is pure padding — frozen constants,
not draws from the test distribution — so exchangeability fails exactly in the
late-horizon regime where long-horizon failures live. Reducing an episode of any length to
a single scalar peak sidesteps alignment entirely.

**Where the conformal guarantee actually bites.** The exchangeable unit is the *episode*,
not the chunk: ``eta`` is an order statistic of per-episode CUSUM peaks. That matters
because the ~28-52 chunk scores inside one episode are strongly dependent (same scene,
adjacent observations, overlapping conditioning), so pooling them as if they were
independent draws would inflate the effective sample size and the realized per-episode
false-alarm rate would exceed ``alpha``. ``mu_0`` and ``sigma`` *are* estimated from the
pooled chunk scores, but they are location/scale nuisance parameters feeding the
statistic, not the conformal object.

**The bf16 floor.** OpenTau serves policies in bfloat16, and the velocity projection runs
in that dtype before being widened to float32. ``accel``'s numerator is a difference of
nearly-equal vectors — textbook catastrophic cancellation — so there is a positive-biased
noise floor beneath the score that *compresses the certain end*, precisely where the
"certain field => accel -> 0" premise lives. Measure that floor on the actual checkpoint
before trusting a threshold: see ``opentau.scripts.diagnose_accel``. A calibration whose
successful-rollout scores sit within a small multiple of the floor is measuring rounding,
not geometry.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opentau.policies.accel import AccelProvenance, assert_comparable

logger = logging.getLogger(__name__)

# Paper's operating point (§4.3): slack in units of the calibration standard deviation.
DEFAULT_SLACK_C = 0.25
# Paper's target false-alarm rate (§4.3).
DEFAULT_ALPHA = 0.1

CALIBRATION_FILENAME = "opentau_accel_calibration.json"


def _clean(scores: Sequence[float]) -> np.ndarray:
    """Return ``scores`` as float64, with non-finite entries dropped."""
    arr = np.asarray(list(scores), dtype=np.float64)
    return arr[np.isfinite(arr)]


def cusum_stream(scores: Sequence[float], *, mu0: float, k: float) -> np.ndarray:
    """Run the one-sided CUSUM recursion over one episode's score stream.

    Non-finite scores (the NaN :class:`~opentau.policies.accel.AccelMeter` emits when a
    score is undefined) carry ``S`` forward unchanged rather than poisoning the running
    sum — a chunk with no valid measurement is not evidence of drift in either direction.

    Args:
        scores: Per-chunk ``accel`` values in temporal order.
        mu0: Reference level (mean of pooled successful-rollout scores).
        k: Slack, ``c * sigma``. Absorbs transient spikes that do not persist.

    Returns:
        ``(len(scores),)`` float64 array of ``S_t``.
    """
    out = np.empty(len(scores), dtype=np.float64)
    s = 0.0
    for i, z in enumerate(scores):
        if math.isfinite(z):
            s = max(0.0, s + float(z) - mu0 - k)
        out[i] = s
    return out


def episode_peak(scores: Sequence[float], *, mu0: float, k: float) -> float:
    """Return ``max_t S_t`` — the single scalar an episode of any length reduces to.

    Args:
        scores: Per-chunk ``accel`` values in temporal order.
        mu0: Reference level.
        k: Slack.

    Returns:
        The peak, or ``0.0`` for an empty stream.
    """
    if len(scores) == 0:
        return 0.0
    return float(cusum_stream(scores, mu0=mu0, k=k).max())


def conformal_rank(num_calibration: int, alpha: float) -> int:
    """Return the 1-based split-conformal order statistic ``ceil((M+1)(1-alpha))``.

    Args:
        num_calibration: Number of calibration episodes ``M``.
        alpha: Target false-alarm rate.

    Returns:
        The rank ``r`` such that ``eta = P_(r)`` of the sorted peaks. With ``M = 50`` and
        ``alpha = 0.1`` this is 46, matching the paper.

    Raises:
        ValueError: When ``M`` is too small for the requested ``alpha`` — i.e. the rank
            would exceed ``M``, so no finite threshold can certify that rate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    rank = math.ceil((num_calibration + 1) * (1.0 - alpha))
    if rank > num_calibration:
        minimum = math.ceil(1.0 / alpha) - 1
        raise ValueError(
            f"alpha={alpha} needs at least {minimum} calibration episodes for a finite "
            f"split-conformal threshold, got {num_calibration}. Collect more successful "
            "rollouts or raise alpha."
        )
    return rank


@dataclass(frozen=True)
class CusumCalibration:
    """A fitted alarm threshold plus everything needed to refuse misuse.

    Attributes:
        mu0: Reference level, mean of pooled successful-rollout chunk scores.
        sigma: Scale, std of the same pool.
        slack_c: Slack in units of ``sigma``.
        eta: Alarm threshold on the CUSUM height.
        alpha: Target false-alarm rate the threshold was fitted for.
        num_calibration_episodes: ``M``.
        num_calibration_chunks: Total chunk scores behind ``mu0``/``sigma``.
        provenance: The configuration the calibration was fitted under.
    """

    mu0: float
    sigma: float
    slack_c: float
    eta: float
    alpha: float
    num_calibration_episodes: int
    num_calibration_chunks: int
    provenance: AccelProvenance | None = None

    @property
    def k(self) -> float:
        """Slack in score units."""
        return self.slack_c * self.sigma

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy."""
        return {
            "mu0": self.mu0,
            "sigma": self.sigma,
            "slack_c": self.slack_c,
            "eta": self.eta,
            "alpha": self.alpha,
            "num_calibration_episodes": self.num_calibration_episodes,
            "num_calibration_chunks": self.num_calibration_chunks,
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CusumCalibration:
        """Rebuild from :meth:`to_dict` output."""
        prov = payload.get("provenance")
        return cls(
            mu0=float(payload["mu0"]),
            sigma=float(payload["sigma"]),
            slack_c=float(payload["slack_c"]),
            eta=float(payload["eta"]),
            alpha=float(payload["alpha"]),
            num_calibration_episodes=int(payload["num_calibration_episodes"]),
            num_calibration_chunks=int(payload["num_calibration_chunks"]),
            provenance=AccelProvenance.from_dict(prov) if prov else None,
        )

    def save(self, path: str | Path) -> Path:
        """Write to ``path`` (a file, or :data:`CALIBRATION_FILENAME` inside a directory)."""
        path = Path(path)
        if path.is_dir():
            path = path / CALIBRATION_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CusumCalibration:
        """Read from ``path`` (a file, or :data:`CALIBRATION_FILENAME` inside a directory)."""
        path = Path(path)
        if path.is_dir():
            path = path / CALIBRATION_FILENAME
        return cls.from_dict(json.loads(path.read_text()))


def calibrate(
    successful_episodes: Iterable[Sequence[float]],
    *,
    alpha: float = DEFAULT_ALPHA,
    slack_c: float = DEFAULT_SLACK_C,
    provenance: AccelProvenance | None = None,
) -> CusumCalibration:
    """Fit a CUSUM alarm threshold from held-out **successful** rollouts.

    Two-stage, per the paper: pool the chunk scores to estimate the reference level and
    scale, then take the split-conformal order statistic of the per-episode CUSUM peaks.

    Only successful rollouts are used, which is the method's main practical appeal — no
    failure labels are needed. The cost is a real one though: at a policy with 20-60%
    success you need roughly ``M / success_rate`` episodes to bank ``M`` successes.

    Args:
        successful_episodes: One sequence of per-chunk ``accel`` scores per episode.
        alpha: Target false-alarm rate.
        slack_c: Slack in units of ``sigma``.
        provenance: Configuration these scores were produced under; recorded so
            :func:`detect` can refuse a mismatched stream.

    Returns:
        The fitted :class:`CusumCalibration`.

    Raises:
        ValueError: On an empty calibration set, a set with no finite scores, or ``M`` too
            small for ``alpha``.
    """
    episodes = [list(ep) for ep in successful_episodes]
    if not episodes:
        raise ValueError("accel calibration needs at least one successful episode.")

    pooled = _clean([z for ep in episodes for z in ep])
    if pooled.size == 0:
        raise ValueError(
            "accel calibration set contains no finite scores. Every value was NaN, which "
            "means the meter never had a valid measurement — check the action-dim mask and "
            "the denoise schedule length."
        )

    mu0 = float(pooled.mean())
    sigma = float(pooled.std())
    if sigma == 0.0:
        logger.warning(
            "accel calibration scale is exactly zero (%d pooled chunks, all equal to %.6g). "
            "The slack term vanishes and the detector reduces to 'any increase alarms'.",
            pooled.size,
            mu0,
        )
    k = slack_c * sigma

    peaks = np.array([episode_peak(ep, mu0=mu0, k=k) for ep in episodes], dtype=np.float64)
    rank = conformal_rank(len(episodes), alpha)
    eta = float(np.sort(peaks)[rank - 1])

    logger.info(
        "accel calibration: M=%d episodes, %d chunks, mu0=%.6g, sigma=%.6g, k=%.6g, "
        "eta=P_(%d)=%.6g at alpha=%.3g",
        len(episodes),
        int(pooled.size),
        mu0,
        sigma,
        k,
        rank,
        eta,
        alpha,
    )
    return CusumCalibration(
        mu0=mu0,
        sigma=sigma,
        slack_c=slack_c,
        eta=eta,
        alpha=alpha,
        num_calibration_episodes=len(episodes),
        num_calibration_chunks=int(pooled.size),
        provenance=provenance,
    )


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of running a calibrated detector over one episode.

    Attributes:
        alarm_index: Index of the first chunk whose CUSUM height exceeded ``eta``, or
            ``None`` if the episode never alarmed.
        lead: Chunks remaining after the alarm (``n_chunks - 1 - alarm_index``), the
            paper's detection-lead metric. ``None`` without an alarm.
        peak: ``max_t S_t`` for the episode.
        statistic: The full ``S_t`` series.
    """

    alarm_index: int | None
    lead: int | None
    peak: float
    statistic: np.ndarray

    @property
    def alarmed(self) -> bool:
        """Whether an alarm was raised."""
        return self.alarm_index is not None


def detect(
    scores: Sequence[float],
    calibration: CusumCalibration,
    *,
    provenance: AccelProvenance | None = None,
) -> DetectionResult:
    """Run a calibrated detector over one episode's ``accel`` stream.

    Args:
        scores: Per-chunk ``accel`` values in temporal order.
        calibration: A fitted :class:`CusumCalibration`.
        provenance: Configuration this stream was produced under. When both this and the
            calibration's provenance are present they must agree on every
            distribution-shifting field.

    Returns:
        The :class:`DetectionResult`.

    Raises:
        ValueError: When the provenances disagree.
    """
    if provenance is not None and calibration.provenance is not None:
        assert_comparable(calibration.provenance, provenance)

    statistic = cusum_stream(scores, mu0=calibration.mu0, k=calibration.k)
    above = np.flatnonzero(statistic > calibration.eta)
    alarm_index = int(above[0]) if above.size else None
    lead = (len(scores) - 1 - alarm_index) if alarm_index is not None else None
    peak = float(statistic.max()) if statistic.size else 0.0
    return DetectionResult(alarm_index=alarm_index, lead=lead, peak=peak, statistic=statistic)


def evaluate(
    episodes: Sequence[Sequence[float]],
    successes: Sequence[bool],
    calibration: CusumCalibration,
) -> dict[str, float]:
    """Score a detector against labeled rollouts, in the paper's metrics.

    Args:
        episodes: Per-episode ``accel`` streams.
        successes: Ground-truth success label per episode.
        calibration: The fitted threshold.

    Returns:
        ``true_positive_rate`` (fraction of *failing* episodes that alarmed),
        ``false_alarm_rate`` (fraction of *successful* episodes that alarmed — an
        out-of-sample quantity that fluctuates around ``alpha`` rather than equalling it),
        ``median_lead`` over detected failures (NaN when none), and the episode counts.

    Raises:
        ValueError: When ``episodes`` and ``successes`` differ in length.
    """
    if len(episodes) != len(successes):
        raise ValueError(f"episodes ({len(episodes)}) and successes ({len(successes)}) differ in length.")

    results = [detect(ep, calibration) for ep in episodes]
    failed = [r for r, ok in zip(results, successes, strict=True) if not ok]
    passed = [r for r, ok in zip(results, successes, strict=True) if ok]

    leads = [r.lead for r in failed if r.alarmed]
    return {
        "true_positive_rate": float(np.mean([r.alarmed for r in failed])) if failed else float("nan"),
        "false_alarm_rate": float(np.mean([r.alarmed for r in passed])) if passed else float("nan"),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
        "n_failed": float(len(failed)),
        "n_successful": float(len(passed)),
    }
