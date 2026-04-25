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

"""Compare two ``PROFILE_LOSS_LOG`` outputs from ``profile_step.py``.

Used to gate the SDPA-default flip on loss equivalence vs the eager
attention kernel: same seed + same data ordering should yield nearly
identical loss curves, with deviations bounded by bf16 reassociation
noise. This script computes per-step deltas, summary stats, and a
verdict against a configurable tolerance.

Usage:
  compare_loss_logs.py <reference.json> <candidate.json> [--rel-tol=0.02] [--abs-tol=1e-3]

Prints a summary table to stdout and exits 0 if equivalent (per the
configured tolerance), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median, stdev


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", help="Reference loss log (e.g. eager attention).")
    parser.add_argument("candidate", help="Candidate loss log (e.g. sdpa attention).")
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.02,
        help="Relative tolerance for the median per-step ratio. Default 2%%.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for the median per-step delta. Default 1e-3.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Drop the warmup steps from the comparison; report only on measured.",
    )
    args = parser.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    cand = json.loads(Path(args.candidate).read_text())

    print(
        f"reference: {args.reference}  attn={ref.get('attention_implementation')}  bs={ref.get('batch_size')}  seed={ref.get('seed')}"
    )
    print(
        f"candidate: {args.candidate}  attn={cand.get('attention_implementation')}  bs={cand.get('batch_size')}  seed={cand.get('seed')}"
    )

    # Sanity: bs and seed should match for a meaningful A/B.
    if ref.get("batch_size") != cand.get("batch_size"):
        print(
            f"WARN: batch_size mismatch ({ref.get('batch_size')} vs {cand.get('batch_size')})",
            file=sys.stderr,
        )
    if ref.get("seed") != cand.get("seed"):
        print(f"WARN: seed mismatch ({ref.get('seed')} vs {cand.get('seed')})", file=sys.stderr)

    ref_steps = ref["per_step"]
    cand_steps = cand["per_step"]
    if args.skip_warmup:
        ref_steps = [s for s in ref_steps if not s.get("warmup")]
        cand_steps = [s for s in cand_steps if not s.get("warmup")]

    n = min(len(ref_steps), len(cand_steps))
    if n == 0:
        print("No comparable steps.", file=sys.stderr)
        return 1
    ref_steps = ref_steps[:n]
    cand_steps = cand_steps[:n]

    # Per-step deltas on the combined loss.
    deltas = []
    rel_deltas = []
    for r, c in zip(ref_steps, cand_steps, strict=True):
        d = c["loss"] - r["loss"]
        deltas.append(d)
        if abs(r["loss"]) > 1e-9:
            rel_deltas.append(d / r["loss"])

    abs_deltas = [abs(d) for d in deltas]
    abs_rel = [abs(r) for r in rel_deltas]

    print()
    print("=== loss equivalence summary (combined loss) ===")
    print(f"  n_steps_compared: {n}  warmup_skipped={args.skip_warmup}")
    print(
        f"  ref  loss:  start={ref_steps[0]['loss']:.6f}  end={ref_steps[-1]['loss']:.6f}  mean={mean(s['loss'] for s in ref_steps):.6f}"
    )
    print(
        f"  cand loss:  start={cand_steps[0]['loss']:.6f}  end={cand_steps[-1]['loss']:.6f}  mean={mean(s['loss'] for s in cand_steps):.6f}"
    )
    print()
    print("  per-step delta (cand - ref):")
    print(f"    median   = {median(deltas):+.6e}")
    print(f"    mean     = {mean(deltas):+.6e}")
    print(f"    stdev    = {stdev(deltas) if len(deltas) > 1 else 0:.6e}")
    print(f"    abs.med  = {median(abs_deltas):.6e}")
    print(f"    abs.max  = {max(abs_deltas):.6e}")
    print("  per-step relative delta:")
    if rel_deltas:
        print(f"    median   = {median(rel_deltas):+.4%}")
        print(f"    abs.med  = {median(abs_rel):.4%}")
        print(f"    abs.max  = {max(abs_rel):.4%}")

    # Per-component breakdown when available.
    for component in ("mse", "ce"):
        if component in ref_steps[0] and component in cand_steps[0]:
            comp_deltas = [c[component] - r[component] for r, c in zip(ref_steps, cand_steps, strict=True)]
            print()
            print(
                f"  {component} loss delta: median={median(comp_deltas):+.6e}  abs.med={median([abs(d) for d in comp_deltas]):.6e}  abs.max={max(abs(d) for d in comp_deltas):.6e}"
            )

    print()
    abs_med_rel = median(abs_rel) if abs_rel else float("inf")
    abs_med_delta = median(abs_deltas)
    rel_ok = abs_med_rel <= args.rel_tol
    abs_ok = abs_med_delta <= args.abs_tol
    verdict_ok = rel_ok or abs_ok
    print(f"=== verdict (rel_tol={args.rel_tol:.2%}, abs_tol={args.abs_tol:.1e}) ===")
    print(f"  median |relative delta| = {abs_med_rel:.4%}  -> {'PASS' if rel_ok else 'FAIL'}")
    print(f"  median |abs delta|      = {abs_med_delta:.6e}  -> {'PASS' if abs_ok else 'FAIL'}")
    print(f"  EQUIVALENT: {'YES' if verdict_ok else 'NO'}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
