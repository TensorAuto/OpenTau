#!/bin/bash

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

# Run a benchmark with concurrent nvidia-smi memory logging, then write
# a per-GPU peak summary. nvidia-smi is started in the background and a
# trap on EXIT/INT/TERM kills it on every exit path (success, crash,
# Ctrl-C, SIGTERM), so the poller never outlives the benchmark.
#
# Honors ``CUDA_VISIBLE_DEVICES``: when set, nvidia-smi is invoked with
# ``-i $CUDA_VISIBLE_DEVICES`` so only the visible GPUs are polled. This
# matters when the cluster gives you a non-contiguous slice (e.g. GPUs
# 0,2,4,6) and the other GPUs are running unrelated jobs whose memory
# peaks would otherwise pollute the report.
#
# Usage:
#   bench_with_mem.sh <tag> <command> [args...]
#
# Example (4 GPUs, IDs 0,2,4,6):
#   CUDA_VISIBLE_DEVICES=0,2,4,6 \
#   bench_with_mem.sh B1-ddp-fp32-bs8 \
#       accelerate launch --num_processes 4 \
#           --config_file configs/libero/reproduce_pi05_libero_accelerate_config.yaml \
#           src/opentau/scripts/profile_step.py \
#           --config_path=configs/libero/reproduce_pi05_libero.json --batch_size=8
#
# Outputs (in the current directory):
#   gpu-mem-<tag>.csv   raw nvidia-smi log (one row per (timestamp, gpu))
#   gpu-mem-<tag>.json  per-GPU peak memory.used + the global max
#
# The script forwards the benchmark's exit code, so it composes with
# shell `&&` / `||` chains and CI checks.

set -uo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <tag> <command> [args...]" >&2
    exit 64
fi

TAG="$1"
shift

CSV="gpu-mem-${TAG}.csv"
SUMMARY="gpu-mem-${TAG}.json"

# memory.used + memory.total per GPU, sampled every 500 ms, written to CSV.
# nvidia-smi handles its own polling cadence, no python loop in the way.
# Restrict to CUDA_VISIBLE_DEVICES if set, so the report doesn't include
# memory peaks from neighbor jobs on GPUs we don't actually own.
NVIDIA_SMI_FILTER=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NVIDIA_SMI_FILTER=(-i "$CUDA_VISIBLE_DEVICES")
    echo "bench_with_mem: filtering nvidia-smi to CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
fi

nvidia-smi \
    "${NVIDIA_SMI_FILTER[@]}" \
    --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader,nounits \
    -lms 500 \
    -f "$CSV" &
SMI_PID=$!

# Always reap the poller, even on abnormal exit. SIGTERM is enough; the
# `wait` swallows the eventual non-zero exit code so the trap doesn't
# clobber our $RC.
cleanup() {
    if kill -0 "$SMI_PID" 2>/dev/null; then
        kill -TERM "$SMI_PID" 2>/dev/null || true
        wait "$SMI_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Run the benchmark with whatever env vars the caller already exported.
"$@"
RC=$?

# Make sure one extra nvidia-smi sample lands after the benchmark's last
# step, so peak captures the worst-case state at the very end of the run.
sleep 0.6
cleanup
trap - EXIT INT TERM

# Reduce the CSV to per-GPU peak. Using awk instead of jq/python so the
# only runtime dependency is nvidia-smi + awk (already on every node).
if [ ! -s "$CSV" ]; then
    echo "bench_with_mem: CSV $CSV is empty; nvidia-smi may have failed." >&2
    echo "{\"tag\": \"${TAG}\", \"error\": \"empty nvidia-smi log\"}" > "$SUMMARY"
    exit "$RC"
fi

awk -v tag="$TAG" -F', *' '
{
    g    = $1 + 0
    used = $2 + 0
    tot  = $3 + 0
    if (used > peak[g]) peak[g] = used
    cap[g] = tot
    if (g > max_g) max_g = g
}
END {
    printf "{\n"
    printf "  \"tag\": \"%s\",\n", tag
    printf "  \"per_gpu\": [\n"
    # Iterate 0..max_g and skip indices that never appeared. This handles
    # non-contiguous slices like CUDA_VISIBLE_DEVICES=0,2,4,6 — nvidia-smi
    # reports the absolute index, so peak[1], peak[3], peak[5] stay unset
    # and we skip them rather than emitting bogus zero rows.
    first = 1
    for (g = 0; g <= max_g; g++) {
        if (!(g in peak)) continue
        if (!first) printf ",\n"
        printf "    {\"gpu\": %d, \"peak_mib\": %d, \"total_mib\": %d, \"peak_gib\": %.2f, \"headroom_mib\": %d}", \
               g, peak[g], cap[g], peak[g]/1024.0, cap[g]-peak[g]
        first = 0
    }
    # max_peak_mib is the max-across-GPUs of each GPU's peak-over-time:
    # the worst single-GPU footprint observed at any point during the run.
    # That's the value that determines whether the run OOMs, since CUDA
    # OOM is per-device, not aggregated.
    max = 0; max_idx = -1
    for (g in peak) if (peak[g] > max) { max = peak[g]; max_idx = g + 0 }
    printf "\n  ],\n"
    printf "  \"max_peak_mib\": %d,\n", max
    printf "  \"max_peak_gib\": %.2f,\n", max/1024.0
    printf "  \"max_peak_gpu_idx\": %d\n", max_idx
    printf "}\n"
}
' "$CSV" > "$SUMMARY"

cat "$SUMMARY"
exit "$RC"
