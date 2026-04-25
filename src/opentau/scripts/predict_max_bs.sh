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

# Predict the largest ``--batch_size`` that fits under a GPU memory
# ceiling, from two prior runs of ``bench_with_mem.sh`` at different
# batch sizes. Under a fixed model/optimizer/distributed config, peak
# memory scales near-linearly with batch size (the per-sample activation
# buffers grow linearly; the per-rank static state — params, Adam
# state, gradient buckets — is constant). Two probe points fully
# determine the line; floor the prediction by 1 to stay below the
# ceiling.
#
# Usage:
#   predict_max_bs.sh <bs1> <gpu-mem-tag1.json> <bs2> <gpu-mem-tag2.json> [target_gib]
#
# ``target_gib`` defaults to 78, which leaves ~2 GiB of headroom on an
# 80 GiB A100 for the driver, NCCL buffers, and intra-step transient
# spikes that the 500 ms poller may miss.
#
# Example:
#   bench_with_mem.sh probe-bs4 ... --batch_size=4
#   bench_with_mem.sh probe-bs8 ... --batch_size=8
#   predict_max_bs.sh 4 gpu-mem-probe-bs4.json 8 gpu-mem-probe-bs8.json 78

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <bs1> <json1> <bs2> <json2> [target_gib]" >&2
    exit 64
fi

BS1="$1"
JSON1="$2"
BS2="$3"
JSON2="$4"
TARGET_GIB="${5:-78}"

if [ ! -s "$JSON1" ] || [ ! -s "$JSON2" ]; then
    echo "Error: one of $JSON1 / $JSON2 is missing or empty." >&2
    exit 1
fi

# Pull max_peak_mib without depending on jq — both values come from
# bench_with_mem.sh's stable JSON shape.
extract() {
    grep -o '"max_peak_mib"[[:space:]]*:[[:space:]]*[0-9]*' "$1" \
        | awk -F: '{ gsub(/[[:space:]]/, "", $2); print $2 }'
}

MIB1="$(extract "$JSON1")"
MIB2="$(extract "$JSON2")"

if [ -z "$MIB1" ] || [ -z "$MIB2" ]; then
    echo "Error: could not read max_peak_mib from one of the inputs." >&2
    exit 1
fi

awk -v bs1="$BS1" -v mib1="$MIB1" \
    -v bs2="$BS2" -v mib2="$MIB2" \
    -v target_gib="$TARGET_GIB" '
BEGIN {
    if (bs2 == bs1) {
        print "Error: bs1 == bs2; need two distinct probe points." > "/dev/stderr"
        exit 2
    }
    target_mib = target_gib * 1024
    slope = (mib2 - mib1) / (bs2 - bs1)
    intercept = mib1 - slope * bs1
    if (slope <= 0) {
        printf "Error: slope=%.1f mib/sample is non-positive; the two runs disagree on whether memory grows with bs.\n", slope > "/dev/stderr"
        exit 3
    }
    bs_max_real = (target_mib - intercept) / slope
    # Floor to int. ``target_gib`` already builds in ~2 GiB of headroom
    # against the 80 GiB cards; flooring on top of that gives one more
    # sample of safety margin.
    bs_max = int(bs_max_real)
    if (bs_max < 1) bs_max = 1
    pred_peak_mib = slope * bs_max + intercept
    printf "{\n"
    printf "  \"probe1\":  {\"bs\": %d, \"peak_mib\": %d, \"peak_gib\": %.2f},\n", bs1, mib1, mib1 / 1024.0
    printf "  \"probe2\":  {\"bs\": %d, \"peak_mib\": %d, \"peak_gib\": %.2f},\n", bs2, mib2, mib2 / 1024.0
    printf "  \"slope_mib_per_sample\": %.1f,\n", slope
    printf "  \"intercept_mib\":        %.1f,\n", intercept
    printf "  \"target_gib\":           %.1f,\n", target_gib
    printf "  \"predicted_max_bs\":     %d,\n", bs_max
    printf "  \"predicted_peak_gib_at_max_bs\": %.2f\n", pred_peak_mib / 1024.0
    printf "}\n"
}
'
