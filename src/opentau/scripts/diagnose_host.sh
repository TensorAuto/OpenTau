#!/usr/bin/env bash
# Capture a one-shot host snapshot before an inference benchmark run.
#
# Read alongside the benchmark JSON to interpret cross-host numbers — e.g. if
# the faster GPU box isn't ~2x the slower one, is the GPU itself slow
# (driver/clocks), the path to it (PCIe negotiated below x16), or the CPU side
# (governor, idle competing processes)?
#
# Usage:
#   src/opentau/scripts/diagnose_host.sh [out_dir]
#
# Default out_dir is ./benchmark_results. Writes to <out_dir>/<host>_diagnose.txt.

set -u

OUT_DIR="${1:-benchmark_results}"
mkdir -p "$OUT_DIR"

HOST="$(hostname)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_DIR}/${HOST}_diagnose.txt"

section() {
    printf '\n========== %s ==========\n' "$1" >> "$OUT"
}

run() {
    # Tolerate missing tools / non-zero exits without nuking the whole report.
    printf '\n$ %s\n' "$*" >> "$OUT"
    eval "$@" >> "$OUT" 2>&1 || printf '(command failed or unavailable)\n' >> "$OUT"
}

: > "$OUT"
printf 'host=%s\ntimestamp=%s\nuname=%s\n' "$HOST" "$TS" "$(uname -a)" >> "$OUT"

section "GPU snapshot (nvidia-smi)"
run "nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,power.draw,power.limit,clocks.gr,clocks.mem,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current --format=csv"

section "GPU detail: persistence, power, clocks"
run "nvidia-smi -q -d CLOCK,POWER,PERFORMANCE | grep -E 'Persistence|Power Limit|Default Power|Graphics|Memory Clock' | head -40"

section "Currently running GPU processes"
run "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv"

section "CPU"
run "lscpu | grep -E 'Model name|CPU\\(s\\)|CPU MHz|CPU max MHz|NUMA|Architecture'"

section "CPU frequency governor (cpu0)"
run "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

section "Memory"
run "free -h"
run "sudo -n dmidecode -t memory | grep -E 'Speed:|Type:' | head -20"

section "PCIe link (NVIDIA)"
run "lspci -vvv -s \"\$(lspci | grep -i nvidia | awk '{print \$1}' | head -1)\" | grep -E 'LnkCap|LnkSta'"

section "Storage"
run "lsblk -d -o name,rota,size,model 2>/dev/null | grep -v loop"

section "PyTorch / CUDA / cuDNN"
# Prefer a known venv if BENCH_PYTHON is set; otherwise fall back to python3.
PYBIN="${BENCH_PYTHON:-python3}"
run "$PYBIN -c \"import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); print('cudnn', torch.backends.cudnn.version())\""

section "Top processes by CPU"
run "ps -eo pid,%cpu,%mem,cmd --sort=-%cpu | head -10"

printf '\n# Report written to %s\n' "$OUT"
