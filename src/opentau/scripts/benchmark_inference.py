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
"""Inference-speed benchmark.

A dedicated harness for measuring per-call inference latency of a policy under
deployment-like conditions: random-init (no checkpoint download), bf16, single
batch, three cameras, ~50-token prompt. Distinct from
``opentau.scripts.inference``, which is a quick smoke runner with N=10 and no
CUDA sync.

Configured via the same ``TrainPipelineConfig`` JSON used elsewhere. Runtime
flags ride on env vars (so we don't have to break the ``@parser.wrap``
signature):

  BENCH_NO_COMPILE=1          # skip torch.compile entirely
  BENCH_COMPILE_MODE=...      # default "max-autotune"
  BENCH_N_WARMUP=5            # post-compile warmup calls
  BENCH_N_TIMED=50            # timed calls after warmup
  BENCH_OUTPUT_DIR=...        # default benchmark_results/

The script writes a single JSON to ``${BENCH_OUTPUT_DIR}/<host>_<policy>_<ts>.json``
with the config snapshot, host/GPU/driver/torch versions, per-warmup wall-clock
series, all timed samples, and a summary (mean/std/p50/p95/min/max).
"""

import datetime as dt
import json
import logging
import os
import socket
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import make_policy
from opentau.policies.normalize import NormalizationMode
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    auto_torch_device,
    create_dummy_observation,
    init_logging,
)

# ~50-token prompt under SentencePiece tokenizers used by PaliGemma / Gemma 3.
# We additionally pin policy.prompt_max_length=64 in the configs so the
# attended sequence is bounded at 64 regardless of tokenizer-specific drift.
BENCHMARK_PROMPT = (
    "Pick up the small yellow lego block from the table and carefully "
    "place it inside the gray bin on the left side now please."
)

HIGH_LEVEL_POLICY_TYPES = frozenset({"pi07_high_level", "pi07_paligemma_high_level_planner"})


def _is_high_level(policy_type: str) -> bool:
    return policy_type in HIGH_LEVEL_POLICY_TYPES


def _force_identity_normalization(cfg_policy) -> None:
    """Random-init policies have no stats; switch all norm modes to IDENTITY
    so the inf-init buffers in `Normalize` are never read.
    """
    for k in list(cfg_policy.normalization_mapping.keys()):
        cfg_policy.normalization_mapping[k] = NormalizationMode.IDENTITY


def _build_observation(cfg: TrainPipelineConfig, device, dtype) -> dict:
    obs = create_dummy_observation(cfg, device, dtype=dtype)
    # Override prompt with a ~50-token sentence (the create_dummy_observation
    # default is ~11 tokens — too short to exercise the requested workload).
    obs["prompt"] = [BENCHMARK_PROMPT]

    # Video-stack policies (pi05_mem, pi07_low_level, pi07_paligemma_low_level
    # when n_obs_steps > 1) consume a 5D (B, T, C, H, W) camera tensor instead
    # of 4D. They also expect an `obs_history_is_pad` mask shaped (B, T). The
    # base create_dummy_observation builds 4D, so promote the cameras here.
    n_obs = getattr(cfg.policy, "n_obs_steps", 1) or 1
    if n_obs > 1:
        h, w = cfg.resolution
        for cam_key in [k for k in list(obs.keys()) if k.startswith("camera")]:
            obs[cam_key] = torch.zeros((1, n_obs, 3, h, w), dtype=dtype, device=device)
        obs["state"] = torch.zeros((1, n_obs, cfg.max_state_dim), dtype=dtype, device=device)
        obs["obs_history_is_pad"] = torch.zeros((1, n_obs), dtype=torch.bool, device=device)

    if _is_high_level(cfg.policy.type):
        # High-level planner reads batch["past_memory"] as list[str]; default
        # to an empty memory token (tokenizer pads to memory_max_length=50).
        obs["past_memory"] = [""]
        # prepare_metadata reads speed/quality/mistake + *_is_pad and an
        # optional fps. Setting every _is_pad=True drops the values from the
        # metadata string entirely (just "Metadata: " padded to
        # metadata_max_length). Cleanest no-op for a speed benchmark.
        b = 1
        zero_long = torch.zeros(b, dtype=torch.long, device=device)
        all_pad = torch.ones(b, dtype=torch.bool, device=device)
        obs["speed"] = zero_long
        obs["quality"] = zero_long
        obs["mistake"] = zero_long
        obs["speed_is_pad"] = all_pad
        obs["quality_is_pad"] = all_pad
        obs["mistake_is_pad"] = all_pad
        obs["fps"] = zero_long
        obs["fps_is_pad"] = all_pad
    return obs


def _verify_compile(compiled_fn) -> None:
    """Raise if torch.compile silently returned the original callable.

    The historical ``attempt_torch_compile`` helper swallows exceptions and
    returns ``fn`` — we want the opposite, a loud failure.
    """
    if not hasattr(compiled_fn, "_torchdynamo_orig_callable"):
        raise RuntimeError(
            "torch.compile did not produce a Dynamo-wrapped callable. "
            f"Got {type(compiled_fn).__name__} without _torchdynamo_orig_callable. "
            "Either torch.compile is unavailable, or it silently bailed."
        )


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _summarize(samples_ms: list[float]) -> dict:
    sorted_samples = sorted(samples_ms)
    n = len(sorted_samples)

    def pct(p: float) -> float:
        # Lower bound index for the p-th percentile.
        idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return sorted_samples[idx]

    return {
        "n": n,
        "mean_ms": statistics.fmean(samples_ms),
        "std_ms": statistics.stdev(samples_ms) if n > 1 else 0.0,
        "min_ms": sorted_samples[0],
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": sorted_samples[-1],
    }


def _nvidia_driver_version() -> str | None:
    try:
        out = subprocess.run(  # nosec B607 - nvidia-smi is intentionally resolved via PATH
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


@parser.wrap()
def benchmark_main(cfg: TrainPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    no_compile = os.environ.get("BENCH_NO_COMPILE", "0") == "1"
    compile_mode = os.environ.get("BENCH_COMPILE_MODE", "max-autotune")
    n_warmup = int(os.environ.get("BENCH_N_WARMUP", "5"))
    n_timed = int(os.environ.get("BENCH_N_TIMED", "50"))
    out_dir = Path(os.environ.get("BENCH_OUTPUT_DIR", "benchmark_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(auto_torch_device())
    if cfg.seed is not None:
        set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    is_hl = _is_high_level(cfg.policy.type)
    logging.info(
        f"Policy type={cfg.policy.type} (high_level={is_hl}), no_compile={no_compile}, "
        f"compile_mode={compile_mode}, n_warmup={n_warmup}, n_timed={n_timed}"
    )

    # Random init path: bypass stats requirement.
    if cfg.policy.pretrained_path is None:
        _force_identity_normalization(cfg.policy)
        logging.info("pretrained_path is null: forced normalization_mapping -> IDENTITY")

    logging.info("Building policy via make_policy (features pre-set in config).")
    policy = make_policy(cfg.policy)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    policy.reset()

    # Compile + verify.
    compile_status = "no-compile"
    if not no_compile:
        torch._dynamo.reset()
        compiled = torch.compile(policy.model.sample_actions, mode=compile_mode, fullgraph=False)
        _verify_compile(compiled)
        policy.model.sample_actions = compiled
        compile_status = f"compiled:{compile_mode}"
        logging.info(f"torch.compile attached: mode={compile_mode}")

    observation = _build_observation(cfg, device, dtype=torch.bfloat16)
    logging.info(f"Observation keys: {sorted(observation.keys())}")

    # Warmup. Time each call so we can verify compile actually fired (call 1
    # should dominate by ~5-100x).
    warmup_ms: list[float] = []
    with torch.inference_mode():
        for i in range(n_warmup):
            _sync(device)
            t0 = time.perf_counter()
            _ = policy.sample_actions(observation)
            _sync(device)
            t1 = time.perf_counter()
            warmup_ms.append((t1 - t0) * 1000.0)
            logging.info(f"  warmup {i + 1}/{n_warmup}: {warmup_ms[-1]:.2f} ms")

    if not no_compile and n_warmup >= 5:
        ratio = warmup_ms[0] / max(warmup_ms[-1], 1e-6)
        if ratio < 5.0:
            raise RuntimeError(
                f"Warmup call 1 / call {n_warmup} ratio is {ratio:.2f}x — expected >= 5x if "
                f"torch.compile fired. Either compile silently bailed, or the cache is being "
                f"reused (try torch._dynamo.reset() / clearing TORCHINDUCTOR_CACHE_DIR)."
            )

    # Timed loop.
    times_ms: list[float] = []
    with torch.inference_mode():
        _sync(device)
        for _ in range(n_timed):
            _sync(device)
            t0 = time.perf_counter()
            _ = policy.sample_actions(observation)
            _sync(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    summary = _summarize(times_ms)
    logging.info(
        f"Inference (ms): mean={summary['mean_ms']:.2f} std={summary['std_ms']:.2f} "
        f"p50={summary['p50_ms']:.2f} p95={summary['p95_ms']:.2f} "
        f"min={summary['min_ms']:.2f} max={summary['max_ms']:.2f}"
    )

    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
        "driver_version": _nvidia_driver_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "policy_type": cfg.policy.type,
        "compile_status": compile_status,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "warmup_ms": warmup_ms,
        "samples_ms": times_ms,
        "summary": summary,
        "config": asdict(cfg),
    }

    out_path = (
        out_dir / f"{record['host']}_{cfg.policy.type}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(record, indent=2, default=str))
    logging.info(f"Wrote {out_path}")


if __name__ == "__main__":
    benchmark_main()
