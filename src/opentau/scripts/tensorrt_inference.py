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

"""Inference script for exported ONNX models using TensorRT with FP16.

Loads the same ONNX model as onnx_inference.py but runs it with the TensorRT
execution provider in FP16 for faster inference on GPU. The ONNX model can be
exported in FP32; TensorRT will execute it in FP16.

Requires optional dependency: uv sync --extra trt

Usage:
  python -m opentau.scripts.tensorrt_inference --checkpoint_dir <path> \\
    [--num_cams 2] [--resolution_height 224] [--resolution_width 224] \\
    [--predict_response true] [--n_repeats 3] [--engine_cache_dir <path>]
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def _prepend_tensorrt_lib_path() -> None:
    """Prepend TensorRT pip package lib path to LD_LIBRARY_PATH so the TRT EP can load libnvinfer."""
    lib_dirs_to_try = []
    import tensorrt as _trt

    pkg_root = Path(_trt.__path__[0]).resolve().parent
    lib_dirs_to_try.append(pkg_root / "tensorrt_libs")
    lib_dirs_to_try.append(Path(_trt.__path__[0]).resolve())
    lib_dirs_to_try.append(pkg_root / "lib")
    for lib_dir in lib_dirs_to_try:
        if lib_dir.is_dir() and any(lib_dir.glob("libnvinfer.so*")):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{existing}" if existing else str(lib_dir)
            return


_prepend_tensorrt_lib_path()
import onnxruntime as ort  # noqa: E402

from opentau.configs import parser  # noqa: E402
from opentau.scripts.onnx_inference import (  # noqa: E402
    PI05_TOKENIZER_ID,
    OnnxInferenceArgs,
    run_inference,
)
from opentau.utils.utils import init_logging  # noqa: E402


@dataclass
class TensorrtInferenceArgs(OnnxInferenceArgs):
    engine_cache_dir: str | None = None


def load_tensorrt_session(
    checkpoint_dir: Path,
    fp16: bool = True,
    engine_cache_dir: Path | str | None = None,
) -> ort.InferenceSession:
    """Load ONNX model with TensorRT execution provider (FP16 by default).

    Expects model.onnx (and optionally model.onnx.data) in checkpoint_dir.
    First run may take several minutes while TensorRT builds and caches the engine.

    Args:
        checkpoint_dir: Directory containing model.onnx.
        fp16: If True, enable FP16 execution in TensorRT.
        engine_cache_dir: Optional directory to cache the TensorRT engine for faster subsequent loads.

    Returns:
        ONNX InferenceSession using TensorRT.
    """
    onnx_path = checkpoint_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    trt_options: dict[str, str | int | bool] = {
        "device_id": 0,
        "trt_fp16_enable": fp16,
    }
    if engine_cache_dir is not None:
        trt_options["trt_engine_cache_enable"] = True
        trt_options["trt_engine_cache_path"] = str(engine_cache_dir)

    providers = [
        ("TensorrtExecutionProvider", trt_options),
    ]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    logging.info(
        "Building TensorRT session (FP16=%s). First run may take several minutes.",
        fp16,
    )
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )
    return session


@parser.wrap()
def main(args: TensorrtInferenceArgs):
    init_logging()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    engine_cache = Path(args.engine_cache_dir).resolve() if args.engine_cache_dir else None
    if engine_cache is not None:
        engine_cache.mkdir(parents=True, exist_ok=True)

    logging.info("Loading ONNX model with TensorRT from %s", checkpoint_dir)
    session = load_tensorrt_session(
        checkpoint_dir,
        fp16=True,
        engine_cache_dir=engine_cache,
    )

    logging.info("Loading tokenizer %s", PI05_TOKENIZER_ID)
    tokenizer = AutoTokenizer.from_pretrained(PI05_TOKENIZER_ID)

    batch_size = 1
    state = np.zeros((batch_size, args.max_state_dim), dtype=np.float32)
    images = [
        np.zeros((batch_size, 3, args.resolution_height, args.resolution_width), dtype=np.float32)
        for _ in range(args.num_cams)
    ]
    rng = np.random.default_rng(args.seed)

    logging.info("Running %d inference repeat(s)...", args.n_repeats)
    step_times = []
    for step in range(args.n_repeats):
        t0 = time.perf_counter()
        actions = run_inference(
            session,
            tokenizer,
            args,
            prompt=args.prompt,
            state=state,
            images=images,
            rng=rng,
        )
        step_times.append(time.perf_counter() - t0)
        if step == 0:
            logging.info(
                "Output actions shape: %s (batch, n_action_steps, action_dim)",
                actions.shape,
            )
    times_ms = np.array(step_times) * 1000
    logging.info(
        "TensorRT inference: %d runs, total %.3f s, mean %.2f ms/run, std %.2f ms",
        args.n_repeats,
        np.sum(step_times),
        np.mean(times_ms),
        np.std(times_ms),
    )
    logging.info(
        "TensorRT inference latency (ms): min %.2f, max %.2f, median %.2f, p5 %.2f, p95 %.2f",
        np.min(times_ms),
        np.max(times_ms),
        np.median(times_ms),
        np.percentile(times_ms, 5),
        np.percentile(times_ms, 95),
    )


if __name__ == "__main__":
    main()
