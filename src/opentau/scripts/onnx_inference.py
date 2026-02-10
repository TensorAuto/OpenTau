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

"""Inference script for exported ONNX models.

Loads an exported ONNX model (model.onnx + model.onnx.data) and runs inference.
Required shapes and settings are passed via command line.

Usage:
  python -m opentau.scripts.onnx_inference --checkpoint_dir <path> \\
    [--num_cams 2] [--resolution_height 224] [--resolution_width 224] \\
    [--prompt_max_length 256] [--n_action_steps 10] [--max_action_dim 32] [--max_state_dim 32] \\
    [--predict_response true] [--prompt "Pick up the object"] [--n_repeats 3]
"""

import logging
import pickle  # nosec: CWE-502
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from opentau.configs import parser
from opentau.utils.utils import init_logging

PI05_TOKENIZER_ID = "google/paligemma-3b-pt-224"


@dataclass
class OnnxInferenceArgs:
    checkpoint_dir: str
    num_cams: int = 2
    resolution_height: int = 224
    resolution_width: int = 224
    prompt_max_length: int = 256
    n_action_steps: int = 10
    max_action_dim: int = 32
    max_state_dim: int = 32
    predict_response: bool = False
    prompt: str = "Pick up the object and place it in the target location"
    n_repeats: int = 3
    provider: str | None = None
    seed: int = 42
    dump_path: Path | None = (
        None  # if provided, will dump input and output of the first run to this pickle file
    )


def _prepare_discrete_state(state: np.ndarray) -> list[str]:
    """Discretize state into 256 bins and return space-separated string per batch item.

    State is expected to be in [-1, 1]. Values outside are clipped.
    """
    state = np.clip(state.astype(np.float32), -1.0, 1.0)
    bins = np.linspace(-1.0, 1.0, 256 + 1)[:-1]
    discretized = np.digitize(state, bins) - 1
    return [" ".join(map(str, row)) for row in discretized]


def _build_prompt(task: str, state_str: str, predict_response: bool) -> str:
    """Build the full prompt string as expected by the PI05 tokenizer."""
    if predict_response:
        return f"Task: {task}<eos>State: {state_str}<eos>Response:"
    return f"Task: {task}<eos>State: {state_str}<eos>Actions:"


def _tokenize_prompt(tokenizer, prompt: str, prompt_max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a single prompt and return input_ids and attention_mask as numpy."""
    tokenized = tokenizer(
        [prompt],
        padding="max_length",
        padding_side="right",
        max_length=prompt_max_length,
        return_tensors="np",
        truncation=True,
    )
    return tokenized["input_ids"].astype(np.int64), tokenized["attention_mask"].astype(np.int64)


def load_onnx_session(checkpoint_dir: Path, provider: str | None = None) -> ort.InferenceSession:
    """Load ONNX model from checkpoint directory.
    Expects model.onnx (and optionally model.onnx.data for external weights) in checkpoint_dir.

    Args:
        checkpoint_dir: Path to the checkpoint directory containing the ONNX model.
        provider: ONNX runtime provider to use. If None, will use the default provider.

    Returns:
        ONNX inference session.
    """
    onnx_path = checkpoint_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if provider:
        providers = [provider]
    elif ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )
    return session


def run_inference(
    session: ort.InferenceSession,
    tokenizer,
    args: OnnxInferenceArgs,
    *,
    prompt: str,
    state: np.ndarray,
    images: list[np.ndarray],
    noise: np.ndarray | None = None,
    action_prefix: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    dump_path: Path | None = None,
) -> np.ndarray:
    """Run one ONNX inference step.

    Args:
        session: Loaded ONNX InferenceSession.
        tokenizer: HuggingFace tokenizer (e.g. PaliGemma).
        args: OnnxInferenceArgs with shapes and prompt settings.
        prompt: Task description string.
        state: State vector of shape (batch_size, max_state_dim) in [-1, 1].
        images: List of image arrays, each (batch_size, 3, H, W) in [0, 1].
        noise: Optional noise tensor (batch, n_action_steps, max_action_dim). If None, sampled from N(0,1).
        rng: Optional numpy RNG for reproducibility.

    Returns:
        actions: (batch_size, n_action_steps, action_dim) in unnormalized space.
    """
    batch_size = state.shape[0]
    rng = rng or np.random.default_rng()

    state_strs = _prepare_discrete_state(state)
    full_prompts = [_build_prompt(prompt, s, args.predict_response) for s in state_strs]
    p = full_prompts[0]
    lang_tokens, lang_masks = _tokenize_prompt(tokenizer, p, args.prompt_max_length)
    if batch_size > 1:
        lang_tokens = np.repeat(lang_tokens, batch_size, axis=0)
        lang_masks = np.repeat(lang_masks, batch_size, axis=0)

    if noise is None:
        noise = rng.standard_normal(
            (batch_size, args.n_action_steps, args.max_action_dim),
            dtype=np.float32,
        )

    if action_prefix is None:
        action_prefix = rng.standard_normal(noise.shape, dtype=np.float32)

    num_cams = args.num_cams
    if len(images) != num_cams:
        raise ValueError(f"Expected {num_cams} images (num_cams), got {len(images)}")
    resolution = (args.resolution_height, args.resolution_width)
    for i, img in enumerate(images):
        if img.shape[2:] != resolution:
            raise ValueError(f"Image {i} shape {img.shape}: expected H,W={resolution}")
        if img.shape[0] != batch_size:
            raise ValueError(f"Image {i} batch size {img.shape[0]} != state batch size {batch_size}")

    # Build input dict: lang_tokens, lang_masks, noise, image0, image1, ...
    input_feed = {
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks.astype(np.bool_),
        "noise": noise,
        "action_prefix": action_prefix,
    }
    for i in range(num_cams):
        input_feed[f"image{i}"] = images[i].astype(np.float32)

    # Output is "actions" with shape (batch_size, n_action_steps, action_dim)
    outputs = session.run(None, input_feed)
    actions = outputs[0]

    if dump_path:
        assert not dump_path.is_dir(), "Expected a file path, not a directory"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Saving inference input and results to %s", dump_path)
        with open(dump_path, "wb") as f:
            pickle.dump((input_feed, actions), f)

    return actions


@parser.wrap()
def main(args: OnnxInferenceArgs):
    init_logging()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    logging.info("Loading ONNX model from %s", checkpoint_dir)
    session = load_onnx_session(checkpoint_dir, provider=args.provider)

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
            dump_path=args.dump_path if step == 0 else None,  # only dump the first run
        )
        step_times.append(time.perf_counter() - t0)
        if step == 0:
            logging.info(
                "Output actions shape: %s (batch, n_action_steps, action_dim)",
                actions.shape,
            )
    times_ms = np.array(step_times) * 1000
    logging.info(
        "ONNX inference: %d runs, total %.3f s, mean %.2f ms/run, std %.2f ms",
        args.n_repeats,
        np.sum(step_times),
        np.mean(times_ms),
        np.std(times_ms),
    )
    logging.info(
        "ONNX inference latency (ms): min %.2f, max %.2f, median %.2f, p5 %.2f, p95 %.2f",
        np.min(times_ms),
        np.max(times_ms),
        np.median(times_ms),
        np.percentile(times_ms, 5),
        np.percentile(times_ms, 95),
    )


if __name__ == "__main__":
    main()
