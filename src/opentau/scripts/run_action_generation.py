#!/usr/bin/env python3

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

"""Benchmark NVIDIA Cosmos3-Super (64B) action generation ("policy" mode) on a single GPU.

Action generation = a single conditioning **image** + a **language prompt** -> a predicted
action chunk ``[chunk_size, action_dim]`` (no video in or out). This is the embodied-policy
direction (cf. Cosmos3-Nano-Policy-DROID): given a visual observation and an instruction,
generate a robot/agent action trajectory.

The 64B model (~129 GB bf16) is loaded with Hugging Face Accelerate `Big Model Inference`_
(`device_map`) -- it fits entirely on a single >=130 GB GPU (H200/B300, ``--offload none``),
or can be split across GPUs / offloaded to CPU on smaller cards (see ``--offload``/``--devices``).

Actions are produced by **flow matching** (the diffusion tower's denoising loop). The number of
flow-matching steps is exactly ``--num-inference-steps`` -- the action path uses the same
flow-matching ``UniPCMultistepScheduler`` as video (deep-copied per modality); there is no
separate hardcoded action step count. NVIDIA's examples use 30; ``--steps`` sweeps several
counts in one process (load once) to benchmark the speed/quality trade-off.

Experimental: not part of OpenTau's pinned env. Needs ``diffusers`` from source
(``Cosmos3OmniPipeline`` only exists on ``main``) plus ``transformers>=4.57``. Install into a
throwaway env on top of a CUDA 12.8+ torch::

    pip install "diffusers @ git+https://github.com/huggingface/diffusers.git" \
        "transformers>=4.57,<5" "accelerate>=0.31" av imageio imageio-ffmpeg \
        einops "peft>=0.11.1" sentencepiece ftfy huggingface_hub

.. _Big Model Inference:
    https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling

Example (single H200/B300, sweep 30/10/5 flow-matching steps, steady-state timing)::

    SNAP="$HF_HOME/hub/models--nvidia--Cosmos3-Super/snapshots/<sha>"
    python src/opentau/scripts/run_action_generation.py \
        --image "$SNAP/assets/example_action_fd_agibotworld_first_frame.png" \
        --prompt "Pickup items in the supermarket" \
        --domain agibotworld --chunk-size 50 --steps 30,10,5 \
        --offload none --warmup 1 --output action.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an action-generation benchmark."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="nvidia/Cosmos3-Super", help="HF repo id of the Cosmos3 omnimodel.")
    p.add_argument("--image", required=True, help="Single conditioning observation image (png/jpg).")
    p.add_argument("--prompt", default="Pickup items in the supermarket", help="Language instruction.")
    p.add_argument(
        "--domain",
        default="agibotworld",
        help="Embodiment/domain (sets action_dim: agibotworld=29, droid/franka=10, av=9).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of actions to generate -> output shape [chunk_size, action_dim].",
    )
    p.add_argument("--resolution-tier", type=int, default=480, choices=[256, 480, 704, 720])
    p.add_argument(
        "--view-point", default="concat_view", help="e.g. concat_view (agibotworld) or ego_view (av)."
    )
    p.add_argument(
        "--num-inference-steps", type=int, default=30, help="Flow-matching steps if --steps is empty."
    )
    p.add_argument(
        "--steps",
        default="",
        help="Comma-separated flow-matching step counts to benchmark in one load, e.g. '30,10,5'.",
    )
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--flow-shift", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-type",
        default="latent",
        choices=["latent", "pil"],
        help="'latent' skips the VAE decode so NO video frames are produced (only the action chunk is "
        "kept); 'pil' also decodes the jointly-denoised video. NB: Cosmos3-Super's policy mode always "
        "denoises a (chunk_size+1)-frame video latent alongside the actions in the same forward -- only "
        "the final decode-to-pixels is optional, the per-step compute is shared with action generation.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Discarded warmup passes before timing. Use 1 for steady-state (excludes CUDA/cuBLAS "
        "init, autotuning, PTX-JIT). Required with --compile (first pass triggers compilation).",
    )
    p.add_argument(
        "--compile", action="store_true", help="torch.compile the transformer (use with --warmup)."
    )
    p.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    # big-model strategy
    p.add_argument(
        "--offload",
        default="none",
        choices=["device_map", "sequential", "group", "none"],
        help="'none' = whole model on one GPU (fits on >=130GB, no hooks -> needed for --compile); "
        "'device_map' = Accelerate auto/split (1 GPU+CPU, or multi-GPU); 'sequential'/'group' = diffusers offload.",
    )
    p.add_argument(
        "--devices", default="0", help="Comma-separated GPU indices for device_map (e.g. '0' or '0,1')."
    )
    p.add_argument(
        "--gpu-mem-gib", type=int, default=135, help="Per-GPU max_memory cap under device_map='auto'."
    )
    p.add_argument(
        "--cpu-mem-gib",
        type=int,
        default=0,
        help="CPU max_memory cap under device_map; 0 forbids CPU offload.",
    )
    p.add_argument("--output", default="action.json", help="Where to write the predicted action JSON(s).")
    return p.parse_args()


def _build_layer_split_device_map(transformer_cls, model_id: str, devices: list[int]) -> dict:
    """Map every non-``layers`` module to ``devices[0]`` and split decoder layers across GPUs.

    Cosmos3's top-level forward computes index math over many modules at once, which breaks
    under a naive multi-GPU auto split. Keeping the small embedding/proj/norm/rotary modules
    together on one GPU and sharding only the large ``layers`` ModuleList sidesteps that while
    still fitting the full 129GB model on-GPU (no CPU offload).
    """
    from accelerate import init_empty_weights

    cfg = transformer_cls.load_config(model_id, subfolder="transformer")
    with init_empty_weights():
        skel = transformer_cls.from_config(cfg)
    device_map = {n: devices[0] for n, _ in skel.named_children() if n != "layers"}
    device_map.update({n: devices[0] for n, _ in skel.named_parameters(recurse=False)})
    n_layers = len(skel.layers)
    del skel
    for i in range(n_layers):
        device_map[f"layers.{i}"] = devices[i * len(devices) // n_layers]
    return device_map


def _summarize_device_map(module: torch.nn.Module) -> None:
    """Print how many parameters Accelerate placed on GPU vs CPU vs disk."""
    device_map = getattr(module, "hf_device_map", None)
    if not device_map:
        return

    def _device_for(name: str):
        # longest-prefix match (handles the single-device "" root key and per-layer keys)
        best_len, best_dev = -1, None
        for key, dev in device_map.items():
            if (key == "" or name == key or name.startswith(key + ".")) and len(key) > best_len:
                best_len, best_dev = len(key), dev
        return best_dev

    buckets: dict[str, int] = {}
    for n, p in module.named_parameters():
        dev = _device_for(n)
        dev_key = f"cuda:{dev}" if isinstance(dev, int) or str(dev).isdigit() else str(dev)
        buckets[dev_key] = buckets.get(dev_key, 0) + p.numel()
    grand = sum(buckets.values()) or 1
    print("[device_map] parameter placement:")
    for dev, n in sorted(buckets.items()):
        print(f"             {dev:>6}: {n / 1e9:7.2f} B params ({100 * n / grand:5.1f}%)")


def load_pipeline(args: argparse.Namespace):
    """Load ``Cosmos3OmniPipeline`` under the requested big-model strategy."""
    from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler

    dtype = torch.bfloat16  # model card: "Only BF16 precision is tested."

    if args.offload == "device_map":
        devices = [int(d) for d in str(args.devices).split(",") if d != ""]
        if len(devices) > 1:
            # Pin non-`layers` modules to the first GPU; split only decoder layers (a naive
            # auto split crashes -- Cosmos3's top-level forward mixes tensors across devices).
            device_map = _build_layer_split_device_map(Cosmos3OmniTransformer, args.model, devices)
            print(f"[load] device_map=split-layers across {devices} (no CPU offload)")
            transformer = Cosmos3OmniTransformer.from_pretrained(
                args.model, subfolder="transformer", torch_dtype=dtype, device_map=device_map
            )
        else:
            max_memory = dict.fromkeys(devices, f"{args.gpu_mem_gib}GiB")
            if args.cpu_mem_gib > 0:
                max_memory["cpu"] = f"{args.cpu_mem_gib}GiB"
            print(f"[load] device_map='auto' transformer  max_memory={max_memory}")
            transformer = Cosmos3OmniTransformer.from_pretrained(
                args.model,
                subfolder="transformer",
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
            )
        _summarize_device_map(transformer)
        pipe = Cosmos3OmniPipeline.from_pretrained(
            args.model, transformer=transformer, torch_dtype=dtype, enable_safety_checker=False
        )
        for name in ("vae", "vision_encoder", "text_encoder"):
            mod = getattr(pipe, name, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to("cuda")
    elif args.offload == "sequential":
        print("[load] enable_sequential_cpu_offload (submodule-level, fits anything, slow)")
        pipe = Cosmos3OmniPipeline.from_pretrained(args.model, torch_dtype=dtype, enable_safety_checker=False)
        pipe.enable_sequential_cpu_offload()
    elif args.offload == "group":
        print("[load] group offload (block_level, use_stream=True)")
        pipe = Cosmos3OmniPipeline.from_pretrained(args.model, torch_dtype=dtype, enable_safety_checker=False)
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        for name in ("vae", "vision_encoder", "text_encoder"):
            mod = getattr(pipe, name, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to("cuda")
    else:  # "none" -- whole model on one GPU, no accelerate hooks (needed for torch.compile)
        print("[load] full pipeline on cuda (single GPU, no device_map hooks)")
        pipe = Cosmos3OmniPipeline.from_pretrained(
            args.model, torch_dtype=dtype, enable_safety_checker=False
        ).to("cuda")

    # flow_shift lives on the scheduler, not the call.
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )

    if args.compile:
        from torch import _dynamo

        _dynamo.config.suppress_errors = True  # fall back to eager on any graph-break error
        print(f"[compile] torch.compile(transformer, mode={args.compile_mode!r}) -- compiles on first call")
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile_mode)
    return pipe


def main() -> None:
    """Generate an action chunk from a single image + prompt, benchmarking each step count."""
    from diffusers import CosmosActionCondition
    from PIL import Image

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    print(f"[gpu] {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")

    steps_list = [int(s) for s in str(args.steps).split(",") if s.strip() != ""] or [args.num_inference_steps]
    print(f"[cfg] policy action gen  domain={args.domain} chunk_size={args.chunk_size} steps={steps_list}")

    t0 = time.time()
    pipe = load_pipeline(args)
    print(f"[load] pipeline ready in {time.time() - t0:.1f}s")

    image = Image.open(args.image).convert("RGB")
    action = CosmosActionCondition(
        mode="policy",
        chunk_size=args.chunk_size,
        domain_name=args.domain,
        resolution_tier=args.resolution_tier,
        image=image,
        view_point=args.view_point,
    )

    def _generate(n_steps: int, output_type: str):
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        return pipe(
            prompt=args.prompt,
            action=action,
            fps=args.fps,
            num_inference_steps=n_steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            output_type=output_type,
            use_system_prompt=False,
            enable_safety_check=False,
        )

    # Warmup at the largest step count so timed runs are steady-state (the per-step kernels /
    # compiled graph are identical across step counts, so one warmup covers all).
    for w in range(args.warmup):
        _generate(max(steps_list), args.output_type)
        torch.cuda.synchronize()
        print(f"[warmup] pass {w + 1}/{args.warmup} done")

    results: dict[int, float] = {}
    for n in steps_list:
        torch.cuda.synchronize()
        t1 = time.time()
        res = _generate(n, args.output_type)
        torch.cuda.synchronize()
        dt = time.time() - t1
        if getattr(res, "action", None) is None:
            raise SystemExit(f"Pipeline returned no action. Got: {res}")
        act = res.action[0]
        act_t = (
            act.detach().float().cpu() if torch.is_tensor(act) else torch.as_tensor(act, dtype=torch.float32)
        )
        results[n] = dt
        print(f"[bench] steps={n:>3}  latency={dt:6.2f}s  action_shape={tuple(act_t.shape)}")
        out = (
            args.output[:-5] + f"_steps{n}.json"
            if args.output.endswith(".json")
            else f"{args.output}_steps{n}.json"
        )
        Path(out).write_text(
            json.dumps({"data": act_t.tolist(), "shape": list(act_t.shape), "dtype": "float32"})
        )

    # Quantify the optional video VAE-decode cost: time the opposite output_type at max steps.
    other = "pil" if args.output_type == "latent" else "latent"
    nmax = max(steps_list)
    torch.cuda.synchronize()
    t1 = time.time()
    res_other = _generate(nmax, other)
    torch.cuda.synchronize()
    dt_other = time.time() - t1
    vframes = len(res_other.video) if other == "pil" and hasattr(res_other.video, "__len__") else None
    print(
        f"[decode] steps={nmax} output_type={other!r}: {dt_other:.2f}s "
        f"(vs {results[nmax]:.2f}s at {args.output_type!r}) -> video-decode adds "
        f"~{abs(dt_other - results[nmax]):.2f}s; video frames generated={vframes}"
    )

    print(f"[summary] flow-matching steps -> action-generation latency (output_type={args.output_type!r}):")
    for n in steps_list:
        print(f"          {n:>3} steps: {results[n]:.2f} s")
    free, total = torch.cuda.mem_get_info()
    print(
        f"[gpu] peak alloc={torch.cuda.max_memory_allocated() / 1e9:.1f}GB  free={free / 1e9:.1f}/{total / 1e9:.1f}GB"
    )


if __name__ == "__main__":
    main()
