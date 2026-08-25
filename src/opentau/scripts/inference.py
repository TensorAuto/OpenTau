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

"""Single-policy inference benchmark.

Optionally reports the denoising-acceleration uncertainty proxy of
:mod:`opentau.policies.accel` alongside the timings; enable it with
``OPENTAU_ACCEL_PREFIX=auto`` (see :func:`opentau.policies.accel.configure_accel`).
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

import torch

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.accel import configure_accel
from opentau.policies.candidates import configure_candidates
from opentau.policies.factory import get_policy_class
from opentau.policies.utils import maybe_compile_sample_actions, to_dtype_preserving_siglip_float32
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    auto_torch_device,
    create_dummy_observation,
    init_logging,
)


@parser.wrap()
def inference_main(cfg: TrainPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = auto_torch_device()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    # Preserve the float32-pinned SigLIP embeddings across the bf16 cast (openpi parity).
    to_dtype_preserving_siglip_float32(policy, device=device, dtype=torch.bfloat16)
    policy.eval()
    policy.model.sample_actions = maybe_compile_sample_actions(
        policy, policy.model.sample_actions, device_hint=device
    )

    # Always reset policy before episode to clear out action cache.
    policy.reset()

    # Denoising-acceleration uncertainty proxy — a no-op unless explicitly requested.
    # Done before the warmup calls so the compiled graph and any unsupported-checkpoint
    # refusal both land here rather than on the first timed run.
    accel_prefix = configure_accel(policy, cfg)

    # Best-of-N candidate sampling; also a no-op unless the config asks for it. Same reason
    # for the placement: critic loading, the dtype cast, the shape smoke-test and any
    # out-of-memory all land here rather than inside a timed run.
    configure_candidates(policy, cfg, device=device, dtype=torch.bfloat16)

    observation = create_dummy_observation(cfg, device, dtype=torch.bfloat16)

    print(observation.keys())

    with torch.inference_mode():
        # two warmup calls are needed right after compiling
        # the first warmup call is needed for compiling
        # the second warmup call is needed for kernel autotuning
        _ = policy.sample_actions(observation)
        _ = policy.sample_actions(observation)

        # Run 10 times and record inference times
        n_runs = 10
        times_ms = []
        accel_runs: list[list[float]] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            actions = policy.sample_actions(observation)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
            if accel_prefix is not None:
                # Already plain floats (`AccelMeter.to_list`), so reading it here costs
                # nothing and cannot carry an inference-mode tensor out of the block.
                last_accel = getattr(policy, "last_accel", None)
                if last_accel:
                    accel_runs.append(list(last_accel))

        actions = actions.to("cpu", torch.float32).numpy()
        print(f"Output shape: {actions.shape}")

        times_ms = torch.tensor(times_ms)
        print(
            f"Inference time (ms) over {n_runs} runs: min={times_ms.min().item():.2f}, max={times_ms.max().item():.2f}, avg={times_ms.mean().item():.2f}, std={times_ms.std().item():.2f}"
        )

    if accel_prefix is not None:
        if accel_runs:
            # Per-sample by construction; this benchmark's dummy observation is batch-1, so
            # flattening is a no-op here and still correct if that ever changes.
            flat = torch.tensor([v for run in accel_runs for v in run])
            print(
                f"accel (prefix={accel_prefix}) over {len(accel_runs)} runs x {len(accel_runs[0])} "
                f"sample(s): min={flat.min().item():.4f}, max={flat.max().item():.4f}, "
                f"avg={flat.mean().item():.4f}"
            )
            logging.info("accel provenance: %s", getattr(policy, "last_accel_provenance", None))
        else:
            # accel_prefix resolved, so `make_meter` should have produced a score — an empty
            # stream means this policy's `sample_actions` never publishes `last_accel`.
            logging.warning(
                "accel was enabled (prefix=%d) but no score was published; policy type %r "
                "does not populate `last_accel` in `sample_actions`.",
                accel_prefix,
                cfg.policy.type,
            )

    logging.info("End of inference")


if __name__ == "__main__":
    init_logging()
    inference_main()
