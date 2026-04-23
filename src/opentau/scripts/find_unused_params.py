#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Audit policy parameters that DDP would flag as unused.

Builds the policy from a training config, runs one forward + backward
on a real batch (single GPU, no DDP or DeepSpeed wrapping), and lists
every parameter where ``param.requires_grad and param.grad is None``.
Those are exactly the parameters DDP would refuse to sync with
``find_unused_parameters=False``.

Usage — single Python invocation is enough (this script never needs
``accelerate launch``):

    python src/opentau/scripts/find_unused_params.py \
        --config_path=configs/libero/reproduce_pi05_libero.json

If the ``UNUSED`` section is empty you can safely set
``FIND_UNUSED_PARAMS=false`` when running training, reclaiming the
per-step graph-walk cost DDP otherwise pays. If not, each reported
tensor is either an orphan in the model (fix or freeze it) or a
conditionally-touched parameter (add an unconditional graph edge).

Environment variables:

  - ``INCLUDE_ZERO_GRAD=true``  —  also list parameters whose grad
    tensor exists but is all zero. Often indicates a code path that
    touches the parameter but produces no learning signal. Usually
    safe to ignore.

Output sections:

  - ``UNUSED`` — params with ``requires_grad=True`` and ``grad is None``
    after backward. These are the ones we need to fix or freeze.
  - ``ZERO-GRAD`` (only when ``INCLUDE_ZERO_GRAD=true``) — see above.
  - ``FROZEN`` — ``requires_grad=False`` params, listed for context.
"""

import logging
import math
import os
from collections import defaultdict
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset_mixture
from opentau.datasets.utils import cycle
from opentau.policies.factory import make_policy
from opentau.utils.utils import init_logging


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if device.type == "cuda" else obj
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
        out = [_to_device(v, device) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    return obj


def _module_root(name: str, depth: int = 2) -> str:
    """Group params by the first `depth` dotted components.

    Args:
        name: Parameter name like ``policy.paligemma_with_expert.gemma_expert.embed_tokens.weight``.
        depth: Number of leading components to keep.

    Returns:
        The root grouping key.
    """
    parts = name.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else name


@parser.wrap()
def find(cfg: TrainPipelineConfig):
    cfg.validate()
    init_logging(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    include_zero_grad = os.environ.get("INCLUDE_ZERO_GRAD", "false").lower() == "true"

    logging.info("Building dataset (one batch is enough)...")
    if cfg.val_freq > 0:
        train_dataset, _ = make_dataset_mixture(cfg)
    else:
        train_dataset = make_dataset_mixture(cfg)

    logging.info("Building policy...")
    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta)
    policy.to(device=device, dtype=torch.bfloat16)
    policy.train()

    train_dataloader = train_dataset.get_dataloader()
    it = cycle(train_dataloader)
    batch = _to_device(next(it), device)

    logging.info("Running one forward + backward...")
    losses = policy.forward(batch)
    loss = cfg.loss_weighting["MSE"] * losses["MSE"] + cfg.loss_weighting["CE"] * losses["CE"]
    # Don't use any optimizer / scaler — just raw autograd.
    loss.backward()

    unused: list[tuple[str, tuple[int, ...]]] = []
    zero_grad: list[tuple[str, tuple[int, ...]]] = []
    frozen: list[tuple[str, tuple[int, ...]]] = []
    used: list[tuple[str, tuple[int, ...]]] = []

    for name, param in policy.named_parameters():
        shape = tuple(param.shape)
        if not param.requires_grad:
            frozen.append((name, shape))
            continue
        if param.grad is None:
            unused.append((name, shape))
            continue
        # grad is not None — could still be all zero
        if include_zero_grad:
            with torch.no_grad():
                if torch.count_nonzero(param.grad) == 0:
                    zero_grad.append((name, shape))
                    continue
        used.append((name, shape))

    def _summarize(label: str, items: list[tuple[str, tuple[int, ...]]]) -> None:
        n = len(items)
        # math.prod avoids allocating a CPU tensor per shape, which matters
        # when `items` is long (hundreds of tensors per audit is normal).
        n_params = sum(math.prod(s) for _, s in items) if items else 0
        print(f"\n========== {label} ({n} tensors, {n_params:,} params) ==========")
        if not items:
            return
        # Group by root module so output is digestible
        by_root: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
        for name, shape in items:
            by_root[_module_root(name, depth=3)].append((name, shape))
        for root in sorted(by_root.keys()):
            group = by_root[root]
            group_n = sum(math.prod(s) for _, s in group)
            print(f"  [{root}]  ({len(group)} tensors, {group_n:,} params)")
            for name, shape in group:
                print(f"    - {name}  shape={shape}")

    # Use the policy type from the config so the header is accurate when this
    # script is run against pi0, value, etc. rather than being hard-coded to pi05.
    policy_name = getattr(cfg.policy, "type", "policy")
    print("\n#" + "=" * 78)
    print(f"# {policy_name} parameter audit — single forward + backward, single GPU")
    print(f"# include_zero_grad={include_zero_grad}")
    print("#" + "=" * 78)

    _summarize(
        "UNUSED (requires_grad=True, grad is None) — DDP will refuse without find_unused_parameters=True",
        unused,
    )
    if include_zero_grad:
        _summarize("ZERO-GRAD (grad exists but all zero) — usually safe to ignore", zero_grad)
    _summarize("FROZEN (requires_grad=False) — context", frozen)
    print(f"\n# USED (requires_grad=True, grad is non-trivial): {len(used)} tensors")
    print(
        "# Tip: if UNUSED list is empty, you can flip "
        "DistributedDataParallelKwargs(find_unused_parameters=False) safely."
    )


if __name__ == "__main__":
    find()
