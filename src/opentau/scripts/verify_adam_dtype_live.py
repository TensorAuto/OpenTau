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

"""Print Adam optimizer-state dtypes under a live accelerate configuration.

This script is a minimal reproducer of the bug in issue #181. It replicates
the pre-fix prefix of ``train.py`` (build policy, ``model.to(bf16)`` on every
backend, construct raw ``torch.optim.AdamW`` directly — no master-weights
wrapper) on a synthetic ``nn.Linear``, then prints the dtypes that materialise
in ``optimizer.state``.

Expected A/B (independent of which branch you run on, because this script
deliberately does NOT route through ``train.py``'s wrapper):

* DDP / single-process / FSDP — ``exp_avg`` and ``exp_avg_sq`` are bf16.
  This is the bug: ``torch.zeros_like(p)`` inherits bf16 from the param.
* DeepSpeed ZeRO-2 — ``exp_avg`` and ``exp_avg_sq`` are fp32. DeepSpeed
  replaces the user optimiser with its own ZeRO optimiser, which always
  allocates fp32 master copies (``single_partition_of_fp32_groups``).

The wrapper itself is exercised by ``tests/optim/test_master_weights.py``.

Usage:
    accelerate launch --config_file <cfg> \
        src/opentau/scripts/verify_adam_dtype_live.py
"""

from __future__ import annotations

import logging

import accelerate
import torch
from torch import nn


def build_model(in_features: int = 1024, out_features: int = 1024) -> nn.Module:
    """Construct a minimal stand-in for the real policy.

    Args:
        in_features: Width of the linear layer's input.
        out_features: Width of the linear layer's output.

    Returns:
        A two-parameter ``nn.Linear`` model. Kept intentionally small so
        the script fits on CPU/single-GPU smoke tests.
    """
    return nn.Linear(in_features, out_features)


def cast_to_bf16(model: nn.Module) -> None:
    """Cast model in-place to bfloat16, mirroring ``train.py:217`` unconditionally.

    The bug in #181 only manifests when model params are bf16 *before*
    ``torch.optim.AdamW`` is constructed: state tensors are then allocated
    via ``torch.zeros_like(p)`` and inherit bf16. We always cast here so
    the script is a faithful repro of the production code path.

    Args:
        model: The module to cast in-place to bf16.
    """
    model.to(torch.bfloat16)


def apply_deepspeed_dataloaderless_workaround(accelerator: accelerate.Accelerator) -> None:
    """Set ``train_micro_batch_size_per_gpu`` so DS ``prepare()`` works without a DataLoader.

    accelerate's DeepSpeed path normally infers the per-rank batch size from
    a dataloader passed to ``prepare()``. This script doesn't use one (a
    single synthetic step is enough), so we set the required field directly
    on the DeepSpeed plugin config. The default value is the string ``"auto"``
    (see ``accelerate/utils/dataclasses.py``); we must overwrite it
    unconditionally — ``setdefault`` is a no-op since the key already exists.
    No-op for non-DeepSpeed backends.

    Args:
        accelerator: The already-constructed accelerator. Must be inspected
            *before* ``accelerator.prepare`` is called.
    """
    if accelerator.distributed_type != accelerate.DistributedType.DEEPSPEED:
        return
    plugin = accelerator.state.deepspeed_plugin
    if plugin is None:
        return
    plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1


def step_once(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: accelerate.Accelerator,
) -> None:
    """Run a single backward + optimizer step with a dummy loss.

    Uses ``accelerator.backward(loss)`` rather than raw ``loss.backward()``
    because under DeepSpeed the prepared optimizer's ``step()`` is a no-op
    (see ``accelerate/utils/deepspeed.py``: ``DeepSpeedOptimizerWrapper.step``);
    DS performs the real optimiser step inside ``accelerator.backward`` via
    ``DeepSpeedEngineWrapper.backward`` → ``engine.step()``. Calling raw
    ``loss.backward()`` + ``optimizer.step()`` would leave ``optimizer.state``
    empty under DS while still working under DDP.

    Args:
        model: The prepared model (possibly an accelerate wrapper).
        optimizer: The prepared optimizer (an ``AcceleratedOptimizer`` or
            ``DeepSpeedOptimizerWrapper``).
        accelerator: The live accelerator; routed through here so the
            DS path triggers its internal step.
    """
    underlying = model.module if hasattr(model, "module") else model
    param_dtype = next(underlying.parameters()).dtype
    x = torch.randn(4, 1024, device=accelerator.device, dtype=param_dtype)
    y = model(x)
    loss = y.pow(2).mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


def _walk_to_populated_state(opt: object) -> object:
    """Walk a chain of ``.optimizer`` attributes until ``.state`` is non-empty.

    Under DDP the prepared optimiser is an ``AcceleratedOptimizer`` whose
    ``state`` property already delegates to the inner torch optimiser, so
    the first level usually has populated state. Under DeepSpeed the
    chain is two layers deep (wrapper → DSZero → inner torch AdamW over
    fp32 flat masters) but each layer's ``state`` proxies via property,
    so the top-level ``.state`` is also populated *after* the DS step ran.
    The walk is a defence-in-depth: if any intermediate level returned an
    empty dict for some accelerate version, we keep descending.

    Args:
        opt: The prepared optimiser (any accelerate / DS wrapper).

    Returns:
        The deepest object in the ``.optimizer`` chain whose ``.state``
        is a non-empty mapping. Falls back to ``opt`` if no level has
        populated state.
    """
    seen: set[int] = set()
    cur = opt
    while id(cur) not in seen:
        seen.add(id(cur))
        state = getattr(cur, "state", None)
        if isinstance(state, dict) and len(state) > 0:
            return cur
        nxt = getattr(cur, "optimizer", None)
        if nxt is None or id(nxt) in seen:
            break
        cur = nxt
    return opt


def print_state_dtypes(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Walk ``optimizer.state`` and print dtype info for every parameter.

    The keys in the populated state are the parameters the *inner* torch
    optimiser actually steps. Under DDP these match ``model.named_parameters()``
    one-to-one; under DeepSpeed they are fp32 flat-master partition tensors
    that don't appear in the model's named-parameter list, so the lookup
    falls back to ``<flat-master-N>``.

    Args:
        model: The prepared model; used to recover per-parameter names
            for the DDP path.
        optimizer: The prepared optimizer (any accelerate / DS wrapper).
    """
    inner = _walk_to_populated_state(optimizer)
    underlying = model.module if hasattr(model, "module") else model
    id_to_name = {id(p): n for n, p in underlying.named_parameters()}

    print(f"  (state dump from {type(inner).__name__})")
    print(f"{'name':<30} {'param':<14} {'exp_avg':<14} {'exp_avg_sq':<14}")
    state = getattr(inner, "state", {})
    if not state:
        print("  (empty optimizer state — accelerator.backward(loss) probably did not run)")
        return
    for i, (param, param_state) in enumerate(state.items()):
        name = id_to_name.get(id(param), f"<flat-master-{i}>")
        exp_avg = param_state.get("exp_avg")
        exp_avg_sq = param_state.get("exp_avg_sq")
        exp_avg_dtype = exp_avg.dtype if exp_avg is not None else None
        exp_avg_sq_dtype = exp_avg_sq.dtype if exp_avg_sq is not None else None
        print(f"{name:<30} {str(param.dtype):<14} {str(exp_avg_dtype):<14} {str(exp_avg_sq_dtype):<14}")


def main() -> None:
    """Entry point: prepare a minimal model + optimizer and dump dtypes."""
    logging.basicConfig(level=logging.INFO)
    accelerator = accelerate.Accelerator()
    apply_deepspeed_dataloaderless_workaround(accelerator)

    print(f"distributed_type = {accelerator.distributed_type}")
    print(f"mixed_precision  = {accelerator.state.mixed_precision}")

    model = build_model()
    cast_to_bf16(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model, optimizer = accelerator.prepare(model, optimizer)
    step_once(model, optimizer, accelerator)
    print_state_dtypes(model, optimizer)


if __name__ == "__main__":
    main()
