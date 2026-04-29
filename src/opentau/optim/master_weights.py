#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""fp32 master-weights wrapper for bf16 training.

This module provides :class:`MasterWeightOptimizer`, a duck-typed proxy
around a real ``torch.optim.Optimizer`` that mirrors the design used by
DeepSpeed's ``BF16_Optimizer`` (fp32 master copies + fp32 Adam state +
post-upcast gradient clipping). It is intended for the DDP / single /
FSDP code paths where accelerate does not otherwise allocate fp32
master parameters; under DeepSpeed ZeRO this wrapper should not be
used because ZeRO already provides equivalent semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.nn import Parameter


class MasterWeightOptimizer(torch.optim.Optimizer):
    """Duck-typed optimizer that keeps fp32 master copies of bf16 params.

    The wrapper holds a parallel list of fp32 master tensors built from
    the live (typically bf16) model parameters, owns a "real"
    ``torch.optim.Optimizer`` (e.g. AdamW) constructed over those fp32
    masters, and at each ``step()`` performs the classic three-phase
    update:

    1. **Upcast grads.** For every (live, master) pair, copy the live
       parameter's gradient into the fp32 master's gradient slot.
       :meth:`torch.Tensor.copy_` performs the dtype cast for us.
    2. **Step.** Run the underlying fp32 optimizer; all moment buffers
       and the master parameters update entirely in fp32.
    3. **Downcast weights.** Copy each updated fp32 master back into the
       corresponding live bf16 parameter, again relying on
       :meth:`torch.Tensor.copy_` to handle the cast.

    Under DDP this runs per-rank: gradients are already all-reduced in
    bf16 during backward, so the local fp32 upcast (and any subsequent
    :meth:`clip_grad_norm_`) yields identical per-rank fp32 norms with
    no extra cross-rank reduction.

    The class subclasses :class:`torch.optim.Optimizer` so that
    accelerate's ``isinstance`` checks recognise it and wrap it in
    :class:`accelerate.optimizer.AcceleratedOptimizer` during
    ``Accelerator.prepare``. We deliberately do *not* call
    ``super().__init__()`` — the base class would try to populate
    ``param_groups`` / ``state`` / ``defaults`` from our master list,
    but those attributes are exposed here as ``@property`` proxies onto
    the inner optimizer (so any mutation lands in one place). The
    upstream :class:`AcceleratedOptimizer` itself uses this same
    skip-super pattern.

    Attributes:
        inner: The underlying ``torch.optim.Optimizer`` operating on
            fp32 master parameters.
    """

    def __init__(
        self,
        inner_factory: Callable[[list[Parameter]], torch.optim.Optimizer],
        bf16_params: list[Parameter],
    ) -> None:
        """Build fp32 masters and instantiate the inner optimizer.

        Note:
            This intentionally does **not** call ``super().__init__()``;
            see the class docstring. ``param_groups`` / ``state`` /
            ``defaults`` are proxied to ``self.inner`` instead.

        Args:
            inner_factory: A callable that takes the list of fp32 master
                parameters and returns the real optimizer (e.g.
                ``lambda ps: torch.optim.AdamW(ps, lr=1e-4)``). The
                resulting optimizer's state therefore lives in fp32 by
                construction.
            bf16_params: The live (typically bf16) model parameters
                whose updates will be backed by fp32 master copies.
                Only parameters with ``requires_grad=True`` get a master
                copy; parameters without grads are skipped.
        """
        trainable: list[Parameter] = [p for p in bf16_params if p.requires_grad]
        masters: list[Parameter] = []
        for live in trainable:
            master = Parameter(live.detach().to(torch.float32).clone(), requires_grad=True)
            masters.append(master)

        self._live_params: list[Parameter] = trainable
        self._fp32_params: list[Parameter] = masters
        self.inner: torch.optim.Optimizer = inner_factory(masters)
        # When clip_grad_norm_ is called explicitly, it performs the
        # bf16->fp32 grad copy. step() must skip the copy in that case
        # to avoid clobbering any in-place modifications (e.g. clipping)
        # that were applied to the fp32 grads. zero_grad() resets it.
        self._upcast_done_this_step: bool = False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_existing(cls, optimizer: torch.optim.Optimizer) -> MasterWeightOptimizer:
        """Rebuild a wrapper around an already-constructed optimizer.

        This convenience classmethod takes an existing
        ``torch.optim.Optimizer`` (built by, e.g.,
        ``opentau.optim.factory.make_optimizer_and_scheduler``), extracts
        its ``defaults`` and ``param_groups``, builds fp32 masters from
        the current live params, and re-constructs the same optimizer
        type over the fp32 masters with identical kwargs and per-group
        overrides. The original optimizer is then discarded.

        This keeps ``make_optimizer_and_scheduler`` untouched and lets
        the train script opt into the wrapper with a single call.

        Args:
            optimizer: A built ``torch.optim.Optimizer`` whose ``param_groups``
                reference the live (typically bf16) model parameters.

        Returns:
            A :class:`MasterWeightOptimizer` whose inner optimizer has
            the same type, defaults, and per-group overrides as the
            input optimizer, but operates over fp32 master copies of
            the input's params.
        """
        cls_ = type(optimizer)
        original_defaults: dict[str, Any] = dict(optimizer.defaults)

        # Capture each group's kwargs (lr, weight_decay, etc.) so we can
        # rebuild the optimizer with the same per-group overrides. The
        # 'params' entry will be replaced with fp32 masters. We only keep
        # params-with-grads here so the master list length matches what
        # __init__ will build.
        live_groups: list[dict[str, Any]] = []
        for group in optimizer.param_groups:
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            live_params = [p for p in group["params"] if p.requires_grad]
            live_groups.append({"params": live_params, "kwargs": group_kwargs})

        # Flat list of all live params (in group order), used to build masters.
        all_live: list[Parameter] = []
        for g in live_groups:
            all_live.extend(g["params"])

        def _factory(masters: list[Parameter]) -> torch.optim.Optimizer:
            # Slice masters back into per-group lists, preserving order.
            # The per-group dicts already carry every hyperparameter
            # (lr, betas, eps, weight_decay, ...) so we do not pass
            # ``**original_defaults`` separately — that would re-supply
            # (or conflict with) keys like ``decoupled_weight_decay``
            # which appear in ``defaults`` but are not valid ctor kwargs.
            # After construction we overwrite ``defaults`` with the
            # original to preserve any extra keys the optimizer reads
            # back later.
            groups_for_inner: list[dict[str, Any]] = []
            offset = 0
            for g in live_groups:
                n = len(g["params"])
                group_dict = {"params": masters[offset : offset + n], **g["kwargs"]}
                groups_for_inner.append(group_dict)
                offset += n
            new_inner = cls_(groups_for_inner)
            new_inner.defaults = original_defaults
            return new_inner

        return cls(_factory, all_live)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _upcast_grads(self) -> None:
        """Copy live-parameter grads into the fp32 master grad slots.

        Allocates ``fp32_master.grad`` lazily on first use. After this
        runs, every fp32 master whose live param had a grad will have a
        same-shape fp32 grad.
        """
        for live, master in zip(self._live_params, self._fp32_params, strict=True):
            if live.grad is None:
                # Leave master.grad as-is (None or stale zero) — the
                # underlying optimizer will skip params without grads.
                continue
            if master.grad is None:
                master.grad = torch.zeros_like(master)
            # tensor.copy_() handles dtype conversion (bf16 -> fp32).
            master.grad.copy_(live.grad)

    def _downcast_params(self) -> None:
        """Copy each updated fp32 master back into the live bf16 param.

        The ``copy_`` call performs the fp32 -> bf16 cast in-place; any
        downstream consumer of the live params (forward pass, DDP grad
        bucketing) sees the rounded bf16 values.
        """
        for live, master in zip(self._live_params, self._fp32_params, strict=True):
            live.data.copy_(master.data)

    # ------------------------------------------------------------------
    # Optimizer-protocol methods
    # ------------------------------------------------------------------
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Run one optimizer step.

        Phases:
            1. Upcast bf16 grads to the fp32 master grad slots (skipped
               if :meth:`clip_grad_norm_` already did this for the
               current step).
            2. Step the inner fp32 optimizer.
            3. Downcast updated fp32 masters back to bf16 live params.

        Args:
            closure: Optional closure forwarded to the inner optimizer.

        Returns:
            Whatever the inner optimizer's ``step`` returns (typically
            ``None``, or the loss when a closure is provided).
        """
        if not self._upcast_done_this_step:
            self._upcast_grads()
        loss = self.inner.step(closure)
        self._downcast_params()
        # Reset the flag so the next training iteration re-runs the
        # upcast (zero_grad() also resets it; we do it here too to be
        # robust against callers that skip zero_grad).
        self._upcast_done_this_step = False
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on both live params and fp32 masters.

        Args:
            set_to_none: If True, set ``.grad`` to ``None`` (the modern
                default; cheaper and matches ``torch.optim.Optimizer``).
                If False, ``.grad.zero_()`` is called instead.
        """
        for live in self._live_params:
            if live.grad is None:
                continue
            if set_to_none:
                live.grad = None
            else:
                live.grad.detach_()
                live.grad.zero_()
        for master in self._fp32_params:
            if master.grad is None:
                continue
            if set_to_none:
                master.grad = None
            else:
                master.grad.detach_()
                master.grad.zero_()
        self._upcast_done_this_step = False

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """Clip the *fp32* master grads in place and return the fp32 norm.

        This first performs the bf16 -> fp32 grad upcast so that fp32
        masters have up-to-date grads, then applies
        :func:`torch.nn.utils.clip_grad_norm_` to the fp32 master
        gradients. The corresponding live bf16 grads are not modified;
        they will be ignored by :meth:`step`, which now reads from the
        clipped fp32 grads instead.

        Under DDP, gradients are already cross-rank-synced in bf16
        during ``accelerator.backward(...)``. The per-rank fp32 upcast
        of synced bf16 grads yields identical fp32 grads on every rank,
        so the resulting fp32 norm is identical on every rank without
        any extra reduction.

        Args:
            max_norm: Maximum allowed gradient norm.
            norm_type: Type of p-norm; defaults to 2 (L2 norm).

        Returns:
            The total fp32 grad norm (post-upcast, pre-clip), as
            returned by :func:`torch.nn.utils.clip_grad_norm_`.
        """
        self._upcast_grads()
        self._upcast_done_this_step = True
        return torch.nn.utils.clip_grad_norm_(self._fp32_params, max_norm=max_norm, norm_type=norm_type)

    # ------------------------------------------------------------------
    # State dict / param-group plumbing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return the inner optimizer's ``state_dict``.

        The fp32 masters are *parameters* of the inner optimizer, so
        their values are persisted there; on resume we recover them
        through :meth:`load_state_dict` plus
        :meth:`rebuild_masters_from_live`.

        Returns:
            The inner optimizer's state dict.
        """
        return self.inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore Adam state into the inner optimizer.

        ``torch.optim.Optimizer.state_dict`` captures only optimizer
        state (e.g. ``exp_avg``, ``exp_avg_sq``) and per-group
        hyperparameters; it does *not* persist parameter tensor values.
        We therefore do not attempt to recover fp32 master values from
        ``state_dict`` here. Callers should ensure the fp32 masters are
        consistent with the live bf16 parameters before/after this call
        (typically by invoking :meth:`rebuild_masters_from_live` after
        ``accelerator.load_state`` repopulates the live bf16 weights).

        Args:
            state_dict: A state dict previously produced by
                :meth:`state_dict`.
        """
        self.inner.load_state_dict(state_dict)
        self._upcast_done_this_step = False

    def rebuild_masters_from_live(self, live_params: Iterable[Parameter] | None = None) -> None:
        """Re-initialize fp32 masters from the current live parameters.

        Useful on resume: after ``accelerator.load_state(...)`` repopulates
        the bf16 model weights, call this to recreate the fp32 masters
        so the inner optimizer can subsequently load Adam state on top
        of consistent fp32 masters.

        Args:
            live_params: Optional iterable of live parameters in the
                same order this wrapper was built with. If ``None``,
                the wrapper uses the live params it was constructed
                with (this is the common case; the argument is
                provided for callers that hold an unwrapped reference,
                e.g. ``policy.parameters()``).
        """
        if live_params is None:
            live_iter = self._live_params
        else:
            live_iter = [p for p in live_params if p.requires_grad]
            if len(live_iter) != len(self._fp32_params):
                raise ValueError(
                    f"rebuild_masters_from_live: got {len(live_iter)} live params, "
                    f"expected {len(self._fp32_params)}"
                )
            # Refresh our internal live-param handles in case the caller
            # rebuilt the model (e.g. accelerate.prepare reordered).
            self._live_params = list(live_iter)

        for live, master in zip(self._live_params, self._fp32_params, strict=True):
            new_data = live.data.detach().to(torch.float32).clone()
            target_device = new_data.device
            # If Adam state was already populated and lives on a different
            # device than the live param now uses (e.g. masters constructed
            # at wrap-time when live was on CPU; live since moved to GPU by
            # ``accelerator.prepare``), migrate the state tensors to the new
            # device before re-pointing master.data. Without this the next
            # ``step()`` would crash on a device-mismatch addmul.
            if master in self.inner.state:
                for k, v in self.inner.state[master].items():
                    if isinstance(v, torch.Tensor) and v.device != target_device:
                        self.inner.state[master][k] = v.to(target_device)
            # ``master.data = ...`` re-points the Parameter at a fresh
            # storage tensor (potentially on a different device). The
            # Parameter object identity is preserved, so the inner
            # optimizer's ``param_groups`` references remain valid.
            master.data = new_data
            if master.grad is not None:
                master.grad = None
        self._upcast_done_this_step = False

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Proxy to the inner optimizer's ``param_groups``.

        Returning the inner groups (whose ``params`` are fp32 masters)
        is what we want: LR schedulers mutate ``param_groups[i]["lr"]``,
        and ``optimizer.param_groups[0]["lr"]`` in train.py reads the
        same dict back.
        """
        return self.inner.param_groups

    @param_groups.setter
    def param_groups(self, groups: list[dict[str, Any]]) -> None:
        self.inner.param_groups = groups

    @property
    def state(self) -> dict[Any, Any]:
        """Proxy to the inner optimizer's ``state`` dict."""
        return self.inner.state

    @state.setter
    def state(self, value: dict[Any, Any]) -> None:
        self.inner.state = value

    @property
    def defaults(self) -> dict[str, Any]:
        """Proxy to the inner optimizer's ``defaults`` dict.

        ``AcceleratedOptimizer`` reads this; LR schedulers may also
        check it for ``initial_lr``.
        """
        return self.inner.defaults

    @defaults.setter
    def defaults(self, value: dict[str, Any]) -> None:
        self.inner.defaults = value

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Register a new param group, building fp32 masters as needed.

        Args:
            param_group: A param-group dict whose ``params`` entry
                contains live (typically bf16) parameters. Each
                trainable param gets a corresponding fp32 master, and
                the inner optimizer is given the master-backed group.
        """
        live_new: list[Parameter] = [p for p in param_group["params"] if p.requires_grad]
        master_new: list[Parameter] = [
            Parameter(p.detach().to(torch.float32).clone(), requires_grad=True) for p in live_new
        ]
        self._live_params.extend(live_new)
        self._fp32_params.extend(master_new)

        inner_group = {**param_group, "params": master_new}
        self.inner.add_param_group(inner_group)
