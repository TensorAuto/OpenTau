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

"""CPU-only unit tests for :class:`MasterWeightOptimizer`.

These tests pin the contract of the fp32 master-weights wrapper used
under DDP / single / FSDP backends (see issue #181). Everything runs on
bf16 + CPU; total wall-time is well under five seconds.
"""

from __future__ import annotations

import pytest
import torch

from opentau.optim.master_weights import MasterWeightOptimizer


def _make_bf16_linear(in_features: int = 8, out_features: int = 4) -> torch.nn.Linear:
    """Build a bf16 ``nn.Linear`` with deterministic weights for testing.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.

    Returns:
        A bf16 ``nn.Linear`` whose weights and bias are seeded.
    """
    torch.manual_seed(0)
    layer = torch.nn.Linear(in_features, out_features)
    return layer.to(torch.bfloat16)


def _adamw_factory(lr: float = 1e-2):
    """Return an inner-optimizer factory that builds AdamW over fp32 masters.

    Args:
        lr: Learning rate for the inner ``AdamW``.

    Returns:
        Callable taking a list of fp32 params and returning a fresh
        ``torch.optim.AdamW``.
    """

    def _factory(params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    return _factory


def test_construction_creates_fp32_masters_with_matching_shapes():
    """fp32 masters should exist, be fp32, and match bf16 param shapes."""
    layer = _make_bf16_linear()
    bf16_params = list(layer.parameters())
    opt = MasterWeightOptimizer(_adamw_factory(), bf16_params)

    assert len(opt._fp32_params) == len(bf16_params)
    for live, master in zip(bf16_params, opt._fp32_params, strict=True):
        assert master.dtype == torch.float32
        assert master.shape == live.shape
        assert master.requires_grad is True


def test_step_converges_synthetic_regression():
    """Running a few steps of fp32 AdamW through the wrapper drives loss down."""
    torch.manual_seed(42)
    layer = _make_bf16_linear(in_features=4, out_features=1)
    target_w = torch.randn(1, 4, dtype=torch.bfloat16)
    target_b = torch.randn(1, dtype=torch.bfloat16)

    opt = MasterWeightOptimizer(_adamw_factory(lr=5e-2), list(layer.parameters()))

    x = torch.randn(64, 4, dtype=torch.bfloat16)
    y = x @ target_w.T + target_b

    initial_loss = None
    final_loss = None
    for step in range(80):
        opt.zero_grad()
        pred = layer(x)
        loss = ((pred - y) ** 2).float().mean()
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    # bf16 noise floor is non-trivial; just require clear improvement.
    assert final_loss < 0.5 * initial_loss, (
        f"expected loss to drop materially; initial={initial_loss}, final={final_loss}"
    )


def test_step_updates_live_bf16_params_to_match_master_downcast():
    """After step(), live bf16 weights must equal master.to(bf16) bitwise."""
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))

    x = torch.randn(16, 8, dtype=torch.bfloat16)
    target = torch.randn(16, 4, dtype=torch.bfloat16)
    loss = ((layer(x) - target) ** 2).float().mean()
    loss.backward()
    opt.step()

    for live, master in zip(layer.parameters(), opt._fp32_params, strict=True):
        assert live.dtype == torch.bfloat16
        torch.testing.assert_close(live.data, master.data.to(torch.bfloat16), rtol=0, atol=0)


def test_adam_state_is_fp32():
    """``exp_avg`` / ``exp_avg_sq`` must be fp32 when the masters are fp32."""
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))

    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = torch.randn(8, 4, dtype=torch.bfloat16)
    loss = ((layer(x) - target) ** 2).float().mean()
    loss.backward()
    opt.step()

    inner_state = opt.inner.state
    assert len(inner_state) == len(opt._fp32_params)
    for state in inner_state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


def test_clip_grad_norm_returns_fp32_and_actually_clips():
    """clip_grad_norm_ must scale the fp32 master grads in place to <= max_norm."""
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(lr=1e-2), list(layer.parameters()))

    # Force a large gradient by using a huge target.
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = 1e3 * torch.randn(8, 4, dtype=torch.bfloat16)

    loss = ((layer(x) - target) ** 2).float().mean()
    loss.backward()
    norm = opt.clip_grad_norm_(max_norm=0.5)

    # Return type / dtype: fp32 norm tensor.
    assert isinstance(norm, torch.Tensor)
    assert norm.dtype == torch.float32
    # Pre-clip norm should clearly exceed the threshold.
    assert norm.item() > 0.5

    # Post-clip, the actual fp32 master-grad norm must be <= max_norm
    # (with a small numerical tolerance).
    grads = [m.grad for m in opt._fp32_params if m.grad is not None]
    post_clip_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g) for g in grads])
    ).item()
    assert post_clip_norm <= 0.5 + 1e-5, f"post-clip norm {post_clip_norm} exceeds 0.5"


def test_clip_then_step_does_not_double_upcast():
    """clip_grad_norm_ + step should yield the same weights as a hand-rolled clip."""
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(lr=1e-2), list(layer.parameters()))

    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = 1e2 * torch.randn(8, 4, dtype=torch.bfloat16)
    loss = ((layer(x) - target) ** 2).float().mean()
    loss.backward()

    pre_clip_master_grads = [m.grad.clone() if m.grad is not None else None for m in opt._fp32_params]
    # No master grads exist yet because step()/clip didn't run.
    assert all(g is None for g in pre_clip_master_grads)

    # Clip; this performs the upcast and writes into master.grad.
    opt.clip_grad_norm_(max_norm=0.1)
    clipped_master_grads = [m.grad.clone() for m in opt._fp32_params]

    # Step shouldn't redo the upcast — i.e. master.grad just before
    # inner.step() must equal what clip_grad_norm_ left.
    # We verify indirectly by checking the post-step weights match a
    # hand-rolled equivalent: clone state, do the same clip, step
    # manually using the same fp32 grads.
    opt.step()

    # Reference: rebuild a fresh wrapper, install identical clipped grads, step.
    layer_ref = _make_bf16_linear()
    opt_ref = MasterWeightOptimizer(_adamw_factory(lr=1e-2), list(layer_ref.parameters()))
    for m, g in zip(opt_ref._fp32_params, clipped_master_grads, strict=True):
        m.grad = g.clone()
    # Mark upcast done so step() won't overwrite the grads we just installed.
    opt_ref._upcast_done_this_step = True
    opt_ref.step()

    for m1, m2 in zip(opt._fp32_params, opt_ref._fp32_params, strict=True):
        torch.testing.assert_close(m1.data, m2.data)


def test_state_dict_round_trip_preserves_optimizer_state():
    """state_dict / load_state_dict should round-trip the Adam state exactly.

    ``torch.optim.Optimizer.state_dict`` does not persist parameter
    values — only ``state`` (Adam moments) and per-group hyperparams.
    The wrapper inherits that contract, so this test focuses on Adam
    state round-tripping; consistency of the fp32 masters with the
    live bf16 params is the caller's responsibility on resume (see
    :meth:`rebuild_masters_from_live`).
    """
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))

    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = torch.randn(8, 4, dtype=torch.bfloat16)
    loss = ((layer(x) - target) ** 2).float().mean()
    loss.backward()
    opt.step()

    saved = opt.state_dict()

    # Build a fresh wrapper and load the saved state.
    layer2 = _make_bf16_linear()
    opt2 = MasterWeightOptimizer(_adamw_factory(), list(layer2.parameters()))
    opt2.load_state_dict(saved)

    # Inner Adam state must round-trip and remain fp32.
    for s1, s2 in zip(opt.inner.state.values(), opt2.inner.state.values(), strict=True):
        torch.testing.assert_close(s1["exp_avg"], s2["exp_avg"])
        torch.testing.assert_close(s1["exp_avg_sq"], s2["exp_avg_sq"])
        assert s1["exp_avg"].dtype == torch.float32
        assert s2["exp_avg"].dtype == torch.float32

    # Per-group hyperparams (lr, betas, eps, weight_decay, ...) must match.
    for g1, g2 in zip(opt.param_groups, opt2.param_groups, strict=True):
        for key in ("lr", "betas", "eps", "weight_decay"):
            assert g1[key] == g2[key]


def test_zero_grad_zeros_both_bf16_and_fp32_grads():
    """zero_grad() must clear grads on live params *and* fp32 masters."""
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))

    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = torch.randn(8, 4, dtype=torch.bfloat16)
    loss = ((layer(x) - target) ** 2).float().mean()
    loss.backward()

    # After backward, live grads exist; master grads do not yet.
    assert all(p.grad is not None for p in layer.parameters())

    # Force the upcast so master grads are populated, then zero everything.
    opt.clip_grad_norm_(max_norm=1e9)  # trivial clip just to upcast
    assert all(m.grad is not None for m in opt._fp32_params)

    # set_to_none=True path
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in layer.parameters())
    assert all(m.grad is None for m in opt._fp32_params)
    assert opt._upcast_done_this_step is False

    # set_to_none=False path: rebuild grads, then zero in place.
    loss2 = ((layer(x) - target) ** 2).float().mean()
    loss2.backward()
    opt.clip_grad_norm_(max_norm=1e9)
    opt.zero_grad(set_to_none=False)
    for p in layer.parameters():
        assert p.grad is not None
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))
    for m in opt._fp32_params:
        assert m.grad is not None
        torch.testing.assert_close(m.grad, torch.zeros_like(m.grad))


def test_from_existing_preserves_defaults_and_state_dtypes():
    """``from_existing`` must rebuild masters and keep fp32 Adam state."""
    layer = _make_bf16_linear()
    inner = torch.optim.AdamW(
        list(layer.parameters()), lr=3e-4, betas=(0.9, 0.95), eps=1e-7, weight_decay=0.1
    )
    wrapped = MasterWeightOptimizer.from_existing(inner)

    assert wrapped.defaults["lr"] == 3e-4
    assert wrapped.defaults["betas"] == (0.9, 0.95)
    assert wrapped.defaults["eps"] == 1e-7
    assert wrapped.defaults["weight_decay"] == 0.1

    # Per-group lr override should also propagate.
    layer2 = _make_bf16_linear()
    inner2 = torch.optim.AdamW(
        [
            {"params": [layer2.weight], "lr": 1e-3},
            {"params": [layer2.bias], "lr": 1e-2},
        ],
        weight_decay=0.0,
    )
    wrapped2 = MasterWeightOptimizer.from_existing(inner2)
    assert wrapped2.param_groups[0]["lr"] == 1e-3
    assert wrapped2.param_groups[1]["lr"] == 1e-2

    # Step once and confirm Adam state is fp32 on the second wrapper.
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = torch.randn(8, 4, dtype=torch.bfloat16)
    loss = ((layer2(x) - target) ** 2).float().mean()
    loss.backward()
    wrapped2.step()
    for state in wrapped2.inner.state.values():
        assert state["exp_avg"].dtype == torch.float32


def test_rebuild_masters_from_live_resyncs_after_external_load():
    """After external bf16 weight load, rebuild_masters_from_live restores fp32 masters."""
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))

    # Simulate accelerator.load_state by overwriting bf16 weights directly.
    new_weight = torch.randn_like(layer.weight)
    layer.weight.data.copy_(new_weight)

    # Masters are now stale.
    assert not torch.equal(opt._fp32_params[0].data, layer.weight.data.to(torch.float32))

    opt.rebuild_masters_from_live(layer.parameters())

    # Masters now match the live params (in fp32).
    for live, master in zip(layer.parameters(), opt._fp32_params, strict=True):
        torch.testing.assert_close(master.data, live.data.to(torch.float32))


def test_wrapper_is_torch_optimizer_subclass():
    """The wrapper must satisfy ``isinstance(_, torch.optim.Optimizer)``.

    Without this, accelerate's ``_prepare_one`` (``accelerator.py:1403``)
    skips the wrapper entirely — no ``AcceleratedOptimizer`` wrapping,
    no gradient-accumulation gating on ``step()`` / ``zero_grad()``, no
    inclusion in ``Accelerator._optimizers`` (so the LR scheduler can't
    find it via ``prepare_scheduler``'s identity match either).
    """
    layer = _make_bf16_linear()
    inner = torch.optim.AdamW(list(layer.parameters()))
    wrapped = MasterWeightOptimizer.from_existing(inner)

    assert isinstance(wrapped, torch.optim.Optimizer)


def test_lr_scheduler_reaches_inner_after_from_existing_with_rebind():
    """Mimics ``train.py``'s prefix and pins the scheduler-rebind contract.

    ``make_optimizer_and_scheduler`` builds the scheduler bound to the
    original ``torch.optim.AdamW``. ``MasterWeightOptimizer.from_existing``
    discards that AdamW and creates a fresh inner with new ``param_groups``
    dicts. Without rebinding ``scheduler.optimizer = wrapped``,
    ``scheduler.step()`` mutates the orphaned optimizer's groups and the
    inner AdamW's lr never moves — schedule silently applied to nothing.
    """
    layer = _make_bf16_linear(in_features=4, out_features=1)
    inner_opt = torch.optim.AdamW(list(layer.parameters()), lr=1.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(inner_opt, start_factor=1.0, end_factor=0.5, total_iters=4)
    initial_lr = scheduler.optimizer.param_groups[0]["lr"]
    assert initial_lr == 1.0

    wrapped = MasterWeightOptimizer.from_existing(inner_opt)
    # The rebind that train.py performs immediately after from_existing.
    scheduler.optimizer = wrapped

    x = torch.randn(8, 4, dtype=torch.bfloat16)
    target = torch.randn(8, 1, dtype=torch.bfloat16)
    for _ in range(2):
        ((layer(x) - target) ** 2).float().mean().backward()
        wrapped.step()
        wrapped.zero_grad()
        scheduler.step()

    # The inner AdamW's lr must reflect the LinearLR decay.
    inner_lr = wrapped.inner.param_groups[0]["lr"]
    assert inner_lr < initial_lr, (
        f"LR decay did not reach the inner optimizer: still {inner_lr} after 2 scheduler steps"
    )
    # And the wrapper's exposed lr must equal the inner's (single source of truth).
    assert wrapped.param_groups[0]["lr"] == inner_lr


def test_lr_scheduler_does_not_reach_inner_without_rebind():
    """Negative control for the rebind contract above.

    Documents the failure mode if a future change drops the
    ``scheduler.optimizer = wrapped`` line in ``train.py``: the inner
    AdamW's lr stays frozen at the construction value while the
    scheduler happily mutates an orphaned optimizer.
    """
    layer = _make_bf16_linear(in_features=4, out_features=1)
    inner_opt = torch.optim.AdamW(list(layer.parameters()), lr=1.0, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(inner_opt, start_factor=1.0, end_factor=0.5, total_iters=4)

    wrapped = MasterWeightOptimizer.from_existing(inner_opt)
    # Deliberately do NOT rebind: scheduler.optimizer is still the original AdamW.

    x = torch.randn(8, 4, dtype=torch.bfloat16)
    target = torch.randn(8, 1, dtype=torch.bfloat16)
    for _ in range(2):
        ((layer(x) - target) ** 2).float().mean().backward()
        wrapped.step()
        wrapped.zero_grad()
        scheduler.step()

    # The orphaned optimizer's lr decayed (proves the scheduler IS stepping).
    assert inner_opt.param_groups[0]["lr"] < 1.0
    # But the wrapper's inner is stuck — the bug we're guarding against.
    assert wrapped.inner.param_groups[0]["lr"] == 1.0


@pytest.mark.gpu
def test_rebuild_masters_migrates_to_live_device_on_fresh_run():
    """Fresh-run regression for the dd1154e migration bug.

    The wrapper is constructed before ``accelerator.prepare`` runs, so the
    fp32 masters are cloned while the live policy is still on CPU.
    ``accelerator.prepare`` later migrates the live params to GPU; the next
    call to ``rebuild_masters_from_live`` must follow the live device, not
    silently leave masters on CPU. (Pre-fix, ``master.data.copy_(...)``
    preserved master's old device — Adam ran on CPU and every step paid a
    GPU<->CPU memcpy.)
    """
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))
    for master in opt._fp32_params:
        assert master.device.type == "cpu"

    layer.cuda()
    opt.rebuild_masters_from_live(layer.parameters())

    for master in opt._fp32_params:
        assert master.device.type == "cuda"
        assert master.dtype == torch.float32
    for live, master in zip(layer.parameters(), opt._fp32_params, strict=True):
        torch.testing.assert_close(master.data, live.data.to(torch.float32))


@pytest.mark.gpu
def test_rebuild_masters_migrates_populated_state_and_step_succeeds():
    """Resume-path regression: populated Adam state must follow the new device.

    On resume, ``accelerator.load_state`` restores Adam state on whatever
    device it was checkpointed from; ``accelerator.prepare`` may have
    independently migrated the live policy. ``rebuild_masters_from_live``
    must move ``exp_avg`` / ``exp_avg_sq`` to the live device before
    re-pointing ``master.data``, otherwise the next ``step()`` crashes on a
    device-mismatch ``addmul_``.
    """
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))

    # Populate Adam state on CPU.
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    target = torch.randn(8, 4, dtype=torch.bfloat16)
    ((layer(x) - target) ** 2).float().mean().backward()
    opt.step()
    for master in opt._fp32_params:
        for value in opt.inner.state[master].values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"

    layer.cuda()
    opt.rebuild_masters_from_live(layer.parameters())

    for master in opt._fp32_params:
        assert master.device.type == "cuda"
        for value in opt.inner.state[master].values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cuda"

    # Next step on GPU must succeed (no device-mismatch addmul_).
    x_gpu = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda")
    target_gpu = torch.randn(8, 4, dtype=torch.bfloat16, device="cuda")
    ((layer(x_gpu) - target_gpu) ** 2).float().mean().backward()
    opt.step()


@pytest.mark.gpu
def test_rebuild_masters_preserves_parameter_identity_across_migration():
    """``master.data = ...`` must keep the ``Parameter`` object identical.

    The inner ``torch.optim.AdamW`` holds references to the master
    ``nn.Parameter`` objects in its ``param_groups``. If
    ``rebuild_masters_from_live`` swapped in fresh ``Parameter`` objects
    (rather than re-assigning ``.data`` on the existing ones), the inner
    optimizer would step the stale tensors and the wrapper would silently
    diverge from the inner state.
    """
    layer = _make_bf16_linear()
    opt = MasterWeightOptimizer(_adamw_factory(), list(layer.parameters()))
    masters_before = list(opt._fp32_params)
    inner_param_ids_before = {id(p) for group in opt.inner.param_groups for p in group["params"]}

    layer.cuda()
    opt.rebuild_masters_from_live(layer.parameters())

    for before, after in zip(masters_before, opt._fp32_params, strict=True):
        assert before is after
    inner_param_ids_after = {id(p) for group in opt.inner.param_groups for p in group["params"]}
    assert inner_param_ids_before == inner_param_ids_after
