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

"""Test-Time Training (TTT) layer with an MLP fast model.

This is the sequence-modeling primitive behind RoboTTT (`arXiv:2607.15275
<https://arxiv.org/abs/2607.15275>`_): a recurrent layer whose recurrent state
is a set of *fast weights* — the parameters of a small two-layer MLP that is
updated by gradient descent on a self-supervised reconstruction loss at every
step, during training *and* inference.

The math follows the reference implementation in
`test-time-training/ttt-video-dit <https://github.com/test-time-training/ttt-video-dit>`_
(``ttt/models/ssm/ops/ttt_mlp.py``), which is itself the "dual form" of the
update in Sun et al., *Learning to (Learn at Test Time)*. Two things differ
here, both required by the robot-policy setting:

1. **Fast weights are carried in and out.** The reference always starts a
   forward from the learned initialization :math:`W_0`. Truncated
   backpropagation through time (TBPTT) needs the *final* fast weights of
   segment *n* to become the *initial* fast weights of segment *n+1*, detached
   at the boundary, and closed-loop inference needs them to persist across
   policy calls. So :meth:`TTTMLPLayer.forward` accepts an optional incoming
   :class:`TTTFastWeights` and returns the outgoing one.
2. **Positional embeddings are 1-D over robot timesteps**, not 3-D over video
   latents, and they take an explicit offset so a TBPTT segment that resumes
   mid-trajectory does not restart its RoPE phase at zero.

Terminology used throughout, matching the paper:

* *slow weights* — ordinary ``nn.Parameter``s (Q/K/V/O projections, the fast
  model's initialization :math:`W_0`, the inner learning-rate gate). Updated by
  the outer optimizer.
* *fast weights* — ``W1/b1/W2/b2``, the two-layer MLP that *is* the recurrent
  state. Updated by the inner gradient step, once per mini-batch.
* *mini-batch* — the group of tokens processed by one inner gradient step. For
  a robot policy this is exactly one timestep's worth of tokens, so one policy
  call performs one fast-weight update (the paper's "one mini batch per
  inference").

The dual form computes a whole mini-batch's outputs *and* the end-of-mini-batch
fast weights with matrix multiplies rather than a token-by-token loop, which is
what makes the layer trainable at sequence lengths in the thousands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from einops import einsum, rearrange, repeat
from torch import Tensor, nn

# The inner reconstruction loss normalizes over the head dimension. This
# epsilon matches the reference implementation's `ln_fwd` / `ln_fused_l2_bwd`
# (1e-8) and is deliberately *not* the policy-level normalization epsilon.
TTT_LN_EPS = 1e-8


@dataclass
class TTTFastWeights:
    """The recurrent state of a :class:`TTTMLPLayer`: one fast MLP per head.

    Attributes:
        w1: First-layer weights, shape ``(B, H, head_dim, hidden_dim)``.
        b1: First-layer bias, shape ``(B, H, 1, hidden_dim)``.
        w2: Second-layer weights, shape ``(B, H, hidden_dim, head_dim)``.
        b2: Second-layer bias, shape ``(B, H, 1, head_dim)``.
    """

    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor

    def detach(self) -> TTTFastWeights:
        """Returns a copy holding the same values with the autograd graph cut.

        This is the TBPTT boundary operation: the *values* cross into the next
        segment so TTT keeps running over the whole trajectory, while the graph
        does not, so activation memory is bounded by the segment length rather
        than the trajectory length.

        Returns:
            A new :class:`TTTFastWeights` whose tensors are detached leaves.
        """
        return TTTFastWeights(
            w1=self.w1.detach(),
            b1=self.b1.detach(),
            w2=self.w2.detach(),
            b2=self.b2.detach(),
        )

    def to(self, *args: Any, **kwargs: Any) -> TTTFastWeights:
        """Applies ``Tensor.to`` to every field.

        Args:
            *args: Positional arguments forwarded to ``Tensor.to``.
            **kwargs: Keyword arguments forwarded to ``Tensor.to``.

        Returns:
            A new :class:`TTTFastWeights` with converted tensors.
        """
        return TTTFastWeights(
            w1=self.w1.to(*args, **kwargs),
            b1=self.b1.to(*args, **kwargs),
            w2=self.w2.to(*args, **kwargs),
            b2=self.b2.to(*args, **kwargs),
        )


def _layer_norm_forward(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = TTT_LN_EPS) -> Tensor:
    """Per-head layer norm over the last dimension.

    Args:
        x: Input, shape ``(..., head_dim)``.
        gamma: Scale, broadcastable to ``x``.
        beta: Shift, broadcastable to ``x``.
        eps: Variance epsilon.

    Returns:
        The normalized, scaled and shifted tensor, same shape as ``x``.
    """
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta


def _layer_norm_fused_l2_backward(
    x: Tensor, l2_target: Tensor, gamma: Tensor, beta: Tensor, eps: float = TTT_LN_EPS
) -> Tensor:
    """Gradient of ``0.5 * ||LN(x) - l2_target||^2`` with respect to ``x``.

    This is the inner loss's backward pass, written in closed form rather than
    obtained from autograd. Doing it by hand is what keeps the inner gradient
    step cheap enough to run once per token group: autograd would have to
    retain a graph per mini-batch across the whole sequence.

    Args:
        x: Pre-norm activations, shape ``(..., head_dim)``.
        l2_target: Reconstruction target, same shape as ``x``.
        gamma: Layer-norm scale, broadcastable to ``x``.
        beta: Layer-norm shift, broadcastable to ``x``.
        eps: Variance epsilon.

    Returns:
        The gradient with respect to ``x``, same shape as ``x``.
    """
    dim = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    grad_output = (gamma * x_hat + beta) - l2_target
    grad_x_hat = grad_output * gamma
    return (
        (1.0 / dim)
        * (
            dim * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )


def _gelu_backward(x: Tensor) -> Tensor:
    """Derivative of the tanh-approximated GeLU used by the fast model.

    Args:
        x: Pre-activation input.

    Returns:
        ``d/dx gelu_tanh(x)``, same shape as ``x``.
    """
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


def _rotary_1d(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Applies 1-D rotary position embeddings over the sequence axis.

    Adjacent channel pairs ``(0, 1), (2, 3), ...`` are treated as one complex
    number and rotated, matching the interleaved convention the reference
    implementation gets from ``torch.view_as_complex`` on a ``(..., d/2, 2)``
    view. Computed with real arithmetic instead, because ``view_as_complex``
    requires a float32/float64 input and a contiguous final stride — both
    awkward inside a bf16 model.

    Args:
        x: Input, shape ``(B, L, H, head_dim)``. ``head_dim`` must be even.
        positions: Integer positions, shape ``(B, L)`` or ``(L,)``.
        theta: RoPE base. RoboTTT uses 10000.

    Returns:
        The rotated tensor, same shape and dtype as ``x``.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

    if positions.ndim == 1:
        positions = repeat(positions, "l -> b l", b=x.shape[0])

    # float32 for the trig regardless of model dtype: the angle grows linearly
    # with position, and at 8K timesteps a bf16 angle has ~1e-2 absolute error.
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = rearrange(positions.to(torch.float32), "b l -> b l 1") * inv_freq
    cos = rearrange(torch.cos(angles), "b l d -> b l 1 d")
    sin = rearrange(torch.sin(angles), "b l d -> b l 1 d")

    pairs = rearrange(x.to(torch.float32), "b l h (d two) -> b l h d two", two=2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rearrange(rotated, "b l h d two -> b l h (d two)", two=2).to(x.dtype)


def _ttt_mini_batch_step(
    fast_weights: TTTFastWeights,
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    eta: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
) -> tuple[TTTFastWeights, Tensor]:
    """Runs one inner gradient step and produces the mini-batch's outputs.

    This is the dual form: instead of updating the fast weights token by token
    and applying them one at a time, it computes every token's output against
    the *running* fast weights with two attention-like matmuls, then computes
    the end-of-mini-batch fast weights once. The two are algebraically
    identical for a single gradient-descent step per token, which is why the
    layer can be trained at long sequence lengths.

    Shapes use ``B`` batch, ``H`` heads, ``C`` mini-batch size (tokens per
    inner step), ``D`` head dim, ``F`` fast-MLP hidden dim.

    Args:
        fast_weights: Incoming fast weights (the recurrent state).
        xq: Query projections, shape ``(B, H, C, D)``.
        xk: Key projections, shape ``(B, H, C, D)``.
        xv: Value projections (already turned into a reconstruction target),
            shape ``(B, H, C, D)``.
        eta: Per-token inner learning rates, shape ``(B, H, C, C)``.
        ln_weight: Inner layer-norm scale, shape ``(H, 1, D)``.
        ln_bias: Inner layer-norm shift, shape ``(H, 1, D)``.

    Returns:
        A tuple of the outgoing fast weights and the mini-batch output of shape
        ``(B, H, C, D)``.
    """
    w1, b1, w2, b2 = fast_weights.w1, fast_weights.b1, fast_weights.w2, fast_weights.b2

    # Forward pass of the fast model on the keys.
    z1 = xk @ w1 + b1
    x2 = F.gelu(z1, approximate="tanh")
    z2 = x2 @ w2 + b2

    # The self-supervised inner loss reconstructs (V - K) from K, so the fast
    # model learns the *residual* association rather than the identity.
    reconstruction_target = xv - xk
    grad_z2 = _layer_norm_fused_l2_backward(z2, reconstruction_target, ln_weight, ln_bias)
    grad_z1 = grad_z2 @ w2.transpose(-2, -1) * _gelu_backward(z1)

    # Apply the *running* (per-token) updated weights to the queries. The
    # `eta * Attn` terms are what make this equivalent to having taken one
    # gradient step per preceding token inside the mini-batch.
    attn1 = xq @ xk.transpose(-2, -1)
    b1_bar = b1 - eta @ grad_z1
    z1_bar = xq @ w1 - (eta * attn1) @ grad_z1 + b1_bar
    x2_bar = F.gelu(z1_bar, approximate="tanh")

    attn2 = x2_bar @ x2.transpose(-2, -1)
    b2_bar = b2 - eta @ grad_z2
    z2_bar = x2_bar @ w2 - (eta * attn2) @ grad_z2 + b2_bar

    # End-of-mini-batch fast weights: the cumulative effect of every token's
    # gradient step, read off the last row of eta.
    last_eta = eta[..., -1, :, None]
    next_fast_weights = TTTFastWeights(
        w1=w1 - (last_eta * xk).transpose(-1, -2) @ grad_z1,
        b1=b1 - torch.sum(last_eta * grad_z1, dim=-2, keepdim=True),
        w2=w2 - (last_eta * x2).transpose(-1, -2) @ grad_z2,
        b2=b2 - torch.sum(last_eta * grad_z2, dim=-2, keepdim=True),
    )

    # Residual around the fast model, so a freshly initialized layer is close
    # to a pass-through of the queries.
    output = xq + _layer_norm_forward(z2_bar, ln_weight, ln_bias)
    return next_fast_weights, output


def _scan_mini_batches(
    fast_weights: TTTFastWeights,
    xq: Tensor,
    xk: Tensor,
    xv: Tensor,
    eta: Tensor,
    ln_weight: Tensor,
    ln_bias: Tensor,
    checkpoint_group_size: int = 0,
) -> tuple[TTTFastWeights, Tensor]:
    """Scans :func:`_ttt_mini_batch_step` over the mini-batch axis.

    Args:
        fast_weights: Incoming fast weights.
        xq: Queries, shape ``(NC, B, H, C, D)`` with ``NC`` mini-batches.
        xk: Keys, same shape as ``xq``.
        xv: Values, same shape as ``xq``.
        eta: Inner learning rates, shape ``(NC, B, H, C, C)``.
        ln_weight: Inner layer-norm scale, shape ``(H, 1, D)``.
        ln_bias: Inner layer-norm shift, shape ``(H, 1, D)``.
        checkpoint_group_size: When > 0, gradient-checkpoint the scan in groups
            of this many mini-batches, trading recompute for activation memory.
            0 disables checkpointing.

    Returns:
        A tuple of the final fast weights and the stacked outputs of shape
        ``(NC, B, H, C, D)``.
    """
    num_mini_batch = xq.shape[0]

    def run_group(carry: TTTFastWeights, start: int, end: int) -> tuple[TTTFastWeights, Tensor]:
        outputs = []
        for i in range(start, end):
            carry, out = _ttt_mini_batch_step(carry, xq[i], xk[i], xv[i], eta[i], ln_weight, ln_bias)
            outputs.append(out)
        return carry, torch.stack(outputs)

    if checkpoint_group_size <= 0:
        return run_group(fast_weights, 0, num_mini_batch)

    # torch.utils.checkpoint cannot carry a dataclass through its
    # pack/unpack, so flatten to a tuple of tensors at the boundary.
    def run_group_flat(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, start: int, end: int):
        carry, out = run_group(TTTFastWeights(w1=w1, b1=b1, w2=w2, b2=b2), start, end)
        return carry.w1, carry.b1, carry.w2, carry.b2, out

    carry = fast_weights
    chunks = []
    for group_start in range(0, num_mini_batch, checkpoint_group_size):
        group_end = min(group_start + checkpoint_group_size, num_mini_batch)
        w1, b1, w2, b2, out = torch.utils.checkpoint.checkpoint(
            run_group_flat,
            carry.w1,
            carry.b1,
            carry.w2,
            carry.b2,
            group_start,
            group_end,
            use_reentrant=False,
        )
        carry = TTTFastWeights(w1=w1, b1=b1, w2=w2, b2=b2)
        chunks.append(out)
    return carry, torch.cat(chunks, dim=0)


class TTTMLPLayer(nn.Module):
    """A TTT layer whose fast model is a two-layer MLP.

    One instance is attached to each action-expert decoder layer. It is the
    only component in the policy that carries information *across* timesteps —
    attention stays strictly within a timestep — which is the division of
    labour RoboTTT's architecture is built on.

    The layer is a no-op contributor at initialization only in combination with
    :class:`TanhGate`; on its own its output is nonzero (its fast model has a
    random :math:`W_0`).

    Args:
        width: Model width, i.e. the action expert's hidden size.
        num_heads: Number of TTT heads. ``width`` must divide evenly.
        mlp_hidden_multiplier: Fast-MLP hidden dim as a multiple of
            ``head_dim``. The reference video model uses 4; RoboTTT's own
            budget (~10M per layer on a 1536-wide backbone) works out to
            roughly 2, which is also what keeps this layer from doubling the
            size of a 1024-wide action expert.
        base_lr: Base inner learning rate, scaled per token by a learned gate.
            RoboTTT uses 0.1.
        rope_theta: RoPE base for the 1-D positional embedding. RoboTTT uses
            10000.
        scan_checkpoint_group_size: Gradient-checkpoint group size for the
            mini-batch scan; 0 disables.

    Raises:
        ValueError: If ``width`` is not divisible by ``num_heads``, or the
            resulting head dim is odd (RoPE needs pairs).
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_hidden_multiplier: int = 2,
        base_lr: float = 0.1,
        rope_theta: float = 10000.0,
        scan_checkpoint_group_size: int = 0,
    ):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")
        head_dim = width // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for rotary embeddings")

        self.width = width
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = head_dim * mlp_hidden_multiplier
        self.base_lr = base_lr
        self.rope_theta = rope_theta
        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        self.wq = nn.Linear(width, width, bias=True)
        self.wk = nn.Linear(width, width, bias=True)
        self.wv = nn.Linear(width, width, bias=True)
        self.wo = nn.Linear(width, width, bias=True)

        # Learned per-head, per-token inner learning-rate gate. Kept as raw
        # parameters rather than an nn.Linear because it is applied per head
        # with a single einsum over the pre-projection hidden states.
        self.ttt_lr_weight = nn.Parameter(torch.normal(0.0, 0.02, size=(num_heads, 1, width)))
        self.ttt_lr_bias = nn.Parameter(torch.zeros(num_heads, 1))

        # Inner-loss layer norm, one per head.
        self.ttt_norm_weight = nn.Parameter(torch.ones(num_heads, head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(num_heads, head_dim))

        # The fast model's learned initialization W_0. These are slow weights:
        # meta-learned through the inner update by the outer task loss, which
        # is what tailors the fast-weight dynamics to robot trajectories. They
        # only receive gradient through the *first* TBPTT segment, whose update
        # originates directly from them.
        self.w1_init = nn.Parameter(torch.normal(0.0, 0.02, size=(num_heads, head_dim, self.hidden_dim)))
        self.b1_init = nn.Parameter(torch.zeros(num_heads, 1, self.hidden_dim))
        self.w2_init = nn.Parameter(torch.normal(0.0, 0.02, size=(num_heads, self.hidden_dim, head_dim)))
        self.b2_init = nn.Parameter(torch.zeros(num_heads, 1, head_dim))

        self.post_norm = nn.LayerNorm(width, eps=1e-6)

    def initial_fast_weights(self, batch_size: int) -> TTTFastWeights:
        """Materializes :math:`W_0` for a batch, starting a fresh rollout.

        Args:
            batch_size: Number of independent trajectories.

        Returns:
            Per-sample copies of the learned initialization. These are *not*
            detached: the outer loss must reach ``w1_init`` and friends through
            the first segment, otherwise the initialization never trains and
            stays a random draw.
        """
        return TTTFastWeights(
            w1=repeat(self.w1_init, "h d f -> b h d f", b=batch_size),
            b1=repeat(self.b1_init, "h one f -> b h one f", b=batch_size),
            w2=repeat(self.w2_init, "h f d -> b h f d", b=batch_size),
            b2=repeat(self.b2_init, "h one d -> b h one d", b=batch_size),
        )

    def _inner_learning_rates(self, hidden_states: Tensor, mini_batch_size: int) -> Tensor:
        """Computes the per-token inner learning rate matrix ``eta``.

        Args:
            hidden_states: Layer input, shape ``(B, L, width)``.
            mini_batch_size: Tokens per inner gradient step.

        Returns:
            ``eta`` of shape ``(B, H, NC, C, C)``, already divided by the
            mini-batch size so a mini-batch's tokens share one gradient step
            (they are one observation, so no causal ordering within it).
        """
        grouped = rearrange(hidden_states, "b (nc c) w -> b nc c w", c=mini_batch_size)
        gate = einsum(grouped, self.ttt_lr_weight.to(grouped.dtype), "b nc c w, h one w -> b h nc c one")
        gate = gate + rearrange(self.ttt_lr_bias, "h one -> 1 h 1 1 one")
        lr = self.base_lr * torch.sigmoid(gate) / self.head_dim
        # (B, H, NC, C, 1) -> (B, H, NC, C, C): every token in the mini-batch
        # sees the same learning-rate column, matching the non-causal
        # mini-batch treatment in the reference implementation.
        lr = rearrange(lr, "b h nc c one -> b h nc one c")
        return repeat(lr, "b h nc one c -> b h nc (one r) c", r=mini_batch_size) / mini_batch_size

    def forward(
        self,
        hidden_states: Tensor,
        mini_batch_size: int,
        fast_weights: TTTFastWeights | None = None,
        position_offset: int = 0,
    ) -> tuple[Tensor, TTTFastWeights]:
        """Runs the layer over a sequence, updating and returning fast weights.

        Args:
            hidden_states: Input, shape ``(B, L, width)``, where ``L`` is
                ``num_timesteps * mini_batch_size``. Position ``0`` of the
                sequence axis is the oldest timestep.
            mini_batch_size: Tokens per timestep, i.e. per inner gradient step.
                ``L`` must be an exact multiple.
            fast_weights: Recurrent state to resume from. ``None`` starts from
                the learned :math:`W_0` — correct for the first TBPTT segment
                and for the first call of a rollout, wrong (a memory reset) in
                the middle of either.
            position_offset: Absolute index of the first token in the sequence,
                used so a resumed segment continues its RoPE phase instead of
                restarting at zero.

        Returns:
            A tuple of the layer output, shape ``(B, L, width)`` and dtype
            matching the input, and the outgoing fast weights.

        Raises:
            ValueError: If ``L`` is not a multiple of ``mini_batch_size``.
        """
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len % mini_batch_size != 0:
            raise ValueError(
                f"sequence length {seq_len} must be a multiple of mini_batch_size {mini_batch_size}"
            )

        input_dtype = hidden_states.dtype
        # Dtype policy, in two stages, because the linear projections and the
        # inner loop want different things:
        #
        #  * The projections run in the module's own parameter dtype. This model
        #    is served in bfloat16, so `wq`/`wk`/`wv` hold bf16 weights and
        #    handing them float32 activations is a hard dtype error, not a slow
        #    path.
        #  * The inner loop then runs in float32. It takes an explicit gradient
        #    step and divides by a standard deviation, thousands of times over a
        #    long trajectory; in bf16 that drifts enough to change the
        #    recurrence. The reference implementation gets away with low
        #    precision only by fusing the whole update into a CUDA kernel.
        #
        # "Promote", not "pin to float32": a float64 module — which the unit
        # tests use, to get tolerances tight enough to catch a real algebra
        # error rather than hide it under float32 noise — must not be silently
        # downcast.
        param_dtype = self.wq.weight.dtype
        compute_dtype = torch.float32 if param_dtype in (torch.float16, torch.bfloat16) else param_dtype

        projected = hidden_states.to(param_dtype)
        xq = rearrange(self.wq(projected), "b l (h d) -> b l h d", h=self.num_heads).to(compute_dtype)
        xk = rearrange(self.wk(projected), "b l (h d) -> b l h d", h=self.num_heads).to(compute_dtype)
        xv = rearrange(self.wv(projected), "b l (h d) -> b l h d", h=self.num_heads).to(compute_dtype)

        # L2-normalizing Q and K bounds the inner loss's curvature, so one
        # gradient step with a shared base learning rate behaves consistently
        # across layers and sequence positions.
        xq = F.normalize(xq, p=2, dim=-1)
        xk = F.normalize(xk, p=2, dim=-1)

        positions = torch.arange(position_offset, position_offset + seq_len, device=hidden_states.device)
        xq = _rotary_1d(xq, positions, self.rope_theta)
        xk = _rotary_1d(xk, positions, self.rope_theta)

        # Turn the values into the reconstruction target the inner loss uses.
        #
        # Deliberate deviation from the reference: ttt-video-dit normalizes this
        # target with an *unbiased* std and divides by ``std + eps``
        # (`ln_reconstruction_target`), while normalizing the fast model's own
        # output with a biased variance and ``sqrt(var + eps)`` (`ln_fwd`). The
        # two are inconsistent with each other upstream. We use the `ln_fwd`
        # convention for both, so the target lives on the same scale as the
        # prediction the inner loss compares it against. The difference is
        # O(1/head_dim) and vanishes as head_dim grows, but a reviewer diffing
        # against the reference will see it, so it is called out here rather
        # than looking like a porting slip.
        ln_weight = rearrange(self.ttt_norm_weight, "h d -> h 1 d").to(compute_dtype)
        ln_bias = rearrange(self.ttt_norm_bias, "h d -> h 1 d").to(compute_dtype)
        xv = xk + _layer_norm_forward(xv - xk, ln_weight.squeeze(1), ln_bias.squeeze(1))

        eta = self._inner_learning_rates(hidden_states.to(compute_dtype), mini_batch_size)

        to_mini_batches = "b (nc c) h d -> nc b h c d"
        xq = rearrange(xq, to_mini_batches, c=mini_batch_size)
        xk = rearrange(xk, to_mini_batches, c=mini_batch_size)
        xv = rearrange(xv, to_mini_batches, c=mini_batch_size)
        eta = rearrange(eta, "b h nc c1 c2 -> nc b h c1 c2")

        if fast_weights is None:
            fast_weights = self.initial_fast_weights(batch_size)
        fast_weights = fast_weights.to(compute_dtype)

        fast_weights, outputs = _scan_mini_batches(
            fast_weights,
            xq,
            xk,
            xv,
            eta,
            ln_weight,
            ln_bias,
            checkpoint_group_size=self.scan_checkpoint_group_size,
        )

        # Back to the parameter dtype for the output projections, then to the
        # caller's dtype so this layer never changes the dtype of the residual
        # stream it feeds.
        out = rearrange(outputs, "nc b h c d -> b (nc c) (h d)").to(param_dtype)
        out = self.post_norm(out)
        out = self.wo(out)
        return out.to(input_dtype), fast_weights


class TanhGate(nn.Module):
    """Blends a TTT output into an attention output through ``tanh(alpha)``.

    Implements ``O = tanh(alpha) * O_ttt + O_attn`` with a learned per-channel
    ``alpha`` initialized near zero. This is what makes it safe to bolt TTT
    onto an already-pretrained VLA: at step 0 the gate is ~0.001, so the model
    reproduces the base policy's behavior, and training decides how far to open
    it. Without it, a randomly initialized memory layer injects noise into
    every layer of a pretrained action expert from the first step.

    Args:
        width: Number of channels; ``alpha`` is learned per channel.
        init_value: Initial ``alpha``. RoboTTT uses 0.001.
    """

    def __init__(self, width: int, init_value: float = 0.001):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((width,), init_value))

    def forward(self, attention_output: Tensor, ttt_output: Tensor) -> Tensor:
        """Adds the gated TTT contribution to the attention output.

        Args:
            attention_output: The attention branch, shape ``(..., width)``.
            ttt_output: The TTT branch, same shape.

        Returns:
            The blended output, same shape and dtype as ``attention_output``.
        """
        gate = torch.tanh(self.alpha).to(ttt_output.dtype)
        return attention_output + gate * ttt_output


@dataclass
class TTTSequenceState:
    """Per-forward runtime state threaded into the action expert's TTT layers.

    :class:`~opentau.policies.pi05.paligemma_with_expert.PaliGemmaWithExpertModel`
    folds the sequence axis into the batch axis so attention runs per timestep
    (which is the whole point of the architecture: attention is spatial, TTT is
    temporal). The TTT layers need the axis back, plus the carried recurrent
    state, so this object travels alongside the hidden states.

    ``incoming`` and ``outgoing`` are keyed by decoder-layer index because each
    layer owns an independent fast model. A missing ``incoming`` key means
    "start this layer from its learned :math:`W_0`" — correct for the first
    TBPTT segment and the first call of a rollout, and a silent memory reset
    anywhere else.

    Attributes:
        num_timesteps: Number of policy calls folded into the batch axis. The
            expert's hidden states arrive as ``(B * num_timesteps, S, W)`` and
            are unfolded to ``(B, num_timesteps * S, W)`` for TTT.
        position_offset: Absolute token index of the first token in this
            segment, so a resumed segment continues its RoPE phase.
        incoming: Carried fast weights per layer index.
        outgoing: Filled by the forward with each layer's end-of-segment fast
            weights, for the caller to detach and pass to the next segment.
    """

    num_timesteps: int
    position_offset: int = 0
    incoming: dict[int, TTTFastWeights] = field(default_factory=dict)
    outgoing: dict[int, TTTFastWeights] = field(default_factory=dict)
