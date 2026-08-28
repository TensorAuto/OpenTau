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

"""π₀.₅ with Test-Time-Training memory in the action expert.

A port of RoboTTT (`arXiv:2607.15275 <https://arxiv.org/abs/2607.15275>`_) onto
π₀.₅. The paper instantiates its recipe on GR00T N1.7 and frames it as
backbone-agnostic; π₀.₅ is a structural match (VLM + flow-matching action
expert), so the port is mostly a matter of deciding which token stream carries
the memory.

What this adds to π₀.₅, and nothing else:

* A :class:`~opentau.policies.ttt_layer.TTTMLPLayer` after the attention block
  of each of the action expert's 18 layers, blended in through a
  :class:`~opentau.policies.ttt_layer.TanhGate` initialized at ~0.001. Attention
  keeps operating strictly within one timestep; the TTT layer is the only path
  that crosses timesteps.

  **What the gate does and does not buy.** At ``alpha = 0`` the TTT branch
  contributes exactly nothing — verified bit-identical against the same forward
  with no ``ttt_state``. It does *not* follow that the whole policy reproduces
  stock π₀.₅ at step 0, and an earlier version of this file claimed that it
  did. The register block is a second change to the expert's input and no gate
  covers it: the tokens occupy attention slots, so they take softmax mass from
  the action tokens. Two things narrow the gap as far as it can go — the
  register table is zero-initialized, and the position ids are built so the
  action block keeps the RoPE phase it has without registers (see
  :meth:`PI05TTTFlowMatching._expert_position_ids`). The honest statement is:
  the *memory* is inert at init, the register block is a small ungated
  perturbation, and at ``n_register_tokens=0`` the policy is stock π₀.₅.
* ``n_register_tokens`` learned register tokens prepended to the expert's token
  stream each timestep. π₀.₅ carries robot state on the *language* side, inside
  the frozen VLM prefix, and the VL tokens deliberately bypass TTT for cost
  reasons — so without registers the memory would never see vision or state.
  The registers are the courier.
* Sequence training: per-timestep flow-matching noise levels ("sequence action
  forcing") and truncated backpropagation through time with fast weights carried
  across segment boundaries and their gradients cut there.
* Loss masking, so a timestep can act as pure context — updating the fast
  weights without contributing an imitation target. This is the hook the paper's
  one-shot-video-imitation and DAgger-Distillation capabilities are built on.

Known gaps, deliberately out of scope for this change and tracked in the PR:

* **The dataloader does not yet emit multi-timestep trajectory sequences.**
  :meth:`PI05TTTPolicy.forward` accepts them and unit tests construct them by
  hand, but training on real long trajectories needs a dataset-side change. At
  ``sequence_length=1`` — the only value today's loader can supply — TTT still
  runs and every TTT parameter receives gradients, so the plumbing is exercised
  and distributed training is well-formed; the fast weights simply take one
  update per sequence and cannot learn anything spanning timesteps.
* **Truncated gradients, untruncated activation memory.** Gradients are
  truncated exactly as the paper specifies, so the optimization is correct. The
  *activation memory* benefit additionally needs one backward per segment, which
  the shared training loop does not do. An earlier revision exposed a
  ``tbptt_backward_fn`` hook for that; it was removed because it could not be
  used as documented — the same graph-carrying tensors were also returned for
  the caller's own backward, so the natural wiring raised "Trying to backward
  through the graph a second time", and the callback bypassed
  ``cfg.loss_weighting``. It belongs with the segmented loop that will use it,
  not ahead of it.
* **The VLM prefix is recomputed for every timestep in a sequence.** Since the
  VLM is frozen during the paper's pretraining stage, its outputs can be
  precomputed and cached offline, which removes most of the cost. That is a
  data-pipeline change, not a modeling one.
* **The response (subtask) cross-entropy is not computed on the sequence path**,
  so sequence training raises when ``predict_response=True`` rather than
  silently dropping the term. Routing π₀.₅'s predicted subtask into the memory
  stream — so the fast weights carry "did step 1, did step 2" for free — is the
  natural follow-up, and is the one place this backbone is *easier* to work with
  than GR00T, which has no equivalent stage.
* DAgger Distillation is a data-collection and mixture procedure; the loss
  masking it needs is here, the procedure is not.
"""

from functools import partial
from typing import Any

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn

from opentau.policies.accel import AccelMeter
from opentau.policies.pi05.modeling_pi05 import (
    PI05FlowMatching,
    PI05Policy,
    make_att_2d_masks,
)
from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from opentau.policies.ttt_layer import (
    TanhGate,
    TTTFastWeights,
    TTTMLPLayer,
    TTTSequenceState,
)
from opentau.policies.utils import PerSampleLoss, ce_per_sample, flow_matching_masked_mse


def _accumulate_per_sample(
    running: PerSampleLoss | None,
    segment: PerSampleLoss,
    batch_size: int,
    segment_length: int,
) -> PerSampleLoss:
    """Folds a segment's per-row loss decomposition into a per-trajectory running total.

    A segment's rows are ``(B, segment_length)`` flattened, so summing over the
    timestep axis turns per-row numerators and denominators into per-trajectory
    ones. Summing rather than averaging is the whole point of carrying
    ``(sum, count)``: the masked mean for any grouping is ``Σsum / Σcount``, so
    both "combine a trajectory's timesteps" and "combine a sequence's segments"
    are the same addition, and an all-context timestep contributes ``(0, 0)``
    without skewing the result.

    Args:
        running: Accumulated total so far, or None on the first segment.
        segment: This segment's per-row decomposition, ``(B * segment_length,)``.
        batch_size: Number of trajectories.
        segment_length: Timesteps in this segment.

    Returns:
        The updated per-trajectory total, ``(batch_size,)``.
    """
    folded = PerSampleLoss(
        sum=reduce(segment.sum, "(b t) -> b", "sum", b=batch_size, t=segment_length),
        count=reduce(segment.count, "(b t) -> b", "sum", b=batch_size, t=segment_length),
    )
    return folded if running is None else running + folded


class PI05TTTFlowMatching(PI05FlowMatching):
    """π₀.₅ flow matching with TTT memory in the action expert.

    Args:
        config: Policy configuration.
        discrete_action_vocab_size: Size of the discrete action vocabulary.
        language_tokenizer: Optional pre-loaded PaliGemma tokenizer shared with
            the enclosing policy.
    """

    def __init__(
        self,
        config: PI05TTTConfig,
        discrete_action_vocab_size: int | None = None,
        language_tokenizer: Any = None,
    ):
        super().__init__(
            config,
            discrete_action_vocab_size=discrete_action_vocab_size,
            language_tokenizer=language_tokenizer,
        )
        self.config: PI05TTTConfig = config

        # Learned register tokens, shared across timesteps. RoPE distinguishes
        # one timestep's registers from another's, so they need no per-timestep
        # parameters of their own.
        if config.n_register_tokens > 0:
            # Zero-init so a warm-start is perturbed as little as possible: a
            # random table would inject `N(0, 0.02)` vectors into every action
            # token's attention on step 0, which no gate covers.
            #
            # The `use_modality_embedding` precedent in `PI05FlowMatching` is
            # only a partial analogy, and worth not overstating: that table is
            # *added* to a non-zero token embedding, so zero really is a no-op
            # there. A register token *is* the whole embedding, so zero means an
            # exact zero vector entering RMSNorm and occupying an attention
            # slot. Small, and the smallest available, but not nothing.
            #
            # This does not make the register block a *complete* no-op: the
            # tokens still occupy attention slots, so they take softmax mass
            # away from the action tokens even when their values are zero. See
            # the class docstring for the exact claim.
            self.register_tokens = nn.Parameter(torch.zeros(config.n_register_tokens, config.proj_width))
        else:
            self.register_tokens = None

        self._attach_ttt_layers()

        # `super().__init__()` above already ran both default-deny freeze
        # sweeps to completion — `set_requires_grad()` inside
        # `PaliGemmaWithExpertModel.__init__` and
        # `freeze_policy_level_params_for_state_action_representation_only(...)`
        # as the last statement of `PI05FlowMatching.__init__`. Everything
        # created *after* that (the register table and all 343 TTT tensors) is
        # born with `requires_grad=True` and would never be swept, which routes
        # around the very guarantee the comment on that sweep claims
        # ("so a module added later cannot silently keep training"). So re-apply
        # the two exclusive-training flags to the parameters this class adds.
        if config.train_state_action_representation_only or config.train_vision_encoder_only:
            for param in self.ttt_parameters():
                param.requires_grad_(False)

        # Fast weights carried across inference calls, and the token position
        # the next call's RoPE should start from. Both are rollout state, reset
        # by ``PI05TTTPolicy.reset``.
        self._carried_fast_weights: dict[int, TTTFastWeights] = {}
        # First-Euler-step fast-weight capture for
        # `config.ttt_inference_update_adoption == "first"`; None outside a call.
        self._first_step_adoption: dict[int, TTTFastWeights] | None = None
        self._inference_token_position: int = 0
        # Set for the duration of a ``sample_actions`` call so the overridden
        # ``denoise_step`` can reach it without changing the parent's signature.
        self._active_ttt_state: TTTSequenceState | None = None

    def _attach_ttt_layers(self) -> None:
        """Attaches a TTT layer and a tanh gate to every action-expert layer.

        The submodules are attached to the decoder layers themselves rather
        than held in a separate ``nn.ModuleList`` on this module. That is
        deliberate: CLAUDE.md rule 5 requires a composite forward unit to be a
        single ``nn.Module`` so FSDP's all-gather hook prefetches every
        sub-component together. A parallel list would be wrapped separately and
        produce mismatched all-gather sizes across ranks.

        The resulting state-dict keys (``...layers.N.ttt.*``, ``...layers.N.ttt_gate.*``)
        are new, so an existing π₀.₅ checkpoint loads cleanly under the
        ``strict=False`` path every ``from_pretrained`` override already uses.
        """
        expert_layers = self.paligemma_with_expert.gemma_expert.model.layers
        width = self.paligemma_with_expert.config.gemma_expert_config.hidden_size
        for layer in expert_layers:
            layer.ttt = TTTMLPLayer(
                width=width,
                num_heads=self.config.ttt_num_heads,
                mlp_hidden_multiplier=self.config.ttt_mlp_hidden_multiplier,
                base_lr=self.config.ttt_base_lr,
                rope_theta=self.config.ttt_rope_theta,
                scan_checkpoint_group_size=self.config.ttt_scan_checkpoint_group_size,
                # Pin the inner-step size to exactly one timestep's tokens. The
                # hook in `_run_layer` derives it from `out_emb.shape[1]`, so
                # without this the coupling would be incidental — and a
                # mini-batch spanning two timesteps silently leaks the later
                # one into the earlier one's output.
                expected_mini_batch_size=self.config.n_expert_tokens_per_timestep,
            )
            layer.ttt_gate = TanhGate(width, init_value=self.config.ttt_gate_init)
            layer.ttt_gate.inference_alpha_scale = self.config.ttt_inference_alpha_scale

    def ttt_parameters(self) -> list[nn.Parameter]:
        """Returns every parameter this policy adds on top of stock π₀.₅.

        Returns:
            The TTT layers' parameters, the gates' ``alpha``, and the register
            tokens.
        """
        params: list[nn.Parameter] = []
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            params.extend(layer.ttt.parameters())
            params.extend(layer.ttt_gate.parameters())
        if self.register_tokens is not None:
            params.append(self.register_tokens)
        return params

    def freeze_pretrained_parameters(self) -> None:
        """Freezes everything except the newly added TTT parameters.

        This is the paper's pretraining stage, which "tunes only the newly added
        sequence-modeling layers and freezes the other components". Its
        post-training stage fine-tunes all parameters, so this must not be
        applied for a task-specific finetune (``config.train_ttt_only=False``).
        """
        new_param_ids = {id(p) for p in self.ttt_parameters()}
        for param in self.parameters():
            if id(param) not in new_param_ids:
                param.requires_grad_(False)

    def reset_memory(self) -> None:
        """Drops the carried fast weights so the next call starts from ``W_0``.

        Must be called at every environment reset. Skipping it leaks one
        episode's memory into the next, which is exactly the failure the
        architecture is supposed to prevent.
        """
        self._carried_fast_weights = {}
        self._inference_token_position = 0

    def sample_time_sequence(self, batch_size: int, num_timesteps: int, device: torch.device | str) -> Tensor:
        """Samples one independent flow-matching noise level per timestep.

        This is *sequence action forcing*. The paper finds it necessary for
        stable sequence training: sharing one noise level across a whole
        sequence makes entire sequences uniformly easy or uniformly hard, and
        training degrades badly without it (their ablation reports the model
        unable to make meaningful progress).

        The per-sample distribution is unchanged from stock π₀.₅ — the same
        ``Beta(1.5, 1)`` draw, rescaled the same way. Only the *shape* changes,
        from ``(B,)`` to ``(B, T)``.

        Args:
            batch_size: Number of trajectories.
            num_timesteps: Number of policy calls in the sequence.
            device: Device to create the tensor on.

        Returns:
            Noise levels of shape ``(batch_size, num_timesteps)``.
        """
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((batch_size, num_timesteps)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def _register_table_for_forward(self) -> Tensor:
        """Selects the register table the forward embeds.

        Returns the trained table, except under the
        ``ttt_inference_zero_registers`` diagnostic in inference mode, where the
        step-0 (zero) table is substituted so the trained registers' ungated
        perturbation of the frozen expert can be isolated from the gated memory
        contribution. Training mode always uses the trained table.

        Returns:
            The ``(n_register_tokens, proj_width)`` table to embed.
        """
        if self.config.ttt_inference_zero_registers and not self.training:
            return torch.zeros_like(self.register_tokens)
        return self.register_tokens

    def embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        register_timestep: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embeds register tokens and noisy actions for the action expert.

        Extends the parent by prepending ``config.n_register_tokens`` learned
        tokens. They join the action tokens in a single attention block, so
        registers and actions attend to each other freely and both cross-attend
        to the whole VLM prefix — which is what lets a register absorb the
        vision-language and state information that TTT never sees directly.

        Args:
            noisy_actions: Noised action chunk, shape ``(B, chunk_size, action_dim)``.
            timestep: Per-action-token noise level, shape ``(B, chunk_size)``.
            register_timestep: Noise level to condition the register tokens'
                AdaRMSNorm on, shape ``(B,)``. Defaults to the last action
                token's level, which is never zeroed by the inference-delay
                mask (that mask only covers a ``max_delay``-length prefix of
                the chunk).

        Returns:
            A tuple of embeddings, padding masks, attention masks and AdaRMS
            conditioning, each covering ``n_register_tokens + chunk_size``
            tokens.
        """
        if self.register_tokens is None:
            return super().embed_suffix(noisy_actions, timestep)

        num_registers = self.config.n_register_tokens
        if register_timestep is None:
            register_timestep = timestep[:, -1]

        # Extend the per-token noise level over the register block before
        # calling the parent, so the parent's single sinusoidal-embedding call
        # produces conditioning for every suffix token in one go.
        register_time = repeat(register_timestep, "b -> b r", r=num_registers)
        embs, pad_masks, att_masks, adarms_cond = super().embed_suffix(
            noisy_actions, torch.cat([register_time, timestep], dim=1)
        )

        # The parent embeds `timestep` for AdaRMS but derives the token
        # embeddings only from `noisy_actions`, so `embs` covers the action
        # block while `adarms_cond` already covers registers + actions.
        register_emb = repeat(
            self._register_table_for_forward().to(embs.dtype), "r w -> b r w", b=embs.shape[0]
        )
        embs = torch.cat([register_emb, embs], dim=1)

        pad_masks = torch.cat(
            [
                torch.ones(embs.shape[0], num_registers, dtype=pad_masks.dtype, device=pad_masks.device),
                pad_masks,
            ],
            dim=1,
        )
        # One attention block spanning registers + actions: `1` opens the block,
        # `0` continues it, so every token in it sees every other.
        att_masks = torch.cat(
            [
                torch.ones(embs.shape[0], 1, dtype=att_masks.dtype, device=att_masks.device),
                torch.zeros(
                    embs.shape[0],
                    num_registers - 1 + self.config.chunk_size,
                    dtype=att_masks.dtype,
                    device=att_masks.device,
                ),
            ],
            dim=1,
        )
        return embs, pad_masks, att_masks, adarms_cond

    def _expert_position_ids(self, prefix_offsets: Tensor, suffix_pad_masks: Tensor) -> Tensor:
        """Position ids for the expert suffix that leave the action block where π₀.₅ puts it.

        The obvious construction — ``prefix_offsets + cumsum(suffix_pad_masks) - 1``
        over the whole suffix — counts the prepended registers, so every action
        token's RoPE phase shifts by ``n_register_tokens`` relative to stock
        π₀.₅. On a warm-start that moves the action readout the pretrained
        weights were trained against, and no gate covers it: it happens at step
        0 regardless of ``alpha``, and under ``train_ttt_only`` the weights that
        could adapt to the shift are frozen. The symptom looks like "TTT hurt
        the policy" when the cause is the displaced readout.

        So the action block keeps positions ``prefix .. prefix + chunk - 1``,
        exactly as without registers, and the registers are placed *after* it in
        position space (``prefix + chunk ..``) while staying first in token
        order. They share one bidirectional attention block with the actions, so
        their position relative to the actions carries no ordering semantics —
        only their displacement of the actions did.

        Args:
            prefix_offsets: ``(B, 1)`` count of prefix tokens the suffix follows.
            suffix_pad_masks: ``(B, S)`` suffix padding mask.

        Returns:
            ``(B, S)`` position ids, register block first in token order.
        """
        num_registers = self.config.n_register_tokens
        if num_registers == 0:
            return prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Pad-aware, like the `num_registers == 0` branch above. π₀.₅'s suffix
        # mask is all ones today so a plain `arange` would be equivalent, but
        # having one branch consult the mask and the other ignore it is the kind
        # of quiet disagreement that outlives the reason for it.
        register_pad = suffix_pad_masks[:, :num_registers]
        action_pad = suffix_pad_masks[:, num_registers:]
        action_positions = torch.cumsum(action_pad, dim=1) - 1
        # Registers start after the action block, so they never displace it.
        action_span = action_pad.sum(dim=1, keepdim=True)
        register_positions = action_span + torch.cumsum(register_pad, dim=1) - 1
        return prefix_offsets + torch.cat([register_positions, action_positions], dim=1)

    def forward_sequence(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions: Tensor,
        num_timesteps: int,
        actions_is_pad: Tensor | None = None,
        response_tokens: Tensor | None = None,
        response_masks: Tensor | None = None,
        discrete_actions: Tensor | None = None,
        discrete_action_masks: Tensor | None = None,
        state: Tensor | None = None,
        real_action_dim: Tensor | None = None,
        loss_mask: Tensor | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, Tensor | PerSampleLoss]:
        """Runs sequence training with TBPTT over a trajectory.

        Every tensor argument arrives already flattened over the sequence axis,
        i.e. with leading dimension ``B * num_timesteps`` in row-major
        ``(batch, timestep)`` order — the order :func:`einops.rearrange`
        produces for ``"b t ... -> (b t) ..."`` and the order the TTT hook
        assumes when it unfolds the axis again.

        The sequence is cut into ``config.tbptt_segment_length``-timestep
        segments. Fast weights cross each boundary by value; their gradients do
        not. ``W_0`` therefore receives gradient only through the first segment,
        whose update originates directly from it — without that path the
        learned initialization would never train and would stay a random draw.

        Args:
            images: Per-camera image tensors, each ``(B * T, ...)``.
            img_masks: Per-camera masks, each ``(B * T,)``.
            lang_tokens: Language tokens, ``(B * T, L)``.
            lang_masks: Language masks, ``(B * T, L)``.
            actions: Ground-truth action chunks, ``(B * T, chunk_size, action_dim)``.
            num_timesteps: ``T``, the number of policy calls per trajectory.
            actions_is_pad: Optional action padding mask.
            response_tokens: Optional response tokens.
            response_masks: Optional response masks.
            discrete_actions: Discrete action targets.
            discrete_action_masks: Discrete action masks.
            state: Optional continuous state, ``(B * T, max_state_dim)``.
            real_action_dim: Optional per-sample real action dimensionality.
            loss_mask: Per-timestep supervision mask, ``(B, T)``, True where the
                timestep contributes an imitation target. Timesteps masked out
                still update the fast weights — that asymmetry is the whole
                point, and is what makes in-context video demonstrations and
                DAgger-style failure-to-correction distillation expressible.

        Returns:
            A dict with the mean ``MSE`` and ``CE`` losses over the sequence.

        Raises:
            ValueError: If ``num_timesteps`` is not a multiple of the configured
                TBPTT segment length, or ``loss_mask`` has the wrong shape.

        Note:
            A sequence whose ``loss_mask`` is entirely False yields a zero loss,
            not an error. Raising would be a data-dependent branch, which must
            not decide control flow that fires collectives.
        """
        segment_length = self.config.tbptt_segment_length
        if num_timesteps % segment_length != 0:
            raise ValueError(
                f"num_timesteps={num_timesteps} must be a multiple of tbptt_segment_length={segment_length}"
            )
        total_rows = actions.shape[0]
        batch_size = total_rows // num_timesteps

        if loss_mask is not None and tuple(loss_mask.shape) != (batch_size, num_timesteps):
            raise ValueError(
                f"loss_mask must have shape ({batch_size}, {num_timesteps}), got {tuple(loss_mask.shape)}"
            )

        tokens_per_timestep = self.config.n_expert_tokens_per_timestep
        carried: dict[int, TTTFastWeights] = {}

        mse_terms: list[Tensor] = []
        ce_terms: list[Tensor] = []
        weights: list[Tensor] = []
        # Per-trajectory (numerator, denominator) pairs, accumulated across
        # segments. `PerSampleLoss` carries the pair rather than a mean exactly
        # so it composes by addition, which is what makes "sum a trajectory's
        # timesteps" and "sum a segment's contributions" the same operation.
        mse_per_sample: PerSampleLoss | None = None
        ce_per_sample_total: PerSampleLoss | None = None

        for segment_index in range(num_timesteps // segment_length):
            start = segment_index * segment_length
            stop = start + segment_length
            rows = self._segment_rows(batch_size, num_timesteps, start, stop, actions.device)

            ttt_state = TTTSequenceState(
                num_timesteps=segment_length,
                position_offset=start * tokens_per_timestep,
                incoming=carried,
            )

            segment_loss_mask = None if loss_mask is None else loss_mask[:, start:stop]
            # Gradient-checkpoint each segment when asked. This is what makes
            # activation memory depend on `tbptt_segment_length` instead of
            # `sequence_length` — measured at 6.75 GiB fixed + 0.304 GiB per
            # timestep without it, i.e. ~47 GiB for a median LIBERO episode at
            # stride 1, against a 23.57 GiB card.
            #
            # Chosen over a per-segment `backward()` (the removed
            # `tbptt_backward_fn` hook) because it needs no change to the
            # training loop's single-backward contract: the returned losses stay
            # graph-carrying and `train.py` backwards them exactly once. The cost
            # is one extra forward per segment.
            #
            # Safe here specifically because a segment is already independent:
            # its incoming fast weights are detached at the boundary, so they are
            # leaf inputs and the recompute cannot diverge. The `ttt_state`
            # side-effect write is idempotent for the same reason the KV-cache
            # write is — deterministic recompute stores an equal value, and the
            # caller reads `outgoing` after the forward completes, before any
            # recompute can run.
            segment_forward = self._forward_segment
            if self.config.checkpoint_tbptt_segments and torch.is_grad_enabled():
                segment_forward = partial(
                    torch.utils.checkpoint.checkpoint,
                    self._forward_segment,
                    use_reentrant=False,
                )
            losses = segment_forward(
                images=[img[rows] for img in images],
                img_masks=[mask[rows] for mask in img_masks],
                lang_tokens=lang_tokens[rows],
                lang_masks=lang_masks[rows],
                actions=actions[rows],
                actions_is_pad=None if actions_is_pad is None else actions_is_pad[rows],
                response_tokens=None if response_tokens is None else response_tokens[rows],
                response_masks=None if response_masks is None else response_masks[rows],
                discrete_actions=None if discrete_actions is None else discrete_actions[rows],
                discrete_action_masks=(
                    None if discrete_action_masks is None else discrete_action_masks[rows]
                ),
                state=None if state is None else state[rows],
                real_action_dim=None if real_action_dim is None else real_action_dim[rows],
                num_timesteps=segment_length,
                loss_mask=segment_loss_mask,
                ttt_state=ttt_state,
                return_per_sample=return_per_sample,
            )
            if return_per_sample:
                mse_per_sample = _accumulate_per_sample(
                    mse_per_sample, losses["MSE_per_sample"], batch_size, segment_length
                )
                ce_per_sample_total = _accumulate_per_sample(
                    ce_per_sample_total, losses["CE_per_sample"], batch_size, segment_length
                )

            # Weight each segment by how many supervised timesteps it holds, so
            # the sequence mean is over supervised timesteps and does not shift
            # when some segments are pure context.
            #
            # Kept as a *tensor*, and appended unconditionally. Reading it into
            # a Python float (`.item()`) and branching on it would make the
            # number of backward calls depend on the local micro-batch's mask,
            # so ranks whose segment is all-context would skip a collective the
            # others enter — a NCCL hang, and precisely the failure CLAUDE.md
            # rule 5 describes. A zero weight contributes a zero term instead.
            weight = (
                torch.tensor(float(segment_length), device=actions.device, dtype=torch.float32)
                if segment_loss_mask is None
                else segment_loss_mask.sum().to(torch.float32)
            )
            mse_terms.append(losses["MSE"] * weight)
            ce_terms.append(losses["CE"] * weight)
            weights.append(weight)

            # The TBPTT boundary. Values cross, the graph does not.
            carried = {idx: fw.detach() for idx, fw in ttt_state.outgoing.items()}

        # Clamped, so a sequence that is entirely context yields a zero loss
        # rather than a NaN that would propagate into every parameter.
        denominator = torch.stack(weights).sum().clamp(min=1.0)
        out: dict[str, Tensor | PerSampleLoss] = {
            "MSE": torch.stack(mse_terms).sum() / denominator,
            "CE": torch.stack(ce_terms).sum() / denominator,
        }
        if return_per_sample:
            out["MSE_per_sample"] = mse_per_sample
            out["CE_per_sample"] = ce_per_sample_total
        return out

    @staticmethod
    def _segment_rows(
        batch_size: int, num_timesteps: int, start: int, stop: int, device: torch.device
    ) -> Tensor:
        """Row indices selecting a timestep window out of a ``(b t)``-flattened batch.

        Args:
            batch_size: Number of trajectories.
            num_timesteps: Total timesteps per trajectory.
            start: First timestep of the window, inclusive.
            stop: Last timestep of the window, exclusive.
            device: Device for the index tensor.

        Returns:
            A 1-D index tensor of length ``batch_size * (stop - start)``, in
            ``(batch, timestep)`` row-major order so the selected rows can be
            unfolded with ``"(b t) ... -> b t ..."``.
        """
        offsets = torch.arange(start, stop, device=device)
        base = rearrange(torch.arange(batch_size, device=device) * num_timesteps, "b -> b 1")
        return rearrange(base + rearrange(offsets, "t -> 1 t"), "b t -> (b t)")

    def _forward_segment(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions: Tensor,
        actions_is_pad: Tensor | None,
        response_tokens: Tensor | None,
        response_masks: Tensor | None,
        discrete_actions: Tensor | None,
        discrete_action_masks: Tensor | None,
        state: Tensor | None,
        real_action_dim: Tensor | None,
        num_timesteps: int,
        loss_mask: Tensor | None,
        ttt_state: TTTSequenceState,
        return_per_sample: bool = False,
    ) -> dict[str, Tensor | PerSampleLoss]:
        """Runs one TBPTT segment: prefix pass, expert pass with TTT, losses.

        Mirrors the structure of :meth:`PI05FlowMatching.forward` — the same
        prefix embedding, cross-attention token count, knowledge-insulation
        detach and masked-MSE reduction — with two differences: the noise level
        is drawn per timestep rather than per sample, and the expert pass
        carries ``ttt_state``.

        Args:
            images: Per-camera image tensors for this segment.
            img_masks: Per-camera masks for this segment.
            lang_tokens: Language tokens for this segment.
            lang_masks: Language masks for this segment.
            actions: Ground-truth action chunks for this segment.
            actions_is_pad: Optional action padding mask.
            response_tokens: Optional response tokens.
            response_masks: Optional response masks.
            discrete_actions: Discrete action targets.
            discrete_action_masks: Discrete action masks.
            state: Optional continuous state.
            real_action_dim: Optional per-sample real action dimensionality.
            num_timesteps: Timesteps in this segment.
            loss_mask: Per-timestep supervision mask, ``(B, num_timesteps)``.
            ttt_state: Sequence state carrying the fast weights in and out.

        Returns:
            A dict with the segment's mean ``MSE`` and ``CE`` over its
            supervised timesteps.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            response_tokens,
            response_masks,
            discrete_actions,
            discrete_action_masks,
            state=state,
        )
        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        num_cross_att_tokens = (
            prefix_embs.shape[1]
            - self.config.discrete_action_indicator_max_length
            - self.config.discrete_action_max_length
        )

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=vlm_2d_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=False,
            fill_kv_cache=True,
        )

        rows = actions.shape[0]
        batch_size = rows // num_timesteps
        noise = self.sample_noise(actions.shape, actions.device)

        # Sequence action forcing: one independent noise level per timestep,
        # flattened to match the `(b t)` row order of everything else.
        time_per_timestep = self.sample_time_sequence(batch_size, num_timesteps, actions.device)
        time_flat = rearrange(time_per_timestep, "b t -> (b t)")

        delay = torch.randint(0, self.config.max_delay + 1, (rows,))
        prefix_mask = rearrange(torch.arange(self.config.chunk_size), "c -> 1 c") < rearrange(
            delay, "b -> b 1"
        )
        prefix_mask = prefix_mask.to(device=actions.device)
        time = torch.where(prefix_mask, 0, rearrange(time_flat, "b -> b 1"))

        time_expanded = rearrange(time, "b c -> b c 1")
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time, register_timestep=time_flat
        )

        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        prefix_offsets = torch.sum(
            prefix_pad_masks[
                :,
                : -self.config.discrete_action_indicator_max_length - self.config.discrete_action_max_length,
            ],
            dim=-1,
        )[:, None]
        action_expert_position_ids = self._expert_position_ids(prefix_offsets, suffix_pad_masks)

        if self.config.knowledge_insulation:
            for layer_idx in past_key_values:
                past_key_values[layer_idx]["key_states"] = past_key_values[layer_idx]["key_states"].detach()
                past_key_values[layer_idx]["value_states"] = past_key_values[layer_idx][
                    "value_states"
                ].detach()

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
            ttt_state=ttt_state,
        )

        # Registers are prepended, so slicing the trailing chunk drops them.
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out).to(dtype=torch.float32)

        # Fold the per-timestep supervision mask into the action padding mask,
        # which `flow_matching_masked_mse` already ANDs into its denominator.
        # A context-only timestep therefore contributes no target while still
        # having driven the fast-weight update above.
        if loss_mask is not None:
            timestep_is_context = rearrange(~loss_mask, "b t -> (b t)")
            context_pad = repeat(timestep_is_context, "r -> r c", c=self.config.chunk_size)
            actions_is_pad = context_pad if actions_is_pad is None else (actions_is_pad | context_pad)

        mse_result = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            max_action_dim=self.config.max_action_dim,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            real_action_dim=real_action_dim,
            return_per_sample=return_per_sample,
        )
        mse_loss, mse_rows = mse_result if return_per_sample else (mse_result, None)

        ce_loss = self._discrete_action_ce(
            prefix_out=prefix_out,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
            loss_mask=loss_mask,
        ) + self._response_ce(
            prefix_out=prefix_out,
            response_tokens=response_tokens,
            response_masks=response_masks,
            loss_mask=loss_mask,
        )
        out: dict[str, Tensor | PerSampleLoss] = {"MSE": mse_loss, "CE": ce_loss}
        if return_per_sample:
            out["MSE_per_sample"] = mse_rows
            out["CE_per_sample"] = self._discrete_action_ce_per_sample(
                prefix_out=prefix_out,
                discrete_actions=discrete_actions,
                discrete_action_masks=discrete_action_masks,
                loss_mask=loss_mask,
            )
        return out

    def _discrete_action_ce(
        self,
        prefix_out: Tensor,
        discrete_actions: Tensor | None,
        discrete_action_masks: Tensor | None,
        loss_mask: Tensor | None,
    ) -> Tensor:
        """Cross-entropy over π₀.₅'s discrete-action tokens, with loss masking.

        Args:
            prefix_out: VLM prefix output, ``(B * T, prefix_len, hidden)``.
            discrete_actions: Discrete action targets, ``(B * T, L)``.
            discrete_action_masks: True on real (non-pad) target tokens.
            loss_mask: Per-timestep supervision mask, ``(B, T)``.

        Returns:
            The cross-entropy summed over supervised, non-pad tokens and divided
            by the total slot count — the same reduction stock π₀.₅ uses — or a
            zero scalar when there are no discrete-action targets.
        """
        if discrete_actions is None:
            return torch.zeros((), device=prefix_out.device, dtype=torch.float32)

        discrete_token_start = -self.config.discrete_action_max_length
        # The last response token predicts the first discrete-action token, so
        # the logits are shifted one position left of the labels.
        discrete_action_out = prefix_out[:, slice(discrete_token_start - 1, -1)]
        logits = self.paligemma_with_expert.da_head(discrete_action_out).to(dtype=torch.float32)

        loss = torch.nn.functional.cross_entropy(
            rearrange(logits, "b s d -> (b s) d"),
            rearrange(discrete_actions, "b s -> (b s)"),
            reduction="none",
        )
        loss = rearrange(loss, "(b s) -> b s", b=discrete_actions.shape[0])

        valid = (
            torch.ones_like(loss, dtype=torch.bool)
            if discrete_action_masks is None
            else discrete_action_masks
        )
        if loss_mask is not None:
            valid = valid & rearrange(loss_mask, "b t -> (b t) 1")

        # Zero the excluded slots, then divide by (supervised rows x slots).
        #
        # This deliberately matches `PI05FlowMatching.forward` rather than
        # dividing by the number of valid tokens. Dividing by `valid.sum()`
        # scales the term by `discrete_action_max_length / mean valid tokens per
        # row` — greater than one and data-dependent per batch, because FAST
        # emits variable-length sequences. Since `train.py::_assemble_weighted_loss`
        # multiplies this scalar by `loss_weighting["CE"]` directly, the two
        # conventions optimize different MSE:CE balances and log a `Train/CE`
        # that cannot be compared against any existing π₀.₅ run. One policy
        # must not compute CE two ways depending on the batch's shape.
        # `policies/utils.py::ce_per_sample` documents the same distinction.
        #
        # The denominator counts *supervised* rows, not all rows. Dividing by all
        # rows would make this term already proportional to the segment's
        # supervised fraction, and `forward_sequence` then weights it by that
        # same fraction again — CE scaled twice while MSE (normalized over
        # unmasked slots by `flow_matching_masked_mse`) is scaled once. Measured
        # at 2.15x on a half-masked segment. With every row supervised this
        # reduces to exactly the parent's `.mean()` over `B * S`, so parity with
        # stock pi05 is preserved.
        rows, slots = loss.shape
        supervised_rows = (
            torch.tensor(float(rows), device=loss.device)
            if loss_mask is None
            else loss_mask.sum().to(loss.dtype)
        )
        return (loss * valid).sum() / (supervised_rows * slots).clamp(min=1.0)

    def _discrete_action_ce_per_sample(
        self,
        prefix_out: Tensor,
        discrete_actions: Tensor | None,
        discrete_action_masks: Tensor | None,
        loss_mask: Tensor | None,
    ) -> PerSampleLoss:
        """Per-row ``(Σ valid CE, #valid tokens)`` for the validation breakdown.

        Normalized over *valid tokens* rather than all slots, which is the
        convention `ce_per_sample` documents for the per-group breakdown — it
        differs from the scalar reduction on purpose, and the scalar is
        unaffected by whether this is computed.

        Args:
            prefix_out: VLM prefix output, ``(B * T, prefix_len, hidden)``.
            discrete_actions: Discrete action targets.
            discrete_action_masks: True on real (non-pad) target tokens.
            loss_mask: Per-timestep supervision mask, ``(B, T)``.

        Returns:
            A per-row decomposition, or an all-zero one when there are no targets.
        """
        rows = prefix_out.shape[0]
        if discrete_actions is None:
            zeros = torch.zeros(rows, device=prefix_out.device, dtype=torch.float32)
            return PerSampleLoss(sum=zeros, count=zeros)

        start = -self.config.discrete_action_max_length
        logits = self.paligemma_with_expert.da_head(prefix_out[:, slice(start - 1, -1)]).to(
            dtype=torch.float32
        )
        loss = torch.nn.functional.cross_entropy(
            rearrange(logits, "b s d -> (b s) d"),
            rearrange(discrete_actions, "b s -> (b s)"),
            reduction="none",
        )
        loss = rearrange(loss, "(b s) -> b s", b=discrete_actions.shape[0])

        valid = (
            torch.ones_like(loss, dtype=torch.bool)
            if discrete_action_masks is None
            else discrete_action_masks
        )
        if loss_mask is not None:
            valid = valid & rearrange(loss_mask, "b t -> (b t) 1")
        return ce_per_sample(loss * valid, valid)

    def _response_ce(
        self,
        prefix_out: Tensor,
        response_tokens: Tensor | None,
        response_masks: Tensor | None,
        loss_mask: Tensor | None,
    ) -> Tensor:
        """Cross-entropy over π₀.₅'s response (subtask) tokens, with loss masking.

        Mirrors the ``predict_response`` block of :meth:`PI05FlowMatching.forward`,
        including the two off-by-one slices: the last language token predicts the
        response's ``<BOS>`` and the last response token predicts the first
        discrete-action token, so neither is scored. Normalized over supervised
        rows for the same reason as :meth:`_discrete_action_ce`.

        Implemented rather than refused. Refusing meant ``predict_response=True``
        was unusable with this policy at all once the flat-batch fall-through was
        removed, which is a capability regression against stock π₀.₅ rather than
        a missing extra.

        Args:
            prefix_out: VLM prefix output, ``(B * T, prefix_len, hidden)``.
            response_tokens: Response token targets, ``(B * T, L)``.
            response_masks: True on real (non-pad) response tokens.
            loss_mask: Per-timestep supervision mask, ``(B, T)``.

        Returns:
            The masked response cross-entropy, or a zero scalar when response
            prediction is off or there are no response targets.
        """
        if not self.config.predict_response or response_tokens is None:
            return torch.zeros((), device=prefix_out.device, dtype=torch.float32)

        rows, seq_len = response_tokens.shape
        start = (
            -self.config.response_max_length
            - self.config.discrete_action_max_length
            - self.config.discrete_action_indicator_max_length
        )
        end = -self.config.discrete_action_max_length - self.config.discrete_action_indicator_max_length - 1
        response_out = prefix_out[:, slice(start, end)]
        logits = self.paligemma_with_expert.paligemma.lm_head(response_out).to(dtype=torch.float32)

        label_slice = slice(1, None)
        loss = torch.nn.functional.cross_entropy(
            rearrange(logits, "b s d -> (b s) d"),
            rearrange(response_tokens[:, label_slice], "b s -> (b s)"),
            reduction="none",
        )
        loss = rearrange(loss, "(b s) -> b s", b=rows, s=seq_len - 1)

        valid = (
            torch.ones_like(loss, dtype=torch.bool)
            if response_masks is None
            else response_masks[:, label_slice]
        )
        if loss_mask is not None:
            valid = valid & rearrange(loss_mask, "b t -> (b t) 1")

        supervised_rows = (
            torch.tensor(float(rows), device=loss.device)
            if loss_mask is None
            else loss_mask.sum().to(loss.dtype)
        )
        return (loss * valid).sum() / (supervised_rows * loss.shape[1]).clamp(min=1.0)

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: dict[int, dict[str, Tensor]],
        x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        """One Euler denoising step, with TTT reading the carried fast weights.

        Mirrors :meth:`PI05FlowMatching.denoise_step`; the only change is that
        the expert pass receives ``self._active_ttt_state``.

        The fast weights are *read* on every step of the Euler loop but the
        rollout only *adopts* the update produced by the final step — see
        :meth:`sample_actions`. One policy call must perform exactly one
        fast-weight update ("one mini batch per inference"), not one per
        denoising step, or memory would advance ``config.num_steps`` times
        faster at inference than it ever did in training.

        **Known train/inference mismatch in the update's input distribution.**
        The update *count* is right, but the flow-matching time it is computed at
        is not. Training runs a single expert forward at
        ``tau ~ Beta(1.5, 1) * 0.999 + 0.001`` (mean about 0.6, i.e. mostly
        noisy actions), while the Euler loop runs ``tau = 1 -> 0`` and the
        adopted update is the one from the final step, at ``tau ~ dt/2``, from
        nearly-clean actions. Since the action tokens' embeddings depend directly
        on ``x_t``, the fast weights ingest a systematically different input at
        deployment than they were trained on.

        The final step is adopted deliberately — it is the update driven by the
        chunk the robot actually executes, which is the quantity the memory
        should carry forward — but the mismatch is real and unmeasured. The
        alternatives are to adopt the *first* step's update (noisy, closer to the
        training marginal, but derived from a chunk that was discarded) or to add
        a dedicated write pass at a training-matched ``tau`` (correct, one extra
        expert forward per call). Deferred until there is a long-context training
        run to measure it against, because picking between them on reasoning
        alone is how a plausible-but-wrong default gets locked in.

        Args:
            prefix_pad_masks: Prefix padding masks.
            past_key_values: Cached prefix keys and values.
            x_t: Current noised action chunk.
            time: Per-token noise level, ``(B, chunk_size)``.

        Returns:
            The predicted velocity, ``(B, chunk_size, action_dim)``.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        num_cross_att_tokens = prefix_pad_masks.shape[1]
        action_expert_2d_attention_mask = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross_att_tokens,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross_att_tokens],
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        action_expert_position_ids = self._expert_position_ids(prefix_offsets, suffix_pad_masks)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=action_expert_2d_attention_mask,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
            ttt_state=self._active_ttt_state,
        )
        if (
            self.config.ttt_inference_update_adoption == "first"
            and self._active_ttt_state is not None
            and self._first_step_adoption is None
            and self._active_ttt_state.outgoing
        ):
            # The first Euler step runs at tau = 1: pure-noise action tokens, the
            # mode of the training marginal — capture its update before later
            # (nearly-clean) steps overwrite `outgoing`.
            self._first_step_adoption = {
                idx: fw.detach() for idx, fw in self._active_ttt_state.outgoing.items()
            }
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]
        return self.action_out_proj(suffix_out).to(dtype=torch.float32)

    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        action_prefix: Tensor,
        delay: Tensor,
        noise: Tensor | None = None,
        state: Tensor | None = None,
        accel: AccelMeter | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Samples an action chunk, advancing the carried fast weights by one step.

        The signature matches :meth:`PI05FlowMatching.sample_actions` for every
        parameter a positional caller can reach, with ``accel`` trailing and
        defaulting to ``None`` — that ordering is a pinned contract, because
        this method is compiled at several serving entry points and
        ONNX-exported at one, all calling it positionally.

        It is *not* parameter-for-parameter identical: ``n_candidates`` is
        deliberately absent. The enclosing wrapper passes it by keyword, so
        ``**kwargs`` receives it, and naming it would make the AST wiring sweep
        classify this family as best-of-N-wired, which it is not — see the
        refusal below and ``policies.candidates.configure_candidates``.

        Args:
            images: Per-camera image tensors.
            img_masks: Per-camera masks.
            lang_tokens: Language tokens.
            lang_masks: Language masks.
            action_prefix: Frozen action prefix for real-time inference.
            delay: Number of frozen prefix actions.
            noise: Optional starting noise.
            state: Optional continuous state.
            accel: Optional denoising-acceleration meter.
            **kwargs: Remaining keyword arguments, forwarded to the parent.

        Returns:
            The sampled action chunk.

        Raises:
            NotImplementedError: If ``n_candidates > 1``. Best-of-N widens the
                batch to ``B * n_candidates`` rows while the carried fast
                weights hold ``B``, and there is no defined answer for which
                candidate's fast-weight update should become the rollout's
                memory — the chunks the losing candidates produced were never
                executed, so adopting their memory is wrong, and adopting the
                winner's means the memory depends on a critic the training run
                never saw.

                The refusal is unconditional rather than conditioned on the
                memory being non-empty: a conditional one would let the first
                call of a rollout succeed and the second raise, which is a far
                more confusing failure than refusing outright.
        """
        if int(kwargs.get("n_candidates", 1)) > 1:
            raise NotImplementedError(
                "pi05_ttt does not support best-of-N candidate sampling: n_candidates>1 "
                "widens the batch, so the carried fast weights would not line up, and it is "
                "undefined which candidate's fast-weight update should be adopted as the "
                "rollout's memory. Use n_candidates=1."
            )

        # A memory built at one batch size would broadcast silently against a
        # wider batch, mixing one rollout's history into another's.
        if self._carried_fast_weights:
            carried_batch = next(iter(self._carried_fast_weights.values())).w1.shape[0]
            incoming_batch = lang_tokens.shape[0]
            if carried_batch != incoming_batch:
                raise ValueError(
                    f"carried TTT memory was built at batch size {carried_batch} but this call "
                    f"has {incoming_batch} rows. Call `reset()` between rollouts, and do not "
                    "change batch size mid-rollout: the fast weights are per-trajectory state."
                )

        ttt_state = TTTSequenceState(
            num_timesteps=1,
            position_offset=self._inference_token_position,
            incoming=self._carried_fast_weights,
        )
        self._active_ttt_state = ttt_state
        # Clear any capture a mid-call exception may have stranded, so this
        # call's first Euler step is the one captured.
        self._first_step_adoption = None
        try:
            actions = super().sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                action_prefix,
                delay,
                noise=noise,
                state=state,
                accel=accel,
                **kwargs,
            )
        finally:
            self._active_ttt_state = None

        # Adopt exactly one step's update per policy call, and detach: a rollout
        # is not a training graph, and keeping one would grow without bound.
        # Which step is `config.ttt_inference_update_adoption`: "last" (historic)
        # ingests nearly-clean self-generated actions; "first" ingests the pure-
        # noise tokens matching the mode of the training marginal.
        if self.config.ttt_inference_update_adoption == "first" and self._first_step_adoption:
            self._carried_fast_weights = self._first_step_adoption
            self._inference_token_position += self.config.n_expert_tokens_per_timestep
        elif ttt_state.outgoing:
            self._carried_fast_weights = {idx: fw.detach() for idx, fw in ttt_state.outgoing.items()}
            self._inference_token_position += self.config.n_expert_tokens_per_timestep
        self._first_step_adoption = None
        return actions


class PI05TTTPolicy(PI05Policy):
    """π₀.₅ with Test-Time-Training memory, wrapped for OpenTau training and inference.

    Args:
        config: Policy configuration.
        per_dataset_stats: Ordered per-dataset normalization stats.
        dataset_names: Ordered dataset names parallel to ``per_dataset_stats``.
    """

    config_class = PI05TTTConfig
    name = "pi05_ttt"
    # Read by `policies.candidates.configure_candidates`, which otherwise probes
    # `hasattr(policy, "n_candidates")` — an attribute this class inherits from
    # `PI05Policy.__init__`, so the probe alone would let best-of-N arm and fail
    # only on the first robot request.
    supports_candidate_sampling = False
    # Read by the gRPC server and the ONNX exporter: the fast weights are
    # per-rollout state that must be reset per episode and cannot be traced.
    carries_rollout_state = True
    # The sequence path drives a Python-level loop over TBPTT segments whose
    # trip count depends on the batch's timestep count, which is exactly the
    # shape of graph break torch.compile handles worst. Left off until the
    # sequence data path exists and can be profiled.
    supports_torch_compile = False

    def __init__(
        self,
        config: PI05TTTConfig,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
    ):
        super().__init__(config, per_dataset_stats=per_dataset_stats, dataset_names=dataset_names)
        self.config: PI05TTTConfig = config
        if config.train_ttt_only:
            self.model.freeze_pretrained_parameters()

    def _build_flow_matching(
        self, config: PI05TTTConfig, discrete_action_vocab_size: int | None
    ) -> PI05TTTFlowMatching:
        """Builds the TTT-augmented inner module in place of the plain one.

        Overriding the parent's factory rather than reassigning ``self.model``
        after ``super().__init__`` matters: the parent would otherwise build a
        complete PaliGemma tower only for it to be dropped on the floor.

        Args:
            config: Policy configuration.
            discrete_action_vocab_size: Size of the discrete action vocabulary.

        Returns:
            The TTT flow-matching module.
        """
        return PI05TTTFlowMatching(
            config,
            discrete_action_vocab_size=discrete_action_vocab_size,
            language_tokenizer=self.language_tokenizer,
        )

    def reset(self) -> None:
        """Resets the action queue and drops the carried TTT memory."""
        super().reset()
        # Called by the parent's __init__ before self.model exists.
        model = getattr(self, "model", None)
        if isinstance(model, PI05TTTFlowMatching):
            model.reset_memory()

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        return_per_sample: bool = False,
    ) -> dict[str, Tensor | PerSampleLoss]:
        """Computes the training loss over a trajectory sequence.

        There is one path. A batch whose ``actions`` carry a leading timestep
        axis (``(B, T, chunk_size, action_dim)``) is trained as a trajectory
        with sequence action forcing and TBPTT; a flat
        ``(B, chunk_size, action_dim)`` batch is the ``T = 1`` case of the same
        path, and ``config.sequence_length`` must say so.

        An earlier revision delegated the flat case to ``PI05Policy.forward``,
        which never passes a ``ttt_state`` — so the TTT branch was skipped and
        not one TTT parameter reached the autograd graph. See the comment in the
        body.

        Args:
            batch: Training batch, optionally with a leading timestep axis on
                every per-timestep entry, plus an optional ``loss_mask`` of
                shape ``(B, T)`` marking supervised timesteps.
            noise: Must be ``None``; the sequence path draws its own noise.
            time: Must be ``None``; the sequence path draws one noise level per
                timestep (sequence action forcing).
            return_per_sample: Additionally return ``MSE_per_sample`` /
                ``CE_per_sample`` as :class:`PerSampleLoss`, decomposed *per
                trajectory* — a trajectory is the sample, so a sequence's
                timesteps are pooled into one ``(numerator, denominator)`` pair.
                The validation loop selects this by signature introspection and
                calls it without a guard, so it must not raise.

        Returns:
            A dict with the ``MSE`` and ``CE`` loss components, plus
            ``MSE_per_sample`` / ``CE_per_sample`` when requested.

        Raises:
            NotImplementedError: If ``noise`` or ``time`` is supplied.
        """
        # Every batch goes through the sequence path, including a flat one,
        # which is treated as a single-timestep sequence.
        #
        # Delegating a flat batch to `PI05Policy.forward` (the previous
        # behaviour) meant `PI05FlowMatching.forward` ran without a
        # `ttt_state`, so the TTT branch in `_run_layer` was skipped and *no
        # TTT parameter reached the autograd graph at all*. Combined with
        # `train_ttt_only=True` that left 85.3M parameters nominally trainable
        # of which only the 16K register table received a gradient — and under
        # DDP with FIND_UNUSED_PARAMS=false, or under ZeRO-3, an unused
        # parameter is a reducer error or an NCCL hang rather than a slow run.
        #
        # The number of timesteps comes from `config.sequence_length`, not from
        # the batch, and the batch is validated against it. That keeps the TBPTT
        # segment count — and therefore the number of backward calls — identical
        # on every rank by construction (CLAUDE.md rule 5) instead of depending
        # on what each rank's micro-batch happened to contain.
        num_timesteps = self.config.sequence_length
        batch = self._as_sequence_batch(batch, num_timesteps)

        if noise is not None or time is not None:
            raise NotImplementedError(
                "pi05_ttt's sequence path draws its own per-timestep noise level (sequence "
                "action forcing); passing `noise` or `time` explicitly would defeat it."
            )
        loss_mask = batch.get("loss_mask")
        batch_size = batch["actions"].shape[0]
        flat_batch = self._flatten_sequence_batch(batch, batch_size, num_timesteps)

        dataset_index = self._resolve_dataset_index(flat_batch)
        flat_batch = self.normalize_inputs(flat_batch, dataset_index)
        flat_batch["discrete_actions"] = self.normalize_discrete_actions(dict(flat_batch), dataset_index)[
            "actions"
        ]
        flat_batch = self.normalize_targets(flat_batch, dataset_index)

        images, img_masks = self.prepare_images(flat_batch)
        lang_tokens, lang_masks = self.prepare_language(flat_batch)
        response_tokens, response_masks = self.prepare_response(flat_batch)
        discrete_actions, discrete_action_masks = self.prepare_discrete_actions(flat_batch)
        state = self.prepare_state(flat_batch) if self.config.state_type == "continuous" else None

        return self.model.forward_sequence(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            actions=flat_batch["actions"],
            num_timesteps=num_timesteps,
            actions_is_pad=flat_batch.get("action_is_pad"),
            response_tokens=response_tokens,
            response_masks=response_masks,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
            state=state,
            real_action_dim=flat_batch.get("real_action_dim"),
            loss_mask=loss_mask,
        )

    @staticmethod
    def _flatten_sequence_batch(batch: dict[str, Any], batch_size: int, num_timesteps: int) -> dict[str, Any]:
        """Folds a trajectory batch's timestep axis into its batch axis.

        Everything downstream of here — normalization, the ``prepare_*``
        helpers, the VLM prefix pass — is written for a flat batch of
        independent rows, and attention is supposed to run per timestep anyway.
        So the sequence axis is folded in here and unfolded again inside the TTT
        layers, which are the only components that need it.

        Rows come out in ``(batch, timestep)`` row-major order, which is what
        ``"b t ... -> (b t) ..."`` produces and what the TTT hook's inverse
        ``rearrange`` assumes. Getting this order wrong would silently interleave
        trajectories inside the memory.

        Classification rule:

        * a tensor whose leading dims are exactly ``(B, T)`` is per-timestep and
          is flattened;
        * a tensor whose leading dim is ``B`` but which is not per-timestep is
          trajectory-level and is repeated ``T`` times per row;
        * a list of length ``B`` (e.g. task strings) is likewise repeated
          element-wise;
        * ``loss_mask`` is left alone, since it is consumed in ``(B, T)`` form.

        Args:
            batch: Trajectory batch.
            batch_size: ``B``.
            num_timesteps: ``T``.

        Returns:
            A new dict with every per-timestep entry flattened to ``B * T`` rows.
            At ``num_timesteps == 1`` the batch is already in that shape and is
            returned as a shallow copy.

        Note:
            The rule is shape-based, so a trajectory-level tensor that happens
            to be shaped ``(B, T)`` — say a token sequence whose length equals
            the timestep count — would be misread as per-timestep. No such key
            exists in today's batches, and the ambiguity should be removed by
            having the dataloader label its per-timestep keys explicitly when
            the sequence data path lands.
        """
        # Short-circuit on the batch being *already flat*, not on
        # `num_timesteps == 1`. At sequence_length 1 a caller may legitimately
        # pass either shape: a flat `(B, chunk, dim)` batch (what the dataloader
        # emits today) or an explicit `(B, 1, chunk, dim)` one. Keying off the
        # timestep count returned the latter untouched, so its 5-D camera
        # tensors reached `prepare_images` and it raised
        # `(b,c,h,w) expected, but torch.Size([1, 1, 3, 224, 224])`.
        actions = batch.get("actions")
        if actions is not None and actions.ndim == 3:
            return dict(batch)

        flat: dict[str, Any] = {}
        for key, value in batch.items():
            if key == "loss_mask":
                flat[key] = value
            elif (
                isinstance(value, Tensor)
                and value.ndim >= 2
                and value.shape[:2]
                == (
                    batch_size,
                    num_timesteps,
                )
            ):
                flat[key] = rearrange(value, "b t ... -> (b t) ...")
            elif isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == batch_size:
                flat[key] = value.repeat_interleave(num_timesteps, dim=0)
            elif isinstance(value, list) and len(value) == batch_size:
                flat[key] = [item for item in value for _ in range(num_timesteps)]
            else:
                flat[key] = value
        return flat

    @staticmethod
    def _as_sequence_batch(batch: dict[str, Any], num_timesteps: int) -> dict[str, Any]:
        """Normalizes a batch to carry a leading timestep axis of ``num_timesteps``.

        A batch whose ``actions`` are ``(B, chunk, dim)`` is a single timestep
        per row; it is unsqueezed to ``(B, 1, chunk, dim)`` so the sequence path
        can run it. A batch that already carries a timestep axis must match
        ``config.sequence_length`` exactly — a mismatch is a config error, and
        silently adopting the batch's length would make the segment count
        rank-dependent.

        Args:
            batch: Training batch, with or without a timestep axis.
            num_timesteps: The configured ``sequence_length``.

        Returns:
            The batch, unchanged if it already has the right shape.

        Raises:
            ValueError: If ``actions`` is missing, has an unusable rank, or
                carries a timestep axis that disagrees with the config.
        """
        actions = batch.get("actions")
        if actions is None:
            raise ValueError("pi05_ttt.forward requires an `actions` entry in the batch")

        if actions.ndim == 3:
            if num_timesteps != 1:
                raise ValueError(
                    f"config.sequence_length={num_timesteps} but the batch carries no timestep "
                    "axis (actions is (B, chunk, dim)), so only sequence_length=1 can consume "
                    "it. Today's dataloader emits one timestep per row; set sequence_length=1, "
                    "or supply (B, T, chunk, dim) batches."
                )
            # Left flat on purpose: unsqueezing to (B, 1, ...) and flattening
            # straight back is the identity, so the round trip would only add a
            # chance to misclassify which keys are per-timestep.
            return batch

        if actions.ndim != 4:
            raise ValueError(
                f"`actions` must be (B, chunk, dim) or (B, T, chunk, dim), got rank {actions.ndim}"
            )
        if actions.shape[1] != num_timesteps:
            raise ValueError(
                f"batch carries T={actions.shape[1]} timesteps but "
                f"config.sequence_length={num_timesteps}. The configured value is the source of "
                "truth so the TBPTT segment count is identical on every rank; fix the config or "
                "the dataloader rather than letting the batch decide."
            )
        return batch
