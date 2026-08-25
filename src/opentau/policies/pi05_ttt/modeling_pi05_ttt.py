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
  hand, but training on real long trajectories needs a dataset-side change.
  With a plain single-timestep batch this policy trains like stock π₀.₅ with an
  extra, near-zero-gated memory path.
* **Truncated gradients, untruncated memory.** Gradients are truncated exactly
  as the paper specifies, so the optimization is correct. The *activation
  memory* benefit additionally needs one backward per segment, which the shared
  training loop does not do; see ``tbptt_backward_fn``.
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

from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

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
from opentau.policies.utils import flow_matching_masked_mse


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
            self.register_tokens = nn.Parameter(
                torch.normal(0.0, 0.02, size=(config.n_register_tokens, config.proj_width))
            )
        else:
            self.register_tokens = None

        self._attach_ttt_layers()

        # Fast weights carried across inference calls, and the token position
        # the next call's RoPE should start from. Both are rollout state, reset
        # by ``PI05TTTPolicy.reset``.
        self._carried_fast_weights: dict[int, TTTFastWeights] = {}
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
            )
            layer.ttt_gate = TanhGate(width, init_value=self.config.ttt_gate_init)

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
        register_emb = repeat(self.register_tokens.to(embs.dtype), "r w -> b r w", b=embs.shape[0])
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
        tbptt_backward_fn: Callable[[Tensor], None] | None = None,
    ) -> dict[str, Tensor]:
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
            tbptt_backward_fn: Optional callback invoked with each segment's
                loss as soon as it is computed. Supplying it is what makes TBPTT
                bound activation memory by the segment length; without it the
                per-segment graphs are all retained until the caller's single
                backward, so memory scales with the full sequence even though
                the *gradients* are already correctly truncated.

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
            losses = self._forward_segment(
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
            if tbptt_backward_fn is not None:
                tbptt_backward_fn(losses["MSE"] * weight + losses["CE"] * weight)

            # The TBPTT boundary. Values cross, the graph does not.
            carried = {idx: fw.detach() for idx, fw in ttt_state.outgoing.items()}

        # Clamped, so a sequence that is entirely context yields a zero loss
        # rather than a NaN that would propagate into every parameter.
        denominator = torch.stack(weights).sum().clamp(min=1.0)
        return {
            "MSE": torch.stack(mse_terms).sum() / denominator,
            "CE": torch.stack(ce_terms).sum() / denominator,
        }

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
    ) -> dict[str, Tensor]:
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
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

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

        mse_loss = flow_matching_masked_mse(
            u_t=u_t,
            v_t=v_t,
            max_action_dim=self.config.max_action_dim,
            prefix_mask=prefix_mask,
            actions_is_pad=actions_is_pad,
            real_action_dim=real_action_dim,
            return_per_sample=False,
        )

        ce_loss = self._discrete_action_ce(
            prefix_out=prefix_out,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
            loss_mask=loss_mask,
            num_timesteps=num_timesteps,
        )
        return {"MSE": mse_loss, "CE": ce_loss}

    def _discrete_action_ce(
        self,
        prefix_out: Tensor,
        discrete_actions: Tensor | None,
        discrete_action_masks: Tensor | None,
        loss_mask: Tensor | None,
        num_timesteps: int,
    ) -> Tensor:
        """Cross-entropy over π₀.₅'s discrete-action tokens, with loss masking.

        Args:
            prefix_out: VLM prefix output, ``(B * T, prefix_len, hidden)``.
            discrete_actions: Discrete action targets, ``(B * T, L)``.
            discrete_action_masks: True on real (non-pad) target tokens.
            loss_mask: Per-timestep supervision mask, ``(B, T)``.
            num_timesteps: Timesteps in this segment.

        Returns:
            The mean cross-entropy over supervised, non-pad tokens, or a zero
            scalar when there are no discrete-action targets.
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

        # Mean over supervised, non-pad tokens. Clamped so an all-context
        # segment yields 0 rather than a NaN that would poison the whole step.
        return (loss * valid).sum() / valid.sum().clamp(min=1)

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
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

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
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :]
        return self.action_out_proj(suffix_out).to(dtype=torch.float32)

    def sample_actions(self, *args: Any, **kwargs: Any) -> Tensor:
        """Samples an action chunk, advancing the carried fast weights by one step.

        Args:
            *args: Forwarded to :meth:`PI05FlowMatching.sample_actions`.
            **kwargs: Forwarded to :meth:`PI05FlowMatching.sample_actions`.

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

        state = TTTSequenceState(
            num_timesteps=1,
            position_offset=self._inference_token_position,
            incoming=self._carried_fast_weights,
        )
        self._active_ttt_state = state
        try:
            actions = super().sample_actions(*args, **kwargs)
        finally:
            self._active_ttt_state = None

        # Adopt only the final Euler step's update, and detach: a rollout is not
        # a training graph, and keeping one would grow without bound.
        if state.outgoing:
            self._carried_fast_weights = {idx: fw.detach() for idx, fw in state.outgoing.items()}
            self._inference_token_position += self.config.n_expert_tokens_per_timestep
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
        # Optional per-segment backward, for a training loop that opts into
        # bounded activation memory. See `forward_sequence`.
        self.tbptt_backward_fn: Callable[[Tensor], None] | None = None

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
    ) -> dict[str, Tensor]:
        """Computes the training loss, taking the sequence path when given sequences.

        A batch whose ``actions`` carry an extra leading timestep axis
        (``(B, T, chunk_size, action_dim)``) is trained as a trajectory with
        sequence action forcing and TBPTT. A plain ``(B, chunk_size, action_dim)``
        batch falls through to stock π₀.₅ behavior, which is what makes this
        policy usable with today's single-timestep dataloader — though with
        nothing for the memory to learn.

        Args:
            batch: Training batch, optionally with a leading timestep axis on
                every per-timestep entry, plus an optional ``loss_mask`` of
                shape ``(B, T)`` marking supervised timesteps.
            noise: Optional noise tensor. Sequence path only accepts ``None``.
            time: Optional time tensor. Sequence path only accepts ``None``,
                since it draws one level per timestep itself.
            return_per_sample: Per-sample loss breakdown. Not available on the
                sequence path.

        Returns:
            A dict with the ``MSE`` and ``CE`` loss components.

        Raises:
            NotImplementedError: If ``noise``, ``time`` or ``return_per_sample``
                is used together with a sequence batch.
        """
        num_timesteps = self._sequence_length(batch)
        if num_timesteps is None:
            return super().forward(batch, noise=noise, time=time, return_per_sample=return_per_sample)

        if noise is not None or time is not None:
            raise NotImplementedError(
                "pi05_ttt's sequence path draws its own per-timestep noise level (sequence "
                "action forcing); passing `noise` or `time` explicitly would defeat it."
            )
        if return_per_sample:
            raise NotImplementedError(
                "return_per_sample is not implemented on pi05_ttt's sequence path: the "
                "validation breakdown buckets by sample, and a sequence's loss is a mean over "
                "timesteps whose provenance the loop does not currently carry."
            )
        if self.config.predict_response:
            # The sequence path computes the discrete-action CE but not the
            # response CE. Raising beats returning a loss that quietly omits a
            # term the config asked for — that reads as a converging run right
            # up until the subtask head turns out to be untrained.
            raise NotImplementedError(
                "pi05_ttt's sequence path does not yet compute the response (subtask) "
                "cross-entropy, so running it with predict_response=True would silently drop "
                "that loss term. Set predict_response=False for sequence training, or use the "
                "single-timestep path. Feeding the predicted subtask into the memory stream is "
                "a natural follow-up: pi05 hands the fast weights a ready-made progress summary."
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
            tbptt_backward_fn=self.tbptt_backward_fn,
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

        Note:
            The rule is shape-based, so a trajectory-level tensor that happens
            to be shaped ``(B, T)`` — say a token sequence whose length equals
            the timestep count — would be misread as per-timestep. No such key
            exists in today's batches, and the ambiguity should be removed by
            having the dataloader label its per-timestep keys explicitly when
            the sequence data path lands.
        """
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
    def _sequence_length(batch: dict[str, Tensor]) -> int | None:
        """Detects a trajectory batch and returns its timestep count.

        Args:
            batch: Candidate training batch.

        Returns:
            ``T`` when ``actions`` carries a leading timestep axis, else None.
        """
        actions = batch.get("actions")
        if actions is None or actions.ndim != 4:
            return None
        return actions.shape[1]
