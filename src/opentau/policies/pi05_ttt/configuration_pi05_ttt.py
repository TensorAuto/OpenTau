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

"""Configuration for the π₀.₅ + Test-Time-Training policy.

:class:`PI05TTTConfig` extends :class:`~opentau.policies.pi05.configuration_pi05.PI05Config`
with the fields RoboTTT (`arXiv:2607.15275 <https://arxiv.org/abs/2607.15275>`_)
adds on top of a flow-matching VLA: the TTT layer's own hyperparameters, the
register-token count, the tanh gate's initial value, and the sequence-training
knobs (context length and TBPTT segment length).

Every added field has a default that reproduces the paper's setting where the
paper states one, so a config that only sets ``policy.type=pi05_ttt`` is the
paper's recipe scaled to π₀.₅'s narrower action expert.
"""

import logging
from dataclasses import dataclass

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_ttt")
@dataclass
class PI05TTTConfig(PI05Config):
    """Configuration for π₀.₅ with Test-Time-Training memory in the action expert.

    Inherits every π₀.₅ field unchanged and adds the TTT-specific ones below.
    A TTT layer is inserted after the attention block of each action-expert
    layer, gated by ``tanh(alpha)`` with ``alpha`` initialized to
    ``ttt_gate_init``, so the TTT *memory* contributes nothing at step 0.

    That is not the same as reproducing stock π₀.₅: with ``n_register_tokens >
    0`` the register block is an ungated change to the expert's input, taking
    softmax mass from the action tokens even though the table is zero-initialized
    and the action block's position ids are held fixed. Only
    ``n_register_tokens=0`` is bit-identical to stock π₀.₅.

    Note on the meaning of a "timestep": in RoboTTT a timestep is one control
    step at 30 Hz, and 8K timesteps is about five minutes. Here a timestep is
    **one policy call**, which decodes a ``chunk_size``-step action chunk. At
    ``chunk_size=50`` and 30 Hz control, one call covers up to ~1.7 s of
    motion, so ``sequence_length=512`` already spans several minutes of
    wall-clock. That buys wall-clock coverage at a *constant* number of
    fast-weight updates — the paper's context-scaling gains are measured in
    update steps, so do not read ``sequence_length=512`` here as equivalent to
    the paper's 512.

    Args:
        n_register_tokens: Number of learned register tokens prepended to the
            action-expert token stream at each timestep. They attend to
            everything in their own timestep (including the VLM prefix through
            cross-attention) and are the only carrier of vision-language and
            state information into the memory, since the VL tokens themselves
            bypass TTT for cost reasons. The paper uses 16 and reports them as
            worth +18% *only* in combination with TTT.
        ttt_num_heads: Number of TTT heads. Must divide the action expert's
            hidden size (1024 for π₀.₅), and the resulting head dim must be
            even for rotary embeddings.
        ttt_mlp_hidden_multiplier: Fast-MLP hidden width as a multiple of the
            TTT head dim. The reference video implementation uses 4; 2 keeps
            the added parameters near the paper's ~2% of backbone per layer
            instead of doubling a 300M-parameter action expert.
        ttt_base_lr: Base inner-loop learning rate, modulated per token by a
            learned sigmoid gate. The paper uses 0.1.
        ttt_rope_theta: RoPE base for the TTT layer's 1-D positional embedding
            over the flattened timestep axis. The paper uses 10000.
        ttt_gate_init: Initial value of the per-channel tanh gate. The paper
            uses 0.001, small enough that the pretrained policy is preserved.
        ttt_scan_checkpoint_group_size: Gradient-checkpoint group size for the
            TTT mini-batch scan, in timesteps. 0 disables checkpointing. This
            trades recompute for activation memory *within* a TBPTT segment and
            is independent of ``gradient_checkpointing``, which wraps the
            decoder layers.
        sequence_length: Number of consecutive policy calls in one training
            sequence — the training context length. 1 reduces the policy to
            stock π₀.₅ plus an unused memory path.
        tbptt_segment_length: Number of timesteps per truncated-BPTT segment.
            Fast weights are carried across boundaries; their gradients are
            not. Peak activation memory scales with this, not with
            ``sequence_length``. Must divide ``sequence_length``.
        train_ttt_only: Freeze every pretrained parameter and train only the
            newly added ones (TTT layers, gates, register tokens). This is the
            paper's pretraining stage; its post-training stage unfreezes
            everything, so set this to False for task-specific fine-tuning.
    """

    n_register_tokens: int = 16

    ttt_num_heads: int = 8
    ttt_mlp_hidden_multiplier: int = 2
    ttt_base_lr: float = 0.1
    ttt_rope_theta: float = 10000.0
    ttt_gate_init: float = 0.001
    ttt_scan_checkpoint_group_size: int = 0

    # Defaults to 1 — the only length today's dataloader can supply. A
    # default-constructed config otherwise raises on the batch shape every
    # existing loader emits, which makes the class unusable without an override.
    sequence_length: int = 1
    tbptt_segment_length: int = 1

    train_ttt_only: bool = True

    # Gradient-checkpoint each TBPTT segment's forward, so activation memory
    # scales with `tbptt_segment_length` rather than `sequence_length`. Measured
    # on an RTX 3090 at chunk 10: 6.75 GiB fixed + 0.304 GiB per timestep
    # without it, so a median LIBERO episode at stride 1 (T~131) would need
    # ~47 GiB against a 23.57 GiB card. Costs one extra forward per segment.
    #
    # Off by default: it changes nothing numerically, but it is pure overhead at
    # the sequence lengths that already fit.
    checkpoint_tbptt_segments: bool = False
    # Which Euler step's fast-weight update a rollout adopts ("one mini batch per
    # inference"). "last" (historic default) uses the final step: nearly-clean,
    # self-generated action tokens — a noise level the training marginal
    # (tau ~ Beta(1.5, 1), mass toward pure noise) almost never visits, so the
    # memory ingests out-of-distribution inputs exactly when the gates open.
    # "first" uses the first step: pure-noise action tokens, the mode of the
    # training marginal, so the update is driven by the observation (registers +
    # proprioception) the way training drove it. Inference-only semantics; safe
    # to flip on an existing checkpoint.
    ttt_inference_update_adoption: str = "last"
    # Inference-only diagnostics for isolating a trained checkpoint's damage
    # vector without retraining. `ttt_inference_alpha_scale` multiplies every
    # tanh gate at rollout (0.0 = memory contribution off; training unaffected).
    # `ttt_inference_zero_registers` feeds the zero-init register table instead
    # of the trained one at rollout, reproducing the step-0 register condition.
    # Both at their "off" pair (0.0, True) reproduce the *step-0 wrapper
    # condition* (zero-init registers, silent memory) — close to, but not
    # bit-identical with, the stock base policy: zeroed registers still occupy
    # attention slots (only `n_register_tokens=0` removes them). NOTE these
    # knobs are read whenever the module is in eval mode — which includes
    # in-training validation — so leave them at defaults in training configs.
    ttt_inference_alpha_scale: float = 1.0
    ttt_inference_zero_registers: bool = False

    # `PI05TTTPolicy.supports_torch_compile` is False (the sequence path drives a
    # Python-level loop over TBPTT segments), so inheriting π₀.₅'s `True` meant
    # every default run hit `maybe_compile_for_training`'s warn-and-skip path.
    use_torch_compile: bool = False

    def __post_init__(self):
        """Validates the TTT fields on top of π₀.₅'s own validation.

        Raises:
            ValueError: If any TTT field is out of range, or if
                ``tbptt_segment_length`` does not divide ``sequence_length``.
        """
        super().__post_init__()

        if self.n_register_tokens < 0:
            raise ValueError(f"n_register_tokens must be >= 0, got {self.n_register_tokens}")
        if self.ttt_num_heads <= 0:
            raise ValueError(f"ttt_num_heads must be > 0, got {self.ttt_num_heads}")
        # The documented contract, enforced here rather than at model-build
        # time: a bad value used to survive config validation and die only after
        # a 3.4B-parameter model had been constructed. `proj_width` is the
        # action expert's hidden size.
        # NOTE: validated against `proj_width`, while `TTTMLPLayer` is actually
        # built from `gemma_expert_config.hidden_size`. Those agree for every
        # shipped configuration (both are 1024) but they are not the same knob,
        # and the expert config is not reachable from here — it is constructed
        # inside `PaliGemmaWithExpertConfig`. `_attach_ttt_layers` re-checks
        # against the real width, so a divergence fails at model build with a
        # clear message rather than silently.
        if self.proj_width % self.ttt_num_heads != 0:
            raise ValueError(
                f"ttt_num_heads={self.ttt_num_heads} must divide the action expert's width "
                f"proj_width={self.proj_width}"
            )
        if (self.proj_width // self.ttt_num_heads) % 2 != 0:
            raise ValueError(
                f"ttt_num_heads={self.ttt_num_heads} gives an odd TTT head dim "
                f"({self.proj_width // self.ttt_num_heads}); rotary embeddings need it even"
            )
        if self.ttt_mlp_hidden_multiplier <= 0:
            raise ValueError(f"ttt_mlp_hidden_multiplier must be > 0, got {self.ttt_mlp_hidden_multiplier}")
        if self.ttt_base_lr <= 0:
            raise ValueError(f"ttt_base_lr must be > 0, got {self.ttt_base_lr}")
        if self.ttt_rope_theta <= 0:
            raise ValueError(f"ttt_rope_theta must be > 0, got {self.ttt_rope_theta}")
        if self.ttt_scan_checkpoint_group_size < 0:
            raise ValueError(
                f"ttt_scan_checkpoint_group_size must be >= 0, got {self.ttt_scan_checkpoint_group_size}"
            )
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.sequence_length}")
        if self.ttt_inference_update_adoption not in ("last", "first"):
            raise ValueError(
                f"ttt_inference_update_adoption must be 'last' or 'first', got "
                f"{self.ttt_inference_update_adoption!r}."
            )
        if self.tbptt_segment_length <= 0:
            raise ValueError(f"tbptt_segment_length must be > 0, got {self.tbptt_segment_length}")
        if self.sequence_length % self.tbptt_segment_length != 0:
            # A ragged final segment is not wrong in principle, but it would
            # make the number of segments (and so the number of backward calls
            # per step) depend on the sequence length, which must stay
            # identical across ranks — see CLAUDE.md rule 5.
            raise ValueError(
                f"tbptt_segment_length={self.tbptt_segment_length} must divide "
                f"sequence_length={self.sequence_length}"
            )

        # `train_ttt_only` freezes everything *except* the TTT parameters, while
        # `train_state_action_representation_only` / `train_vision_encoder_only`
        # freeze the TTT parameters (they are default-deny sweeps re-applied in
        # `PI05TTTFlowMatching.__init__`). The two sets are complementary, so
        # combining them freezes the entire model: measured at 0 trainable
        # tensors, 0 elements. pi05 raises for exactly this class of conflict in
        # `validate_state_action_representation_only_config`; do the same rather
        # than let a run start with nothing to optimize.
        exclusive = [
            name
            for name, enabled in (
                ("train_state_action_representation_only", self.train_state_action_representation_only),
                ("train_vision_encoder_only", self.train_vision_encoder_only),
                ("train_expert_only", self.train_expert_only),
            )
            if enabled
        ]
        if self.train_ttt_only and exclusive:
            raise ValueError(
                f"train_ttt_only=True is mutually exclusive with {' and '.join(exclusive)}: "
                "train_ttt_only trains only the TTT parameters while those flags freeze exactly "
                "those parameters, so together they leave nothing trainable. Pick one."
            )

        if self.ttt_gate_init != 0.0 and abs(self.ttt_gate_init) > 0.1:
            logging.warning(
                "ttt_gate_init=%s is far from the paper's 0.001. The gate exists to keep the "
                "randomly initialized TTT layers from perturbing a pretrained action expert on "
                "step 0; a large initial value forfeits that protection.",
                self.ttt_gate_init,
            )

        if self.sequence_length == 1:
            logging.warning(
                "sequence_length=1 trains the TTT layers on a single timestep per sequence. "
                "Every TTT parameter is still on the autograd graph and receives gradients, so "
                "the plumbing is exercised and distributed training is well-formed — but the "
                "fast weights only ever take one update per sequence, so the memory cannot "
                "learn anything that spans timesteps. This is the only value today's dataloader "
                "can supply; raise it once the data path emits trajectory sequences."
            )

    @property
    def n_expert_tokens_per_timestep(self) -> int:
        """Number of action-expert tokens per timestep, i.e. the TTT mini-batch size.

        One inner gradient step consumes exactly this many tokens, so one
        policy call performs exactly one fast-weight update.

        Returns:
            ``n_register_tokens + chunk_size``.
        """
        return self.n_register_tokens + self.chunk_size
