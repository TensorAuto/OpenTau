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

"""Utility functions for policy implementations in OpenTau.

This module provides helper functions for managing data queues, inspecting model
properties (device, dtype), determining output shapes, and logging model loading
information.
"""

import logging
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce
from torch import Tensor, nn


def assert_gemma3_input_resolution(input_image_size: tuple[int, int] | None, tower_image_size: int) -> None:
    """Fail fast when a Gemma3-family policy would run at a native resolution.

    ``Gemma3MultiModalProjector`` hard-codes a square ``patches_per_image =
    image_size // patch_size`` reshape + avg-pool, so a non-default patch
    grid crashes deep inside the projector with an unrelated-looking reshape
    error. The pi06/pi07 constructors call this to surface the real
    diagnosis instead. (The PaliGemma-family policies support native
    resolutions.)

    Args:
        input_image_size: ``(H, W)`` the vision tower will receive
            (``PreTrainedConfig.input_image_size``), or ``None`` when
            underivable (no check possible).
        tower_image_size: The Gemma 3 SigLIP config's square ``image_size``.

    Raises:
        ValueError: When ``input_image_size`` is known and differs from the
            tower's square resolution.
    """
    if input_image_size is not None and tuple(input_image_size) != (tower_image_size, tower_image_size):
        raise ValueError(
            f"input resolution {tuple(input_image_size)} (resize_imgs_with_padding, or the bound "
            f"image-feature resolution when it is null) != the Gemma 3 vision tower's "
            f"image_size ({tower_image_size}). Native resolutions are not yet supported "
            "for the Gemma3-family policies; set resize_imgs_with_padding (and resolution) to "
            f"({tower_image_size}, {tower_image_size})."
        )


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
) -> dict[str, deque]:
    """Populates queues with batch data.

    If a queue is not full (e.g. at the start of an episode), it is filled by repeating
    the first observation. Otherwise, the latest observation is appended.

    Args:
        queues: A dictionary of deques to be populated.
        batch: A dictionary containing the data to add to the queues.
        exclude_keys: A list of keys to exclude from population. Defaults to None.

    Returns:
        dict[str, deque]: The updated dictionary of queues.
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note:
        Assumes that all parameters have the same device.

    Args:
        module: The PyTorch module to inspect.

    Returns:
        torch.device: The device of the module's parameters.
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note:
        Assumes that all parameters have the same dtype.

    Args:
        module: The PyTorch module to inspect.

    Returns:
        torch.dtype: The data type of the module's parameters.
    """
    return next(iter(module.parameters())).dtype


# Full parameter-name suffixes of the SigLIP patch/position embeddings that
# PaliGemmaWithExpertModel.to_bfloat16_like_physical_intelligence pins to float32.
_SIGLIP_FLOAT32_PARAM_SUFFIXES = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
)


def maybe_compile_sample_actions(policy, sample_actions, device_hint=None):
    """Compile a policy's ``sample_actions`` unless the policy carries rollout state.

    Every serving and scoring entry point wants the compiled sampler, and each
    one previously called ``attempt_torch_compile`` directly. That is a manual
    sweep, and manual sweeps over ``scripts/`` are exactly what CLAUDE.md rule 6
    records going wrong — a six-file pass missed ``robocasa/server.py``, and a
    later four-site pass missed it again. Routing every site through one helper
    makes the gate single-sourced and lets an AST test pin that no script calls
    the raw helper on a sampler (see
    ``tests/policies/test_rollout_state_entry_points.py``).

    The gate itself: a policy whose ``carries_rollout_state`` is True mutates
    Python-level state inside ``sample_actions`` — pi05_ttt carries fast weights
    and an integer token position that feeds RoPE. Best case that recompiles per
    distinct position and then falls back to eager forever; worst case the
    position specializes into the graph and the phase silently freezes.

    Args:
        policy: The policy owning ``sample_actions``.
        sample_actions: The bound method to compile.
        device_hint: Optional device hint forwarded to ``attempt_torch_compile``.

    Returns:
        Either a compiled callable or ``sample_actions`` unchanged.
    """
    from opentau.utils.utils import attempt_torch_compile

    if getattr(policy, "carries_rollout_state", False):
        logging.info(
            "Skipping torch.compile of sample_actions: %s carries recurrent rollout state.",
            type(policy).__name__,
        )
        return sample_actions
    return attempt_torch_compile(sample_actions, device_hint=device_hint)


def to_dtype_preserving_siglip_float32(
    module: nn.Module,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> nn.Module:
    """``module.to(...)`` that preserves the float32-pinned SigLIP patch/position embeddings.

    The SigLIP patch-embedding conv and position-embedding table are pinned to float32 at
    build time (openpi parity, see
    ``PaliGemmaWithExpertModel.to_bfloat16_like_physical_intelligence``): Big Vision keeps
    "patch extraction and posemb in float32", and they are the only float32-master vision
    weights. A blanket ``policy.to(torch.bfloat16)`` in a serving / inference entry point
    would round those tables back to bfloat16 and permanently lose precision. This wrapper
    snapshots them before the cast and restores them (float32, on the target device)
    afterwards, so the mixed float32-embeddings / bfloat16-encoder state the forward dtype
    bridges expect is preserved.

    Only parameters that are float32 *before* the cast are preserved, so this is a no-op for a
    uniform-bfloat16 tower (e.g. the Gemma3 pi06/pi07 policies, whose embeddings are not
    pinned) and for a float32 cast. This is inference/serving-only: it must not be used on the
    distributed-training cast path, where re-introducing float32 params after the cast would
    change how DeepSpeed/DDP partition parameters.

    Known limitation (only the *weights* are kept float32, not the *input*): the serving entry
    points still feed a bfloat16 image — e.g. ``grpc/server.py`` casts the decoded image to the
    policy dtype — so the patch conv runs on bfloat16-precision pixels. HF
    ``SiglipVisionEmbeddings.forward`` upcasts ``pixel_values`` to the (float32) weight dtype, so
    there is no dtype mismatch, but the JAX recipe is a float32 *image* into the patch conv
    ("do patch extraction and posemb in float32"). Empirically the bfloat16 image costs ~1%
    (~1.5 max) on ``embed_image`` — larger than the ~0.14% weight-rounding this fix removes — so
    full JAX-recipe fidelity would additionally require keeping the served image float32. Tracked
    as a known limitation alongside issue #483.

    Args:
        module: The policy (or any module) to cast in place.
        dtype: Target dtype for the blanket cast.
        device: Optional target device for the cast.

    Returns:
        The same module, cast in place.
    """
    saved = {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if param.dtype == torch.float32
        and any(name.endswith(suffix) for suffix in _SIGLIP_FLOAT32_PARAM_SUFFIXES)
    }
    if device is not None:
        module.to(device=device, dtype=dtype)
    else:
        module.to(dtype=dtype)
    if saved:
        params_by_name = dict(module.named_parameters())
        for name, tensor in saved.items():
            param = params_by_name.get(name)
            if param is not None:
                param.data = tensor.to(device=param.data.device)
    return module


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module: A PyTorch module.
        input_shape: A tuple representing the input shape, e.g., (batch_size, channels, height, width).

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


@dataclass
class PerSampleLoss:
    """Per-sample decomposition of a masked loss, with the batch dim kept.

    ``sum`` and ``count`` are both ``(B,)`` and hold, for each sample, the
    summed unmasked loss and the number of unmasked slots that fed it. The
    masked *mean* for any group of samples is ``Σsum / Σcount`` — carrying the
    (numerator, denominator) pair rather than a per-sample mean is what lets a
    caller (e.g. the validation loop) regroup samples by provenance and recover
    an exact masked mean per group. Averaging per-sample means instead would
    double-normalize and weight a 1-slot sample the same as a 200-slot one.
    """

    sum: Tensor
    count: Tensor

    def __add__(self, other: "PerSampleLoss") -> "PerSampleLoss":
        # Pool several loss components (e.g. discrete-action CE + response CE)
        # into one (numerator, denominator); the pooled per-slot mean is then
        # Σ(sum_i) / Σ(count_i) over the pooled slots.
        return PerSampleLoss(sum=self.sum + other.sum, count=self.count + other.count)


def make_action_dim_mask(
    real_action_dim: Tensor | None,
    max_action_dim: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Per-sample bool mask over action dims; True for real dims, False for zero-pad.

    Heterogeneous datasets are zero-padded to ``max_action_dim`` along the last
    action axis to keep batches rectangular, but the flow-matching MSE on the
    velocity field should only score real dims for each sample. This helper
    builds the per-dim mask that callers AND into their existing per-timestep
    mask before reducing.

    Args:
        real_action_dim: Optional ``(B,)`` long tensor of the real (pre-pad)
            action dimensionality for each sample (the batch key emitted by
            ``LeRobotDataset._to_standard_data_format``). When ``None``, the
            returned mask is all-True so the dim-mask AND in the caller's
            reduction is a no-op (pi0 additionally harmonized its `.mean()`
            to `sum / mask.sum()` in this PR — see the PR body's "pi0
            loss-magnitude shift" note; the dim-mask itself is still a
            no-op when ``real_action_dim`` is None).
        max_action_dim: The padded action dim (last-axis length of ``actions``).
        batch_size: Used to construct the all-True fallback shape; when
            ``real_action_dim`` is provided, must match
            ``real_action_dim.shape[0]``.
        device: Output device.

    Returns:
        ``(batch_size, max_action_dim)`` bool tensor.
    """
    if real_action_dim is None:
        return torch.ones((batch_size, max_action_dim), dtype=torch.bool, device=device)
    if real_action_dim.shape != (batch_size,):
        # Catch caller drift (e.g. a sliced `real_action_dim` passed with the
        # original `batch_size`) at the helper boundary — silent shape
        # mismatches propagate into broadcast errors deep in the loss reduction.
        raise ValueError(
            f"real_action_dim.shape {tuple(real_action_dim.shape)} does not match "
            f"batch_size={batch_size}; expected ({batch_size},)"
        )
    arange = torch.arange(max_action_dim, device=device)
    return rearrange(arange, "d -> 1 d") < rearrange(real_action_dim.to(device=device), "b -> b 1")


def flow_matching_masked_mse(
    u_t: Tensor,
    v_t: Tensor,
    *,
    max_action_dim: int,
    prefix_mask: Tensor | None = None,
    actions_is_pad: Tensor | None = None,
    real_action_dim: Tensor | None = None,
    return_per_sample: bool = False,
) -> Tensor | tuple[Tensor, PerSampleLoss]:
    """Masked MSE for flow-matching velocity-field training.

    Shared across pi05, pi05_mem, pi06, pi07 (low_level), and pi07_paligemma
    (low_level). Builds a `(B, chunk_size, max_action_dim)` mask that AND-s
    together up to three conditions and reduces ``F.mse_loss(u_t, v_t)``
    over the unmasked slots:

      1. **Frozen-prefix (RTI delay):** ``~prefix_mask`` — False where the
         model isn't asked to predict (the action prefix is the actually
         executed action from a previous inference, frozen as ground truth).
         Pass ``None`` to disable (non-RTI policies); the helper builds an
         all-False prefix mask internally so every step is supervised.
      2. **Per-timestep chunk padding:** ``~actions_is_pad`` — False where
         the action chunk extends past episode end. Pass ``None`` to skip.
         Also covers VQA-style items (``actions_is_pad`` all-True ⇒ loss = 0).
      3. **Per-sample real action dim:** built from ``real_action_dim`` via
         :func:`make_action_dim_mask`. False on the zero-pad tail dims of
         each sample. Pass ``None`` to score all ``max_action_dim`` columns.

    Args:
        u_t: Target velocity field, shape ``(B, chunk_size, D)`` (D ≥ max_action_dim).
        v_t: Predicted velocity field, same shape as ``u_t``.
        max_action_dim: Number of leading action dims to score against; trailing
            dims are dropped before reduction. Keyword-only.
        prefix_mask: Optional bool ``(B, chunk_size)`` — True where the step is
            frozen (RTI delay). ``None`` ⇒ all-False (non-RTI behavior).
        actions_is_pad: Optional bool ``(B, chunk_size)`` — True where the
            action chunk is padded (no real action target). ``None`` ⇒ all-False.
        real_action_dim: Optional long ``(B,)`` — real (pre-pad) action dim per
            sample. ``None`` ⇒ all-True (every dim is real).
        return_per_sample: When True, additionally return a :class:`PerSampleLoss`
            holding the per-sample ``(Σ over masked slots, #masked slots)`` so the
            caller can regroup the loss by provenance. The scalar is computed
            exactly as in the default path (bit-identical), so toggling this flag
            never perturbs the training reduction.

    Returns:
        Scalar tensor (masked mean of ``(u_t - v_t)**2`` over the unmasked slots)
        when ``return_per_sample`` is False; otherwise ``(scalar, PerSampleLoss)``
        where the per-sample ``sum``/``count`` are over the same masked slots
        (so each sample's mean is ``sum / count``).
    """
    mse_loss = F.mse_loss(u_t, v_t, reduction="none")
    bsz, chunk_size = u_t.shape[:2]
    if prefix_mask is None:
        prefix_mask = torch.zeros((bsz, chunk_size), dtype=torch.bool, device=u_t.device)
    postfix_mask = rearrange(torch.logical_not(prefix_mask), "b c -> b c 1")
    if actions_is_pad is not None:
        in_episode_bound = rearrange(~actions_is_pad, "b c -> b c 1")
        postfix_mask = torch.logical_and(postfix_mask, in_episode_bound)
    mse_loss = mse_loss[:, :, :max_action_dim]
    dim_mask = make_action_dim_mask(real_action_dim, max_action_dim, batch_size=bsz, device=u_t.device)
    full_mask = postfix_mask & rearrange(dim_mask, "b d -> b 1 d")
    masked = mse_loss * full_mask
    scalar = masked.sum() / (full_mask.sum() + 1e-8)
    if not return_per_sample:
        return scalar
    per_sample = PerSampleLoss(
        sum=reduce(masked, "b c d -> b", "sum"),
        count=reduce(full_mask.float(), "b c d -> b", "sum"),
    )
    return scalar, per_sample


def ce_per_sample(masked_ce: Tensor, valid_mask: Tensor) -> PerSampleLoss:
    """Per-sample numerator/denominator for a masked token cross-entropy.

    Policies compute their CE as ``F.cross_entropy(..., reduction="none")``
    reshaped to ``(B, S)`` and zeroed at pad positions, then reduce it with
    ``.mean()`` to a scalar. This helper takes that same pad-zeroed ``(B, S)``
    tensor plus the per-token validity mask and returns the per-sample
    ``(Σ over valid tokens, #valid tokens)``, so a caller can pool CE per
    provenance group as ``Σsum / Σcount`` — the mean cross-entropy per valid
    token. Multiple CE components (e.g. discrete-action + response) pool by
    adding their :class:`PerSampleLoss` objects.

    Note this normalizes by *valid* token count, unlike the legacy scalar
    ``.mean()`` which divides by the full ``B * S`` (pad slots included); the
    per-group breakdown is therefore over valid tokens only.

    Args:
        masked_ce: ``(B, S)`` cross-entropy already zeroed at pad positions.
        valid_mask: ``(B, S)`` bool, True at non-pad (scored) tokens.

    Returns:
        ``PerSampleLoss`` whose ``sum`` and ``count`` are ``(B,)``.
    """
    return PerSampleLoss(
        sum=reduce(masked_ce, "b s -> b", "sum"),
        count=reduce(valid_mask.float(), "b s -> b", "sum"),
    )


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model.

    Args:
        missing_keys: Keys that were expected but not found.
        unexpected_keys: Keys that were found but not expected.
    """
    if missing_keys:
        # DO NOT UPDATE THIS MESSAGE WITHOUT UPDATING THE REGEX IN .gitlab/scripts/check_pi0_state_keys.py
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        # DO NOT UPDATE THIS MESSAGE WITHOUT UPDATING THE REGEX IN .gitlab/scripts/check_pi0_state_keys.py
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")


def freeze_policy_level_params_for_vision_only(
    policy_module: nn.Module, with_expert_module: nn.Module
) -> None:
    """Freeze the policy-level (outer) parameters for ``train_vision_encoder_only``.

    The vision/video encoder — the SigLIP / Gemma3 / Qwen3-VL tower plus its
    multimodal projector — lives *inside* ``with_expert_module``; its
    ``set_requires_grad`` has already left exactly that pathway trainable and frozen
    the LLM backbone, action expert and discrete heads. What remains are the
    *policy-level* projections that live on the enclosing flow-matching module
    (``state_proj`` / ``action_in_proj`` / ``action_out_proj`` / ``time_mlp_*`` /
    ``action_time_mlp_*`` / ``adarms_proj`` / optional modality embeddings). Those
    must be frozen so that **only** the vision/video encoder trains.

    Rather than enumerate every projection per policy (a single omission would let a
    head silently keep training and quietly break the "vision only" contract), this
    freezes every outer parameter that is neither part of ``with_expert_module`` nor
    a video-encoder ``motion_module`` — the pi05_mem RLDX encoder's own temporal
    block, which *is* part of the video encoder and stays trainable.

    Args:
        policy_module: the enclosing flow-matching / planner ``nn.Module``.
        with_expert_module: the inner ``*WithExpertModel`` submodule whose own
            ``set_requires_grad`` has already configured the vision pathway.
    """
    protected = {id(p) for p in with_expert_module.parameters()}
    for name, param in policy_module.named_parameters():
        if id(param) in protected or "motion_module" in name:
            continue
        param.requires_grad = False


#: Policy-level projections that bridge a *dataset's* raw state/action vector space
#: and the model's hidden space. Their in- or out-features are ``max_state_dim`` /
#: ``max_action_dim``, so they are the only outer projections whose shape is a
#: function of the robot rather than of the architecture — which is exactly what
#: makes them dataset-specific. Deliberately excludes ``time_mlp_in``/``time_mlp_out``
#: (and pi0's ``action_time_mlp_*``, cosmos3's ``adarms_proj``): those are
#: hidden-size-to-hidden-size, are fed a sinusoidal embedding of flow-matching time
#: alone, and condition the action expert — the same line ``per_group_projection``
#: already draws when it gives per-(robot_type, control_mode) copies to these three
#: and notes that "time_mlp_in/out stay shared".
STATE_ACTION_PROJECTION_ATTRS: tuple[str, ...] = ("state_proj", "action_in_proj", "action_out_proj")

#: With-expert-level modules holding the discrete-action (FAST token) representation:
#: the input embedding table and the untied head producing logits over the action
#: vocabulary. Both are pure functions of the fitted FAST tokenizer's vocabulary and
#: are invalidated by the same event (a tokenizer re-fit), so they move as one unit —
#: as every existing freeze flag already treats them.
DISCRETE_ACTION_REPRESENTATION_ATTRS: tuple[str, ...] = ("discrete_action_embedding", "da_head")


def validate_state_action_representation_only_config(
    config: object, *, policy_name: str, has_discrete_actions: bool
) -> None:
    """Validate ``train_state_action_representation_only`` on a policy config.

    Shared by every policy config's ``__post_init__`` so the rules cannot drift, and
    so a policy added later gets them by calling one function rather than by copying
    a block (the copy-a-block approach is how ``train_vision_encoder_only`` ended up
    reaching the pi07_paligemma planner but not the pi07 one).

    Enforces mutual exclusion with the two other "train only X" modes, and warns —
    rather than raises — in the two situations where the flag is legal but trains
    less than a reader might expect:

    * **No discrete-action pathway** (pi0, cosmos3/cosmos3_nano): bucket (1) is empty
      and the flag degenerates to "train the state/action projections only". That is
      a real and useful mode, so it is allowed, but a user who set the flag expecting
      embedding training must be told.
    * **Knowledge insulation on** (the default): KI detaches the prefix KV cache
      before the action expert, which splits the trainable set across two losses —
      ``state_proj`` and the discrete-action modules are reachable only from CE, while
      ``action_in_proj``/``action_out_proj`` are reachable only from MSE. Zeroing
      either ``loss_weighting`` term starves one group entirely.

    Args:
        config: the policy config being validated (read via ``getattr`` so configs
            that do not declare a given sibling flag are handled).
        policy_name: policy name used in the messages, e.g. ``"pi05"``.
        has_discrete_actions: whether this policy has a discrete-action pathway at
            all. A static per-policy fact, passed explicitly so the empty-bucket
            warning cannot be silently skipped by an attribute lookup returning None.

    Raises:
        ValueError: if combined with ``train_expert_only`` or ``train_vision_encoder_only``.
    """
    if not getattr(config, "train_state_action_representation_only", False):
        return

    for conflicting in ("train_expert_only", "train_vision_encoder_only"):
        if getattr(config, conflicting, False):
            raise ValueError(
                f"`train_state_action_representation_only=True` and `{conflicting}=True` are mutually "
                f"exclusive on {policy_name}: this mode freezes the VLM, the vision encoder and the "
                f"action expert, so it cannot be combined with a mode that trains one of them."
            )

    if not has_discrete_actions:
        logging.warning(
            "train_state_action_representation_only=True on %s, which has no discrete-action "
            "pathway: there is no discrete-action embedding or logit head to train, so this run "
            "trains ONLY the state/action projections (state_proj / action_in_proj / "
            "action_out_proj). That is a supported mode, but if you expected the discrete-action "
            "embeddings to train, you want one of the policies that has a FAST discrete-action "
            "branch (pi05, pi05_mem, pi06, pi07_low_level, pi07_paligemma_low_level).",
            policy_name,
        )

    if getattr(config, "knowledge_insulation", False):
        logging.warning(
            "train_state_action_representation_only=True with knowledge_insulation=True on %s: "
            "knowledge insulation detaches the VLM prefix KV cache before the action expert, so "
            "the trainable parameters split across two losses — state_proj and the discrete-action "
            "embedding/head receive gradient ONLY from the CE loss, while action_in_proj and "
            "action_out_proj receive gradient ONLY from the flow-matching MSE loss. If either "
            "`loss_weighting` term is 0, that half of the trainable set gets no gradient at all.",
            policy_name,
        )


def _param_ids_of_attrs(module: nn.Module, attrs: tuple[str, ...]) -> set[int]:
    """Collect ``id()`` of every parameter under ``attrs`` that exists on ``module``.

    Resolution is by *module object*, never by parameter name. Name matching is
    unsafe here: ``Normalize``/``Unnormalize`` register their per-dataset statistics
    as ``nn.Parameter(..., requires_grad=False)`` (see :mod:`opentau.policies.normalize`),
    so they appear in ``named_parameters()`` as ``normalize_*.buffer_state`` /
    ``buffer_actions``. Any pattern matching ``state`` or ``action`` would flip those
    normalization statistics into learned parameters, and the optimizer factory —
    which selects on ``requires_grad`` alone — would happily optimize them.

    Args:
        module: the module to resolve attributes against.
        attrs: attribute names to look up; missing attributes are skipped, since
            which projections/heads exist varies by policy (pi06 has no
            ``state_proj`` at all, pi05 only builds one for ``state_type="continuous"``).

    Returns:
        The set of ``id()`` values of the collected parameters.
    """
    ids: set[int] = set()
    for attr in attrs:
        submodule = getattr(module, attr, None)
        if isinstance(submodule, nn.Module):
            ids.update(id(p) for p in submodule.parameters())
    return ids


def freeze_with_expert_params_for_state_action_representation_only(
    with_expert_module: nn.Module,
) -> list[str]:
    """Freeze the whole VLM/vision/expert stack, keeping only the discrete-action
    representation trainable, for ``train_state_action_representation_only``.

    This is the inner half of the flag: it configures the ``*WithExpertModel``
    wrapper. Everything it owns — the VLM backbone and its tied ``lm_head``, the
    SigLIP/Gemma3/Qwen3-VL vision tower, the multimodal projector, the action expert
    and its AdaRMS conditioning — is frozen, and only
    :data:`DISCRETE_ACTION_REPRESENTATION_ATTRS` is left trainable.

    The polarity is deliberately *default-deny*: every parameter is set to
    ``requires_grad = id(param) in keep`` in one pass, rather than freezing an
    enumerated list of submodules. Enumerating is how a head silently keeps training
    when a new module is added — and it is why ``freeze_vision_encoder`` leaves the
    multimodal projector trainable to this day (it freezes ``vision_tower`` only).
    Here the projector is covered for free.

    Args:
        with_expert_module: the ``*WithExpertModel`` wrapper (PaliGemma, Gemma 3 or
            Qwen3-VL flavour).

    Returns:
        Sorted names of the parameters left trainable, for logging and tests. Empty
        for policies with no discrete-action pathway (pi0, cosmos3), which is a
        supported — and warned-about — degenerate case.
    """
    keep = _param_ids_of_attrs(with_expert_module, DISCRETE_ACTION_REPRESENTATION_ATTRS)
    trainable: list[str] = []
    for name, param in with_expert_module.named_parameters():
        param.requires_grad = id(param) in keep
        if param.requires_grad:
            trainable.append(name)
    return sorted(trainable)


def set_with_expert_train_mode_for_state_action_representation_only(
    with_expert_module: nn.Module, mode: bool
) -> None:
    """Pin every frozen submodule of the wrapper to ``eval()`` under the flag.

    ``requires_grad=False`` stops weights from updating but does **not** stop
    ``nn.Dropout`` from firing inside the frozen trunk, and ``update_policy`` calls
    ``policy.train()`` on every step. Under this flag the trainable set is a single
    embedding table plus one linear head, so trunk dropout is pure variance injected
    between the frozen features and the tiny head that reads them — with no
    regularization benefit, since no trunk weight can move. The wrapper's own
    ``self.dropout`` (applied to attention and MLP outputs inside the decoder loop)
    is therefore pinned to eval as well, which means ``config.dropout`` has no effect
    under this flag.

    Every direct child is evaluated and only the discrete-action modules follow
    ``mode`` — correct by construction, because under this flag everything that is
    not a discrete-action module is frozen.

    Args:
        with_expert_module: the ``*WithExpertModel`` wrapper.
        mode: the training mode requested by the caller's ``train(mode)``.
    """
    for child in with_expert_module.children():
        child.eval()
    for attr in DISCRETE_ACTION_REPRESENTATION_ATTRS:
        submodule = getattr(with_expert_module, attr, None)
        if isinstance(submodule, nn.Module):
            submodule.train(mode)


def freeze_policy_level_params_for_state_action_representation_only(
    policy_module: nn.Module, with_expert_module: nn.Module
) -> list[str]:
    """Freeze the policy-level (outer) parameters for
    ``train_state_action_representation_only``, keeping only the state/action
    projections trainable.

    This is the outer half of the flag, and the mirror image of
    :func:`freeze_policy_level_params_for_vision_only`: the inner wrapper's
    ``set_requires_grad`` has already frozen everything it owns except the
    discrete-action representation, and what remains are the projections and other
    modules that live on the enclosing flow-matching module. Only
    :data:`STATE_ACTION_PROJECTION_ATTRS` survives; ``time_mlp_*`` /
    ``action_time_mlp_*``, the optional modality embeddings, and any video-encoder
    parameters are frozen.

    Note that unlike :func:`freeze_policy_level_params_for_vision_only` this function
    has **no ``motion_module`` carve-out**. The pi05_mem RLDX temporal block is part
    of the video encoder, and it is the one module a generic pretrained checkpoint
    does *not* contain — it loads fresh and zero-gated. Inheriting the carve-out would
    silently train a randomly-initialized temporal block in a run whose entire premise
    is that only the dataset-facing adapters move.

    Args:
        policy_module: the enclosing flow-matching ``nn.Module``.
        with_expert_module: the inner ``*WithExpertModel`` submodule, already
            configured by its own ``set_requires_grad``; its parameters are left
            untouched here.

    Returns:
        Sorted names of the outer parameters left trainable, for logging and tests.
    """
    protected = {id(p) for p in with_expert_module.parameters()}
    keep = _param_ids_of_attrs(policy_module, STATE_ACTION_PROJECTION_ATTRS)
    trainable: list[str] = []
    for name, param in policy_module.named_parameters():
        if id(param) in protected:
            continue
        param.requires_grad = id(param) in keep
        if param.requires_grad:
            trainable.append(name)
    return sorted(trainable)
