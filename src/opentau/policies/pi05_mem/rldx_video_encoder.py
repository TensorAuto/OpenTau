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

"""RLDX SigLIP video encoder: plain SigLIP + STSS motion module.

``RLDXVideoEncoder`` is a standalone video encoder that runs a stock SigLIP ViT
on each frame and injects the RLDX-1 space-time self-similarity (STSS) motion
module (https://github.com/RLWRLD/RLDX-1) as a residual update at a single
encoder layer. The motion module is the *only* cross-frame mechanism here:
unlike :class:`~opentau.policies.pi07.video_encoder.SpaceTimeSiglipVideoEncoder`,
this encoder has **no space-time / temporal attention layers** — the SigLIP
tower is left exactly as pretrained, and temporal dynamics are captured solely
by the STSS motion module (see :mod:`opentau.policies.pi05_mem.motion_module`).

It deliberately does not subclass or import the space-time encoder, so the two
implementations stay fully independent.

Contract (identical I/O to the space-time encoder, so it is a drop-in):
    forward(video, obs_history_is_pad=None)
        video: ``(B, T, C, H, W)`` pixel values in ``[0, 1]``, ``1 <= T``.
        returns: ``(B, num_video_tokens, vlm_hidden_size)`` current-frame tokens.

Notes:
  - **Adds new learnable parameters** (the motion module). A plain pi05
    checkpoint loads with ``strict=False``; with ``motion_zero_init=True``
    (default) the motion residual is zero-gated at init so the policy is
    byte-identical at step 0 and the motion contribution warms up.
  - The motion module is registered under ``motion_module.*`` and stays
    trainable even when the SigLIP tower is frozen by the caller.
  - Motion runs only when there is a real time axis (``T > 1``); at ``T = 1``
    the encoder is a plain single-frame SigLIP forward.
  - ``obs_history_is_pad`` is honored: padded (start-of-episode zero) frames are
    replaced by the nearest real frame before the STSS correlation, so they
    contribute "no motion" instead of spurious motion against a blank frame.
    This mirrors the space-time encoder, which masks exactly these frames out of
    temporal attention. (The RLDX-1 reference does not mask them; we do, to avoid
    a train/inference distribution mismatch at episode start.)
"""

import torch
from einops import rearrange
from torch import Tensor, nn
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

# Import triggers the transformers patch that drops the `/ sqrt(hidden_size)`
# scaling stock HuggingFace applies after the multi_modal_projector, matching
# the rest of the OpenTau vision path.
import opentau.utils.transformers_patch  # noqa: F401
from opentau.policies.pi05_mem.motion_module import MotionModule


class RLDXVideoEncoder(nn.Module):
    """Plain SigLIP video encoder with an RLDX-1 STSS motion module.

    The caller owns ``vision_tower`` and ``multi_modal_projector`` (constructed
    by a ``PaliGemmaWithExpertModel`` / ``Gemma3WithExpertModel``); this module
    holds them by reference (via a list, so their parameters are not
    re-registered under this module's path) and adds only the motion module's
    parameters.

    Args:
        vision_tower: A ``SiglipVisionModel`` (left unmodified — no layer wrapping).
        multi_modal_projector: SigLIP-hidden -> VLA-hidden projector.
        max_num_frames: Upper bound on ``T`` accepted by ``forward`` (validation cap).
        gradient_checkpointing: Wrap each SigLIP layer forward in
            ``torch.utils.checkpoint`` during training.
        motion_insert_layer: 0-indexed encoder layer after which the motion
            residual is injected. ``None`` -> mid-stack (``n_layers // 3``),
            which is layer 9 for the 27-layer so400m tower (RLDX placement).
        motion_hidden_dim: Internal correlation/feature width (``in_proj`` target).
        motion_window: ``(L, kh, kw)`` space-time correlation window; spatial
            size must fit the patch grid.
        motion_corr_func: "cosine" / "dotproduct" / "dotproduct_softmax".
        motion_n_encoders: Number of stacked STSS encoders (outputs summed).
        motion_norm: "groupnorm" (default; per-sample, no cross-rank sync — safe
            under FSDP/DeepSpeed/multi-rank) / "batchnorm" (RLDX-faithful) / "syncbn".
        motion_int_mode: "lite" (single fuse conv) or "full" (3x3 conv stack).
        motion_zero_init: Zero-init the residual so the module starts as a no-op.
    """

    def __init__(
        self,
        vision_tower: SiglipVisionModel,
        multi_modal_projector: nn.Module,
        max_num_frames: int,
        gradient_checkpointing: bool = False,
        *,
        motion_insert_layer: int | None = None,
        motion_hidden_dim: int = 256,
        motion_window: tuple[int, int, int] = (5, 9, 9),
        motion_corr_func: str = "cosine",
        motion_n_encoders: int = 1,
        motion_norm: str = "groupnorm",
        motion_int_mode: str = "lite",
        motion_zero_init: bool = True,
    ):
        super().__init__()
        if max_num_frames < 1:
            raise ValueError(f"max_num_frames ({max_num_frames}) must be >= 1.")

        self.max_num_frames = max_num_frames
        self.gradient_checkpointing = gradient_checkpointing

        # Hold references in lists so nn.Module.__setattr__ does not re-register
        # these caller-owned modules under this encoder's path (would duplicate
        # ~400M params in state_dict).
        self._vision_tower_ref: list[SiglipVisionModel] = [vision_tower]
        self._multi_modal_projector_ref: list[nn.Module] = [multi_modal_projector]

        vision_cfg = vision_tower.config
        num_patches = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
        self.num_video_tokens = num_patches
        self.siglip_hidden_size = vision_cfg.hidden_size

        n_layers = len(vision_tower.vision_model.encoder.layers)

        # Square patch grid (e.g. 16x16 for 224/14). The motion module needs the
        # spatial grid shape to build the local correlation window.
        grid = int(round(num_patches**0.5))
        if grid * grid != num_patches:
            raise ValueError(f"Motion module requires a square patch grid; got {num_patches} patches.")
        self.motion_grid_hw = grid
        if max(motion_window[1], motion_window[2]) > grid:
            raise ValueError(
                f"motion_window spatial size {motion_window[1:]} exceeds patch grid {grid}x{grid}."
            )

        # Default insert layer: mid-stack (n_layers // 3). For the 27-layer
        # so400m tower this is layer 9, matching the RLDX-1 placement.
        insert = n_layers // 3 if motion_insert_layer is None else motion_insert_layer
        if not 0 <= insert < n_layers:
            raise ValueError(f"motion_insert_layer ({insert}) out of range [0, {n_layers}).")
        self.motion_insert_layer = insert

        self.motion_module = MotionModule(
            d_in=self.siglip_hidden_size,
            d_hid=motion_hidden_dim,
            d_out=self.siglip_hidden_size,
            window=motion_window,
            corr_func=motion_corr_func,
            n_encoders=motion_n_encoders,
            norm=motion_norm,
            int_mode=motion_int_mode,
            zero_init_residual=motion_zero_init,
        )

    @property
    def vision_tower(self) -> SiglipVisionModel:
        return self._vision_tower_ref[0]

    @property
    def multi_modal_projector(self) -> nn.Module:
        return self._multi_modal_projector_ref[0]

    def _apply_motion(self, hidden: Tensor, b: int, t: int, obs_history_is_pad: Tensor | None) -> Tensor:
        """Add the STSS motion residual to the per-patch hidden states.

        Args:
            hidden: ``(B*T, N, D)`` encoder hidden states at the insert layer.
            b: batch size. t: number of frames.
            obs_history_is_pad: Optional ``(B, T)`` bool mask, ``True`` for padded
                (start-of-episode zero) frames. When given, padded frames are
                replaced by the nearest real frame before the STSS correlation,
                so they contribute "no motion" instead of spurious motion against
                a blank frame. The current frame (t = T-1) is treated as real
                even when flagged (the ``history_state_drop_prob`` augmentation
                marks the whole mask True while keeping the current frame).
                This mirrors the space-time encoder, which masks exactly these
                frames out of temporal attention.
        Returns:
            ``(B*T, N, D)`` hidden states with the motion residual added.
        """
        n = hidden.shape[1]
        gh = self.motion_grid_hw
        x = rearrange(hidden, "(b t) n d -> b t n d", b=b, t=t)

        # Neutralize padded history frames. The dataloader / select_action
        # zero-pads the EARLIEST history slots (a prefix) and the current frame
        # (t=T-1) is always real, so forward-filling each padded frame with the
        # FIRST real frame removes the blank-frame "jump" the STSS correlation
        # would otherwise read as motion. Padded frames are dropped after the
        # encoder anyway; this only protects the current frame's residual.
        if obs_history_is_pad is not None:
            real = ~obs_history_is_pad  # (b, t) — `~` allocates a fresh tensor
            # The current frame (t = T-1) is ALWAYS real even when the
            # dataset's history_state_drop_prob augmentation marks
            # obs_history_is_pad all-True. Without this override, argmax over
            # an all-False row would silently pick the zeroed frame 0 as the
            # fill and replace the current frame's hidden state with it too.
            real[:, -1] = True
            # First real index per row (argmax returns the first max=1 position).
            first_real = real.float().argmax(dim=1)  # (b,)
            fill = x[torch.arange(b, device=x.device), first_real]  # (b, n, d)
            pad_mask = rearrange(~real, "b t -> b t 1 1")
            x = torch.where(pad_mask, rearrange(fill, "b n d -> b 1 n d"), x)

        # (B, T, N, D) -> (B*T*N, D), preserving the (b t h w) token order the
        # motion module expects (n is the row-major (h w) patch grid).
        flat = rearrange(x, "b t n d -> (b t n) d")
        grid_sizes = torch.tensor([[t, gh, gh]] * b, dtype=torch.long, device=hidden.device)
        residual = self.motion_module(flat, grid_sizes)  # (B*T*N, D)
        residual = rearrange(residual, "(b t n) d -> (b t) n d", b=b, t=t, n=n)
        return hidden + residual

    def forward(self, video: Tensor, obs_history_is_pad: Tensor | None = None) -> Tensor:
        """Encode a video clip and return the current-frame tokens.

        Args:
            video: ``(B, T, C, H, W)`` pixel values in ``[0, 1]``.
            obs_history_is_pad: Optional ``(B, T)`` bool mask, ``True`` for padded
                history frames. Padded frames are replaced by the nearest real
                frame before the STSS motion correlation so they don't inject
                spurious motion into the current-frame residual.

        Returns:
            ``(B, num_video_tokens, vlm_hidden_size)`` current-frame tokens.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W); got {tuple(video.shape)}.")
        b, t = video.shape[0], video.shape[1]
        if t < 1:
            raise ValueError(f"Expected T >= 1; got {t}.")
        if t > self.max_num_frames:
            raise ValueError(
                f"Expected T <= max_num_frames ({self.max_num_frames}); got {t}. "
                "Reinstantiate the encoder with a larger max_num_frames."
            )

        # SigLIP expects pixel values in [-1, 1]; the dataset loader yields [0, 1].
        video = video * 2.0 - 1.0
        flat = rearrange(video, "b t c h w -> (b t) c h w")
        hidden = self.vision_tower.vision_model.embeddings(flat)

        # The SigLIP patch/position embeddings are pinned to float32 (openpi parity, see
        # PaliGemmaWithExpertModel.to_bfloat16_like_physical_intelligence). This encoder
        # hand-rolls the SigLIP forward instead of routing through the patched
        # SiglipVisionTransformer.forward, so it has to reproduce that forward's dtype
        # bridge here: cast the float32 embeddings to the (possibly bfloat16) encoder dtype
        # before the first encoder layer. No-op when the tower is single-dtype.
        # ``post_layernorm`` is the unambiguous SigLIP encoder dtype.
        encoder_dtype = self.vision_tower.vision_model.post_layernorm.weight.dtype
        if hidden.dtype != encoder_dtype:
            hidden = hidden.to(encoder_dtype)

        use_ckpt = self.gradient_checkpointing and self.training
        # Motion runs only with a real time axis (T > 1). It is intentionally NOT
        # gradient-checkpointed: its BatchNorm3d (default) running stats would be
        # updated twice under recompute. Memory stays bounded (runs at one layer).
        run_motion = t > 1
        for idx, layer in enumerate(self.vision_tower.vision_model.encoder.layers):
            # transformers >= 4.57 ``SiglipEncoderLayer.forward(hidden_states,
            # attention_mask)`` returns a bare hidden-state tensor (no
            # ``output_attentions`` arg, no tuple). Matches video_encoder.py.
            if use_ckpt:
                hidden = torch.utils.checkpoint.checkpoint(layer, hidden, None, use_reentrant=False)
            else:
                hidden = layer(hidden, None)

            if run_motion and idx == self.motion_insert_layer:
                hidden = self._apply_motion(hidden, b, t, obs_history_is_pad)

        hidden = self.vision_tower.vision_model.post_layernorm(hidden)

        # Drop past-timestep tokens: keep only the current frame (t = T-1).
        hidden = rearrange(hidden, "(b t) n d -> b t n d", b=b, t=t)
        current = hidden[:, -1]

        # multi_modal_projector: SigLIP hidden -> VLA hidden. The `/ sqrt(...)`
        # division is intentionally omitted to match the patched
        # ``PaliGemmaModel.get_image_features``.
        return self.multi_modal_projector(current)
