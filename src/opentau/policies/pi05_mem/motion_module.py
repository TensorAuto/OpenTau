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

"""Space-Time Self-Similarity (STSS) motion module.

Ported from RLDX-1 (``rldx/model/modules/backbone/motion.py``,
https://github.com/RLWRLD/RLDX-1), whose motion module in turn builds on the
SELFY space-time self-similarity formulation (Kwon et al., "Learning
Self-Similarity in Space and Time as Generalized Motion for Video Action
Recognition", ICCV 2021).

Rather than optical flow or raw frame differences, the module captures temporal
dynamics by computing **local space-time self-similarity**: for every
spatio-temporal patch feature it correlates against its neighbors within a
``(L, kh, kw)`` window across frames, producing a similarity volume that a small
3D-conv encoder turns into a motion feature. The result is a **residual** that
is added back onto the vision features:

    v~(t) = v(t) + S_theta(STSS(v(t)))

In OpenTau this residual is injected at a single SigLIP encoder layer (see
``SpaceTimeSiglipVideoEncoder`` in ``video_encoder.py``).

Interface (matching the RLDX-1 reference so the math is easy to compare):

    forward(x, grid_sizes)
        x:          (sum_i T_i*H_i*W_i, C) flattened patch features, token order
                    ``(b t h w)`` (batch-major, then time, then row-major patches).
        grid_sizes: (B, 3) int tensor; each row is ``[T, H, W]`` for one video.
        returns:    (sum_i T_i*H_i*W_i, C) residual, same layout as ``x``.

Differences from the RLDX-1 reference, all behind flags:
  - ``norm`` (default "groupnorm"): GroupNorm(1, C) — per-sample, no cross-rank
    stat sync, safe under FSDP/DeepSpeed/multi-rank (CLAUDE.md rule #5). Pass
    "batchnorm" for the RLDX-1-faithful BatchNorm3d, or "syncbn".
  - ``zero_init_residual`` (default True): zero-initializes the output projection
    (or the LayerScale) so the module starts as an exact no-op. A policy
    fine-tuned from a pre-existing pi05 checkpoint is therefore byte-identical at
    step 0 and the motion contribution warms up during training. RLDX-1 dropped
    this zero-init so motion contributes from step 1 — pass
    ``zero_init_residual=False`` to match that.
  - The gradient-monitoring print hook from the reference is omitted.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn

_NormKind = str  # one of {"batchnorm", "groupnorm", "syncbn"}


def _make_norm3d(num_channels: int, norm: _NormKind) -> nn.Module:
    """Norm layer for the STSS 3D-conv stack.

    ``groupnorm`` uses ``GroupNorm(1, C)`` (a.k.a. per-sample LayerNorm over
    channels+space) — it carries no running statistics and needs no cross-rank
    synchronization, so it is the safe choice under FSDP / DeepSpeed / multi-rank
    (CLAUDE.md rule #5). ``batchnorm`` matches the RLDX-1 default but, like any
    BatchNorm, keeps running stats that diverge across ranks unless wrapped in
    ``syncbn``.
    """
    if norm == "groupnorm":
        return nn.GroupNorm(1, num_channels)
    if norm == "syncbn":
        return nn.SyncBatchNorm(num_channels)
    if norm == "batchnorm":
        return nn.BatchNorm3d(num_channels)
    raise ValueError(f"Unknown norm '{norm}'; expected one of batchnorm/groupnorm/syncbn.")


class STSSTransformation(nn.Module):
    """Compute the local space-time self-similarity volume.

    For each spatio-temporal patch feature, correlate it against the features of
    nearby frames within a ``window[0]`` temporal span, then keep only a local
    ``window[1] x window[2]`` spatial neighborhood of the correlation.
    """

    def __init__(self, window: tuple[int, int, int] = (5, 9, 9), corr_func: str = "cosine"):
        super().__init__()
        self.window = window
        assert window[1] == window[2], "spatial correlation window must be square"
        self.corr_func = corr_func
        if self.corr_func == "dotproduct_softmax":
            self.pad_value = -float("inf")
        else:
            self.pad_value = 0.0

    def _convert_global_to_local(self, corr_g: Tensor) -> Tensor:
        """Absolute (h,w)x(h,w) correlation -> local-window correlation.

        Args:
            corr_g: ``(b, h, w, h, w)`` global correlation tensor.
        Returns:
            ``(b, h, w, kh, kw)`` local-window correlation tensor.
        """
        max_d = self.window[1] // 2

        # Extract spatial-row offsets via diagonals, padding out-of-range entries.
        corr_l = [
            F.pad(
                torch.diagonal(corr_g, offset=i, dim1=1, dim2=3),
                (abs(i) if i < 0 else 0, abs(i) if i >= 0 else 0),
                value=self.pad_value,
            )
            for i in range(-max_d, max_d + 1)
        ]
        corr_l = torch.stack(corr_l, dim=-1)  # B, W1, W2, H1, H2 -> U

        # Extract spatial-col offsets the same way.
        corr_l = [
            F.pad(
                torch.diagonal(corr_l, offset=i, dim1=1, dim2=2),
                (abs(i) if i < 0 else 0, abs(i) if i >= 0 else 0),
                value=self.pad_value,
            )
            for i in range(-max_d, max_d + 1)
        ]
        corr_l = torch.stack(corr_l, dim=-1)  # B, H1, H2 -> U, W1, W2 -> V
        corr_l = corr_l.transpose(2, 3).contiguous()  # B, H1, W1, U, V

        return corr_l

    def _correlation(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        if self.corr_func == "cosine":
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
        elif self.corr_func in ("dotproduct", "dotproduct_softmax"):
            scale = feat1.size(1) ** -0.5
            feat1 = feat1 * scale

        corr = torch.einsum("bchw,bcuv->bhwuv", feat1, feat2)
        corr = self._convert_global_to_local(corr)

        if self.corr_func == "dotproduct_softmax":
            corr_shape = corr.shape
            corr = rearrange(corr, "b h w u v -> b h w (u v)")
            corr = F.softmax(corr, dim=-1)
            corr = corr.reshape(corr_shape)

        return corr

    def forward(self, x: Tensor, grid_sizes: Tensor) -> Tensor:
        t, h, w = (int(v) for v in grid_sizes[0])
        if self.window[0] > 1:
            x = rearrange(x, "(b t h w) c -> b t c h w", t=t, h=h, w=w)
            x_src = repeat(x, "b t c h w -> (b t l) c h w", l=self.window[0])
            # Replicate-pad the temporal axis: edge frames repeat instead of being
            # zero-padded, so boundary correlations reflect "no motion" rather than
            # "motion against a blank frame".
            pad_t = self.window[0] // 2
            x_pad = torch.cat(
                [
                    x[:, :1].expand(-1, pad_t, -1, -1, -1),
                    x,
                    x[:, -1:].expand(-1, pad_t, -1, -1, -1),
                ],
                dim=1,
            )
            x_tgt = x_pad.unfold(1, self.window[0], 1)
            x_tgt = rearrange(x_tgt, "b t c h w l -> (b t l) c h w")
        else:
            # L=1: correlate each frame against the previous one (frame 0 against
            # itself). Both operands must be 4D (b t) c h w for ``_correlation``'s
            # einsum — rearrange x_src too (do NOT keep the raw 2D input), so it
            # lines up with the l=1 case of the trailing stss rearrange.
            x_src = rearrange(x, "(b t h w) c -> (b t) c h w", t=t, h=h, w=w)
            x = rearrange(x, "(b t h w) c -> b t c h w", t=t, h=h, w=w)
            x_tgt = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1)
            x_tgt = rearrange(x_tgt, "b t c h w -> (b t) c h w")

        stss = self._correlation(x_src, x_tgt)
        stss = rearrange(stss, "(b t l) h w u v -> b t h w 1 l u v", t=t, l=self.window[0])

        return stss


class STSSExtraction(nn.Module):
    """Project the ``kh*kw`` similarity channels of the STSS volume to features."""

    def __init__(
        self,
        window: tuple[int, int, int] = (5, 9, 9),
        chnls: tuple[int, ...] = (256,),
        norm: _NormKind = "groupnorm",
    ):
        super().__init__()
        self.window = window
        self.chnls = chnls

        self.conv0 = nn.Sequential(
            nn.Conv3d(
                self.window[1] * self.window[2],
                chnls[0],
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),
            _make_norm3d(chnls[0], norm),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (b, t, h, w, 1, l, u, v) -> (b*l, u*v, t, h, w). All axes are
        # merges/known literals, so einops infers every size from the shape.
        x = rearrange(x, "b t h w 1 l u v -> (b l) (u v) t h w")
        x = self.conv0(x)
        return x


class STSSIntegration(nn.Module):
    """Fuse the ``L`` temporal-window slices into a single motion feature map."""

    def __init__(
        self,
        d_in: int,
        window: tuple[int, int, int] = (5, 9, 9),
        chnls: tuple[int, int, int] = (64, 64, 64),
        norm: _NormKind = "groupnorm",
        mode: str = "lite",
    ):
        super().__init__()
        self.window = window
        self.mode = mode

        if mode == "lite":
            # Single 1x1 Conv3d: L fuse + channel projection, no spatial mixing, no
            # norm. Replaces the 3-layer 3x3 conv stack so the module contributes
            # without a deep warm-up path.
            self.fuse = nn.Sequential(
                Rearrange("(b l) c t h w -> b (l c) t h w", l=self.window[0]),
                nn.Conv3d(d_in * self.window[0], chnls[-1], kernel_size=(1, 1, 1), bias=False),
                nn.GELU(),
            )
            return

        self.conv0 = nn.Sequential(
            nn.Conv3d(d_in, chnls[0], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            _make_norm3d(chnls[0], norm),
            nn.GELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                chnls[0], chnls[1], kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False
            ),
            _make_norm3d(chnls[1], norm),
            nn.GELU(),
        )
        self.conv2_fuse = nn.Sequential(
            Rearrange("(b l) c t h w -> b (l c) t h w", l=self.window[0]),
            nn.Conv3d(
                chnls[1] * self.window[0],
                chnls[2],
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            _make_norm3d(chnls[2], norm),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "lite":
            return self.fuse(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2_fuse(x)
        return x


class STSSEncoder(nn.Module):
    """One STSS block: LN -> in_proj -> transform -> extract -> integrate -> out_proj."""

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        window: tuple[int, int, int] = (5, 9, 9),
        ext_chnls: tuple[int, ...] = (256,),
        int_chnls: tuple[int, int, int] = (256, 256, 512),
        corr_func: str = "cosine",
        norm: _NormKind = "groupnorm",
        int_mode: str = "lite",
    ):
        super().__init__()
        self.window = window
        self.ln_pre = nn.LayerNorm(d_in, eps=1e-6)
        self.in_proj = nn.Linear(d_in, d_hid)
        self.stss_transformation = STSSTransformation(window=window, corr_func=corr_func)
        self.stss_extraction = STSSExtraction(window=window, chnls=ext_chnls, norm=norm)
        self.stss_integration = STSSIntegration(
            ext_chnls[-1], window=window, chnls=int_chnls, norm=norm, mode=int_mode
        )
        self.out_proj = nn.Linear(int_chnls[-1], d_out)

    def forward(self, x: Tensor, grid_sizes: Tensor) -> Tensor:
        x = self.in_proj(self.ln_pre(x))
        x = self.stss_transformation(x, grid_sizes)
        x = self.stss_extraction(x)
        x = self.stss_integration(x)
        x = self.out_proj(rearrange(x, "b c t h w -> (b t h w) c"))
        return x


class MotionModule(nn.Module):
    """STSS motion module producing a residual update to vision features.

    Args:
        d_in: Input feature dimension (must equal the residual target dim).
        d_hid: Internal correlation/feature dimension (``in_proj`` target).
        d_out: Output dimension; set equal to ``d_in`` so the residual adds back.
        window: ``(L, kh, kw)`` space-time correlation window.
        ext_chnls / int_chnls: channel widths for the extraction / integration convs.
        corr_func: "cosine" (default), "dotproduct", or "dotproduct_softmax".
        n_encoders: number of stacked STSS encoders (their outputs are summed).
        use_layerscale: gate the output with a learnable per-channel LayerScale
            instead of an ``out_proj`` linear.
        layerscale_init: initial LayerScale value (used only with ``use_layerscale``).
        norm: "groupnorm" (default, distributed-safe) / "batchnorm" (RLDX-faithful) / "syncbn".
        int_mode: "lite" (single fuse conv) or the full 3x3 conv stack.
        zero_init_residual: when True, zero-initialize the residual output so the
            module starts as an exact no-op (warm start from a pretrained
            checkpoint); the contribution ramps up during training.
    """

    def __init__(
        self,
        d_in: int,
        d_hid: int,
        d_out: int,
        window: tuple[int, int, int] = (5, 9, 9),
        ext_chnls: tuple[int, ...] = (256,),
        int_chnls: tuple[int, int, int] = (256, 256, 512),
        corr_func: str = "cosine",
        n_encoders: int = 1,
        use_layerscale: bool = False,
        layerscale_init: float = 1e-5,
        norm: _NormKind = "groupnorm",
        int_mode: str = "lite",
        zero_init_residual: bool = True,
    ):
        super().__init__()
        # All window dims must be positive and ODD: each STSS axis builds a
        # centered window of 2*(k//2)+1 entries, and the temporal unfold only
        # yields exactly T windows for an odd span. An even dim silently changes
        # the entry count and crashes downstream (conv channel / einsum batch
        # mismatch), so reject it up front. Spatial dims must also be square.
        if len(window) != 3 or any(w < 1 or w % 2 == 0 for w in window):
            raise ValueError(f"window dims must be positive odd ints (L, kh, kw); got {window}.")
        if window[1] != window[2]:
            raise ValueError(f"window spatial dims must be square; got {window[1:]}.")

        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.zero_init_residual = zero_init_residual

        self.stss_encoders = nn.ModuleList(
            [
                STSSEncoder(
                    d_in=d_in if i == 0 else (d_out if self.use_layerscale else d_hid),
                    d_hid=d_hid,
                    d_out=d_out if self.use_layerscale else d_hid,
                    window=window,
                    ext_chnls=ext_chnls,
                    int_chnls=int_chnls,
                    corr_func=corr_func,
                    norm=norm,
                    int_mode=int_mode,
                )
                for i in range(n_encoders)
            ]
        )

        if self.use_layerscale:
            self.layerscale = nn.Parameter(torch.ones(d_out) * layerscale_init, requires_grad=True)
        else:
            self.out_proj = nn.Linear(d_hid, d_out)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize all submodule weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(
                m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm)
            ):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # No-op warm start: gate the residual at zero so a checkpoint loaded into a
        # policy with this module behaves identically at step 0; the motion
        # contribution then warms up during training.
        if self.use_layerscale:
            init_val = 0.0 if self.zero_init_residual else self.layerscale_init
            self.layerscale.data.fill_(init_val)
        elif self.zero_init_residual:
            nn.init.constant_(self.out_proj.weight, 0.0)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: Tensor, grid_sizes: Tensor) -> Tensor:
        """Args:
            x: ``(sum_i T_i*H_i*W_i, C)`` flattened patch features, order ``(b t h w)``.
            grid_sizes: ``(B, 3)`` int tensor; each row ``[T, H, W]``.
        Returns:
            ``(sum_i T_i*H_i*W_i, d_out)`` residual, same layout as ``x``.
        """
        all_same_grid = bool((grid_sizes == grid_sizes[0]).all())

        if all_same_grid:
            out = x
            encoder_outputs = []
            for stss_encoder in self.stss_encoders:
                out = stss_encoder(out, grid_sizes=grid_sizes)
                encoder_outputs.append(out)
            out = torch.stack(encoder_outputs, dim=0).sum(dim=0)
        else:
            num_tokens_per_video = grid_sizes.prod(dim=1).tolist()
            x_splits = x.split(num_tokens_per_video, dim=0)

            processed_videos = []
            for x_video, grid_size in zip(x_splits, grid_sizes, strict=False):
                video_out = x_video
                encoder_outputs = []
                for stss_encoder in self.stss_encoders:
                    video_out = stss_encoder(video_out, grid_sizes=grid_size.unsqueeze(0))
                    encoder_outputs.append(video_out)
                video_out = torch.stack(encoder_outputs, dim=0).sum(dim=0)
                processed_videos.append(video_out)
            out = torch.cat(processed_videos, dim=0)

        out = out * self.layerscale if self.use_layerscale else self.out_proj(out)

        return out
