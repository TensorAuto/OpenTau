# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Self-contained BF16 inference transformer for the released DROID policy.

This module deliberately owns its attention, transformer blocks, prepared
state, and RoPE layout.  It shares checkpoint names with the training model,
but never calls training/reference blocks at runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

VIDEO = "video"
VIDEO_COND = "video_cond"
ACTION = "action_prediction_droid"
ACTION_COND = "action_prediction_droid_cond"
STREAM_ORDER = (VIDEO, VIDEO_COND, ACTION, ACTION_COND)
CHANNELS = {VIDEO: 96, VIDEO_COND: 96, ACTION: 8, ACTION_COND: 8}
HIDDEN_SIZE = 3072
NUM_HEADS = 24
MLP_SIZE = 9216
EARLY_DEPTH = 5
JOINT_DEPTH = 28
CONTEXT_DIM = 20480
VECTOR_DIM = 768
AXES_DIM = (32, 32, 32, 32)
THETA = 10000

ModulationTuple = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class _PreparedText:
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    txt_len: int
    txt: Tensor
    rope: Tensor
    early_video_cond: ModulationTuple
    early_action_cond: ModulationTuple
    single_txt: ModulationTuple
    single_video_cond: ModulationTuple
    single_action_cond: ModulationTuple
    vector_embedding: Tensor


@dataclass(frozen=True)
class _PreparedRequest:
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    txt_len: int
    video_len: int
    video_cond_len: int
    action_len: int
    action_cond_len: int
    txt: Tensor
    video_cond: Tensor
    action_cond: Tensor
    rope: Tensor
    single_txt: ModulationTuple
    single_video_cond: ModulationTuple
    single_action_cond: ModulationTuple
    vector_embedding: Tensor


@dataclass(frozen=True)
class _PreparedStep:
    early_video: ModulationTuple
    early_action: ModulationTuple
    single_video: ModulationTuple
    single_action: ModulationTuple
    final_video_shift: Tensor
    final_video_scale: Tensor
    final_action_shift: Tensor
    final_action_scale: Tensor


@dataclass(frozen=True)
class DroidPrediction:
    video: Tensor
    action: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {"x_video": self.video, "x_action_prediction_droid": self.action}


def _rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    angles = torch.einsum("...n,d->...nd", pos, omega)
    matrix = torch.stack(
        (torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)), dim=-1
    )
    return matrix.reshape(*matrix.shape[:-1], 2, 2).float()


def _apply_rope(q: Tensor, k: Tensor, rope: Tensor) -> tuple[Tensor, Tensor]:
    q_pairs = q.float().reshape(*q.shape[:-1], -1, 1, 2)
    k_pairs = k.float().reshape(*k.shape[:-1], -1, 1, 2)
    q_out = rope[..., 0] * q_pairs[..., 0] + rope[..., 1] * q_pairs[..., 1]
    k_out = rope[..., 0] * k_pairs[..., 0] + rope[..., 1] * k_pairs[..., 1]
    return q_out.reshape_as(q).to(q.dtype), k_out.reshape_as(k).to(k.dtype)


def _timestep_embedding(timestep: Tensor, dim: int = 256) -> Tensor:
    scaled = timestep * 1000.0
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
    )
    angles = scaled[:, None].float() * frequencies[None]
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class _RopeEmbedder(nn.Module):
    def __init__(self, head_dim: int, theta: int, axes_dim: tuple[int, int, int, int]):
        super().__init__()
        self.dim = head_dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        embeddings = torch.cat(
            tuple(_rope(ids[..., axis], dim, self.theta) for axis, dim in enumerate(self.axes_dim)),
            dim=-3,
        )
        return embeddings.unsqueeze(1)


class _MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + 1e-6)
        return normalized.to(dtype) * self.scale


class _QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = _RMSNorm(dim)
        self.key_norm = _RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class _SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        gate, value = x.chunk(2, dim=-1)
        return self.gate_fn(gate) * value


class _Modulation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, vector: Tensor) -> ModulationTuple:
        output = self.lin(F.silu(vector))
        if output.ndim == 2:
            output = output[:, None, :]
        shift, scale, gate = output.chunk(3, dim=-1)
        return shift, scale, gate


class _LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))


class _Attention(nn.Module):
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        attended = F.scaled_dot_product_attention(q, k, v)
        return attended.transpose(1, 2).flatten(2)


class _ModeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = HIDDEN_SIZE
        self.num_heads = NUM_HEADS
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = MLP_SIZE
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * MLP_SIZE, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(MLP_SIZE, hidden_size, bias=False)
        self.norm = _QKNorm(hidden_size // NUM_HEADS)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = _SwiGLU()
        self.attention = _Attention()
        self.register_parameter("in_proj_weight", None)

    def pack_input_projection(self) -> None:
        assert self.in_proj_weight is None, "input projection already packed"
        packed = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.mlp_in.weight), dim=0
        )
        self.in_proj_weight = nn.Parameter(packed, requires_grad=False)
        del self.q_proj, self.k_proj, self.v_proj, self.mlp_in

    def forward(self, x: Tensor, rope: Tensor, modulation: ModulationTuple) -> Tensor:
        shift, scale, gate = modulation
        modulated = (1 + scale) * self.pre_norm(x) + shift
        batch, length, _ = modulated.shape
        assert self.in_proj_weight is not None, "input projection is not packed"
        q, k, v, mlp = F.linear(modulated, self.in_proj_weight).split(
            (self.hidden_size, self.hidden_size, self.hidden_size, 2 * self.mlp_hidden_dim), dim=-1
        )
        q = q.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        q, k = self.norm(q, k, v)
        q, k = _apply_rope(q, k, rope)
        output = self.attn_out(self.attention(q, k, v)) + self.mlp_out(self.mlp_act(mlp))
        return x + gate * output


def _modulate_segments(
    normalized: Tensor,
    lengths: tuple[int, int, int, int, int],
    modulations: tuple[ModulationTuple, ModulationTuple, ModulationTuple, ModulationTuple, ModulationTuple],
) -> Tensor:
    segments = torch.split(normalized, lengths, dim=1)
    return torch.cat(
        tuple(
            (1 + modulation[1]) * segment + modulation[0]
            for segment, modulation in zip(segments, modulations)
        ),
        dim=1,
    )


def _gate_segments(
    output: Tensor,
    lengths: tuple[int, int, int, int, int],
    modulations: tuple[ModulationTuple, ModulationTuple, ModulationTuple, ModulationTuple, ModulationTuple],
) -> Tensor:
    segments = torch.split(output, lengths, dim=1)
    return torch.cat(
        tuple(modulation[2] * segment for segment, modulation in zip(segments, modulations)), dim=1
    )


class _JointBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = HIDDEN_SIZE
        self.num_heads = NUM_HEADS
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = MLP_SIZE
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * MLP_SIZE, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(MLP_SIZE, hidden_size, bias=False)
        self.norm = _QKNorm(hidden_size // NUM_HEADS)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = _SwiGLU()
        self.attention = _Attention()
        self.register_parameter("in_proj_weight", None)

    def pack_input_projection(self) -> None:
        assert self.in_proj_weight is None, "input projection already packed"
        packed = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.mlp_in.weight), dim=0
        )
        self.in_proj_weight = nn.Parameter(packed, requires_grad=False)
        del self.q_proj, self.k_proj, self.v_proj, self.mlp_in

    def forward(
        self,
        sequence: Tensor,
        rope: Tensor,
        lengths: tuple[int, int, int, int, int],
        modulations: tuple[
            ModulationTuple,
            ModulationTuple,
            ModulationTuple,
            ModulationTuple,
            ModulationTuple,
        ],
    ) -> Tensor:
        modulated = _modulate_segments(self.pre_norm(sequence), lengths, modulations)
        batch, length, _ = modulated.shape
        assert self.in_proj_weight is not None, "input projection is not packed"
        q, k, v, mlp = F.linear(modulated, self.in_proj_weight).split(
            (self.hidden_size, self.hidden_size, self.hidden_size, 2 * self.mlp_hidden_dim), dim=-1
        )
        q = q.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, -1).transpose(1, 2)
        q, k = self.norm(q, k, v)
        q, k = _apply_rope(q, k, rope)
        output = self.attn_out(self.attention(q, k, v)) + self.mlp_out(self.mlp_act(mlp))
        return sequence + _gate_segments(output, lengths, modulations)


class BF16InferenceDiT(nn.Module):
    """Released DROID BF16 DiT with opaque, explicitly owned prepared state."""

    def __init__(self):
        super().__init__()
        in_channels = CHANNELS
        self.depth = EARLY_DEPTH

        head_dim = HIDDEN_SIZE // NUM_HEADS
        self.pe_embedder = _RopeEmbedder(head_dim, THETA, AXES_DIM)
        self.emb_in = nn.ModuleDict(
            {name: nn.Linear(channels, HIDDEN_SIZE, bias=False) for name, channels in in_channels.items()}
        )
        self.txt_in = nn.Linear(CONTEXT_DIM, HIDDEN_SIZE, bias=False)
        self.time_in = _MLPEmbedder(256, HIDDEN_SIZE)
        self.vector_in = _MLPEmbedder(VECTOR_DIM, HIDDEN_SIZE)

        sorted_modalities = sorted(in_channels)
        self.early_stream_modulations = nn.ModuleDict(
            {
                **{name: _Modulation(HIDDEN_SIZE) for name in sorted_modalities},
                "txt": _Modulation(HIDDEN_SIZE),
            }
        )
        self.single_stream_modulations = nn.ModuleDict(
            {
                **{name: _Modulation(HIDDEN_SIZE) for name in sorted_modalities},
                "txt": _Modulation(HIDDEN_SIZE),
            }
        )
        self.content_mode_blocks = nn.ModuleDict(
            {name: nn.ModuleList(_ModeBlock() for _ in range(EARLY_DEPTH)) for name in sorted_modalities}
        )
        self.txt_mode_blocks = nn.ModuleList(_ModeBlock() for _ in range(EARLY_DEPTH))
        self.single_blocks = nn.ModuleList(_JointBlock() for _ in range(JOINT_DEPTH))
        self.final_layer = nn.ModuleDict(
            {name: _LastLayer(HIDDEN_SIZE, channels) for name, channels in in_channels.items()}
        )

        self._compiled_text = None
        self._compiled_observation = None
        self._compiled_step = None
        self._compiled_hot = None
        self._packed = False

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
    ) -> BF16InferenceDiT:
        """Strictly load materialized BF16 tensors."""
        with torch.device("meta"):
            model = cls()
        devices = {tensor.device for tensor in state_dict.values()}
        dtypes = {tensor.dtype for tensor in state_dict.values()}
        assert len(devices) == 1 and next(iter(devices)).type != "meta", "weights need one real device"
        assert dtypes == {torch.bfloat16}, "expected BF16 weights"
        model.load_state_dict(state_dict, strict=True, assign=True)
        model._pack_for_inference()
        model.eval()
        return model

    def _pack_for_inference(self) -> None:
        assert not self._packed, "model already packed"
        for collection in (self.content_mode_blocks.values(), (self.txt_mode_blocks,), (self.single_blocks,)):
            for group in collection:
                for block in group:
                    block.pack_input_projection()
        del self.final_layer[VIDEO_COND]
        del self.final_layer[ACTION_COND]
        self._packed = True

    def train(self, mode: bool = True) -> BF16InferenceDiT:
        assert not mode, "BF16InferenceDiT is inference-only"
        super().train(False)
        return self

    def _apply(self, fn, recurse: bool = True):
        assert not self._packed, "prepare inference after device placement"
        return super()._apply(fn, recurse=recurse)

    def compile_static(self) -> None:
        assert not any(
            compiled is not None
            for compiled in (self._compiled_text, self._compiled_observation, self._compiled_step)
        ), "static path already compiled"
        options = {"dynamic": False, "fullgraph": True, "mode": "reduce-overhead"}
        self._compiled_text = torch.compile(self._prepare_text_math, **options)
        self._compiled_observation = torch.compile(self._prepare_observation_math, **options)
        self._compiled_step = torch.compile(self._prepare_step_math, **options)

    def compile_hot(self) -> None:
        assert self._compiled_hot is None, "hot path already compiled"
        self._compiled_hot = torch.compile(
            self._forward_math, dynamic=False, fullgraph=True, mode="reduce-overhead"
        )

    def _weight_spec(self) -> tuple[torch.device, torch.dtype]:
        weight = self.txt_in.weight
        assert weight.dtype == torch.bfloat16 and weight.device.type != "meta", "expected BF16 weights"
        return weight.device, weight.dtype

    def validate_runtime_device(self) -> None:
        device, _ = self._weight_spec()
        assert device.type == "cuda", "prepare after CUDA placement"

    def _embed_timestep(self, timestep: Tensor) -> Tensor:
        with torch.autocast(device_type=timestep.device.type, enabled=False):
            return self.time_in(_timestep_embedding(timestep).to(self.time_in.in_layer.weight.dtype))

    def _prepare_text_math(self, ctx: Tensor, ctx_ids: Tensor, vector: Tensor) -> tuple[Tensor, ...]:
        vector_embedding = self.vector_in(vector)
        zero = torch.zeros(ctx.shape[0], dtype=torch.float32, device=ctx.device)
        static_vector = self._embed_timestep(zero).to(ctx.dtype) + vector_embedding
        early_txt = self.early_stream_modulations["txt"](static_vector)
        early_video_cond = self.early_stream_modulations[VIDEO_COND](static_vector)
        early_action_cond = self.early_stream_modulations[ACTION_COND](static_vector)
        single_txt = self.single_stream_modulations["txt"](static_vector)
        single_video_cond = self.single_stream_modulations[VIDEO_COND](static_vector)
        single_action_cond = self.single_stream_modulations[ACTION_COND](static_vector)
        text_rope = self.pe_embedder(ctx_ids)
        txt = self.txt_in(ctx)
        for depth in range(self.depth):
            txt = self.txt_mode_blocks[depth](txt, text_rope, early_txt)
        return (
            txt,
            text_rope,
            *early_video_cond,
            *early_action_cond,
            *single_txt,
            *single_video_cond,
            *single_action_cond,
            vector_embedding,
        )

    def _prepare_observation_math(
        self,
        text_rope: Tensor,
        video_ids: Tensor,
        video_cond: Tensor,
        video_cond_ids: Tensor,
        action_ids: Tensor,
        action_cond: Tensor,
        action_cond_ids: Tensor,
        early_video_cond: ModulationTuple,
        early_action_cond: ModulationTuple,
    ) -> tuple[Tensor, Tensor, Tensor]:
        video_rope = self.pe_embedder(video_ids)
        video_cond_rope = self.pe_embedder(video_cond_ids)
        action_rope = self.pe_embedder(action_ids)
        action_cond_rope = self.pe_embedder(action_cond_ids)
        encoded_video_cond = self.emb_in[VIDEO_COND](video_cond)
        encoded_action_cond = self.emb_in[ACTION_COND](action_cond)
        for depth in range(self.depth):
            encoded_video_cond = self.content_mode_blocks[VIDEO_COND][depth](
                encoded_video_cond, video_cond_rope, early_video_cond
            )
            encoded_action_cond = self.content_mode_blocks[ACTION_COND][depth](
                encoded_action_cond, action_cond_rope, early_action_cond
            )
        joint_rope = torch.cat((text_rope, video_rope, video_cond_rope, action_rope, action_cond_rope), dim=2)
        return encoded_video_cond, encoded_action_cond, joint_rope

    def _prepare_step_math(
        self,
        vector_embedding: Tensor,
        video_timestep: Tensor,
        action_timestep: Tensor,
    ) -> tuple[Tensor, ...]:
        video_vector = self._embed_timestep(video_timestep).to(vector_embedding.dtype) + vector_embedding
        action_vector = self._embed_timestep(action_timestep).to(vector_embedding.dtype) + vector_embedding
        early_video = self.early_stream_modulations[VIDEO](video_vector)
        early_action = self.early_stream_modulations[ACTION](action_vector)
        single_video = self.single_stream_modulations[VIDEO](video_vector)
        single_action = self.single_stream_modulations[ACTION](action_vector)
        final_values = []
        for name, vector in ((VIDEO, video_vector), (ACTION, action_vector)):
            head = self.final_layer[name]
            activated = head.adaLN_modulation[0](vector)
            shift_weight, scale_weight = head.adaLN_modulation[1].weight.chunk(2)
            final_values.extend(
                (
                    F.linear(activated, shift_weight).unsqueeze(1),
                    F.linear(activated, scale_weight).add_(1).unsqueeze(1),
                )
            )
        return (*early_video, *early_action, *single_video, *single_action, *final_values)

    @torch.no_grad()
    def prepare_text(self, ctx: Tensor, ctx_ids: Tensor, vector: Tensor) -> _PreparedText:
        device, dtype = self._weight_spec()
        batch = ctx.shape[0]
        core = self._compiled_text if self._compiled_text is not None else self._prepare_text_math
        txt, rope, *values = core(ctx, ctx_ids, vector)
        return _PreparedText(
            batch_size=batch,
            dtype=dtype,
            device=device,
            txt_len=ctx.shape[1],
            txt=txt.detach().clone(),
            rope=rope.detach().clone(),
            early_video_cond=tuple(x.detach().clone() for x in values[0:3]),  # type: ignore[arg-type]
            early_action_cond=tuple(x.detach().clone() for x in values[3:6]),  # type: ignore[arg-type]
            single_txt=tuple(x.detach().clone() for x in values[6:9]),  # type: ignore[arg-type]
            single_video_cond=tuple(x.detach().clone() for x in values[9:12]),  # type: ignore[arg-type]
            single_action_cond=tuple(x.detach().clone() for x in values[12:15]),  # type: ignore[arg-type]
            vector_embedding=values[15].detach().clone(),
        )

    @torch.no_grad()
    def prepare_observation(
        self,
        text: _PreparedText,
        *,
        video_ids: Tensor,
        video_cond: Tensor,
        video_cond_ids: Tensor,
        action_ids: Tensor,
        action_cond: Tensor,
        action_cond_ids: Tensor,
    ) -> _PreparedRequest:
        batch, device, dtype = text.batch_size, text.device, text.dtype
        core = (
            self._compiled_observation
            if self._compiled_observation is not None
            else self._prepare_observation_math
        )
        encoded_video_cond, encoded_action_cond, rope = core(
            text.rope,
            video_ids,
            video_cond,
            video_cond_ids,
            action_ids,
            action_cond,
            action_cond_ids,
            text.early_video_cond,
            text.early_action_cond,
        )
        return _PreparedRequest(
            batch_size=batch,
            dtype=dtype,
            device=device,
            txt_len=text.txt_len,
            video_len=video_ids.shape[1],
            video_cond_len=video_cond.shape[1],
            action_len=action_ids.shape[1],
            action_cond_len=action_cond.shape[1],
            txt=text.txt,
            video_cond=encoded_video_cond.detach().clone(),
            action_cond=encoded_action_cond.detach().clone(),
            rope=rope.detach().clone(),
            single_txt=text.single_txt,
            single_video_cond=text.single_video_cond,
            single_action_cond=text.single_action_cond,
            vector_embedding=text.vector_embedding,
        )

    def _normalize_timestep(self, timestep: Tensor, request: _PreparedRequest) -> Tensor:
        if timestep.ndim == 0:
            timestep = timestep.expand(request.batch_size)
        elif timestep.shape == (request.batch_size, 1):
            timestep = timestep[:, 0]
        else:
            assert timestep.shape == (request.batch_size,), "invalid timestep shape"
        assert timestep.dtype == torch.float32 and timestep.device == request.device, (
            "invalid timestep placement"
        )
        return timestep

    @torch.no_grad()
    def prepare_step(
        self,
        request: _PreparedRequest,
        video_timestep: Tensor,
        action_timestep: Tensor,
    ) -> _PreparedStep:
        video_timestep = self._normalize_timestep(video_timestep, request)
        action_timestep = self._normalize_timestep(action_timestep, request)
        core = self._compiled_step if self._compiled_step is not None else self._prepare_step_math
        values = core(request.vector_embedding, video_timestep, action_timestep)
        return _PreparedStep(
            early_video=tuple(x.detach().clone() for x in values[0:3]),  # type: ignore[arg-type]
            early_action=tuple(x.detach().clone() for x in values[3:6]),  # type: ignore[arg-type]
            single_video=tuple(x.detach().clone() for x in values[6:9]),  # type: ignore[arg-type]
            single_action=tuple(x.detach().clone() for x in values[9:12]),  # type: ignore[arg-type]
            final_video_shift=values[12].detach().clone(),
            final_video_scale=values[13].detach().clone(),
            final_action_shift=values[14].detach().clone(),
            final_action_scale=values[15].detach().clone(),
        )

    def prepare_steps(
        self,
        request: _PreparedRequest,
        video_timesteps: Tensor,
        action_timesteps: Tensor,
    ) -> tuple[_PreparedStep, ...]:
        assert video_timesteps.ndim == action_timesteps.ndim == 1, "invalid timestep rank"
        assert video_timesteps.shape == action_timesteps.shape, "timestep shape mismatch"
        return tuple(
            self.prepare_step(request, video_timesteps[index], action_timesteps[index])
            for index in range(video_timesteps.numel())
        )

    def _forward_math(
        self,
        request: _PreparedRequest,
        step: _PreparedStep,
        video: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        video_start = request.txt_len
        video_cond_start = video_start + request.video_len
        action_start = video_cond_start + request.video_cond_len
        action_cond_start = action_start + request.action_len
        video_rope = request.rope[:, :, video_start:video_cond_start]
        action_rope = request.rope[:, :, action_start:action_cond_start]
        encoded_video = self.emb_in[VIDEO](video)
        encoded_action = self.emb_in[ACTION](action)
        for depth in range(self.depth):
            encoded_video = self.content_mode_blocks[VIDEO][depth](
                encoded_video, video_rope, step.early_video
            )
            encoded_action = self.content_mode_blocks[ACTION][depth](
                encoded_action, action_rope, step.early_action
            )
        sequence = torch.cat(
            (request.txt, encoded_video, request.video_cond, encoded_action, request.action_cond), dim=1
        )
        lengths = (
            request.txt_len,
            request.video_len,
            request.video_cond_len,
            request.action_len,
            request.action_cond_len,
        )
        modulations = (
            request.single_txt,
            step.single_video,
            request.single_video_cond,
            step.single_action,
            request.single_action_cond,
        )
        for block in self.single_blocks:
            sequence = block(sequence, request.rope, lengths, modulations)
        video_hidden = sequence[:, video_start:video_cond_start]
        action_hidden = sequence[:, action_start:action_cond_start]
        video_hidden = self.final_layer[VIDEO].norm_final(video_hidden)
        video_hidden.mul_(step.final_video_scale).add_(step.final_video_shift)
        action_hidden = self.final_layer[ACTION].norm_final(action_hidden)
        action_hidden.mul_(step.final_action_scale).add_(step.final_action_shift)
        return self.final_layer[VIDEO].linear(video_hidden), self.final_layer[ACTION].linear(action_hidden)

    @torch.no_grad()
    def forward_prepared(
        self,
        request: _PreparedRequest,
        step: _PreparedStep,
        *,
        video: Tensor,
        action: Tensor,
    ) -> DroidPrediction:
        core = self._compiled_hot if self._compiled_hot is not None else self._forward_math
        video_output, action_output = core(request, step, video, action)
        return DroidPrediction(video_output, action_output)

    def forward(self, *_args: Any, **_kwargs: Any):
        raise RuntimeError("use prepare_text/prepare_observation/prepare_steps/forward_prepared")
