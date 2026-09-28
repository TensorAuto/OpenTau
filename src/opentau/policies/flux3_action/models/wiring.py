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
"""Wire an embodiment-specific action modality into the FLUX Action DiT.

The action-pretrained trunk carries the video model
plus the co-trained action branch: ``action_prediction`` / ``action_prediction_cond`` mode blocks and
stream modulations. It may also carry ``emb_in`` / ``final_layer`` heads of action modalities that are
not yours; their channel count and meaning are not yours, so they are dropped. A finetune adds YOUR
modality on top of it:

* mode blocks and modulations for ``<modality>`` / ``<modality>_cond`` are initialized from the
  co-trained ``action_prediction`` branch (dataset-specific action modalities route through one shared
  action backbone);
* ``emb_in.<modality>``, ``emb_in.<modality>_cond``, ``final_layer.<modality>`` and
  ``final_layer.<modality>_cond`` are fresh, sized to your action channels;
* everything else (video model, text path, single-stream blocks) loads as is.

A checkpoint written by a finetune already contains the embodiment keys and loads with
``strict_heads=True``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn

from ..config import CONTENT_STREAMS
from .transformer import JointSingleSeq, JointSingleSeqParams, LastLayer

SHARED_ACTION_MODE = "action_prediction"
SHARED_KINDS = (
    "content_mode_blocks",
    "early_stream_modulations",
    "single_stream_modulations",
    "late_stream_modulations",
    "late_content_mode_blocks",
)
HEAD_KINDS = ("emb_in", "final_layer")
# Modules of the video trunk that are never in an action checkpoint (audio conditioning is unused).
OPTIONAL_KEY_SUBSTRINGS = ("audio_cond",)
_STREAM_MODULE_KINDS = (
    "emb_in",
    "final_layer",
    "content_mode_blocks",
    "early_stream_modulations",
    "single_stream_modulations",
    "late_stream_modulations",
    "late_content_mode_blocks",
)


def restrict_content_streams(base: JointSingleSeqParams, streams: Sequence[str]) -> JointSingleSeqParams:
    """``base`` with only ``streams`` among its content streams (the action modality is added afterwards)."""
    # Historical training configs can explicitly restore the full trunk architecture.
    legacy_channels = {"image": 128, "image_cond": 128, "audio": 64, "audio_cond": 64}
    unknown = [s for s in streams if s not in base.in_channels and s not in legacy_channels]
    assert not unknown, f"unknown content streams: {unknown}"
    keep = set(streams)
    in_channels = {m: c for m, c in base.in_channels.items() if m in keep}
    in_channels.update(
        {m: legacy_channels[m] for m in streams if m in legacy_channels and m not in in_channels}
    )
    sequence = {k: v for k, v in base.sequence.items() if v in keep}
    sequence.update(
        {f"x_{m}": m for m in streams if m in legacy_channels and m not in base.sequence.values()}
    )
    return replace(base, in_channels=in_channels, sequence=sequence)


def stream_of_key(key: str) -> str | None:
    """Name of the stream a DiT parameter belongs to (``video``, ``txt``, a head modality, ...), else ``None``."""
    parts = key.removeprefix("dit.").split(".")
    if len(parts) >= 3 and parts[0] in _STREAM_MODULE_KINDS:
        return parts[1]
    return None


def action_dit_params(
    base: JointSingleSeqParams,
    modality: str,
    channels: int,
    attn_mode: str | None = None,
    *,
    conditioning_channels: int | None = None,
) -> JointSingleSeqParams:
    """``base`` plus the ``<modality>`` and ``<modality>_cond`` content streams."""
    in_channels = dict(base.in_channels)
    in_channels[modality] = channels
    in_channels[f"{modality}_cond"] = channels if conditioning_channels is None else conditioning_channels
    sequence = dict(base.sequence)
    sequence[f"x_{modality}"] = modality
    sequence[f"x_{modality}_cond"] = f"{modality}_cond"
    params = replace(base, in_channels=in_channels, sequence=sequence)
    if attn_mode is not None:
        params = replace(params, attn_mode=attn_mode)
    return params


def fresh_module_names(modality: str) -> list[str]:
    return [f"{kind}.{m}" for kind in HEAD_KINDS for m in (modality, f"{modality}_cond")]


def remap_shared_action_keys(state_dict: dict[str, torch.Tensor], modality: str) -> dict[str, torch.Tensor]:
    """Route the co-trained action backbone to ``modality``; drop every other action head.

    ``content_mode_blocks.action_prediction.*``      -> ``content_mode_blocks.<modality>.*``
    ``*_stream_modulations.action_prediction_cond.*`` -> ``*_stream_modulations.<modality>_cond.*``
    ``emb_in.action_prediction*.*`` / ``final_layer.action_prediction*.*`` -> dropped, unless they
    already belong to ``modality`` (a finetuned checkpoint)
    """
    ours = (modality, f"{modality}_cond")
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] in HEAD_KINDS and parts[1].startswith(SHARED_ACTION_MODE):
            if parts[1] not in ours:
                continue  # generic head or another embodiment's head: channels are not ours
        elif (
            modality != SHARED_ACTION_MODE
            and len(parts) >= 3
            and parts[0] in SHARED_KINDS
            and parts[1] in (SHARED_ACTION_MODE, f"{SHARED_ACTION_MODE}_cond")
        ):
            parts[1] = modality if parts[1] == SHARED_ACTION_MODE else f"{modality}_cond"
            out[".".join(parts)] = value
            continue
        out[key] = value
    return out


def fresh_head_state_dict(
    hidden_size: int, modality: str, channels: int, seed: int = 0, *, conditioning_channels: int | None = None
) -> dict[str, torch.Tensor]:
    """Tensors for fresh embodiment heads with the reference model's initialization.

    The reference initializes every linear layer xavier-uniform and then zeroes the final layers
    (output projection and its modulation), so a new head predicts zero at first and only its output
    projection moves at the first update. Its released action trunk carries untrained embodiment heads
    in exactly this state. ``seed`` drives the xavier draw of the input projections.
    """
    generator = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for m, width in (
        (modality, channels),
        (f"{modality}_cond", channels if conditioning_channels is None else conditioning_channels),
    ):
        weight = torch.empty(hidden_size, width)
        nn.init.xavier_uniform_(weight, generator=generator)
        out[f"emb_in.{m}.weight"] = weight
        with torch.random.fork_rng(devices=[]):
            layer_state = LastLayer(hidden_size, width).state_dict()
        for k, v in layer_state.items():
            out[f"final_layer.{m}.{k}"] = torch.zeros_like(v)
    return out


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def load_action_checkpoint(
    model: JointSingleSeq, ckpt_path: str, modality: str, *, strict_heads: bool = False, head_seed: int = 0
) -> None:
    """Load a trunk / finetuned ``.safetensors`` into an action DiT built with :func:`action_dit_params`.

    ``strict_heads=True`` demands that the checkpoint already contains the embodiment heads (a finetuned
    checkpoint); ``False`` accepts an action-pretrained trunk and gives the missing heads the reference
    initialization of :func:`fresh_head_state_dict`, drawn from ``head_seed`` so every rank builds the same
    heads whatever its own RNG state.
    """
    from safetensors import safe_open

    dtype = next(model.parameters()).dtype
    # Filter unused content streams before materializing tensors from a full trunk.
    unused = set(CONTENT_STREAMS) - set(model.in_channels)
    with safe_open(ckpt_path, framework="pt", device="cpu") as checkpoint:
        sd = {
            key: checkpoint.get_tensor(key) for key in checkpoint.keys() if stream_of_key(key) not in unused
        }
    sd = remap_shared_action_keys(sd, modality)
    fresh_prefixes = tuple(n + "." for n in fresh_module_names(modality))
    if strict_heads:
        missing_heads = [k for k in model.state_dict() if k.startswith(fresh_prefixes) and k not in sd]
        if missing_heads:
            raise ValueError(f"checkpoint missing required embodiment heads: {missing_heads[:6]}")
    res = model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    missing = [
        k
        for k in res.missing_keys
        if not k.startswith(fresh_prefixes) and not any(s in k for s in OPTIONAL_KEY_SUBSTRINGS)
    ]
    if missing:
        raise ValueError(f"checkpoint missing required keys, e.g. {missing[:6]}")
    fresh_missing = [k for k in res.missing_keys if k.startswith(fresh_prefixes)]
    if fresh_missing:
        emb = model.emb_in[modality].weight
        fresh = fresh_head_state_dict(
            emb.shape[0],
            modality,
            emb.shape[1],
            head_seed,
            conditioning_channels=model.emb_in[f"{modality}_cond"].in_features,
        )
        model.load_state_dict({k: fresh[k].to(dtype) for k in fresh_missing}, strict=False)


def build_action_dit(
    params: JointSingleSeqParams,
    ckpt_path: str | None = None,
    *,
    modality: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    strict_heads: bool = False,
    head_seed: int = 0,
) -> JointSingleSeq:
    """Instantiate the DiT (``params`` already carry the action modality) and load ``ckpt_path``.

    ``ckpt_path=None`` -> random init (wiring tests, or a policy whose weights are loaded afterwards by
    ``from_pretrained``).
    """
    with torch.device(device), default_dtype(dtype):
        model = JointSingleSeq(params)  # built directly in ``dtype``: no fp32 copy on the device
    if ckpt_path is not None:
        load_action_checkpoint(model, ckpt_path, modality, strict_heads=strict_heads, head_seed=head_seed)
    model.eval()
    return model
