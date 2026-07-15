#!/usr/bin/env python
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

"""Extract the **reasoning tower** of an NVIDIA ``Cosmos3`` omni model (``Cosmos3-Super``
or ``Cosmos3-Nano``) into a standalone Qwen3-VL checkpoint for the cosmos3 /
cosmos3_nano policy backbones.

The Cosmos3 omni models (HF: ``cosmos3_omni`` / diffusers ``Cosmos3OmniTransformer``,
ungated) are interleaved Mixture-of-Transformers: every layer computes one joint
attention over both token streams but carries **separate per-path weights** -- the
autoregressive **reasoner / text** path (``mlp``, ``self_attn.to_*``) and the diffusion
**generation** path (``mlp_moe_gen``, ``self_attn.add_*``/``*_moe_gen``) -- plus
audio/action/video generation modules. The reasoner geometry is byte-for-byte the
matching stock Qwen3-VL size -- Qwen3-VL-32B for Super (~64.6B total; hidden 5120, 64
layers, 64 heads, 8 KV, head_dim 128, intermediate 25600), Qwen3-VL-8B for Nano
(~15.75B total; hidden 4096, 36 layers, 32 heads, same 8 KV / head_dim 128) -- and the
``vision_encoder/`` is a stock ``Qwen3VLVisionModel`` in both. The geometry is read from
the snapshot's root ``config.json`` (``text_config`` / ``vision_config``), so the same
script extracts every family member.

This script keeps only the reasoner tower and drops the generation tower, producing a
standard ``Qwen3VLForConditionalGeneration`` checkpoint:

  reasoner text (from ``transformer/``):
    embed_tokens                         -> model.language_model.embed_tokens
    norm                                 -> model.language_model.norm
    lm_head                              -> lm_head
    layers.N.input_layernorm             -> model.language_model.layers.N.input_layernorm
    layers.N.post_attention_layernorm    -> model.language_model.layers.N.post_attention_layernorm
    layers.N.mlp.{gate,up,down}_proj     -> model.language_model.layers.N.mlp.{...}
    layers.N.self_attn.to_q/to_k/to_v    -> model.language_model.layers.N.self_attn.{q,k,v}_proj
    layers.N.self_attn.to_out            -> model.language_model.layers.N.self_attn.o_proj
    layers.N.self_attn.norm_q/norm_k     -> model.language_model.layers.N.self_attn.{q,k}_norm
  reasoner vision (from ``vision_encoder/``, already Qwen3VLVisionModel):
    <key>                                -> model.visual.<key>

  DROPPED (generation tower): *mlp_moe_gen*, *_moe_gen norms, self_attn.add_{q,k,v}_proj,
  self_attn.to_add_out, self_attn.norm_added_{q,k}, *_modality_embed, action/audio_proj_*,
  proj_in/proj_out (latent patch), time_embedder.

Usage::

    # one-time, on a box with the (ungated) Cosmos3 download + enough RAM/disk
    huggingface-cli download nvidia/Cosmos3-Nano --include "config.json" "transformer/*" \
        "vision_encoder/*" "text_tokenizer/*" "tokenizer_config.json" \
        "preprocessor_config.json" "video_preprocessor_config.json"
    python -m opentau.scripts.extract_cosmos3_reasoner \
        --cosmos3-path <hf-snapshot-dir> --out-dir <reasoner-checkpoint-dir>

(Same for ``nvidia/Cosmos3-Super``; the root ``config.json`` must be part of the
download since it carries the reasoner geometry.) The resulting directory is a normal
Qwen3-VL checkpoint; point the cosmos3 / cosmos3_nano policy at it via
``--policy.pretrained_backbone_repo_id=<reasoner-checkpoint-dir>``.
"""

import argparse
import glob
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

# Per-layer attention key renames (Cosmos diffusers attention -> Qwen3-VL text attention).
_ATTN_RENAME = {
    "self_attn.to_q": "self_attn.q_proj",
    "self_attn.to_k": "self_attn.k_proj",
    "self_attn.to_v": "self_attn.v_proj",
    "self_attn.to_out": "self_attn.o_proj",
    "self_attn.norm_q": "self_attn.q_norm",
    "self_attn.norm_k": "self_attn.k_norm",
}

# Substrings that mark a weight as belonging to the generation tower / non-reasoner modules.
_DROP_SUBSTRINGS = (
    "moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added",
    "modality_embed",
    "action_proj",
    "audio_proj",
    "time_embedder",
)
# Top-level keys to drop (latent-patch diffusion projections).
_DROP_PREFIXES = ("proj_in", "proj_out")


def _is_reasoner_key(key: str) -> bool:
    drop = any(s in key for s in _DROP_SUBSTRINGS) or any(key.startswith(p) for p in _DROP_PREFIXES)
    return not drop


def _remap_transformer_key(key: str) -> str | None:
    """Map a Cosmos3 omni ``transformer/`` key to its Qwen3-VL name, or None to drop it."""
    if not _is_reasoner_key(key):
        return None
    if key == "embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if key == "norm.weight":
        return "model.language_model.norm.weight"
    if key == "lm_head.weight":
        return "lm_head.weight"
    m = re.match(r"layers\.(\d+)\.(.+)", key)
    if m is None:
        return None  # any other top-level key is generation-side; drop it
    layer, rest = m.group(1), m.group(2)
    for src, dst in _ATTN_RENAME.items():
        if rest.startswith(src):
            rest = dst + rest[len(src) :]
            break
    return f"model.language_model.layers.{layer}.{rest}"


def _iter_safetensors(folder: str):
    """Yield (key, tensor) over all .safetensors shards in ``folder`` (lazy)."""
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {folder}")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 -- safe_open is not dict-iterable
                yield key, f.get_tensor(key)


def _qwen3vl_config_from_snapshot(cosmos3_path: str) -> Qwen3VLConfig:
    """Build the reasoner's ``Qwen3VLConfig`` from the snapshot's root ``config.json``.

    The Cosmos3 omni root config carries the reasoner's ``text_config`` /
    ``vision_config`` verbatim (byte-for-byte the matching stock Qwen3-VL size:
    32B for Super, 8B for Nano) plus the Qwen3-VL special token ids. Reading the
    geometry from the source instead of hardcoding one size lets this script extract
    every family member -- and keeps source fields the old hardcoded config silently
    defaulted (e.g. ``max_position_embeddings`` 262144, vs the transformers default
    128000 that ``TensorAuto/cosmos3-reason-32b`` shipped with; harmless at our
    prompt lengths, but source-verbatim is the safer default going forward).
    """
    cfg_path = os.path.join(cosmos3_path, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f'{cfg_path} not found -- add "config.json" to the snapshot download includes; '
            "the root Cosmos3 config carries the reasoner text/vision geometry."
        )
    with open(cfg_path) as f:
        src = json.load(f)
    for key in (
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        if key not in src:
            raise KeyError(f"'{key}' missing from {cfg_path} -- not a Cosmos3 omni root config?")
    return Qwen3VLConfig(
        text_config=src["text_config"],
        vision_config=src["vision_config"],
        image_token_id=src["image_token_id"],
        video_token_id=src["video_token_id"],
        vision_start_token_id=src["vision_start_token_id"],
        vision_end_token_id=src["vision_end_token_id"],
        tie_word_embeddings=src.get("tie_word_embeddings", False),
        dtype="bfloat16",
    )


def extract(cosmos3_path: str, out_dir: str, verify: bool = True) -> None:
    """Build a standalone Qwen3-VL reasoner checkpoint from a Cosmos3 omni snapshot."""
    # Derive the reasoner geometry first so a missing/bad root config.json fails fast,
    # before the long tensor-loading pass over the multi-GB snapshot.
    config = _qwen3vl_config_from_snapshot(cosmos3_path)
    n_layers = config.text_config.num_hidden_layers
    hidden = config.text_config.hidden_size
    print(f"reasoner geometry from root config.json: {n_layers} layers, hidden {hidden}")

    transformer_dir = os.path.join(cosmos3_path, "transformer")
    vision_dir = os.path.join(cosmos3_path, "vision_encoder")
    state_dict: dict[str, torch.Tensor] = {}

    kept = dropped = 0
    for key, tensor in _iter_safetensors(transformer_dir):
        new_key = _remap_transformer_key(key)
        if new_key is None:
            dropped += 1
            continue
        state_dict[new_key] = tensor
        kept += 1
    print(f"transformer/: kept {kept} reasoner tensors, dropped {dropped} generation tensors")

    n_vis = 0
    for key, tensor in _iter_safetensors(vision_dir):
        state_dict[f"model.visual.{key}"] = tensor
        n_vis += 1
    print(f"vision_encoder/: mapped {n_vis} tensors -> model.visual.*")

    os.makedirs(out_dir, exist_ok=True)

    # Instantiate on meta (no real storage), then assign the loaded bf16 tensors directly.
    # The plain ``Qwen3VLForConditionalGeneration(config)`` constructor would otherwise
    # materialize all ~32B params in fp32 and random-init them (immediately overwritten),
    # roughly doubling peak RAM. ``assign=True`` swaps the meta params for the state-dict
    # tensors as-is, so the saved checkpoint keeps the source bf16 dtype with no fp32 detour.
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)

    if verify:
        expected = set(model.state_dict().keys())
        got = set(state_dict.keys())
        missing_keys, unexpected_keys = expected - got, got - expected
        # tie_word_embeddings=False, so lm_head should be present; tolerate it if tied.
        if missing_keys:
            print(
                f"WARNING: {len(missing_keys)} expected keys missing (first 15): {sorted(missing_keys)[:15]}"
            )
        if unexpected_keys:
            print(
                f"WARNING: {len(unexpected_keys)} unexpected keys (first 15): {sorted(unexpected_keys)[:15]}"
            )
        if not missing_keys and not unexpected_keys:
            print("✓ key sets match Qwen3VLForConditionalGeneration exactly")

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        # With assign=True any unmatched param stays on the meta device; saving it would
        # ship empty weights, so refuse rather than write a silently-broken checkpoint.
        raise RuntimeError(
            f"{len(missing)} params had no tensor in the extracted state dict and would remain on "
            f"meta: {sorted(missing)[:15]} -- aborting so we never save an incomplete checkpoint."
        )

    config.save_pretrained(out_dir)
    # save_pretrained shards + writes a correct index from the (now bf16, real) params.
    model.save_pretrained(out_dir, safe_serialization=True)
    del model, state_dict

    # Copy tokenizer / processor configs so the dir is a usable Qwen3-VL checkpoint.
    for fname in (
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
    ):
        src = os.path.join(cosmos3_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
    tok_dir = os.path.join(cosmos3_path, "text_tokenizer")
    if os.path.isdir(tok_dir):
        for f in os.listdir(tok_dir):
            shutil.copy2(os.path.join(tok_dir, f), os.path.join(out_dir, f))
    print(f"✓ wrote Qwen3-VL reasoner checkpoint ({n_layers} layers, hidden {hidden}) to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cosmos3-path",
        required=True,
        help="Local snapshot dir of nvidia/Cosmos3-Super or nvidia/Cosmos3-Nano (must include the root config.json).",
    )
    ap.add_argument("--out-dir", required=True, help="Output dir for the extracted Qwen3-VL reasoner.")
    ap.add_argument("--no-verify", action="store_true", help="Skip the meta-device key-set verification.")
    args = ap.parse_args()
    extract(args.cosmos3_path, args.out_dir, verify=not args.no_verify)


if __name__ == "__main__":
    main()
