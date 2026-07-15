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

"""Offline tests for the config-derivation half of ``extract_cosmos3_reasoner``.

The tensor-remapping half is exercised end-to-end when extracting a real snapshot;
these lock ``_qwen3vl_config_from_snapshot`` — the only pure function with real
failure modes: a missing root ``config.json`` (e.g. a snapshot downloaded with the
pre-parametrization include list), a non-omni config, and both released family
geometries (Nano = Qwen3-VL-8B, Super = Qwen3-VL-32B)."""

import json

import pytest

from opentau.scripts.extract_cosmos3_reasoner import _qwen3vl_config_from_snapshot


def _root_config(hidden: int, layers: int, heads: int, inter: int) -> dict:
    """A minimal Cosmos3 omni root config carrying the reasoner geometry."""
    return {
        "architectures": ["Cosmos3ForConditionalGeneration"],
        "model_type": "cosmos3_omni",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "max_position_embeddings": 262144,
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "depth": 27,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": hidden,
            "deepstack_visual_indexes": [8, 16, 24],
            "num_position_embeddings": 2304,
            "in_channels": 3,
        },
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": False,
    }


@pytest.mark.parametrize(
    "hidden,layers,heads,inter",
    [
        (4096, 36, 32, 12288),  # Cosmos3-Nano reasoner == Qwen3-VL-8B
        (5120, 64, 64, 25600),  # Cosmos3-Super reasoner == Qwen3-VL-32B
    ],
)
def test_geometry_derived_from_root_config(tmp_path, hidden, layers, heads, inter):
    (tmp_path / "config.json").write_text(json.dumps(_root_config(hidden, layers, heads, inter)))
    cfg = _qwen3vl_config_from_snapshot(str(tmp_path))
    t, v = cfg.text_config, cfg.vision_config
    assert (t.hidden_size, t.num_hidden_layers, t.num_attention_heads, t.intermediate_size) == (
        hidden,
        layers,
        heads,
        inter,
    )
    # Family constants the expert cross-attention depends on, taken verbatim.
    assert (t.num_key_value_heads, t.head_dim) == (8, 128)
    assert t.max_position_embeddings == 262144
    assert (v.out_hidden_size, v.depth) == (hidden, 27)
    assert cfg.image_token_id == 151655
    assert cfg.tie_word_embeddings is False


def test_missing_root_config_fails_fast(tmp_path):
    """A snapshot without the root config.json (e.g. downloaded with the old include
    list) must fail immediately with an actionable message, not mid-extraction."""
    with pytest.raises(FileNotFoundError, match="config.json"):
        _qwen3vl_config_from_snapshot(str(tmp_path))


@pytest.mark.parametrize("missing", ["text_config", "vision_config", "image_token_id"])
def test_non_omni_root_config_raises_diagnostic_keyerror(tmp_path, missing):
    root = _root_config(4096, 36, 32, 12288)
    del root[missing]
    (tmp_path / "config.json").write_text(json.dumps(root))
    with pytest.raises(KeyError, match=missing):
        _qwen3vl_config_from_snapshot(str(tmp_path))
