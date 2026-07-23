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

"""SigLIP float32-embedding / bf16-encoder dtype-bridge regression tests.

Locks the fix that keeps the SigLIP patch/position embeddings in float32 (openpi parity)
while the encoder runs in bfloat16. The pinning itself lives on the full
``PaliGemmaWithExpertModel`` (a ~3B random-init build, exercised by the GPU suite); here we
lock the two things a CPU test can cover cheaply and that would otherwise regress silently:

1. the patched ``SiglipVisionTransformer.forward`` bridges float32 embeddings into a bfloat16
   encoder instead of raising a dtype mismatch, and is a faithful no-op when the tower is
   single-dtype;
2. ``pi05_mem``'s ``RLDXVideoEncoder``, which hand-rolls the SigLIP forward, reproduces the
   same bridge and so survives the mixed-dtype state its shared vision tower is left in.
"""

import torch
from transformers import PaliGemmaConfig, SiglipVisionConfig, SiglipVisionModel
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

# Importing the policy package applies opentau.utils.transformers_patch, which installs the
# patched SiglipVisionTransformer.forward under test.
import opentau.utils.transformers_patch  # noqa: F401
from opentau.policies.pi05_mem.rldx_video_encoder import RLDXVideoEncoder
from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder
from opentau.policies.utils import to_dtype_preserving_siglip_float32

# The three parameter names pinned to float32 by to_bfloat16_like_physical_intelligence.
PINNED_SUFFIXES = (
    "embeddings.patch_embedding.weight",
    "embeddings.patch_embedding.bias",
    "embeddings.position_embedding.weight",
)


def _tiny_siglip():
    cfg = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_image_tokens=256,
        patch_size=14,
        image_size=224,
        vision_use_head=False,
    )
    return SiglipVisionModel(cfg).eval()


def _tiny_projector():
    pali_cfg = PaliGemmaConfig(
        vision_config={"hidden_size": 64, "model_type": "siglip_vision_model", "projection_dim": 32},
        text_config={"hidden_size": 32, "model_type": "gemma", "vocab_size": 64},
        projection_dim=32,
    )
    return PaliGemmaMultiModalProjector(pali_cfg).eval()


def _pin_embeddings_float32(vision_tower):
    """Reproduce the mixed-dtype state: bf16 everywhere, float32 patch/pos embeddings."""
    vision_tower.to(torch.bfloat16)
    emb = vision_tower.vision_model.embeddings
    emb.patch_embedding.to(torch.float32)
    emb.position_embedding.to(torch.float32)
    return vision_tower


def test_embedding_param_names_match_pinning_selectors():
    """Guards the pinning selectors against a transformers rename of the SigLIP embeddings."""
    names = dict(_tiny_siglip().named_parameters())
    for suffix in PINNED_SUFFIXES:
        assert any(n.endswith(suffix) for n in names), f"no SigLIP param ends with {suffix!r}"


def test_patched_forward_is_noop_when_single_dtype():
    """In uniform float32 the bridge does nothing: output matches the hand-rolled forward."""
    vt = _tiny_siglip()
    torch.manual_seed(0)
    pixels = torch.randn(1, 3, 224, 224)

    hidden = vt.vision_model.embeddings(pixels)
    for layer in vt.vision_model.encoder.layers:
        hidden = layer(hidden, None)
    reference = vt.vision_model.post_layernorm(hidden)

    out = vt(pixels).last_hidden_state
    assert out.dtype == torch.float32
    assert torch.equal(out, reference)


def test_patched_forward_bridges_float32_embeddings_into_bf16_encoder():
    """The motivating case: pinned float32 embeddings must not crash the bfloat16 encoder."""
    vt = _pin_embeddings_float32(_tiny_siglip())
    assert vt.vision_model.embeddings.patch_embedding.weight.dtype == torch.float32
    assert vt.vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16

    out = vt(torch.randn(1, 3, 224, 224)).last_hidden_state
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_rldx_video_encoder_survives_mixed_dtype_tower():
    """pi05_mem's hand-rolled SigLIP forward must reproduce the bridge (issue: it bypassed it)."""
    vt = _pin_embeddings_float32(_tiny_siglip())
    projector = _tiny_projector().to(torch.bfloat16)
    encoder = RLDXVideoEncoder(vt, projector, max_num_frames=2).eval()

    video = torch.rand(1, 1, 3, 224, 224)  # (B, T, C, H, W) in [0, 1]
    with torch.no_grad():
        out = encoder(video)

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    assert out.shape[0] == 1


def test_spacetime_video_encoder_survives_mixed_dtype_tower():
    """The DEFAULT pi05_mem / pi07_paligemma encoder also hand-rolls the SigLIP forward."""
    vt = _tiny_siglip()
    projector = _tiny_projector().to(torch.bfloat16)
    encoder = SpaceTimeSiglipVideoEncoder(vt, projector, max_num_frames=2).eval()

    # SpaceTime wraps layers in place inside ``vt``; put the whole tower (incl. any temporal
    # modules and the encoder object) in bfloat16, then pin the patch/position embeddings back
    # to float32 — the mixed-dtype state the pinning + build leaves the encoder in.
    vt.to(torch.bfloat16)
    encoder.to(torch.bfloat16)
    vt.vision_model.embeddings.patch_embedding.to(torch.float32)
    vt.vision_model.embeddings.position_embedding.to(torch.float32)

    with torch.no_grad():
        # T=1 short-circuits temporal attention (isolates the embeddings -> encoder bridge);
        # T=2 additionally builds the temporal mask, which is derived from the bridged dtype.
        out_t1 = encoder(torch.rand(1, 1, 3, 224, 224))
        out_t2 = encoder(
            torch.rand(1, 2, 3, 224, 224), obs_history_is_pad=torch.zeros(1, 2, dtype=torch.bool)
        )

    for out in (out_t1, out_t2):
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()
        assert out.shape[0] == 1


class _DummyPolicy(torch.nn.Module):
    """Names the tower ``vision_tower`` so param names match the helper's suffixes."""

    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower


def test_to_dtype_helper_preserves_pinned_embeddings_across_bf16_cast():
    """The serving/inference entry points cast through this helper to keep the float32 masters."""
    policy = _DummyPolicy(_pin_embeddings_float32(_tiny_siglip()))
    to_dtype_preserving_siglip_float32(policy, dtype=torch.bfloat16)

    emb = policy.vision_tower.vision_model.embeddings
    assert emb.patch_embedding.weight.dtype == torch.float32
    assert emb.patch_embedding.bias.dtype == torch.float32
    assert emb.position_embedding.weight.dtype == torch.float32
    assert policy.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16


def test_to_dtype_helper_is_noop_for_uniform_bf16_tower():
    """Gemma3-style tower: embeddings are not pinned, so the helper must not force float32."""
    policy = _DummyPolicy(_tiny_siglip().to(torch.bfloat16))
    to_dtype_preserving_siglip_float32(policy, dtype=torch.bfloat16)

    emb = policy.vision_tower.vision_model.embeddings
    assert emb.patch_embedding.weight.dtype == torch.bfloat16
    assert emb.position_embedding.weight.dtype == torch.bfloat16
