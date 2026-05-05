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

"""Tests for the published `TensorAuto/pi07-{high,low}-untrained` checkpoints.

Mirrors the pattern from PR #245's pi06 tests (``test_pi06.py:904-1075``).
For each pi07 sub-policy this verifies that:

1. The published checkpoint loads cleanly via ``from_pretrained`` with no
   missing / unexpected keys (allowing for tied embed_tokens/lm_head, which
   ``save_model_as_safetensor`` de-duplicates on disk).
2. The Gemma 3 text tower + multimodal projector inside the policy are
   byte-for-byte identical to ``google/gemma-3-4b-pt``.
3. The SigLIP vision tower is byte-for-byte identical to the vision tower
   bundled in ``google/gemma-3-4b-pt``, modulo a deterministic 4096 → 1024
   bilinear resample of ``position_embedding`` to match pi07's 448×448 input
   resolution (gemma-3-4b-pt ships at 896×896 / 4096 patches).

Notes:

- pi07 does NOT use ``<locNNNN>`` tokens, so unlike pi06 the reference
  fixture does not call ``ensure_loc_tokens``.
- For the low-level checkpoint, the SigLIP encoder layers at indices
  [3, 7, 11, 15, 19, 23] are wrapped in place by
  :class:`~opentau.policies.pi07.video_encoder.SpaceTimeEncoderLayerWrapper`.
  The wrapper adopts the wrapped ``SiglipEncoderLayer``'s submodules under the
  same attribute names (``self_attn`` / ``layer_norm1`` / ``layer_norm2`` /
  ``mlp``); the temporal extras are a non-persistent buffer + a by-list
  reference. The state_dict keys are therefore byte-identical to a vanilla
  ``SiglipVisionModel`` (pinned by
  ``test_pi07_video_encoder_cpu.py::test_state_dict_keys_match_vanilla_siglip``),
  so the same ``vision_tower`` filter works for both sub-policies.

Heavy: each fixture downloads ~10 GB on CPU. Gated behind ``gpu`` + ``slow``
to stay out of CPU CI. They do NOT actually require CUDA — comparison is on
CPU in bf16 — but the markers double as a "needs heavy infra + network" gate
matching the pi06 tests.
"""

from __future__ import annotations

import pytest
import torch

# State_dict key for the SigLIP position-embedding table inside
# ``Gemma3ForConditionalGeneration``. Stable across all pi07 sub-policies because
# they share ``Gemma3WithExpertModel.gemma3``.
_SIGLIP_POS_EMBED_KEY = "model.vision_tower.vision_model.embeddings.position_embedding.weight"

# 448**2 / 14**2 = 1024. Both pi07 sub-policies run SigLIP at 448×448.
# google/gemma-3-4b-pt ships at 896×896 (4096 patches), so the reference
# state_dict needs a deterministic bilinear resample to 1024 patches before
# it can be compared against the policy weights.
_TARGET_NUM_PATCHES = 1024


@pytest.fixture(scope="module")
def pi07_high_untrained_policy():
    """Load ``TensorAuto/pi07-high-untrained`` once on CPU; reused across the module."""
    from opentau.policies.pi07.high_level_planner.modeling_pi07_high_level import (
        PI07HighLevelPlannerPolicy,
    )

    return PI07HighLevelPlannerPolicy.from_pretrained("TensorAuto/pi07-high-untrained")


@pytest.fixture(scope="module")
def pi07_low_untrained_policy():
    """Load ``TensorAuto/pi07-low-untrained`` once on CPU; reused across the module."""
    from opentau.policies.pi07.low_level.modeling_pi07_low_level import PI07LowLevelPolicy

    return PI07LowLevelPolicy.from_pretrained("TensorAuto/pi07-low-untrained")


@pytest.fixture(scope="module")
def gemma3_4b_pt_aligned_state_dict():
    """``google/gemma-3-4b-pt``'s state_dict, aligned to pi07's 1024-patch SigLIP.

    The published Gemma 3-4B PT checkpoint runs SigLIP at 896×896 (4096 patches);
    pi07 runs at 448×448 (1024 patches), so the reference ``position_embedding``
    must be bilinearly resampled before any byte-for-byte comparison. This is
    the same recipe the build script applies before saving the published
    policy weights, so byte-equality holds.

    Returns a ``dict[str, Tensor]`` rather than the model itself because the
    resampled position embedding has a different shape than the original
    ``nn.Embedding`` parameter — so we can't mutate the model in place.

    Unlike the pi06 fixture, this does NOT call ``ensure_loc_tokens``: pi07
    does not extend the Gemma 3 vocabulary with ``<locNNNN>`` tokens.
    """
    from transformers import Gemma3ForConditionalGeneration

    from opentau.utils.vision_utils import bilinear_resample_pos_embed

    model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-pt", torch_dtype=torch.bfloat16)

    state = model.state_dict()
    state[_SIGLIP_POS_EMBED_KEY] = bilinear_resample_pos_embed(
        state[_SIGLIP_POS_EMBED_KEY], target_num_patches=_TARGET_NUM_PATCHES
    )
    del model
    return state


def _diff_state_dicts_against_reference(
    policy_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    *,
    name_filter,
) -> tuple[int, list[str]]:
    """Compare every entry of ``reference_state`` whose name passes ``name_filter``
    against the matching entry in ``policy_state``. Returns ``(checked, mismatches)``.

    ``torch.equal`` is used (not ``allclose``) because the policy weights came
    from the same source bf16 tensors with no precision-changing transform —
    they must be byte-identical, and any drift is a real bug worth surfacing.
    """
    mismatches: list[str] = []
    checked = 0
    for name, ref in reference_state.items():
        if not name_filter(name):
            continue
        pol = policy_state.get(name)
        if pol is None:
            mismatches.append(f"{name}: missing in policy state_dict")
            continue
        if pol.shape != ref.shape:
            mismatches.append(f"{name}: shape mismatch {tuple(pol.shape)} vs {tuple(ref.shape)}")
            continue
        if not torch.equal(pol, ref):
            max_abs = (pol.float() - ref.float()).abs().max().item()
            mismatches.append(f"{name}: tensor not equal (max abs diff {max_abs:g})")
            continue
        checked += 1
    return checked, mismatches


def _assert_no_missing_or_unexpected(policy: torch.nn.Module, hf_repo: str) -> None:
    """Assert every state_dict key is on disk in ``model.safetensors``, modulo
    storage-shared tied weights.

    Each tied group (keys sharing the same ``data_ptr``) must have exactly one
    representative on disk, and the rest are ``allowed_missing``. Non-tied
    keys must be in the safetensors 1:1 — that's the case that actually
    catches a save-side regression (e.g. silently dropping an action-expert
    layer).

    Args:
        policy: The loaded policy instance.
        hf_repo: The Hub repo id whose ``model.safetensors`` to check against
            (e.g. ``"TensorAuto/pi07-high-untrained"``).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ckpt_path = hf_hub_download(hf_repo, "model.safetensors")
    saved_keys = set(load_file(ckpt_path).keys())
    loaded_state = policy.state_dict()
    expected_keys = set(loaded_state.keys())

    # Build groups of state_dict keys that share storage (tied weights).
    ptr_to_names: dict[int, list[str]] = {}
    for name, tensor in loaded_state.items():
        ptr_to_names.setdefault(tensor.data_ptr(), []).append(name)
    tied_groups = [names for names in ptr_to_names.values() if len(names) > 1]

    # Each tied group must have exactly one representative on disk.
    tied_group_violations: list[str] = []
    allowed_missing: set[str] = set()
    for group in tied_groups:
        on_disk = [n for n in group if n in saved_keys]
        if len(on_disk) != 1:
            tied_group_violations.append(
                f"tied group {sorted(group)}: expected exactly 1 on disk, got {len(on_disk)}"
            )
        for n in group:
            if n not in saved_keys:
                allowed_missing.add(n)

    missing = expected_keys - saved_keys - allowed_missing
    unexpected = saved_keys - expected_keys
    assert not tied_group_violations, "\n".join(tied_group_violations)
    assert not missing and not unexpected, (
        f"missing in safetensors ({len(missing)}): {sorted(missing)[:10]}\n"
        f"unexpected in safetensors ({len(unexpected)}): {sorted(unexpected)[:10]}"
    )


# pi07 high-level (TensorAuto/pi07-high-untrained)


@pytest.mark.gpu
@pytest.mark.slow
def test_pi07_high_untrained_loads_with_no_missing_or_unexpected_keys(pi07_high_untrained_policy):
    """Published high-level checkpoint round-trips through ``from_pretrained``
    with every state_dict key accounted for (modulo tied embed_tokens/lm_head)."""
    _assert_no_missing_or_unexpected(pi07_high_untrained_policy, "TensorAuto/pi07-high-untrained")


@pytest.mark.gpu
@pytest.mark.slow
def test_pi07_high_untrained_vlm_matches_gemma3_4b_pt(
    pi07_high_untrained_policy, gemma3_4b_pt_aligned_state_dict
):
    """Gemma 3 text tower + multimodal projector inside the published high-level
    checkpoint are byte-identical to ``google/gemma-3-4b-pt`` (vision tower
    checked separately in
    ``test_pi07_high_untrained_siglip_matches_gemma3_4b_pt``)."""
    pol_state = pi07_high_untrained_policy.model.gemma3_with_expert.gemma3.state_dict()

    checked, mismatches = _diff_state_dicts_against_reference(
        pol_state,
        gemma3_4b_pt_aligned_state_dict,
        name_filter=lambda name: "vision_tower" not in name,
    )
    assert checked > 0, "Sanity: should have compared at least one VLM (non-vision) param"
    assert not mismatches, "VLM (Gemma 3 text + projector) mismatches:\n" + "\n".join(mismatches[:20])


@pytest.mark.gpu
@pytest.mark.slow
def test_pi07_high_untrained_siglip_matches_gemma3_4b_pt(
    pi07_high_untrained_policy, gemma3_4b_pt_aligned_state_dict
):
    """SigLIP vision tower inside the published high-level checkpoint is
    byte-identical to the vision tower bundled in ``google/gemma-3-4b-pt``,
    modulo the deterministic bilinear ``position_embedding`` resample from
    4096 (896×896 published) to 1024 (448×448 pi07) patches that the build
    script applies."""
    pol_state = pi07_high_untrained_policy.model.gemma3_with_expert.gemma3.state_dict()

    checked, mismatches = _diff_state_dicts_against_reference(
        pol_state,
        gemma3_4b_pt_aligned_state_dict,
        name_filter=lambda name: "vision_tower" in name,
    )
    assert checked > 0, "Sanity: should have compared at least one SigLIP param"
    assert not mismatches, "SigLIP vision tower mismatches:\n" + "\n".join(mismatches[:20])


# pi07 low-level (TensorAuto/pi07-low-untrained)


@pytest.mark.gpu
@pytest.mark.slow
def test_pi07_low_untrained_loads_with_no_missing_or_unexpected_keys(pi07_low_untrained_policy):
    """Published low-level checkpoint round-trips through ``from_pretrained``
    with every state_dict key accounted for (modulo tied embed_tokens/lm_head).

    The low-level policy includes a ``SpaceTimeSiglipVideoEncoder`` that wraps
    six SigLIP encoder layers in place; the wrapper exposes vanilla SigLIP
    keys (pinned by ``test_pi07_video_encoder_cpu.py``), so this assertion
    has the same shape as the high-level case."""
    _assert_no_missing_or_unexpected(pi07_low_untrained_policy, "TensorAuto/pi07-low-untrained")


@pytest.mark.gpu
@pytest.mark.slow
def test_pi07_low_untrained_vlm_matches_gemma3_4b_pt(
    pi07_low_untrained_policy, gemma3_4b_pt_aligned_state_dict
):
    """Gemma 3 text tower + multimodal projector are byte-identical for the
    low-level checkpoint as well — the SpaceTime wrappers only touch the
    SigLIP encoder, not the text tower or the projector."""
    pol_state = pi07_low_untrained_policy.model.gemma3_with_expert.gemma3.state_dict()

    checked, mismatches = _diff_state_dicts_against_reference(
        pol_state,
        gemma3_4b_pt_aligned_state_dict,
        name_filter=lambda name: "vision_tower" not in name,
    )
    assert checked > 0, "Sanity: should have compared at least one VLM (non-vision) param"
    assert not mismatches, "VLM mismatches (low-level):\n" + "\n".join(mismatches[:20])


@pytest.mark.gpu
@pytest.mark.slow
def test_pi07_low_untrained_siglip_matches_gemma3_4b_pt(
    pi07_low_untrained_policy, gemma3_4b_pt_aligned_state_dict
):
    """SigLIP vision tower is byte-identical to ``google/gemma-3-4b-pt``'s
    despite ``SpaceTimeEncoderLayerWrapper`` being installed at indices
    [3, 7, 11, 15, 19, 23].

    The wrapper adopts the wrapped ``SiglipEncoderLayer``'s submodules under
    the same attribute names (``self_attn`` / ``layer_norm1`` /
    ``layer_norm2`` / ``mlp``), and the temporal extras are a
    ``persistent=False`` buffer + a by-list reference — neither registers
    new state_dict entries. Structural equality is pinned by
    ``test_pi07_video_encoder_cpu.py``; this test extends the guarantee to
    value equality against the public Gemma 3-4B PT weights, which is the
    actual user-visible contract."""
    pol_state = pi07_low_untrained_policy.model.gemma3_with_expert.gemma3.state_dict()

    checked, mismatches = _diff_state_dicts_against_reference(
        pol_state,
        gemma3_4b_pt_aligned_state_dict,
        name_filter=lambda name: "vision_tower" in name,
    )
    assert checked > 0, "Sanity: should have compared at least one SigLIP param"
    assert not mismatches, "SigLIP mismatches (low-level):\n" + "\n".join(mismatches[:20])
