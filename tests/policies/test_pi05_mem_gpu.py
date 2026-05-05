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

"""GPU regression tests for the SigLIP space-time video encoder.

These tests cannot run on the reporter's Mac (Apple Silicon, no CUDA); every
test is skipped unless ``torch.cuda.is_available()`` and also carries the
``@pytest.mark.gpu`` marker so a standard ``pytest -m 'not gpu'`` keeps them
out of CPU-only CI.

Conservative choices because these are unreviewable without GPU access:
  - Use small ``B`` and ``T`` to keep test runtime low.
  - Compare with ``atol=rtol=1e-3`` rather than chasing fp precision — bf16
    and cross-driver numerics easily drift past ``1e-4``.
  - Every test prints its dtype and device so diagnostic output remains
    useful without a debugger on hand.
"""

from __future__ import annotations

import pytest
import torch

# Import the patched PaliGemma class through opentau so the transformers
# patch is guaranteed active (it rewrites ``PaliGemmaModel.get_image_features``
# to drop the ``/ sqrt(hidden_size)`` scaling the raw HF class applies).
from opentau.utils.transformers_patch import PaliGemmaForConditionalGeneration

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


# Keep these tests reasonably fast: single batch element, few frames.
_PALIGEMMA_REPO = "google/paligemma-3b-pt-224"


def _load_paligemma(dtype=torch.bfloat16):
    """Load PaliGemma once and return (full_model, vision_tower, projector).

    The video encoder requires vision_tower + multi_modal_projector by
    reference — we keep ``full_model`` alive because dropping it would
    garbage-collect the submodules."""
    full = PaliGemmaForConditionalGeneration.from_pretrained(_PALIGEMMA_REPO, torch_dtype=dtype)
    full = full.to("cuda").eval()
    return full, full.vision_tower, full.multi_modal_projector


def _build_encoder(*, num_frames: int, freeze_encoder: bool, dtype=torch.bfloat16):
    """Build a SpaceTimeSiglipVideoEncoder wrapping a fresh PaliGemma's
    vision_tower + projector. Returns (encoder, full_paligemma, vision_tower,
    projector) so the caller can keep references alive and assert on params."""
    from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder

    full, vision_tower, projector = _load_paligemma(dtype=dtype)
    if freeze_encoder:
        for p in vision_tower.parameters():
            p.requires_grad = False
    encoder = SpaceTimeSiglipVideoEncoder(
        vision_tower=vision_tower,
        multi_modal_projector=projector,
        max_num_frames=num_frames,
        spacetime_layer_stride=4,
    ).to("cuda")
    return encoder, full, vision_tower, projector


@pytest.mark.gpu
@pytest.mark.slow
def test_forward_shape_and_dtype_cuda():
    """Shape and dtype plumbing on a realistic T=8 batch."""
    encoder, *_ = _build_encoder(num_frames=8, freeze_encoder=True)
    encoder.eval()
    print("[shape] device=cuda dtype=bfloat16 num_frames=8")

    video = torch.rand(2, 8, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = encoder(video)
    assert out.shape == (2, 256, 2048), f"got {tuple(out.shape)}"
    assert out.dtype == torch.bfloat16, f"got {out.dtype}"


@pytest.mark.gpu
@pytest.mark.slow
def test_single_frame_invariance_cuda():
    """With T=1 and e(T-1)=0, the video encoder must match the patched
    ``PaliGemmaModel.get_image_features`` for the same single frame.

    The encoder and the reference model share the same vision_tower +
    projector instances, so any drift is purely from the encoder's forward
    logic (expected to match at bit-level precision at T=1)."""
    encoder, pg, _, _ = _build_encoder(num_frames=1, freeze_encoder=True)
    encoder.eval()
    print("[invariance] device=cuda dtype=bfloat16 num_frames=1")

    img = torch.rand(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        # The encoder does its own ``* 2 - 1``; apply the same normalization
        # to the reference input.
        ref = pg.get_image_features(img * 2 - 1)
        out = encoder(img.unsqueeze(1))

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.slow
def test_gradient_freeze_cuda():
    """Freezing ``vision_tower`` externally (via
    ``p.requires_grad=False``) propagates through the encoder; the shared
    ``multi_modal_projector`` stays trainable."""
    encoder, _, vision_tower, projector = _build_encoder(num_frames=4, freeze_encoder=True)
    print("[freeze] device=cuda dtype=bfloat16 freeze=external")

    video = torch.rand(1, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    encoder(video).sum().backward()

    assert all(p.grad is None for p in vision_tower.parameters()), "vision_tower must be frozen"
    assert any(p.grad is not None for p in projector.parameters()), (
        "multi_modal_projector must stay trainable"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_gradient_unfreeze_cuda():
    """Without external freezing, gradients reach SigLIP params through the
    shared vision_tower."""
    encoder, _, vision_tower, _ = _build_encoder(num_frames=4, freeze_encoder=False)
    print("[unfreeze] device=cuda dtype=bfloat16 freeze=none")

    video = torch.rand(1, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    encoder(video).sum().backward()

    assert any(p.grad is not None for p in vision_tower.parameters()), (
        "vision_tower should receive gradients when not externally frozen"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_causality_smoke_cuda():
    """Perturbing the current frame (last slot) must dominate over perturbing
    the oldest frame (first slot). Exact ratio depends on pretrained weights."""
    encoder, *_ = _build_encoder(num_frames=4, freeze_encoder=True)
    encoder.eval()
    print("[causality] device=cuda dtype=bfloat16 num_frames=4")

    torch.manual_seed(0)
    video = torch.rand(1, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    noise = torch.randn_like(video[:, 0]) * 0.05

    video_perturb_old = video.clone()
    video_perturb_old[:, 0] = video_perturb_old[:, 0] + noise
    video_perturb_current = video.clone()
    video_perturb_current[:, -1] = video_perturb_current[:, -1] + noise

    with torch.no_grad():
        ref = encoder(video)
        out_old = encoder(video_perturb_old)
        out_cur = encoder(video_perturb_current)

    delta_old = (out_old - ref).float().norm()
    delta_cur = (out_cur - ref).float().norm()
    print(f"  delta_old={delta_old.item():.4f} delta_cur={delta_cur.item():.4f}")
    # Current-frame perturbation dominates but not overwhelmingly: the oldest
    # frame still propagates into the current-frame tokens through the 6
    # temporal-attention sublayers. Empirically on pretrained weights the
    # ratio is ~2.5x on MPS fp32; require >= 2x as a floor.
    assert delta_cur > 2 * delta_old, (
        f"current-frame delta ({delta_cur:.4f}) should dominate oldest-frame "
        f"delta ({delta_old:.4f}) by >= 2x; the current frame lives at t=T-1."
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_policy_end_to_end_cuda():
    """Smoke test: construct ``PI05MemPolicy`` with no pretrained download and
    run one forward + backward on a fake batch. Catches device/dtype/shape
    mismatches across the whole model."""
    from opentau.configs.types import FeatureType, PolicyFeature
    from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
    from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

    # Keep feature dims == max_*_dim so the fake batch below doesn't need
    # separate pad/un-pad logic. This is a plumbing smoke test, not a
    # realistic training setup.
    config = PI05MemConfig(
        n_obs_steps=4,
        history_interval=1,
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
    )
    state_dim = config.max_state_dim
    action_dim = config.max_action_dim
    config.input_features = {
        "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    config.output_features = {
        "actions": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }

    # Supply finite stats so the Normalize module doesn't trip its
    # infinity assertion (pretrained_path=None skips pretrained loads).
    # normalize_discrete_actions uses MIN_MAX, so min/max are also needed.
    dataset_stats = {
        "state": {
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
            "min": -torch.ones(state_dim),
            "max": torch.ones(state_dim),
        },
        "actions": {
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim),
            "min": -torch.ones(action_dim),
            "max": torch.ones(action_dim),
        },
    }

    print("[e2e] device=cuda dtype=bfloat16 pretrained_path=None n_obs_steps=4")
    policy = PI05MemPolicy(config, dataset_stats=dataset_stats).to(device="cuda", dtype=torch.bfloat16)

    batch_size = 1
    batch = {
        "camera0": torch.rand(batch_size, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
        "state": torch.randn(batch_size, 4, state_dim, device="cuda", dtype=torch.bfloat16),
        "actions": torch.randn(
            batch_size, config.chunk_size, action_dim, device="cuda", dtype=torch.bfloat16
        ),
        "prompt": ["pick up the block"],
        "response": ["pick up the block"],
        "img_is_pad": torch.zeros(batch_size, 1, dtype=torch.bool, device="cuda"),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool, device="cuda"),
        "obs_history_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool, device="cuda"),
    }
    # forward returns {"MSE": tensor, "CE": tensor}; sum for a scalar loss.
    losses = policy.forward(batch)
    (losses["MSE"] + losses["CE"]).backward()
