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
# Fallback checkpoint for debugging if the PaliGemma download path is
# unavailable on a given machine. Exposed as a constant so reviewers can flip
# it at runtime; default stays on the public checkpoint.
_POLICY_CHECKPOINT_DEBUG = "TensorAuto/pi05_base"  # noqa: F841 (documented for reviewers)


def _build_encoder(
    *, num_frames: int, freeze_encoder: bool, load_pretrained: bool = True, dtype=torch.bfloat16
):
    from opentau.policies.pi05_mem.video_encoder import SpaceTimeSiglipVideoEncoder

    encoder = SpaceTimeSiglipVideoEncoder(
        paligemma_model_name=_PALIGEMMA_REPO,
        num_frames=num_frames,
        freeze_encoder=freeze_encoder,
        encoder_dtype=dtype,
        load_pretrained=load_pretrained,
    )
    return encoder.to(device="cuda")


@pytest.mark.gpu
@pytest.mark.slow
def test_forward_shape_and_dtype_cuda():
    """Shape and dtype plumbing on a realistic T=8 batch."""
    encoder = _build_encoder(num_frames=8, freeze_encoder=True).eval()
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
    ``PaliGemmaModel.get_image_features`` for the same single frame."""
    encoder = _build_encoder(num_frames=1, freeze_encoder=True).eval()
    print("[invariance] device=cuda dtype=bfloat16 num_frames=1")

    pg = (
        PaliGemmaForConditionalGeneration.from_pretrained(_PALIGEMMA_REPO, torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )

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
    """``freeze_encoder=True`` freezes vision_tower only; multi_modal_projector
    remains trainable."""
    encoder = _build_encoder(num_frames=4, freeze_encoder=True)
    print("[freeze] device=cuda dtype=bfloat16 freeze_encoder=True")

    video = torch.rand(1, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    encoder(video).sum().backward()

    assert all(p.grad is None for p in encoder.vision_tower.parameters()), "vision_tower must be frozen"
    assert any(p.grad is not None for p in encoder.multi_modal_projector.parameters()), (
        "multi_modal_projector must stay trainable"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_gradient_unfreeze_cuda():
    """``freeze_encoder=False`` lets gradients reach SigLIP."""
    encoder = _build_encoder(num_frames=4, freeze_encoder=False)
    print("[unfreeze] device=cuda dtype=bfloat16 freeze_encoder=False")

    video = torch.rand(1, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    encoder(video).sum().backward()

    assert any(p.grad is not None for p in encoder.vision_tower.parameters()), (
        "vision_tower should receive gradients when freeze_encoder=False"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_causality_smoke_cuda():
    """Perturbing the current frame (last slot) must dominate over perturbing
    the oldest frame (first slot). Exact ratio depends on pretrained weights;
    we require current-frame dominance by at least 5x in L2 norm."""
    encoder = _build_encoder(num_frames=4, freeze_encoder=True).eval()
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
    assert delta_cur > 5 * delta_old, (
        f"current-frame delta ({delta_cur:.4f}) should dominate oldest-frame "
        f"delta ({delta_old:.4f}) by >= 5x; the current frame lives at t=T-1."
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

    config = PI05MemConfig(
        init_strategy="no_init",
        n_obs_steps=4,
        n_obs_history=4,
        history_interval=1,
        chunk_size=10,
        n_action_steps=10,
    )
    config.input_features = {
        "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {
        "actions": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }

    print("[e2e] device=cuda dtype=bfloat16 init=no_init n_obs_steps=4")
    policy = PI05MemPolicy(config).to(device="cuda", dtype=torch.bfloat16)

    batch_size = 1
    batch = {
        "camera0": torch.rand(batch_size, 4, 3, 224, 224, device="cuda", dtype=torch.bfloat16),
        "state": torch.randn(batch_size, 4, 8, device="cuda", dtype=torch.bfloat16),
        "actions": torch.randn(batch_size, config.chunk_size, 8, device="cuda", dtype=torch.bfloat16),
        "prompt": ["pick up the block"],
        "response": ["pick up the block"],
        "img_is_pad": torch.zeros(batch_size, 1, dtype=torch.bool, device="cuda"),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool, device="cuda"),
        "obs_history_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool, device="cuda"),
    }
    # The policy returns (loss_dict, _) in training mode via .forward().
    loss, _ = policy.forward(batch)
    loss["loss"].backward()
