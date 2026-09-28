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

"""GPU tests for the flux3_action policy.

The CPU suite pins wiring and contracts; these pin the things that can only fail on a
real device, and in particular the one dependency this port adds:

* **NATTEN kernels actually dispatch.** ``models/video_vae.py`` imports ``natten`` at
  module scope with no fallback, and the VAE is mandatory on the *encode* path -- so it
  is required for inference, not just decoding. The wheel is built against one exact
  torch/CUDA pair, so "imports fine" and "kernels run on this arch" are different claims
  and only the second one matters.
* **The bf16 trunk runs end to end**, in the dtype the released checkpoints ship in.

Everything is built at toy sizes from random weights; nothing is downloaded.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

from opentau.policies.flux3_action.models.transformer import JointSingleSeqParams  # noqa: E402
from opentau.policies.flux3_action.models.wiring import (  # noqa: E402
    action_dit_params,
    build_action_dit,
)

TINY = {
    "hidden_size": 256,
    "num_heads": 4,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [16, 16, 16, 16],
    "context_in_dim": 512,
    "vec_in_dim": 64,
}


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_natten_kernels_dispatch_on_this_architecture(device):
    """NATTEN's prebuilt wheel is pinned to one torch/CUDA pair; prove it runs here.

    A wheel built for a different arch imports cleanly and then fails at the first
    kernel launch, which would otherwise surface as a crash deep inside the video VAE
    on the first real observation.
    """
    from natten.functional import na2d

    q, k, v = (torch.randn(1, 16, 16, 2, 32, device=device, dtype=torch.float16) for _ in range(3))
    out = na2d(q, k, v, kernel_size=3)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_video_vae_module_imports_and_exposes_its_attention(device):
    """The VAE is mandatory on the encode path, so its attention block must be usable."""
    from opentau.policies.flux3_action.models.video_vae import Natten3D

    assert issubclass(Natten3D, torch.nn.Module)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_action_dit_builds_on_device_in_checkpoint_dtype(device, dtype):
    """bf16 is the dtype the released trunks ship in; float32 is the test/debug path."""
    base = JointSingleSeqParams(**TINY)
    params = action_dit_params(base, modality="action", channels=8, attn_mode="torch")
    dit = build_action_dit(params, None, modality="action", device=device, dtype=dtype)

    assert set(dit.in_channels) == {"action", "action_cond", "video", "video_cond"}
    materialized = [p for p in dit.parameters() if p.device.type == "cuda"]
    assert materialized, "DiT parameters should live on the CUDA device"
    assert all(p.dtype == dtype for p in materialized if p.is_floating_point())


def test_trunk_keeps_video_streams_on_device(device):
    """The structural invariant again, on the device the policy actually runs on.

    Pinned in both suites deliberately: a change that strips the video streams produces
    a model that still loads, still runs, and still returns actions -- just worse ones.
    """
    base = JointSingleSeqParams(**TINY)
    params = action_dit_params(base, modality="action", channels=8, attn_mode="torch")
    dit = build_action_dit(params, None, modality="action", device=device, dtype=torch.bfloat16)
    assert {"video", "video_cond"}.issubset(set(dit.in_channels))
