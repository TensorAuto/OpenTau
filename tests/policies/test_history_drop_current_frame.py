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

"""CPU-only regression tests for the history-drop "current step is kept" invariant.

The dataset-side ``history_state_drop_prob`` augmentation
(``LeRobotDataset._emit_optional_keys``) marks ``obs_history_is_pad`` ALL-True
while zeroing only the historical camera frames (``camera[:-1]``) — the
documented contract (``configs/default.py``) is that the current camera frame
and the current state are always kept. The policies' ``prepare_videos``
re-apply the mask at the pixel level, so they must exempt the current (last)
frame, mirroring the state path's ``state_mask[:, -1] = True`` exemption.

Regression coverage for the bug where an all-True ``obs_history_is_pad``
zeroed ALL frames including the current one, so ~30% of training samples (at
the default drop prob of 0.3) conditioned on a blank current observation.

These tests pin the contract for pi07, pi05_mem and pi07_paligemma without
instantiating any backbone — ``prepare_videos`` only reads ``self.config``, so
a ``SimpleNamespace`` stand-in suffices (same pattern as ``test_pi07_cpu.py``'s
``prepare_metadata`` tests). The RLDX motion-fill tests bind
``RLDXVideoEncoder._apply_motion`` the same way.
"""

from __future__ import annotations

import types

import pytest
import torch
from einops import rearrange

from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy
from opentau.policies.pi05_mem.rldx_video_encoder import RLDXVideoEncoder
from opentau.policies.pi07.low_level.modeling_pi07_low_level import PI07LowLevelPolicy
from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
    PI07PaligemmaLowLevelPolicy,
)

CAM_KEY = "camera0"


def _fake_policy(n_obs_steps: int):
    """Minimal ``self`` stand-in for the three ``prepare_videos`` methods.

    They only read ``config.{image_features, n_obs_steps,
    resize_imgs_with_padding, empty_cameras}`` — no backbone needed.
    ``resize_imgs_with_padding=None`` skips the resize so frames can be
    compared bit-exactly against the input.
    """
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            image_features=[CAM_KEY],
            n_obs_steps=n_obs_steps,
            resize_imgs_with_padding=None,
            empty_cameras=0,
        )
    )


def _call_prepare_videos(policy_cls, fake, video, pad):
    """Dispatch on the two signatures: pi07_paligemma reads the mask from the batch."""
    if policy_cls is PI07PaligemmaLowLevelPolicy:
        return policy_cls.prepare_videos(fake, {CAM_KEY: video, "obs_history_is_pad": pad})
    return policy_cls.prepare_videos(fake, {CAM_KEY: video}, obs_history_is_pad=pad)


POLICIES = [PI07LowLevelPolicy, PI05MemPolicy, PI07PaligemmaLowLevelPolicy]


@pytest.mark.parametrize("policy_cls", POLICIES)
class TestHistoryDropKeepsCurrentFrame:
    """A forced history drop (all-True mask) must never zero the current frame."""

    @pytest.mark.parametrize("n_obs_history", [None, 1, 3])
    def test_current_frame_survives_forced_drop(self, policy_cls, n_obs_history):
        b, c, h, w = 2, 3, 8, 8
        t = 1 if n_obs_history is None else n_obs_history
        video_5d = torch.rand(b, t, c, h, w) + 0.1  # strictly nonzero everywhere
        # n_obs_history=None yields rank-4 camera tensors (no time axis);
        # prepare_videos unsqueezes them back to rank-5.
        video_in = video_5d[:, 0].clone() if n_obs_history is None else video_5d.clone()
        pad = torch.ones(b, t, dtype=torch.bool)  # forced history drop

        videos, _ = _call_prepare_videos(policy_cls, _fake_policy(n_obs_steps=t), video_in, pad)

        out = videos[0]
        assert out.shape == (b, t, c, h, w)
        torch.testing.assert_close(out[:, -1], video_5d[:, -1])
        if t > 1:
            assert torch.all(out[:, :-1] == 0)

    def test_caller_mask_not_mutated(self, policy_cls):
        b, t = 2, 3
        video = torch.rand(b, t, 3, 8, 8) + 0.1
        pad = torch.ones(b, t, dtype=torch.bool)
        _call_prepare_videos(policy_cls, _fake_policy(n_obs_steps=t), video, pad)
        # The [:, -1] = True exemption must act on a fresh tensor, not leak out.
        assert torch.all(pad)

    def test_genuine_start_of_episode_padding_still_zeroed(self, policy_cls):
        b, t = 2, 3
        video = torch.rand(b, t, 3, 8, 8) + 0.1
        pad = torch.tensor([[True, False, False]] * b)

        videos, _ = _call_prepare_videos(policy_cls, _fake_policy(n_obs_steps=t), video.clone(), pad)

        out = videos[0]
        assert torch.all(out[:, 0] == 0)
        torch.testing.assert_close(out[:, 1:], video[:, 1:])

    def test_mixed_per_sample_masks_are_independent(self, policy_cls):
        """One batch mixing a dropped row, a prefix-padded row and a full row."""
        t = 3
        video = torch.rand(3, t, 3, 8, 8) + 0.1
        pad = torch.tensor(
            [
                [True, True, True],  # forced history drop
                [True, False, False],  # genuine start-of-episode prefix pad
                [False, False, False],  # full history
            ]
        )

        videos, _ = _call_prepare_videos(policy_cls, _fake_policy(n_obs_steps=t), video.clone(), pad)

        out = videos[0]
        assert torch.all(out[0, :-1] == 0)
        torch.testing.assert_close(out[0, -1], video[0, -1])
        assert torch.all(out[1, 0] == 0)
        torch.testing.assert_close(out[1, 1:], video[1, 1:])
        torch.testing.assert_close(out[2], video[2])

    def test_multi_camera_and_empty_camera_slots(self, policy_cls):
        """Each present camera is masked independently; empty slots stay zero."""
        b, t = 2, 3
        fake = _fake_policy(n_obs_steps=t)
        fake.config.image_features = ["camera0", "camera1", "camera2"]
        fake.config.empty_cameras = 1
        vid0 = torch.rand(b, t, 3, 8, 8) + 0.1
        vid1 = torch.rand(b, t, 3, 8, 8) + 0.1
        pad = torch.ones(b, t, dtype=torch.bool)  # forced history drop

        batch = {"camera0": vid0.clone(), "camera1": vid1.clone()}  # camera2 missing
        if policy_cls is PI07PaligemmaLowLevelPolicy:
            batch["obs_history_is_pad"] = pad
            videos, vid_masks = policy_cls.prepare_videos(fake, batch)
        else:
            videos, vid_masks = policy_cls.prepare_videos(fake, batch, obs_history_is_pad=pad)

        assert len(videos) == 3  # 2 present + 1 empty slot
        for out, src in zip(videos[:2], (vid0, vid1), strict=False):
            torch.testing.assert_close(out[:, -1], src[:, -1])
            assert torch.all(out[:, :-1] == 0)
        assert torch.all(videos[2] == 0)
        assert torch.all(vid_masks[0]) and torch.all(vid_masks[1])
        assert not vid_masks[2].any()

    def test_resize_path_keeps_current_frame_content(self, policy_cls):
        """Masking runs before the resize; the invariant must survive it."""
        b, t = 2, 3
        fake = _fake_policy(n_obs_steps=t)
        fake.config.resize_imgs_with_padding = (16, 16)
        video = torch.rand(b, t, 3, 8, 8) + 0.1
        pad = torch.ones(b, t, dtype=torch.bool)  # forced history drop

        videos, _ = _call_prepare_videos(policy_cls, fake, video.clone(), pad)

        out = videos[0]
        assert out.shape == (b, t, 3, 16, 16)
        assert torch.all(out[:, :-1] == 0)
        assert torch.all(out[:, -1].abs().sum(dim=(-3, -2, -1)) > 0)


class _RecordingMotion(torch.nn.Module):
    """Stub motion module: records its ``(B*T*N, D)`` input, contributes nothing."""

    def __init__(self):
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, flat, grid_sizes):
        self.seen = flat.detach().clone()
        return torch.zeros_like(flat)


class TestRLDXMotionFillTreatsCurrentFrameAsReal:
    """``_apply_motion`` must fill padded slots from a *real* frame under a forced drop.

    Before the fix, an all-True ``obs_history_is_pad`` made
    ``(~pad).float().argmax(dim=1)`` return 0 — silently picking the *zeroed*
    frame 0 as the fill and replacing the current frame's hidden state too, so
    the STSS correlation ran on an all-blank clip.
    """

    B, T, GRID, DIM = 2, 3, 4, 6

    def _run(self, pad):
        n = self.GRID * self.GRID
        hidden = torch.rand(self.B * self.T, n, self.DIM) + 0.1
        motion = _RecordingMotion()
        fake = types.SimpleNamespace(motion_grid_hw=self.GRID, motion_module=motion)

        out = RLDXVideoEncoder._apply_motion(fake, hidden, self.B, self.T, pad)

        seen = rearrange(motion.seen, "(b t n) d -> b t n d", b=self.B, t=self.T, n=n)
        x = rearrange(hidden, "(b t) n d -> b t n d", b=self.B, t=self.T)
        return out, seen, x, hidden

    def test_forced_drop_fills_every_slot_with_current_frame(self):
        pad = torch.ones(self.B, self.T, dtype=torch.bool)
        out, seen, x, hidden = self._run(pad)
        # Every STSS input frame is the (real) current frame — a static clip,
        # i.e. "no motion" — never the zeroed frame 0.
        for i in range(self.T):
            torch.testing.assert_close(seen[:, i], x[:, -1])
        # The fill only feeds the STSS input; with a zero residual the returned
        # hidden states are bit-identical to the input, current frame included.
        torch.testing.assert_close(out, hidden)
        assert torch.all(pad)  # caller's mask untouched

    def test_prefix_padding_forward_fills_first_real_frame(self):
        pad = torch.tensor([[True, False, False]] * self.B)
        _, seen, x, _ = self._run(pad)
        torch.testing.assert_close(seen[:, 0], x[:, 1])  # padded slot <- first real
        torch.testing.assert_close(seen[:, 1], x[:, 1])
        torch.testing.assert_close(seen[:, 2], x[:, 2])

    def test_current_slot_never_overwritten_for_any_mask_shape(self):
        """The current frame's STSS input survives even a pathological mask.

        A suffix-shaped mask ([False, True, True]) cannot occur in training
        (padding is a prefix; the drop is all-True), but it kills a revert of
        the ``pad_mask = ~real`` sub-fix: with the old
        ``pad_mask = obs_history_is_pad``, the current slot would be replaced
        by frame 0 here.
        """
        pad = torch.tensor([[False, True, True]] * self.B)
        _, seen, x, _ = self._run(pad)
        torch.testing.assert_close(seen[:, -1], x[:, -1])
        torch.testing.assert_close(seen[:, 0], x[:, 0])
        torch.testing.assert_close(seen[:, 1], x[:, 0])  # padded slot <- first real
