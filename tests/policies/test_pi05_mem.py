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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from opentau.policies.pi05_mem.configuration_pi05 import PI05MemConfig
from opentau.policies.pi05_mem.modeling_pi05 import (
    PI05MemPolicy,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_discrete_tokens,
    resize_with_pad,
)


class TestPI05MemConfig:
    """Unit tests for PI05MemConfig validation and properties."""

    def test_default_config(self):
        config = PI05MemConfig()
        assert config.n_obs_steps == 8
        assert config.chunk_size == 50
        assert config.n_action_steps == 50
        assert config.history_interval == 1
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32
        assert config.spacetime_layer_stride == 4

    def test_invalid_spacetime_layer_stride(self):
        with pytest.raises(ValueError, match="spacetime_layer_stride"):
            PI05MemConfig(spacetime_layer_stride=0)

    def test_obs_buffer_size_single_frame(self):
        config = PI05MemConfig(n_obs_steps=1)
        assert config.obs_buffer_size == 1

    def test_obs_buffer_size_with_history(self):
        config = PI05MemConfig(n_obs_steps=4, history_interval=2)
        assert config.obs_buffer_size == (4 - 1) * 2 + 1  # 7

    def test_obs_buffer_size_interval_defaults_to_1(self):
        config = PI05MemConfig(n_obs_steps=8)
        assert config.history_interval == 1
        assert config.obs_buffer_size == 8

    def test_invalid_n_obs_steps(self):
        with pytest.raises(ValueError, match="n_obs_steps"):
            PI05MemConfig(n_obs_steps=0)

    def test_invalid_history_interval(self):
        with pytest.raises(ValueError, match="history_interval"):
            PI05MemConfig(n_obs_steps=4, history_interval=-1)

    def test_n_action_steps_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="chunk size"):
            PI05MemConfig(chunk_size=10, n_action_steps=20)

    def test_max_delay_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="max delay"):
            PI05MemConfig(chunk_size=10, n_action_steps=10, max_delay=20)

    def test_validate_features_adds_empty_cameras(self):
        config = PI05MemConfig(empty_cameras=2)
        config.input_features = {}
        config.validate_features()
        assert "observation.images.empty_camera_0" in config.input_features
        assert "observation.images.empty_camera_1" in config.input_features

    def test_action_delta_indices(self):
        config = PI05MemConfig(chunk_size=10, n_action_steps=10)
        assert config.action_delta_indices == list(range(10))


class TestHelperFunctions:
    """Unit tests for standalone helper functions in modeling_pi05."""

    def test_sinusoidal_embedding_shape(self):
        time = torch.rand(2, 10)
        emb = create_sinusoidal_pos_embedding(time, dimension=64, min_period=4e-3, max_period=4.0)
        assert emb.shape == (2, 10, 64)

    def test_sinusoidal_embedding_odd_dimension_raises(self):
        time = torch.rand(2, 10)
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding(time, dimension=63, min_period=4e-3, max_period=4.0)

    def test_sinusoidal_embedding_wrong_ndim_raises(self):
        time = torch.rand(10)
        with pytest.raises(ValueError, match="batch_size"):
            create_sinusoidal_pos_embedding(time, dimension=64, min_period=4e-3, max_period=4.0)

    def test_resize_with_pad_maintains_shape(self):
        img = torch.randn(2, 3, 480, 640)
        result = resize_with_pad(img, width=224, height=224)
        assert result.shape == (2, 3, 224, 224)

    def test_resize_with_pad_wrong_ndim(self):
        img = torch.randn(3, 480, 640)
        with pytest.raises(ValueError, match="expected"):
            resize_with_pad(img, width=224, height=224)

    def test_pad_discrete_tokens_padding(self):
        tokens = [[1, 2, 3], [4, 5]]
        result_tokens, result_masks = pad_discrete_tokens(tokens, max_length=5)
        assert result_tokens.shape == (2, 5)
        assert result_masks.shape == (2, 5)
        np.testing.assert_array_equal(result_tokens[0], [1, 2, 3, 0, 0])
        np.testing.assert_array_equal(result_masks[0], [True, True, True, False, False])
        np.testing.assert_array_equal(result_tokens[1], [4, 5, 0, 0, 0])
        np.testing.assert_array_equal(result_masks[1], [True, True, False, False, False])

    def test_pad_discrete_tokens_truncation(self):
        tokens = [[1, 2, 3, 4, 5]]
        result_tokens, result_masks = pad_discrete_tokens(tokens, max_length=3)
        assert result_tokens.shape == (1, 3)
        np.testing.assert_array_equal(result_tokens[0], [1, 2, 3])
        np.testing.assert_array_equal(result_masks[0], [True, True, True])

    def test_make_att_2d_masks_basic(self):
        pad_masks = torch.ones(1, 4, dtype=torch.bool)
        att_masks = torch.tensor([[1, 0, 0, 1]], dtype=torch.int32)
        result = make_att_2d_masks(pad_masks, att_masks)
        assert result.shape == (1, 4, 4)
        assert result.dtype == torch.bool
        assert result[0, 0, 0].item() is True
        assert result[0, 0, 3].item() is False

    def test_make_att_2d_masks_with_padding(self):
        pad_masks = torch.tensor([[True, True, False]], dtype=torch.bool)
        att_masks = torch.tensor([[1, 0, 0]], dtype=torch.int32)
        result = make_att_2d_masks(pad_masks, att_masks)
        assert result[0, 2, 0].item() is False
        assert result[0, 0, 2].item() is False


class TestBuildHistoryBatch:
    """Test _build_history_batch logic using a lightweight mock policy."""

    @pytest.fixture
    def mock_policy(self):
        """Create a minimal mock that simulates PI05MemPolicy for history testing."""
        from opentau.configs.types import FeatureType, PolicyFeature

        policy = MagicMock()
        policy.config = PI05MemConfig(n_obs_steps=4, history_interval=1)
        policy.config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        policy._state_buffer = None
        policy._obs_buffers = {}

        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy._build_history_batch = PI05MemPolicy._build_history_batch.__get__(policy)
        return policy

    def test_first_call_zero_pads(self, mock_policy):
        batch = {
            "state": torch.ones(1, 8),
            "camera0": torch.ones(1, 3, 4, 4),
            "prompt": ["test"],
        }
        result = mock_policy._build_history_batch(batch)

        assert result["state"].shape == (1, 4, 8)
        assert result["camera0"].shape == (1, 4, 3, 4, 4)
        assert result["prompt"] == ["test"]

        # First 3 frames should be zero-padded, last should be the actual observation
        assert torch.all(result["state"][:, :3, :] == 0)
        assert torch.all(result["state"][:, 3, :] == 1)

    def test_history_fills_up(self, mock_policy):
        batch_template = {
            "state": torch.zeros(1, 8),
            "camera0": torch.zeros(1, 3, 4, 4),
            "prompt": ["test"],
        }

        for i in range(4):
            batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_template.items()}
            batch["state"] = torch.full((1, 8), float(i + 1))
            batch["camera0"] = torch.full((1, 3, 4, 4), float(i + 1))
            result = mock_policy._build_history_batch(batch)

        assert result["state"].shape == (1, 4, 8)
        for t in range(4):
            assert result["state"][0, t, 0].item() == float(t + 1)

    def test_metadata_keys_passed_through(self, mock_policy):
        batch = {
            "state": torch.ones(1, 8),
            "camera0": torch.ones(1, 3, 4, 4),
            "prompt": ["pick up the block"],
            "extra_metadata": "should_pass_through",
        }
        result = mock_policy._build_history_batch(batch)
        assert result["prompt"] == ["pick up the block"]
        assert result["extra_metadata"] == "should_pass_through"


class TestPrepareState:
    """Test prepare_state in isolation."""

    def test_2d_state_unsqueezes(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=8, n_obs_steps=1)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 6)}
        result = policy.prepare_state(batch)
        assert result.shape == (2, 1, 8)

    def test_3d_state_pads(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=16)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 4, 10)}
        result = policy.prepare_state(batch)
        assert result.shape == (2, 4, 16)

    def test_state_exceeding_max_dim_raises(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=8)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 1, 16)}
        with pytest.raises(ValueError, match="exceeds max_state_dim"):
            policy.prepare_state(batch)

    def test_2d_state_with_history_raises(self):
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        policy.config = PI05MemConfig(max_state_dim=8, n_obs_steps=4)
        policy.prepare_state = PI05MemPolicy.prepare_state.__get__(policy)

        batch = {"state": torch.randn(2, 6)}
        with pytest.raises(ValueError, match="Expected 3D"):
            policy.prepare_state(batch)


class TestPrepareVideos:
    """Test prepare_videos preprocessing logic."""

    @pytest.fixture
    def mock_policy(self):
        from opentau.configs.types import FeatureType, PolicyFeature
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = MagicMock(spec=PI05MemPolicy)
        # Default fixture uses single-frame (n_obs_steps=1); individual tests
        # that exercise history behavior bump n_obs_steps themselves.
        policy.config = PI05MemConfig(
            n_obs_steps=1,
            resize_imgs_with_padding=(224, 224),
            empty_cameras=0,
        )
        policy.config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
        policy.prepare_videos = PI05MemPolicy.prepare_videos.__get__(policy)
        return policy

    def test_4d_input_unsqueezes(self, mock_policy):
        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        videos, masks = mock_policy.prepare_videos(batch)
        assert len(videos) == 1
        assert videos[0].shape == (2, 1, 3, 224, 224)
        assert masks[0].shape == (2,)
        assert torch.all(masks[0])

    def test_5d_input_passes_through(self, mock_policy):
        batch = {"camera0": torch.randn(2, 4, 3, 224, 224)}
        videos, masks = mock_policy.prepare_videos(batch)
        assert videos[0].shape == (2, 4, 3, 224, 224)

    def test_all_images_missing_raises(self, mock_policy):
        batch = {"state": torch.randn(2, 8)}
        with pytest.raises(ValueError, match="All image features are missing"):
            mock_policy.prepare_videos(batch)

    def test_obs_history_is_pad_masks_frames(self, mock_policy):
        vid = torch.ones(1, 4, 3, 224, 224)
        obs_pad = torch.tensor([[True, False, False, False]])
        batch = {"camera0": vid}
        videos, masks = mock_policy.prepare_videos(batch, obs_history_is_pad=obs_pad)
        assert torch.all(videos[0][:, 0, :, :, :] == 0)
        assert torch.all(videos[0][:, 1, :, :, :] == 1)

    def test_empty_cameras_appended(self, mock_policy):
        from opentau.configs.types import FeatureType, PolicyFeature

        mock_policy.config.empty_cameras = 1
        mock_policy.config.input_features["empty_cam"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        )

        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        videos, masks = mock_policy.prepare_videos(batch)
        assert len(videos) == 2
        assert torch.all(videos[1] == 0)
        assert torch.all(masks[1] == 0)

    def test_4d_with_history_raises(self, mock_policy):
        mock_policy.config.n_obs_steps = 4
        batch = {"camera0": torch.randn(2, 3, 224, 224)}
        with pytest.raises(ValueError, match="Expected 5D"):
            mock_policy.prepare_videos(batch)


# Standalone tests for the SpaceTime SigLIP video encoder live in
# tests/policies/test_pi07_video_encoder_cpu.py — that file is the canonical
# location and parametrizes over both the Gemma 3 (pi07/low_level) and
# PaliGemma (pi05_mem, pi07_paligemma) projector flavors.


# State mask: current step (t = T-1) must always be marked real, even when
# obs_history_is_pad sets it to True (e.g. dataset's history_state_drop_prob
# augmentation flips the entire tensor to all-True). Without the override the
# policy is conditioned on no state at all when that augmentation fires.


def _make_embed_prefix_stub():
    """Construct a partial PI05MemFlowMatching that exposes only the attrs
    ``embed_prefix`` reads — no PaliGemma init, just enough for the layout
    paths exercised by the state-mask tests.
    """
    import types

    from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

    hidden = 4
    n_video_tokens = 3
    fm = PI05MemFlowMatching.__new__(PI05MemFlowMatching)

    class _FakePaligemma:
        def embed_language_tokens(self, tokens):
            return torch.zeros((*tokens.shape, hidden), dtype=torch.float32)

        def embed_discrete_actions(self, da):
            return torch.zeros((*da.shape, hidden), dtype=torch.float32)

    fm.paligemma_with_expert = _FakePaligemma()
    fm.state_proj = lambda state: torch.zeros(state.shape[0], state.shape[1], hidden, dtype=torch.float32)
    fm.embed_video = lambda video, obs_history_is_pad=None: torch.zeros(
        video.shape[0], n_video_tokens, hidden, dtype=torch.float32
    )
    fm.config = types.SimpleNamespace(discrete_action_max_length=2)
    return fm


def _embed_prefix_default_inputs(*, batch_size: int = 2, prompt_len: int = 3, t_state: int = 1):
    return {
        "videos": [torch.zeros(batch_size, t_state, 3, 4, 4)],
        "vid_masks": [torch.ones(batch_size, dtype=torch.bool)],
        "lang_tokens": torch.zeros(batch_size, prompt_len, dtype=torch.long),
        "lang_masks": torch.ones(batch_size, prompt_len, dtype=torch.bool),
        "state": torch.zeros(batch_size, t_state, 7),
    }


def _state_slice_indices(prompt_len: int, n_video_tokens: int, t_state: int) -> slice:
    """Layout: videos(n_video_tokens) + lang(prompt_len) + state(t_state)."""
    state_lo = n_video_tokens + prompt_len
    return slice(state_lo, state_lo + t_state)


class TestStateMaskCurrentStepAlwaysReal:
    """Pin the post-fix invariant: state_mask[:, -1] is True regardless of
    obs_history_is_pad. (Bug B from the audit; ported from pi07 / PR #205.)
    """

    def test_state_mask_current_step_real_when_all_history_padded(self):
        """``obs_history_is_pad = ones(B, T)`` (the
        ``history_state_drop_prob=1.0`` case) MUST still leave the current
        state token (index T-1) marked real, otherwise attention to it is
        masked out and the policy conditions on no state at all.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        kwargs["obs_history_is_pad"] = torch.ones(bsize, t_state, dtype=torch.bool)

        _, pad_masks, _ = PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        state_slice = _state_slice_indices(prompt_len=3, n_video_tokens=3, t_state=t_state)
        state_mask = pad_masks[:, state_slice]
        assert state_mask.shape == (bsize, t_state)

        for i in range(bsize):
            assert state_mask[i, -1].item() is True, (
                f"sample {i}: current state token (T-1) is masked out — the "
                f"history_state_drop_prob augmentation would condition on no "
                f"state at all. state_mask = {state_mask[i].tolist()}"
            )
        assert (~state_mask[:, :-1]).all().item() is True

    def test_masked_state_zeroed_before_projection_current_preserved(self):
        """Defense-in-depth (pi05_mem): masked state steps are zeroed *after*
        normalization but *before* ``state_proj``; the current step keeps its
        real value, so dropped history cannot leak even if the attention mask
        later regresses.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        state = torch.arange(1, bsize * t_state * 7 + 1, dtype=torch.float32).reshape(bsize, t_state, 7)
        kwargs["state"] = state
        kwargs["obs_history_is_pad"] = torch.ones(bsize, t_state, dtype=torch.bool)

        captured = {}

        def _spy_state_proj(s):
            captured["state"] = s.clone()
            return torch.zeros(s.shape[0], s.shape[1], 4, dtype=torch.float32)

        fake.state_proj = _spy_state_proj
        PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        seen = captured["state"]
        assert torch.all(seen[:, :-1] == 0)
        assert torch.equal(seen[:, -1], state[:, -1].to(seen.dtype))

    def test_state_mask_none_branch_assumes_history_padded_keeps_current_real(self):
        """``obs_history_is_pad = None`` means the caller didn't tell us
        which slots are real. Post-fix: assume all history is padded so the
        encoder cannot attend to garbage history slots — but the current
        step is still real.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        kwargs["obs_history_is_pad"] = None

        _, pad_masks, _ = PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        state_slice = _state_slice_indices(prompt_len=3, n_video_tokens=3, t_state=t_state)
        state_mask = pad_masks[:, state_slice]
        assert state_mask.shape == (bsize, t_state)

        for i in range(bsize):
            assert state_mask[i, -1].item() is True
            assert (~state_mask[i, :-1]).all().item() is True

    def test_state_mask_partial_history_pad_preserves_current(self):
        """Mixed pad pattern (typical of natural episode-boundary padding):
        some history slots padded, current step real -> state_mask matches
        ``~obs_history_is_pad`` exactly, with the override a no-op since the
        current bit was already True.
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 2
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        kwargs["obs_history_is_pad"] = torch.tensor([[True, True, False, False], [True, False, False, False]])

        _, pad_masks, _ = PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        state_slice = _state_slice_indices(prompt_len=3, n_video_tokens=3, t_state=t_state)
        state_mask = pad_masks[:, state_slice]
        torch.testing.assert_close(
            state_mask,
            torch.tensor([[False, False, True, True], [False, True, True, True]]),
        )

    def test_state_mask_does_not_mutate_obs_history_is_pad(self):
        """The override path must not mutate the caller's
        ``obs_history_is_pad`` (it is also threaded into ``embed_video``
        for the temporal attention mask — mutating it there would cause
        cross-call corruption).
        """
        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemFlowMatching

        fake = _make_embed_prefix_stub()
        bsize = 1
        t_state = 4
        kwargs = _embed_prefix_default_inputs(batch_size=bsize, t_state=t_state)
        original_pad = torch.ones(bsize, t_state, dtype=torch.bool)
        kwargs["obs_history_is_pad"] = original_pad
        snapshot = original_pad.clone()

        PI05MemFlowMatching.embed_prefix(fake, **kwargs)

        torch.testing.assert_close(original_pad, snapshot)


# `_build_history_batch` emits ``obs_history_is_pad`` so the encoder can use
# real mid-episode history while still masking start-of-episode zero-fill.
# Without this emit, the encoder's None-fallback masks ALL history at
# inference (mid-episode regression flagged in pi07's PR #253 review and
# inherited from pi05_mem's original design).


class TestBuildHistoryBatchEmitsObsHistoryIsPad:
    @staticmethod
    def _make_policy_stub(*, n_obs_steps: int, history_interval: int, image_keys: list[str]):
        """Construct a partial PI05MemPolicy that exposes only the attrs
        ``_build_history_batch`` reads.
        """
        import types

        from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

        policy = PI05MemPolicy.__new__(PI05MemPolicy)
        buf_size = (n_obs_steps - 1) * history_interval + 1
        policy.config = types.SimpleNamespace(
            n_obs_steps=n_obs_steps,
            history_interval=history_interval,
            obs_buffer_size=buf_size,
            image_features=dict.fromkeys(image_keys),
        )
        policy._state_buffer = None
        policy._obs_buffers = None
        return policy

    def _make_batch(self, image_keys: list[str], state_dim: int = 4) -> dict:
        return {
            "state": torch.zeros(1, state_dim),
            **{k: torch.zeros(1, 3, 8, 8) for k in image_keys},
        }

    def test_first_step_marks_all_but_current_padded(self):
        """At episode start, only the very first observation is in the
        buffer; every other slot in the requested history was zero-filled.
        Mask should be ``[True, ..., True, False]`` — the canonical case
        the Bug A fix protects against contamination from.
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=1, image_keys=["camera0"])
        out = policy._build_history_batch(self._make_batch(["camera0"]))

        assert "obs_history_is_pad" in out
        assert out["obs_history_is_pad"].shape == (1, 4)
        assert out["obs_history_is_pad"].dtype == torch.bool
        assert out["obs_history_is_pad"].tolist() == [[True, True, True, False]]

    def test_buffer_full_emits_all_false(self):
        """Once the buffer is full (after ``obs_buffer_size`` calls), every
        slot maps to a real observation — mask is all-False. This is the
        mid-episode case the previous behavior regressed: with no emit, the
        encoder masked these real frames out via the None-fallback.
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=2, image_keys=["camera0"])
        # obs_buffer_size = (4-1)*2 + 1 = 7. Need 7 calls to fill.
        batch = self._make_batch(["camera0"])
        for _ in range(7):
            out = policy._build_history_batch(batch)
        assert out["obs_history_is_pad"].tolist() == [[False, False, False, False]]

    def test_partial_fill_marks_only_unfilled_slots(self):
        """After ``k < obs_buffer_size`` calls, the leading slots are still
        virtual past-steps. With ``n_obs_steps=4, history_interval=2``
        (buffer_size=7), after 4 calls the deque has 4 entries -> ``missing
        = 3`` -> slots with ``i*interval - 3 < 0`` are padded: i=0 -> -3 (T),
        i=1 -> -1 (T), i=2 -> 1 (F), i=3 -> 3 (F). Mask = [T, T, F, F].
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=2, image_keys=["camera0"])
        batch = self._make_batch(["camera0"])
        for _ in range(4):
            out = policy._build_history_batch(batch)
        assert out["obs_history_is_pad"].tolist() == [[True, True, False, False]]

    def test_mask_is_broadcast_over_batch(self):
        """The buffer is shared across batch elements, so the (B, T) mask is
        the same across the batch dim. Verify by emitting from a B=3 batch.
        """
        policy = self._make_policy_stub(n_obs_steps=4, history_interval=1, image_keys=["camera0"])
        batch = {
            "state": torch.zeros(3, 4),
            "camera0": torch.zeros(3, 3, 8, 8),
        }
        out = policy._build_history_batch(batch)

        assert out["obs_history_is_pad"].shape == (3, 4)
        assert torch.all(out["obs_history_is_pad"] == out["obs_history_is_pad"][0:1])

    def test_state_and_camera_padding_match_emitted_mask(self):
        """The emitted mask must agree slot-for-slot with the actual
        zero-padding pattern of state and camera tensors.
        """
        policy = self._make_policy_stub(n_obs_steps=3, history_interval=1, image_keys=["camera0"])
        batch = {
            "state": torch.full((1, 4), 7.0),
            "camera0": torch.full((1, 3, 8, 8), 5.0),
        }
        out = policy._build_history_batch(batch)
        # After one call: missing = 2; mask = [True, True, False].
        is_pad = out["obs_history_is_pad"][0]  # (T,)
        state = out["state"][0]  # (T, D)
        cam = out["camera0"][0]  # (T, C, H, W)
        for t, padded in enumerate(is_pad.tolist()):
            if padded:
                assert torch.all(state[t] == 0.0), f"state[{t}] not zero-filled"
                assert torch.all(cam[t] == 0.0), f"camera[{t}] not zero-filled"
            else:
                assert torch.all(state[t] == 7.0), f"state[{t}] zero-filled but mask says real"
                assert torch.all(cam[t] == 5.0), f"camera[{t}] zero-filled but mask says real"


class TestPI05MemExecutionHorizon:
    """Regression coverage for ``n_action_steps`` as a short execution horizon.

    ``chunk_size`` is the trained prediction horizon (always decoded);
    ``n_action_steps`` (<= chunk_size) is how many actions are executed before
    re-querying. ``n_action_steps < chunk_size`` used to broadcast-crash the
    denoise/MSE paths and trip the queue assert; these CPU tests guard that path.
    """

    MAX_STATE_DIM = 32
    MAX_ACTION_DIM = 32

    @classmethod
    def _config(cls, chunk_size=10, n_action_steps=3, max_delay=0):
        # n_obs_steps=1 skips the temporal-history branch in select_action.
        return PI05MemConfig(
            n_obs_steps=1,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_delay=max_delay,
            max_state_dim=cls.MAX_STATE_DIM,
            max_action_dim=cls.MAX_ACTION_DIM,
        )

    def test_guard_rejects_short_horizon_with_delay(self):
        # A shortened execution horizon is not compatible with the real-time
        # delay prefix path; the config must reject it loudly.
        with pytest.raises(ValueError, match="max_delay"):
            self._config(chunk_size=10, n_action_steps=3, max_delay=2)

    def test_guard_allows_short_horizon_without_delay(self):
        cfg = self._config(chunk_size=10, n_action_steps=3, max_delay=0)
        assert (cfg.chunk_size, cfg.n_action_steps, cfg.max_delay) == (10, 3, 0)

    def test_guard_allows_full_horizon_with_delay(self):
        # n_action_steps == chunk_size keeps the real-time-delay path available.
        cfg = self._config(chunk_size=10, n_action_steps=10, max_delay=2)
        assert cfg.max_delay == 2

    def test_select_action_executes_first_n_then_requeries(self):
        """Model decodes the full ``chunk_size`` chunk, but ``select_action``
        executes only the first ``n_action_steps`` before re-querying.
        """
        chunk_size, n_steps, bsz = 10, 3, 2
        cfg = self._config(chunk_size=chunk_size, n_action_steps=n_steps, max_delay=0)

        policy = object.__new__(PI05MemPolicy)
        policy.config = cfg
        policy.eval = lambda: None  # bypass nn.Module.eval (no __init__ was run)
        PI05MemPolicy.reset(policy)

        calls = {"n": 0}

        def fake_sample_actions(batch, noise=None, action_prefix=None, delay=None):
            # Return a full chunk_size chunk; element value encodes
            # (call_index * 1000 + timestep) so we can assert which timesteps run.
            calls["n"] += 1
            ts = torch.arange(chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
            return (calls["n"] * 1000 + ts).expand(bsz, chunk_size, self.MAX_ACTION_DIM).clone()

        policy.sample_actions = fake_sample_actions
        batch = {"state": torch.zeros(bsz, self.MAX_STATE_DIM)}

        # First n_steps actions all come from a single decode (call 1), in order.
        acts = [PI05MemPolicy.select_action(policy, batch) for _ in range(n_steps)]
        assert calls["n"] == 1
        assert [tuple(a.shape) for a in acts] == [(bsz, self.MAX_ACTION_DIM)] * n_steps
        assert [a[0, 0].item() for a in acts] == [1000.0, 1001.0, 1002.0]

        # Queue drained after n_action_steps -> next call re-queries (call 2).
        a_next = PI05MemPolicy.select_action(policy, batch)
        assert calls["n"] == 2
        assert a_next[0, 0].item() == 2000.0
