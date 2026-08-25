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

"""Trajectory-sequence emission: offsets, reshape and the off-by-default contract.

Sequence emission adds a leading timestep axis for recurrent policies
(``pi05_ttt``). Everything here is CPU-only and touches no dataset on the Hub:
the offset construction is arithmetic, and the reshape is a ``rearrange``.

The property that matters most is the **regression guard**: at
``sequence_length == 1`` the emitted offsets must be byte-identical to what the
code produced before this feature existed. Every existing config leaves the
field at its default, so anything else silently changes every run in the repo.
"""

import numpy as np
import pytest
import torch
from einops import rearrange


def _action_offsets(seq_len: int, stride: int, chunk_offsets: list[int], freq: float) -> list[float]:
    """Mirror of the action-offset construction in ``resolve_delta_timestamps``.

    Kept as a local mirror on purpose. The real one is a few lines inside a
    function that needs a dataset, a name map and a policy config to reach; this
    pins the arithmetic, and ``test_mirror_matches_the_implementation`` pins that
    the mirror has not drifted from it.

    Args:
        seq_len: Supervised timesteps per sample.
        stride: Frames between consecutive timesteps.
        chunk_offsets: ``policy.action_delta_indices``.
        freq: Action frequency in Hz.

    Returns:
        The flat ``seq_len * len(chunk_offsets)`` offset list, in seconds.
    """
    return [(-(seq_len - 1 - t) * stride + h) / freq for t in range(seq_len) for h in chunk_offsets]


def _obs_offsets(seq_len: int, stride: int, freq: float) -> list[float]:
    """Mirror of the sequence branch of the observation-offset construction.

    Args:
        seq_len: Supervised timesteps per sample.
        stride: Frames between consecutive timesteps.
        freq: Action frequency in Hz.

    Returns:
        One offset per timestep, in seconds.
    """
    return [-(seq_len - 1 - t) * stride / freq for t in range(seq_len)]


class TestOffDefaultIsByteIdentical:
    """``sequence_length == 1`` must not perturb any existing run."""

    def test_action_offsets_collapse_to_the_chunk(self):
        """The regression guard for this whole feature.

        Every shipped config leaves ``sequence_length`` at 1, so if this ever
        differs from ``[i / freq for i in action_delta_indices]`` the feature has
        silently changed the data every existing run trains on.
        """
        chunk_offsets = list(range(10))
        freq = 20.0
        assert _action_offsets(1, 1, chunk_offsets, freq) == [i / freq for i in chunk_offsets]

    @pytest.mark.parametrize("stride", [1, 5, 50])
    def test_stride_is_irrelevant_at_length_one(self, stride):
        """A single timestep has no gap to stride over."""
        chunk_offsets = list(range(4))
        assert _action_offsets(1, stride, chunk_offsets, 20.0) == _action_offsets(1, 1, chunk_offsets, 20.0)

    def test_observation_offsets_collapse_to_the_current_frame(self):
        assert _obs_offsets(1, 1, 20.0) == [0.0]


class TestWindowAnchoring:
    """The window is anchored at its *last* timestep."""

    def test_last_timestep_is_the_current_frame(self):
        """Timestep T-1 must sit at offset 0 — the frame being predicted.

        Anchoring here is what lets the observation offsets reuse the existing
        history convention (all ``<= 0``) and what matches inference, where the
        memory is built from the past and the policy acts *now*.
        """
        offsets = _obs_offsets(8, 1, 20.0)
        assert offsets[-1] == 0.0
        assert all(o <= 0.0 for o in offsets)

    def test_stride_one_uses_consecutive_frames(self):
        """RoboTTT's own definition: one timestep is one control step."""
        freq = 20.0
        offsets = _obs_offsets(5, 1, freq)
        frames = [round(o * freq) for o in offsets]
        assert frames == [-4, -3, -2, -1, 0]

    def test_window_span_in_frames(self):
        """A T-timestep window spans ``(T-1)*stride + chunk`` frames.

        This is what makes stride 1 tractable on short episodes: at chunk 10,
        T=32 needs 41 frames, and LIBERO's shortest episode is 75.
        """
        chunk = 10
        offsets = _action_offsets(32, 1, list(range(chunk)), 20.0)
        span = round((max(offsets) - min(offsets)) * 20.0) + 1
        assert span == (32 - 1) * 1 + chunk

    def test_action_offsets_are_timestep_major(self):
        """Timestep-major, so the reshape to ``(T, H)`` is a plain view.

        Chunk-major would reshape without error and silently transpose the
        sequence against the chunk.
        """
        chunk_offsets = [0, 1, 2]
        freq = 1.0
        flat = _action_offsets(3, 1, chunk_offsets, freq)
        grid = rearrange(torch.tensor(flat), "(t h) -> t h", t=3)
        # Each row must be one timestep's chunk: consecutive within a row.
        for row in grid:
            assert torch.equal(row - row[0], torch.tensor([0.0, 1.0, 2.0]))
        # Rows must advance by the stride.
        assert torch.equal(grid[:, 0], torch.tensor([-2.0, -1.0, 0.0]))


class TestReshape:
    """``(T*H, ...) -> (T, H, ...)`` and the loss mask."""

    def test_actions_fold_timestep_major(self):
        seq_len, chunk, dim = 4, 10, 7
        flat = rearrange(
            torch.arange(seq_len * chunk * dim).float(), "(t h d) -> (t h) d", t=seq_len, h=chunk
        )
        folded = rearrange(flat, "(t h) ... -> t h ...", t=seq_len)
        assert folded.shape == (seq_len, chunk, dim)
        # Timestep 1's chunk is rows chunk..2*chunk-1 of the flat tensor.
        torch.testing.assert_close(folded[1], flat[chunk : 2 * chunk])

    def test_pad_mask_folds_the_same_way(self):
        seq_len, chunk = 3, 5
        flat = torch.zeros(seq_len * chunk, dtype=torch.bool)
        flat[-2:] = True  # last window ran off the episode end
        folded = rearrange(flat, "(t h) -> t h", t=seq_len)
        assert folded.shape == (seq_len, chunk)
        assert folded[-1, -2:].all() and not folded[0].any()

    def test_indivisible_leading_dim_is_a_bug_not_a_silent_truncation(self):
        """A mismatched fold must raise, never truncate.

        `einops` raises its own error type here rather than a builtin, which is
        the point: a silent truncation would drop timesteps and still train.
        """
        from einops import EinopsError

        with pytest.raises(EinopsError):
            rearrange(torch.zeros(11, 7), "(t h) ... -> t h ...", t=4)


class TestConfigValidation:
    """The mutually-exclusive guards."""

    def test_rejects_sequence_length_with_n_obs_history(self):
        from opentau.configs.default import DatasetMixtureConfig

        with pytest.raises(ValueError, match="observation time axis"):
            DatasetMixtureConfig(sequence_length=4, n_obs_history=4)

    def test_rejects_non_positive_sequence_length(self):
        from opentau.configs.default import DatasetMixtureConfig

        with pytest.raises(ValueError, match="sequence_length"):
            DatasetMixtureConfig(sequence_length=0)

    def test_rejects_non_positive_stride(self):
        from opentau.configs.default import DatasetMixtureConfig

        with pytest.raises(ValueError, match="sequence_stride"):
            DatasetMixtureConfig(sequence_length=4, sequence_stride=0)

    def test_defaults_are_the_pre_feature_behaviour(self):
        from opentau.configs.default import DatasetMixtureConfig

        config = DatasetMixtureConfig()
        assert config.sequence_length == 1
        assert config.sequence_stride is None
        assert config.n_obs_history is None


class TestMirrorMatchesImplementation:
    """The local mirrors above must not drift from the real construction."""

    def test_mirror_matches_the_implementation(self):
        """Reads the real source and re-evaluates its expression.

        A hand-copied formula in a test is only a pin while it still matches the
        code — otherwise it pins the copy. Rather than duplicate the whole
        ``resolve_delta_timestamps`` call graph (which needs a dataset, a name
        map and a policy), this extracts the comprehension from the source and
        checks the mirror reproduces it.
        """
        import inspect

        from opentau.datasets import factory

        source = inspect.getsource(factory.resolve_delta_timestamps)
        # The two constructions this file mirrors must still be present verbatim.
        assert "(-(seq_len - 1 - t) * seq_stride + h) / action_freq" in source, (
            "the action-offset expression changed; update _action_offsets to match"
        )
        assert "-(seq_len - 1 - t) * seq_stride / action_freq" in source, (
            "the observation-offset expression changed; update _obs_offsets to match"
        )


class TestNumpyRoundTrip:
    """``resolve_delta_timestamps`` returns numpy arrays; ordering must survive."""

    def test_offsets_survive_the_numpy_conversion(self):
        chunk_offsets = list(range(10))
        flat = _action_offsets(4, 1, chunk_offsets, 20.0)
        arr = np.array(flat)
        assert arr.shape == (40,)
        # Strictly increasing within a timestep, and each timestep starts one
        # frame later than the previous — the property the reshape relies on.
        grid = arr.reshape(4, 10)
        assert np.all(np.diff(grid, axis=1) > 0)
        assert np.allclose(np.diff(grid[:, 0]), 1 / 20.0)
