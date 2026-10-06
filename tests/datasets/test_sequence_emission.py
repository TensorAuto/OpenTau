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
        """The offset math places consecutive timesteps ``stride`` frames apart.

        Exercised at stride 1 as pure offset arithmetic. Note stride 1 is no
        longer a *reachable configuration* — RoboTTT's timestep is one action
        chunk, so the config layer derives ``stride = action_chunk`` and rejects
        anything else (see ``TestStrideEqualsChunkContract``); the mirror math
        itself is stride-agnostic.
        """
        freq = 20.0
        offsets = _obs_offsets(5, 1, freq)
        frames = [round(o * freq) for o in offsets]
        assert frames == [-4, -3, -2, -1, 0]

    def test_window_span_in_frames(self):
        """A T-timestep window spans ``(T-1)*stride + chunk`` frames.

        With the derived ``stride = action_chunk``, the span is ``T * chunk``
        frames — the budget that sizes ``sequence_length`` against a dataset's
        shortest episodes (e.g. chunk 10: T=4 spans 40 frames vs LIBERO's
        shortest episode of 75; the boundary-padding path absorbs the rest).
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
        # The stride derivation is the behavior this contract rests on: `None`
        # must resolve to `action_chunk` (chunk-tiled sequences), and the old
        # stride-1 fallback must stay gone — reverting it re-opens the
        # teacher-forcing leak while every validator-level test still passes.
        assert "seq_stride = cfg.action_chunk if seq_len > 1 else 1" in source, (
            "the stride derivation changed; sequence timesteps must tile in action chunks"
        )
        assert 'getattr(cfg.dataset_mixture, "sequence_stride", None) or 1' not in source, (
            "the stride-1 fallback is back; that recipe leaks overlapping action targets"
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


class TestObsHistoryPadFallback:
    """The interior-sample fallback must match the boundary passthrough's shape.

    The fetch layer attaches per-step state pad flags only to samples whose
    query window crossed an episode boundary; interior samples fall through to
    ``BaseDataset._obs_history_pad_fallback``. Both kinds land in one
    ``default_collate`` batch, so the fallback length must equal the
    passthrough length in every temporal mode — a fixed ``(1,)`` against the
    passthrough's ``(sequence_length,)`` crashed every ``batch_size > 1``
    sequence run with "Trying to resize storage that is not resizable"
    (invisible at batch 1, the only batch shape pi05_ttt had run at).
    """

    @staticmethod
    def _stub(sequence_length: int = 1, n_obs_history: int | None = None):
        from opentau.datasets.lerobot_dataset import BaseDataset

        ds = object.__new__(BaseDataset)
        ds.sequence_length = sequence_length
        ds.n_obs_history = n_obs_history
        return ds

    def test_sequence_mode_matches_the_boundary_passthrough_shape(self):
        seq_len = 32
        ds = self._stub(sequence_length=seq_len)
        fallback = ds._obs_history_pad_fallback(padded=False)
        assert fallback.shape == (seq_len,)
        assert fallback.dtype == torch.bool and not fallback.any()

    def test_interior_and_boundary_samples_collate(self):
        """The actual crash case: one interior + one boundary sample in a batch."""
        from torch.utils.data import default_collate

        seq_len = 32
        ds = self._stub(sequence_length=seq_len)
        interior = {"obs_history_is_pad": ds._obs_history_pad_fallback(padded=False)}
        boundary_passthrough = torch.zeros(seq_len, dtype=torch.bool)
        boundary_passthrough[-3:] = True  # window ran off the episode end
        boundary = {"obs_history_is_pad": boundary_passthrough}
        batch = default_collate([interior, boundary])
        assert batch["obs_history_is_pad"].shape == (2, seq_len)
        assert batch["obs_history_is_pad"][1, -3:].all()

    def test_history_mode_is_unchanged(self):
        ds = self._stub(n_obs_history=5)
        torch.testing.assert_close(
            ds._obs_history_pad_fallback(padded=False), torch.zeros(5, dtype=torch.bool)
        )
        torch.testing.assert_close(ds._obs_history_pad_fallback(padded=True), torch.ones(5, dtype=torch.bool))

    def test_plain_mode_is_byte_identical_to_the_pre_fix_fallback(self):
        ds = self._stub(sequence_length=1)
        torch.testing.assert_close(ds._obs_history_pad_fallback(padded=False), torch.tensor([False]))
        torch.testing.assert_close(ds._obs_history_pad_fallback(padded=True), torch.tensor([True]))

    def test_history_drop_marks_every_sequence_timestep(self):
        """The drop branch (padded=True) must also be sequence-shaped."""
        ds = self._stub(sequence_length=8)
        dropped = ds._obs_history_pad_fallback(padded=True)
        assert dropped.shape == (8,) and dropped.all()


class TestStrideEqualsChunkContract:
    """`sequence_stride` is derived from `action_chunk`, never a free knob.

    RoboTTT's timestep is one H-step action chunk — its sequences tile the
    trajectory in disjoint chunks and the paper has no stride concept. A
    sub-chunk stride overlaps consecutive timesteps' action targets, so the
    mostly teacher-forced context contains the current chunk's answers; the TTT
    layers learn to copy them and closed-loop rollouts collapse (observed at
    stride 1, chunk 20: 0% success from a 33%-success frozen base).
    """

    @staticmethod
    def _cfg(sequence_length: int, sequence_stride: int | None, action_chunk: int = 20):
        import draccus

        from opentau.configs.train import TrainPipelineConfig

        overrides = [
            "--dataset_mixture.datasets",
            '[{"repo_id": "dummy/dummy"}]',
            "--dataset_mixture.sequence_length",
            str(sequence_length),
            "--action_chunk",
            str(action_chunk),
            "--batch_size",
            "2",
            "--dataloader_batch_size",
            "2",
        ]
        if sequence_stride is not None:
            overrides += ["--dataset_mixture.sequence_stride", str(sequence_stride)]
        return draccus.parse(TrainPipelineConfig, args=overrides)

    def test_mismatched_stride_is_rejected(self):
        cfg = self._cfg(sequence_length=8, sequence_stride=1)
        with pytest.raises(ValueError, match="disjoint action chunks"):
            cfg._validate_sequence_stride()

    def test_explicit_equal_stride_is_accepted(self):
        cfg = self._cfg(sequence_length=8, sequence_stride=20)
        cfg._validate_sequence_stride()

    def test_none_is_accepted_and_derives_the_chunk(self):
        cfg = self._cfg(sequence_length=8, sequence_stride=None)
        cfg._validate_sequence_stride()

    def test_stride_is_inert_at_sequence_length_one(self):
        """Non-sequence configs keep any stride value: the field is never read."""
        cfg = self._cfg(sequence_length=1, sequence_stride=3)
        cfg._validate_sequence_stride()


class TestOversamplingGuardStrideAware:
    """The mixed-frequency guard trips only when timesteps land inside one frame.

    Consecutive timesteps are ``seq_stride / action_freq`` seconds apart and
    collide only when that is shorter than one source frame — i.e. when
    ``action_freq > seq_stride * fps``. The stride-1-era predicate
    (``action_freq > fps``) would wrongly reject valid oversampled configs now
    that the stride derives from ``action_chunk``; the boundary cases below
    fail under a revert.
    """

    @staticmethod
    def _resolve(action_freq: float, fps: int, chunk: int = 2, stride: int | None = None):
        from types import SimpleNamespace

        from opentau.configs.default import DatasetConfig
        from opentau.datasets.factory import resolve_delta_timestamps

        train_cfg = SimpleNamespace(
            dataset_mixture=SimpleNamespace(
                action_freq=action_freq,
                n_obs_history=None,
                sequence_length=2,
                sequence_stride=stride,
            ),
            action_chunk=chunk,
            policy=None,
        )
        ds_cfg = DatasetConfig(
            repo_id="_tests/oversampling",
            data_features_name_mapping={"state": "observation.state", "actions": "action"},
        )
        meta = SimpleNamespace(
            features={"observation.state": {}, "action": {}}, fps=fps, control_mode="joint"
        )
        return resolve_delta_timestamps(train_cfg, ds_cfg, meta)

    def test_boundary_equal_passes(self):
        """action_freq == stride * fps: timesteps exactly one frame apart — valid.

        The stride-1-era predicate (``action_freq > fps``) raised here; passing
        is the revert detector.
        """
        self._resolve(action_freq=20.0, fps=10)

    def test_over_boundary_raises(self):
        with pytest.raises(ValueError, match="resolve to the same"):
            self._resolve(action_freq=21.0, fps=10)

    def test_explicit_equal_stride_same_boundary(self):
        self._resolve(action_freq=20.0, fps=10, stride=2)
        with pytest.raises(ValueError, match="resolve to the same"):
            self._resolve(action_freq=21.0, fps=10, stride=2)


class TestPadShiftToFront:
    """Rotating a short episode's real timesteps to the front of the window.

    The window is anchored on the episode's last frame, so a short episode
    arrives with padding at the FRONT. Keeping that placement would hand short
    demonstrations to TTT at a RoPE phase evaluation never produces, because
    demo pools skip episodes shorter than the window. The rotation moves the
    real frames to positions ``0..n_real-1`` without changing which frames were
    selected or their order.
    """

    @staticmethod
    def _item(seq_len: int = 6, n_pad: int = 2):
        """Builds a standard-format item whose first ``n_pad`` timesteps are padding.

        Args:
            seq_len: Timesteps in the window.
            n_pad: Padded timesteps at the front.

        Returns:
            The item dict.
        """
        tp = torch.zeros(seq_len, dtype=torch.bool)
        tp[:n_pad] = True
        return {
            "state": torch.arange(seq_len * 3, dtype=torch.float32).reshape(seq_len, 3),
            "actions": torch.arange(seq_len * 2 * 4, dtype=torch.float32).reshape(seq_len, 2, 4),
            "action_is_pad": torch.zeros(seq_len, 2, dtype=torch.bool),
            "loss_mask": torch.ones(seq_len, dtype=torch.bool),
            "obs_history_is_pad": torch.zeros(seq_len, dtype=torch.bool),
            "timestep_is_pad": tp,
            "camera0": torch.arange(seq_len * 3, dtype=torch.float32).reshape(seq_len, 3),
            "prompt": "not a tensor, must be left alone",
        }

    def test_padding_moves_to_the_back(self):
        from opentau.datasets.lerobot_dataset import LeRobotDataset

        seq_len, n_pad = 6, 2
        item = self._item(seq_len, n_pad)
        LeRobotDataset._shift_real_timesteps_to_front(item, n_pad)

        assert not item["timestep_is_pad"][: seq_len - n_pad].any()
        assert item["timestep_is_pad"][seq_len - n_pad :].all()

    def test_real_frames_keep_their_content_and_order(self):
        """The rotation must reposition frames, never reselect or reorder them."""
        from opentau.datasets.lerobot_dataset import LeRobotDataset

        seq_len, n_pad = 6, 2
        item = self._item(seq_len, n_pad)
        before = item["state"].clone()
        LeRobotDataset._shift_real_timesteps_to_front(item, n_pad)

        # Real frames were rows n_pad..end; they must now be rows 0..n_real-1,
        # in the same order.
        torch.testing.assert_close(item["state"][: seq_len - n_pad], before[n_pad:])
        # And the wrapped padding keeps the rows it had.
        torch.testing.assert_close(item["state"][seq_len - n_pad :], before[:n_pad])

    def test_every_temporal_key_moves_together(self):
        """A key left behind would desynchronise the mask from its frames."""
        from opentau.datasets.lerobot_dataset import LeRobotDataset

        seq_len, n_pad = 6, 2
        item = self._item(seq_len, n_pad)
        originals = {k: v.clone() for k, v in item.items() if torch.is_tensor(v)}
        LeRobotDataset._shift_real_timesteps_to_front(item, n_pad)

        for key, before in originals.items():
            torch.testing.assert_close(item[key], torch.roll(before, shifts=-n_pad, dims=0))

    def test_non_tensor_keys_are_untouched(self):
        from opentau.datasets.lerobot_dataset import LeRobotDataset

        item = self._item()
        LeRobotDataset._shift_real_timesteps_to_front(item, 2)
        assert item["prompt"] == "not a tensor, must be left alone"

    def test_a_missing_temporal_key_raises(self):
        """Guards against a future key reaching the sample after the shift."""
        from opentau.datasets.lerobot_dataset import LeRobotDataset

        item = self._item()
        del item["timestep_is_pad"]

        with pytest.raises(RuntimeError, match="missed required temporal key"):
            LeRobotDataset._shift_real_timesteps_to_front(item, 2)


class TestPadTimestepCount:
    """``frame_index -> n_pad``, including the resampled case.

    A timestep spans ``sequence_stride * fps / action_freq`` SOURCE frames:
    ``resolve_delta_timestamps`` emits the window's offsets in seconds against
    ``action_freq`` and ``get_delta_indices_soft`` resolves them back to frames
    against the dataset's native fps. Reading ``sequence_stride`` as a frame
    count holds only when the two rates agree, and validation rejects only
    *over*sampling, so an undersampled dataset reaches this code.
    """

    @staticmethod
    def _stub(action_chunk: int = 15, fps: int = 15, action_freq: float | None = None):
        from opentau.datasets.lerobot_dataset import BaseDataset

        ds = object.__new__(BaseDataset)
        ds.action_chunk = action_chunk
        ds.fps = fps
        ds._action_freq = action_freq
        return ds

    def test_long_episode_has_no_padding(self):
        ds = self._stub()
        assert ds._pad_timestep_count(frame_index=599, seq_len=31) == 0

    def test_short_episode_pads_the_shortfall(self):
        ds = self._stub()
        # frame 120 at 15 frames/timestep -> 9 real timesteps of 31.
        assert ds._pad_timestep_count(frame_index=120, seq_len=31) == 31 - 9

    def test_first_frame_leaves_one_real_timestep(self):
        ds = self._stub()
        assert ds._pad_timestep_count(frame_index=0, seq_len=31) == 30

    def test_undersampled_dataset_uses_source_frames(self):
        """``action_freq`` below the native fps widens a timestep in source frames.

        At 30 fps with ``action_freq`` 15, one 15-unit timestep spans 30 source
        frames, so frame 120 covers 5 timesteps -- not the 9 that reading the
        stride as frames would give.
        """
        ds = self._stub(action_chunk=15, fps=30, action_freq=15.0)
        assert ds._pad_timestep_count(frame_index=120, seq_len=31) == 31 - 5

    def test_matched_rates_agree_with_the_naive_stride(self):
        """When the rates agree the ratio is 1 and the simple reading is correct."""
        matched = self._stub(action_chunk=15, fps=15, action_freq=15.0)
        unset = self._stub(action_chunk=15, fps=15, action_freq=None)
        for frame_index in (0, 14, 15, 120, 599):
            assert matched._pad_timestep_count(frame_index, 31) == unset._pad_timestep_count(frame_index, 31)

    def test_a_boundary_timestep_is_called_padding(self):
        """Conservative rounding: never learn from a frame the fetch layer clamped."""
        ds = self._stub(action_chunk=15, fps=30, action_freq=15.0)
        # 29 source frames is one short of a full 30-frame timestep.
        assert ds._pad_timestep_count(frame_index=29, seq_len=4) == 3
        assert ds._pad_timestep_count(frame_index=30, seq_len=4) == 2


class TestSequenceAwareDeltaFold:
    """Each timestep's chunk must be offset by its OWN chunk-start state.

    Sequence mode reaches the delta transform while ``actions`` are still flat
    ``(T * chunk, D_a)`` and ``state`` already carries its time axis
    ``(T, D_s)``. Both are rank 2, so the offset broadcast reads the trailing
    axis as a history window and applies the LAST timestep's pose to the whole
    sequence -- every timestep but the final one offset from the wrong pose.
    Folding the timestep axis out first is what makes each chunk relative to
    its own start.
    """

    @staticmethod
    def _fold_and_offset(actions_flat, state, delta_map, seq_len):
        """Mirrors the sequence branch of ``_apply_column_index_and_delta``.

        Args:
            actions_flat: ``(T * chunk, D_a)`` absolute actions.
            state: ``(T, D_s)`` per-timestep state.
            delta_map: ``{action_pos: state_pos}``.
            seq_len: Timesteps.

        Returns:
            ``(T * chunk, D_a)`` actions, made relative per timestep.
        """
        from opentau.datasets.action_indexing import subtract_chunk_start_state

        folded = rearrange(actions_flat, "(t h) ... -> t h ...", t=seq_len)
        folded = subtract_chunk_start_state(folded, state, delta_map)
        return rearrange(folded, "t h ... -> (t h) ...")

    def test_each_timestep_is_offset_by_its_own_start_state(self):
        seq_len, chunk, dim = 2, 3, 4
        # Timestep 0 holds 10s, timestep 1 holds 100s, so a cross-timestep
        # offset is unmistakable in the result.
        actions = torch.cat(
            [
                torch.full((chunk, dim), 10.0),
                torch.full((chunk, dim), 100.0),
            ]
        )
        state = torch.tensor([[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]])
        delta_map = {0: 0, 1: 1}  # dims 2,3 stay absolute

        out = self._fold_and_offset(actions, state, delta_map, seq_len)

        # Timestep 0: 10 - 1 on mapped dims, 10 untouched elsewhere.
        torch.testing.assert_close(out[:chunk, :2], torch.full((chunk, 2), 9.0))
        torch.testing.assert_close(out[:chunk, 2:], torch.full((chunk, 2), 10.0))
        # Timestep 1: 100 - 5, NOT 100 - 1.
        torch.testing.assert_close(out[chunk:, :2], torch.full((chunk, 2), 95.0))
        torch.testing.assert_close(out[chunk:, 2:], torch.full((chunk, 2), 100.0))

    def test_the_last_timesteps_pose_is_not_broadcast_to_all(self):
        """Pins the exact bug the fold prevents."""
        seq_len, chunk, dim = 2, 3, 2
        actions = torch.cat([torch.full((chunk, dim), 10.0), torch.full((chunk, dim), 10.0)])
        state = torch.tensor([[1.0, 1.0], [5.0, 5.0]])

        out = self._fold_and_offset(actions, state, {0: 0, 1: 1}, seq_len)

        # If the last timestep's pose leaked across, timestep 0 would read 5.0.
        assert not torch.allclose(out[:chunk], torch.full((chunk, dim), 5.0))
        torch.testing.assert_close(out[:chunk], torch.full((chunk, dim), 9.0))

    def test_round_trips_through_the_inverse(self):
        from opentau.datasets.action_indexing import add_chunk_start_state

        seq_len, chunk, dim = 3, 4, 5
        torch.manual_seed(0)
        actions = torch.randn(seq_len * chunk, dim)
        state = torch.randn(seq_len, dim)
        delta_map = {0: 0, 2: 2}

        rel = self._fold_and_offset(actions, state, delta_map, seq_len)
        folded = rearrange(rel, "(t h) ... -> t h ...", t=seq_len)
        back = rearrange(add_chunk_start_state(folded, state, delta_map), "t h ... -> (t h) ...")

        torch.testing.assert_close(back, actions)


class TestTemporalKeyMirror:
    """``_TIME_AXIS_KEYS`` and ``_is_temporal`` must name the same keys.

    The shift rotates every key carrying a time axis; the paired loader
    concatenates exactly those across a pair's halves. A key in one list and not
    the other desynchronises a shifted half from an unshifted one — silent, and
    visible only as degraded training. The comments say they mirror each other;
    this pins it.
    """

    def test_the_two_lists_agree(self):
        from opentau.datasets.lerobot_dataset import BaseDataset
        from opentau.datasets.paired_sequence import PairedSequenceDataset

        shifted = set(BaseDataset._TIME_AXIS_KEYS)
        concatenated = set(PairedSequenceDataset._TEMPORAL_KEYS)

        assert shifted == concatenated, (
            "BaseDataset._TIME_AXIS_KEYS and PairedSequenceDataset._TEMPORAL_KEYS "
            f"have drifted: only shifted={sorted(shifted - concatenated)}, "
            f"only concatenated={sorted(concatenated - shifted)}"
        )

    def test_camera_keys_are_temporal_in_both(self):
        """Cameras are matched by prefix rather than listed, in both places."""
        from opentau.datasets.lerobot_dataset import BaseDataset
        from opentau.datasets.paired_sequence import PairedSequenceDataset

        assert PairedSequenceDataset._is_temporal("camera0")
        item = {
            "camera0": torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
            "state": torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
            "actions": torch.zeros(4, 1, 2),
            "timestep_is_pad": torch.tensor([True, False, False, False]),
        }
        before = item["camera0"].clone()
        BaseDataset._shift_real_timesteps_to_front(item, 1)
        torch.testing.assert_close(item["camera0"], torch.roll(before, shifts=-1, dims=0))


class TestSelectedEpisodeStatsFallback:
    """Episode-subset stats: the empty case falls back, the partial case warns.

    Some v3.0 datasets ship an episodes parquet with no flattened ``stats/*``
    columns, leaving ``episodes_stats`` empty. Aggregating that produced an
    empty dict which then overwrote ``meta.stats``; the ImageNet camera override
    layered image keys onto it, and ``DatasetMixtureMetadata`` later died with
    ``KeyError: 'observation.state'``.
    """

    @staticmethod
    def _stub(episodes, episodes_stats):
        """Builds a dataset stand-in carrying only what the aggregation reads.

        Args:
            episodes: Selected episode indices.
            episodes_stats: ``{episode_index: stats}``.

        Returns:
            The stub.
        """
        from types import SimpleNamespace

        from opentau.datasets.lerobot_dataset import BaseDataset

        ds = object.__new__(BaseDataset)
        ds.episodes = episodes
        ds.repo_id = "org/stub"
        ds.meta = SimpleNamespace(episodes_stats=episodes_stats)
        return ds

    @staticmethod
    def _stats(value):
        """One well-formed stats dict.

        Args:
            value: Fill value for every field.

        Returns:
            A stats dict shaped like the real ones.
        """
        return {
            "observation.state": {
                "mean": np.array([value]),
                "std": np.array([1.0]),
                "min": np.array([value]),
                "max": np.array([value]),
                "count": np.array([1]),
            }
        }

    def test_no_episode_carries_stats_yields_an_empty_aggregate(self):
        """The empty dict is what makes the caller keep the dataset-level stats."""
        ds = self._stub([0, 1, 2], {})
        assert ds._aggregate_selected_episode_stats() == {}

    def test_empty_per_episode_dicts_count_as_absent(self):
        ds = self._stub([0, 1], {0: {}, 1: {}})
        assert ds._aggregate_selected_episode_stats() == {}

    def test_empty_case_warns(self, caplog):
        ds = self._stub([0, 1], {})
        with caplog.at_level("WARNING"):
            ds._aggregate_selected_episode_stats()
        assert "no selected episode carries per-episode stats" in caplog.text

    def test_partial_case_warns_and_names_the_counts(self, caplog):
        ds = self._stub([0, 1, 2], {0: self._stats(1.0)})
        with caplog.at_level("WARNING"):
            out = ds._aggregate_selected_episode_stats()
        assert out, "a partial aggregate is still returned"
        assert "1 of 3 selected episodes carry per-episode stats" in caplog.text

    def test_complete_case_does_not_warn(self, caplog):
        ds = self._stub([0, 1], {0: self._stats(1.0), 1: self._stats(3.0)})
        with caplog.at_level("WARNING"):
            out = ds._aggregate_selected_episode_stats()
        assert "observation.state" in out
        assert "per-episode stats" not in caplog.text
