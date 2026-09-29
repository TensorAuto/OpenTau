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

"""Camera windows that run *forward* (``policy.camera_delta_indices``).

Until now every camera offset OpenTau emitted was non-positive: cameras and state got
the observation-history window, and only actions had a forward horizon. A policy that
supervises **predicted frames** needs the frames after the observation as targets, so it
supplies its own camera window instead.

The machinery below `resolve_delta_timestamps` already handled this — it clips
``idx + delta`` into the episode and raises ``<key>_is_pad`` at whichever end overruns,
with no sign assumption anywhere. The tests here pin both halves of that claim: that the
request side now emits forward offsets for cameras only, and that the fetch side treats a
positive offset as the exact mirror of a negative one.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.policies import PreTrainedConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import resolve_delta_timestamps
from opentau.datasets.lerobot_dataset import LeRobotDataset
from opentau.datasets.transforms import ImageTransformsConfig


@dataclass
class DummyPolicyConfig(PreTrainedConfig):
    """Minimal policy config whose camera window is injectable."""

    chunk_size: int = 4
    history_interval: int = 1
    camera_window: tuple[int, ...] | None = None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    @property
    def camera_delta_indices(self):
        return None if self.camera_window is None else list(self.camera_window)

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass


def _make_cfg(camera_window=None, n_obs_history=None, action_freq=30.0, sequence_length=1):
    dataset_cfg = DatasetConfig(
        repo_id="mock_dataset",
        root="/tmp/mock",
        image_transforms=ImageTransformsConfig(enable=False),
        episodes=[0],
        video_backend=None,
    )
    mixture_cfg = DatasetMixtureConfig(
        datasets=[dataset_cfg],
        weights=[1.0],
        action_freq=action_freq,
        n_obs_history=n_obs_history,
        sequence_length=sequence_length,
    )
    cfg = TrainPipelineConfig(
        dataset_mixture=mixture_cfg,
        policy=DummyPolicyConfig(camera_window=camera_window),
        batch_size=8,
    )
    return cfg, dataset_cfg


def _meta(features):
    meta = MagicMock()
    meta.features = features
    return meta


# --------------------------------------------------------------------------- request side
def test_default_is_unchanged_for_every_other_policy():
    """``camera_delta_indices`` is None by default, so nothing else moves.

    The property is concrete-with-a-None-default rather than abstract precisely so no
    existing config has to opt out; this pins that the old path still runs.
    """
    cfg, ds_cfg = _make_cfg(camera_window=None)
    dt, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}, "state": {}}))
    np.testing.assert_array_equal(dt["camera0"], [0.0])
    np.testing.assert_array_equal(dt["state"], [0.0])


def test_forward_camera_window_is_emitted_in_seconds():
    cfg, ds_cfg = _make_cfg(camera_window=(0, 1, 2, 3, 4), action_freq=30.0)
    dt, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}}))
    np.testing.assert_allclose(dt["camera0"], np.array([0, 1, 2, 3, 4]) / 30.0)


def test_state_does_not_get_the_camera_window():
    """It is the *video* stream that needs future targets, not proprioception.

    Handing state the same forward window would silently feed the policy future joint
    positions — a label leak that no shape check would catch.
    """
    cfg, ds_cfg = _make_cfg(camera_window=(0, 1, 2))
    dt, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}, "state": {}}))
    assert len(dt["camera0"]) == 3
    np.testing.assert_array_equal(dt["state"], [0.0])


def test_a_past_facing_window_still_works():
    """The knob is sign-agnostic; it replaces the window, it is not a 'futures' flag."""
    cfg, ds_cfg = _make_cfg(camera_window=(-2, -1, 0), action_freq=30.0)
    dt, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}}))
    np.testing.assert_allclose(dt["camera0"], np.array([-2, -1, 0]) / 30.0)


# --------------------------------------------------------------------------- conflicts
def test_conflict_with_n_obs_history_is_rejected():
    cfg, ds_cfg = _make_cfg(camera_window=(0, 1), n_obs_history=3)
    with pytest.raises(ValueError, match="n_obs_history"):
        resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}}))


def test_conflict_with_sequence_mode_is_rejected():
    cfg, ds_cfg = _make_cfg(camera_window=(0, 1), sequence_length=2)
    with pytest.raises(ValueError, match="sequence_length"):
        resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}}))


def test_unsorted_window_is_rejected():
    """Frames come back in offset order and are stacked as a time axis.

    An unsorted window would reorder time silently — every shape still checks out.
    """
    cfg, ds_cfg = _make_cfg(camera_window=(0, 2, 1))
    with pytest.raises(ValueError, match="ascending"):
        resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}}))


def test_empty_window_is_rejected():
    cfg, ds_cfg = _make_cfg(camera_window=())
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}}))


# --------------------------------------------------------------------------- fetch side
def _query(idx, ep_start, ep_end, offsets, fps=30):
    """Drive ``LeRobotDataset._get_query_indices_soft`` against a stub episode."""
    mean = {"camera0": np.asarray(offsets, dtype=float) / fps}
    zeros = {"camera0": np.zeros(len(offsets))}
    stub = MagicMock()
    stub.episode_data_index = {
        "from": torch.tensor([ep_start]),
        "to": torch.tensor([ep_end]),
    }
    stub.epi2idx = {0: 0}
    stub.delta_timestamps_params = (mean, zeros, mean, mean)
    stub.fps = fps
    return LeRobotDataset._get_query_indices_soft(stub, idx, 0)


def test_positive_offset_clips_at_the_episode_end_and_flags_pad():
    """The mirror of the existing start-of-episode behaviour.

    This is the whole reason the change is small: nothing below the request side assumed
    non-positive offsets. A window running past the last frame clamps to it and reports
    the overrun, exactly as a window running before frame 0 clamps and reports.
    """
    # episode spans [0, 10); sit at 8 and ask for 0..4 ahead -> 8,9,10,11,12
    q, pad = _query(idx=8, ep_start=0, ep_end=10, offsets=[0, 1, 2, 3, 4])
    np.testing.assert_array_equal(q["camera0"], [8, 9, 9, 9, 9])  # clamped to ep_end - 1
    np.testing.assert_array_equal(pad["camera0_is_pad"].numpy(), [False, False, True, True, True])


def test_negative_offset_clips_at_the_episode_start_and_flags_pad():
    """The pre-existing half, pinned here so the symmetry is visible in one place."""
    q, pad = _query(idx=1, ep_start=0, ep_end=10, offsets=[-3, -2, -1, 0])
    np.testing.assert_array_equal(q["camera0"], [0, 0, 0, 1])
    np.testing.assert_array_equal(pad["camera0_is_pad"].numpy(), [True, True, False, False])


def test_window_wholly_inside_the_episode_flags_nothing():
    q, pad = _query(idx=4, ep_start=0, ep_end=10, offsets=[0, 1, 2])
    np.testing.assert_array_equal(q["camera0"], [4, 5, 6])
    assert not pad["camera0_is_pad"].any()


# --------------------------------------------------------------------------- flux3_action
def test_flux3_action_window_matches_upstreams_required_frame_count():
    """Upstream refuses any window that is not exactly ``window_frames`` per camera.

    Deriving the offsets independently of ``window_frames`` and then asserting the two
    agree is what makes a drift between them fail here rather than at the first training
    forward.
    """
    from opentau.policies.flux3_action.configuration_flux3_action import Flux3ActionConfig

    cfg = Flux3ActionConfig()
    offsets = cfg.camera_delta_indices
    assert offsets == list(range(0, cfg.chunk_size + 1))
    assert len(offsets) == cfg.to_policy_config().window_frames

    hist = Flux3ActionConfig(
        n_obs_steps=4, inference_profile="history", history_snapshots=2, gripper_flip_dims=()
    )
    hist_offsets = hist.camera_delta_indices
    assert hist_offsets == list(range(-3, hist.chunk_size + 1))
    assert len(hist_offsets) == hist.to_policy_config().window_frames


def test_flux3_action_window_is_ascending_and_accepted_by_the_factory():
    """End-to-end: the policy's own window survives the factory's validation."""
    from opentau.policies.flux3_action.configuration_flux3_action import Flux3ActionConfig

    policy = Flux3ActionConfig()
    cfg, ds_cfg = _make_cfg(camera_window=tuple(policy.camera_delta_indices), action_freq=15.0)
    dt, _, _, _ = resolve_delta_timestamps(cfg, ds_cfg, _meta({"camera0": {}, "state": {}}))
    assert len(dt["camera0"]) == policy.to_policy_config().window_frames
    np.testing.assert_allclose(dt["camera0"][0], 0.0)
    np.testing.assert_allclose(dt["camera0"][-1], policy.chunk_size / 15.0)
    np.testing.assert_array_equal(dt["state"], [0.0])
