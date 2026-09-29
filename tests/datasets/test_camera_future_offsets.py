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


# ------------------------------------------------- standardization (where it actually broke)
def _standardizer(*, n_obs_history=None, sequence_length=1, camera_window_frames=None, resolution=(224, 224)):
    """A ``BaseDataset`` stub wired for ``_standardize_images``, with the real resize.

    Only the config knobs are faked; ``resize_with_pad``, ``temporal_camera_frames`` and
    the unit-range assertion are the real implementations, so this exercises the actual
    shape handling rather than a restatement of it.
    """
    from opentau.datasets.lerobot_dataset import BaseDataset

    ds = MagicMock(spec=BaseDataset)
    ds.n_obs_history = n_obs_history
    ds.sequence_length = sequence_length
    ds.camera_window_frames = camera_window_frames
    ds.resolution = resolution
    ds._get_name_map = lambda: {"camera0": "observation.images.cam0"}
    ds.resize_with_pad = BaseDataset.resize_with_pad.__get__(ds)
    ds._assert_image_in_unit_range = BaseDataset._assert_image_in_unit_range.__get__(ds)
    ds.temporal_camera_frames = BaseDataset.temporal_camera_frames.fget(ds)
    return ds


def _frames(t, h=360, w=640):
    return {
        "observation.images.cam0": torch.rand(t, 3, h, w),
        "observation.images.cam0_is_pad": torch.zeros(t, dtype=torch.bool),
    }


def test_policy_owned_window_survives_standardization():
    """The regression this PR's first draft missed entirely.

    A policy-owned camera window has ``n_obs_history`` unset and ``sequence_length == 1``
    *by construction* (the factory rejects the alternatives), so keying the shape handling
    on those two alone sent it down the scalar path, where ``_is_pad.item()`` raised
    ``a Tensor with 33 elements cannot be converted to Scalar`` on the first training
    batch — a crash no request-side test could ever have caught.
    """
    from opentau.datasets.lerobot_dataset import BaseDataset

    ds = _standardizer(camera_window_frames=33)
    out = {}
    pads = BaseDataset._standardize_images(ds, _frames(33), out, 1)
    assert out["camera0"].shape == (33, 3, 224, 224)
    assert pads == [False]


def test_absent_camera_still_gets_the_window_shaped_zeros():
    """A missing camera must match the present ones' rank, or collation breaks."""
    from opentau.datasets.lerobot_dataset import BaseDataset

    ds = _standardizer(camera_window_frames=33)
    ds._get_name_map = lambda: {}  # camera0 not present in this dataset
    out = {}
    pads = BaseDataset._standardize_images(ds, {}, out, 1)
    assert out["camera0"].shape == (33, 3, 224, 224)
    assert pads == [True]


def test_single_frame_path_is_untouched():
    """With no window at all, cameras stay rank-3 — the default every other policy uses."""
    from opentau.datasets.lerobot_dataset import BaseDataset

    ds = _standardizer()
    item = {
        "observation.images.cam0": torch.rand(3, 360, 640),
        "observation.images.cam0_is_pad": torch.tensor(False),
    }
    out = {}
    pads = BaseDataset._standardize_images(ds, item, out, 1)
    assert out["camera0"].shape == (3, 224, 224)
    assert pads == [False]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, None),
        ({"n_obs_history": 4}, 4),
        ({"sequence_length": 3}, 3),
        ({"camera_window_frames": 33}, 33),
        # config validation makes these mutually exclusive; precedence is pinned anyway so
        # a future overlap fails loudly here rather than silently picking a window.
        ({"n_obs_history": 4, "camera_window_frames": 33}, 4),
        ({"sequence_length": 3, "camera_window_frames": 33}, 3),
    ],
)
def test_temporal_camera_frames_resolves_all_three_mechanisms(kwargs, expected):
    """One property owns the question, because keying on a subset has broken twice."""
    from opentau.datasets.lerobot_dataset import BaseDataset

    ds = _standardizer(**kwargs)
    assert BaseDataset.temporal_camera_frames.fget(ds) == expected


# ------------------------------------------------- end-to-end config validation
def _train_cfg(policy, n_obs_history=None, action_chunk=32):
    ds = DatasetConfig(
        repo_id="mock_dataset",
        root="/tmp/mock",
        image_transforms=ImageTransformsConfig(enable=False),
        episodes=[0],
        video_backend=None,
    )
    mix = DatasetMixtureConfig(datasets=[ds], weights=[1.0], action_freq=15.0, n_obs_history=n_obs_history)
    return TrainPipelineConfig(
        dataset_mixture=mix,
        policy=policy,
        batch_size=2,
        action_chunk=action_chunk,
        use_policy_training_preset=True,
    )


def _f3a_history():
    from opentau.policies.flux3_action.configuration_flux3_action import Flux3ActionConfig

    return Flux3ActionConfig(
        n_obs_steps=4, inference_profile="history", history_snapshots=2, gripper_flip_dims=()
    )


def test_history_profile_is_configurable_end_to_end():
    """Regression: the F3A history profile used to be dead config space.

    ``validate()`` demanded ``policy.n_obs_steps == dataset_mixture.n_obs_history`` while
    ``resolve_delta_timestamps`` rejected pairing a policy-owned camera window with
    ``n_obs_history`` at all — so every route was refused and each error pointed at the
    other. A policy-owned window is now authoritative, and this builds the config both
    checks see rather than testing either in isolation, which is how the dead end hid.
    """
    cfg = _train_cfg(_f3a_history(), n_obs_history=None)
    cfg.validate()  # must not raise
    assert cfg.policy.n_obs_steps == 4
    assert len(cfg.policy.camera_delta_indices) == cfg.policy.to_policy_config().window_frames


def test_mixture_may_not_also_claim_the_camera_window():
    """The policy owning the window means the mixture must not set one too."""
    cfg = _train_cfg(_f3a_history(), n_obs_history=4)
    with pytest.raises(ValueError, match="n_obs_history must stay unset"):
        cfg.validate()


def test_policies_without_a_camera_window_keep_the_original_pairing_rule():
    """The relaxation is scoped: every other policy still must agree with the mixture.

    Uses a real registered policy rather than the local dummy, so this exercises the same
    path a shipped config takes — ``validate()`` resolves the draccus choice name, which a
    test-local subclass has no entry for.
    """
    from opentau.policies.pi05.configuration_pi05 import PI05Config

    policy = PI05Config()
    assert policy.camera_delta_indices is None, "pi05 must not opt into a camera window"
    # action_chunk matches pi05's own horizon so the failure under test is the
    # n_obs pairing rule, not an unrelated chunk/horizon conflict.
    cfg = _train_cfg(policy, n_obs_history=3, action_chunk=policy.chunk_size)
    with pytest.raises(ValueError, match="n_obs_steps"):
        cfg.validate()
