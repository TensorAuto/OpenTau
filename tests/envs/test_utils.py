#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import warnings
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from opentau.envs.configs import EnvMetadataConfig
from opentau.envs.utils import (
    add_eval_metadata,
    are_all_envs_same_type,
    check_env_attributes_and_types,
    preprocess_observation,
)


def make_type1_env():
    env = make_no_attributes_env()
    env.task_description = "test task description"
    env.task = "test task description"
    return env


def make_type2_env():
    env = make_type1_env()

    class Dummy(env.__class__):
        pass

    env.__class__ = Dummy
    return env


def make_no_attributes_env():
    return gym.make("CartPole-v1")


def make_partial_attributes_env():
    env = make_no_attributes_env()
    env.task_description = "test task description"
    return env


class TestAreAllEnvsSameType:
    """Test cases for are_all_envs_same_type function"""

    def test_all_envs_same_type(self):
        """Test that function returns True when all environments are the same type"""
        vector_env = gym.vector.SyncVectorEnv([make_type1_env for _ in range(3)])

        result = are_all_envs_same_type(vector_env)
        assert result is True

    def test_envs_different_types(self):
        """Test that function returns False when environments have different types"""
        vector_env = gym.vector.SyncVectorEnv([make_type1_env, make_type2_env], observation_mode="different")

        result = are_all_envs_same_type(vector_env)
        assert result is False

    def test_single_env(self):
        """Test that function returns True for single environment"""
        vector_env = gym.vector.SyncVectorEnv([make_type1_env])
        result = are_all_envs_same_type(vector_env)
        assert result is True

    def test_empty_envs_list(self):
        """Test that function handles empty environments list"""
        # This should raise an IndexError since we access envs[0]
        with pytest.raises(IndexError):
            vector_env = gym.vector.SyncVectorEnv([])
            are_all_envs_same_type(vector_env)


class TestCheckEnvAttributesAndTypes:
    """Test cases for check_env_attributes_and_types function"""

    def test_env_with_required_attributes_same_type(self):
        """Test that no warnings are issued when env has required attributes and same types"""
        vector_env = gym.vector.SyncVectorEnv([make_type1_env for _ in range(2)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should not issue any warnings
            assert len(w) == 0

    def test_env_without_required_attributes(self):
        """Test that warning is issued when env lacks required attributes"""
        vector_env = gym.vector.SyncVectorEnv([make_no_attributes_env for _ in range(2)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should issue warning about missing attributes
            assert len(w) == 1
            assert "task_description" in str(w[0].message)
            assert "task" in str(w[0].message)

    def test_envs_different_types(self):
        """Test that warning is issued when environments have different types"""
        vector_env = gym.vector.SyncVectorEnv([make_type1_env, make_type2_env])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should issue warning about different types
            assert len(w) == 1
            assert "different types" in str(w[0].message)

    def test_both_warnings_issued(self):
        """Test that both warnings are issued when both conditions are met"""
        vector_env = gym.vector.SyncVectorEnv([make_no_attributes_env, make_type2_env])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should issue both warnings
            assert len(w) == 2

            # Check that both types of warnings are present
            warning_messages = [str(warning.message) for warning in w]
            assert any("task_description" in msg for msg in warning_messages)
            assert any("different types" in msg for msg in warning_messages)

    def test_warning_filter_scope(self):
        """Test that warning filter is only applied within the function"""
        vector_env = gym.vector.SyncVectorEnv([make_no_attributes_env for _ in range(2)])

        # Set up a warning filter that should be overridden
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)

            # This should not raise an error because the function sets its own filter
            check_env_attributes_and_types(vector_env)

    def test_warning_stacklevel(self):
        """Test that warnings have correct stacklevel"""
        vector_env = gym.vector.SyncVectorEnv([make_no_attributes_env for _ in range(2)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Check that stacklevel is set correctly (should be 2)
            assert w[0].filename.endswith("test_utils.py")  # Should point to test file, not utils.py

    def test_single_env_with_attributes(self):
        """Test with single environment that has required attributes"""
        vector_env = gym.vector.SyncVectorEnv([make_type1_env])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should not issue any warnings
            assert len(w) == 0

    def test_single_env_without_attributes(self):
        """Test with single environment that lacks required attributes"""
        vector_env = gym.vector.SyncVectorEnv([make_no_attributes_env])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should issue warning about missing attributes
            assert len(w) == 1
            assert "task_description" in str(w[0].message)

    def test_partial_attributes(self):
        """Test with environment that has only one of the required attributes"""
        vector_env = gym.vector.SyncVectorEnv([make_partial_attributes_env for _ in range(2)])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env_attributes_and_types(vector_env)

            # Should not issue warning about missing attributes
            assert len(w) == 0


def _make_obs(batch_size: int = 2, device: str = "cpu") -> dict:
    """Minimal observation dict mirroring what ``preprocess_observation`` emits.

    ``add_eval_metadata`` only reads ``observation["state"]`` (for batch size
    and device), so anything richer would be dead weight.
    """
    return {"state": torch.zeros(batch_size, 8, device=device, dtype=torch.float32)}


def _make_cfg(*, fps: int = 30, **metadata_kwargs) -> SimpleNamespace:
    """Build a duck-typed ``cfg`` exposing ``cfg.env.metadata`` and
    ``cfg.env.fps``.

    Avoids constructing a full ``TrainPipelineConfig`` (which pulls in
    optimizer/dataset/policy validation) just to read a handful of attributes.
    ``fps`` defaults to 30 (matches ``EnvConfig.fps``) so the
    ``emit_fps=True`` default path produces a deterministic value.
    """
    return SimpleNamespace(
        env=SimpleNamespace(
            metadata=EnvMetadataConfig(**metadata_kwargs),
            fps=fps,
        )
    )


class TestAddEvalMetadata:
    """Pin the contract between the rollout-time helper and the policy's
    ``prepare_metadata`` reader. The dtypes, shapes, and missing-key skip
    behaviour below are what ``prepare_metadata`` relies on — drift here
    silently changes the prefix the model sees at eval.
    """

    def test_all_none_default_skips_every_key(self):
        """``EnvMetadataConfig`` defaults: every user-controlled field is
        ``None`` and ``emit_fps`` defaults to ``False``, so the helper
        injects **no** metadata keys at all. This preserves pre-PR
        behaviour for any eval config that doesn't opt in.
        """
        obs = _make_obs()
        original_keys = set(obs.keys())
        out = add_eval_metadata(obs, cfg=_make_cfg())
        assert out is obs  # mutated-in-place contract
        assert set(out.keys()) == original_keys, (
            f"unexpected injected keys under all-None metadata default: {set(out.keys()) - original_keys}"
        )

    def test_emit_fps_true_injects_fps(self):
        """``emit_fps=True`` opts in to broadcasting ``cfg.env.fps`` as a
        ``(B,)`` torch.long tensor with ``fps_is_pad=False``.
        """
        obs = _make_obs()
        original_keys = set(obs.keys())
        out = add_eval_metadata(obs, cfg=_make_cfg(emit_fps=True))
        new_keys = set(out.keys()) - original_keys
        assert new_keys == {"fps", "fps_is_pad"}, (
            f"unexpected injected keys: {new_keys}; emit_fps=True with no other "
            f"metadata fields should inject only fps/fps_is_pad"
        )

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_speed_injects_long_tensor_with_pad_flag(self, batch_size):
        obs = _make_obs(batch_size=batch_size)
        add_eval_metadata(obs, cfg=_make_cfg(speed=20))

        assert obs["speed"].dtype == torch.long
        assert obs["speed"].shape == (batch_size,)
        assert torch.equal(obs["speed"], torch.full((batch_size,), 20, dtype=torch.long))

        assert obs["speed_is_pad"].dtype == torch.bool
        assert obs["speed_is_pad"].shape == (batch_size,)
        assert not obs["speed_is_pad"].any()

    def test_quality_injects_long_tensor_with_pad_flag(self):
        obs = _make_obs(batch_size=3)
        add_eval_metadata(obs, cfg=_make_cfg(quality=4))

        assert obs["quality"].dtype == torch.long
        assert torch.equal(obs["quality"], torch.full((3,), 4, dtype=torch.long))
        assert obs["quality_is_pad"].dtype == torch.bool
        assert not obs["quality_is_pad"].any()

    @pytest.mark.parametrize("value", [True, False])
    def test_mistake_injects_bool_tensor_with_pad_flag(self, value):
        obs = _make_obs(batch_size=2)
        add_eval_metadata(obs, cfg=_make_cfg(mistake=value))

        assert obs["mistake"].dtype == torch.bool
        assert obs["mistake"].shape == (2,)
        assert torch.equal(obs["mistake"], torch.full((2,), value, dtype=torch.bool))
        assert obs["mistake_is_pad"].dtype == torch.bool
        assert not obs["mistake_is_pad"].any()

    def test_mistake_false_vs_none_differ(self):
        """``mistake=False`` injects a key; ``mistake=None`` (default) does not.

        This is the foot-gun called out in code review: ``False`` is *not*
        "no mistake info available" — that's ``None``.
        """
        obs_false = _make_obs()
        add_eval_metadata(obs_false, cfg=_make_cfg(mistake=False))
        assert "mistake" in obs_false
        assert "mistake_is_pad" in obs_false

        obs_none = _make_obs()
        add_eval_metadata(obs_none, cfg=_make_cfg(mistake=None))
        assert "mistake" not in obs_none
        assert "mistake_is_pad" not in obs_none

    @pytest.mark.parametrize("key,value", [("robot_type", "UR5"), ("control_mode", "ee")])
    def test_string_fields_broadcast_as_list(self, key, value):
        obs = _make_obs(batch_size=3)
        add_eval_metadata(obs, cfg=_make_cfg(**{key: value}))

        assert obs[key] == [value, value, value]
        assert f"{key}_is_pad" not in obs, "string fields use empty-string as pad signal, not a flag"

    def test_partial_fields_only_inject_specified_keys(self):
        # `emit_fps` defaults to False, so this exercises the user-set fields
        # in isolation from the fps emission path (covered separately).
        obs = _make_obs()
        add_eval_metadata(obs, cfg=_make_cfg(speed=30, robot_type="UR5"))

        assert "speed" in obs and "speed_is_pad" in obs
        assert "robot_type" in obs
        for absent in (
            "quality",
            "quality_is_pad",
            "mistake",
            "mistake_is_pad",
            "control_mode",
            "fps",
            "fps_is_pad",
        ):
            assert absent not in obs, f"unset field {absent} should not be injected"

    def test_device_propagation(self):
        """Newly-injected tensors must live on the same device as ``state``.

        Using "meta" device makes this CPU-host portable (no real CUDA
        required) while still exercising the device-routing branch in the
        helper. Opts in to ``emit_fps=True`` so the fps device-routing
        branch is also exercised.
        """
        obs = _make_obs(batch_size=2, device="meta")
        add_eval_metadata(obs, cfg=_make_cfg(emit_fps=True, speed=20, mistake=True))

        assert obs["speed"].device.type == "meta"
        assert obs["speed_is_pad"].device.type == "meta"
        assert obs["mistake"].device.type == "meta"
        assert obs["mistake_is_pad"].device.type == "meta"
        assert obs["fps"].device.type == "meta"
        assert obs["fps_is_pad"].device.type == "meta"

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_fps_injects_long_tensor_from_env_fps(self, batch_size):
        """``emit_fps=True`` opt-in → broadcast ``cfg.env.fps`` as a
        ``(B,)`` torch.long tensor with ``fps_is_pad=False`` rows."""
        obs = _make_obs(batch_size=batch_size)
        add_eval_metadata(obs, cfg=_make_cfg(emit_fps=True, fps=20))  # LiberoEnv-style fps

        assert obs["fps"].dtype == torch.long
        assert obs["fps"].shape == (batch_size,)
        assert torch.equal(obs["fps"], torch.full((batch_size,), 20, dtype=torch.long))

        assert obs["fps_is_pad"].dtype == torch.bool
        assert obs["fps_is_pad"].shape == (batch_size,)
        assert not obs["fps_is_pad"].any()

    def test_fps_value_comes_from_env_not_metadata(self):
        """The fps value source is ``cfg.env.fps`` (env stepping freq), NOT a
        field on ``EnvMetadataConfig`` (which only carries the on/off toggle)."""
        obs = _make_obs(batch_size=2)
        add_eval_metadata(obs, cfg=_make_cfg(emit_fps=True, fps=50))

        assert torch.equal(obs["fps"], torch.full((2,), 50, dtype=torch.long))

    def test_fps_disabled_skips_only_fps_keys(self):
        """``emit_fps=False`` (the default) skips the fps injection but leaves
        other set fields alone — they go through the regular None-presence
        gating. The explicit ``emit_fps=False`` documents intent."""
        obs = _make_obs(batch_size=2)
        add_eval_metadata(obs, cfg=_make_cfg(emit_fps=False, speed=30, robot_type="UR5"))

        assert "fps" not in obs
        assert "fps_is_pad" not in obs
        # Existing user-set fields still flow.
        assert "speed" in obs
        assert obs["robot_type"] == ["UR5", "UR5"]


class TestPreprocessObservationCameraPadding:
    """Pin the train↔eval input parity contract for ``preprocess_observation``.

    Training-time ``BaseDataset._standardize_images`` (in
    ``opentau.datasets.lerobot_dataset``) always emits a ``num_cams``-wide
    camera stack with zero-fill for missing slots and a matching
    ``img_is_pad`` mask. Eval-time observations from an env that exposes
    fewer cameras (e.g. LIBERO with 2 cameras while a checkpoint was
    trained on a 4-camera mixture) must be padded the same way, otherwise
    the VLM prefix at eval is strictly shorter than at any training step
    and ``Normalize`` emits a missing-key warning every forward call.
    """

    @staticmethod
    def _make_cfg(num_cams: int, resolution: tuple[int, int] = (224, 224), max_state_dim: int = 8):
        return SimpleNamespace(
            num_cams=num_cams,
            resolution=resolution,
            max_state_dim=max_state_dim,
        )

    @staticmethod
    def _make_pixel_obs(
        *,
        batch_size: int = 2,
        height: int = 224,
        width: int = 224,
        n_present_cams: int = 2,
        state_dim: int = 8,
    ) -> dict:
        rng = np.random.default_rng(0)
        pixels = {
            f"camera{i}": rng.integers(0, 256, size=(batch_size, height, width, 3), dtype=np.uint8)
            for i in range(n_present_cams)
        }
        agent_pos = np.zeros((batch_size, state_dim), dtype=np.float32)
        return {"pixels": pixels, "agent_pos": agent_pos}

    def test_missing_cameras_are_zero_filled(self):
        """``num_cams=4`` with 2 present cameras → ``camera2``/``camera3``
        are added as zero tensors and ``img_is_pad[:, 2:] == True``."""
        cfg = self._make_cfg(num_cams=4)
        obs = self._make_pixel_obs(batch_size=2, n_present_cams=2)
        out = preprocess_observation(obs, cfg)

        for i in range(4):
            assert f"camera{i}" in out, f"camera{i} missing from preprocessed output"
            assert out[f"camera{i}"].shape == (2, 3, 224, 224)

        # Real cameras: derived from random uint8 input, so non-zero.
        assert out["camera0"].float().abs().sum().item() > 0
        assert out["camera1"].float().abs().sum().item() > 0
        # Padded cameras: exactly zero.
        assert out["camera2"].float().abs().sum().item() == 0
        assert out["camera3"].float().abs().sum().item() == 0

        expected_pad = torch.tensor(
            [[False, False, True, True], [False, False, True, True]],
            dtype=torch.bool,
        )
        assert torch.equal(out["img_is_pad"].cpu(), expected_pad)

    def test_all_cameras_present_no_padding(self):
        """``num_cams=2`` with 2 present cameras → no zero-fill, ``img_is_pad`` all False."""
        cfg = self._make_cfg(num_cams=2)
        out = preprocess_observation(self._make_pixel_obs(batch_size=3, n_present_cams=2), cfg)

        assert {"camera0", "camera1"}.issubset(out)
        assert "camera2" not in out
        assert torch.equal(out["img_is_pad"].cpu(), torch.zeros((3, 2), dtype=torch.bool))

    def test_img_is_pad_shape_matches_num_cams(self):
        """``img_is_pad`` is always ``(B, num_cams)``; ``num_cams=4`` with 1
        present camera → padded for slots 1/2/3."""
        cfg = self._make_cfg(num_cams=4)
        out = preprocess_observation(self._make_pixel_obs(batch_size=5, n_present_cams=1), cfg)

        assert out["img_is_pad"].shape == (5, 4)
        expected = torch.tensor([[False, True, True, True]] * 5, dtype=torch.bool)
        assert torch.equal(out["img_is_pad"].cpu(), expected)

    def test_zero_filled_camera_shape_uses_cfg_resolution(self):
        """Zero-filled cameras use ``cfg.resolution``, not the env's raw HxW.
        Mirrors the training contract where ``_standardize_images`` allocates
        ``torch.zeros((3, *self.resolution))``."""
        cfg = self._make_cfg(num_cams=3, resolution=(96, 128))
        out = preprocess_observation(
            self._make_pixel_obs(batch_size=2, height=240, width=320, n_present_cams=2),
            cfg,
        )

        assert out["camera0"].shape == (2, 3, 96, 128)
        assert out["camera2"].shape == (2, 3, 96, 128)
