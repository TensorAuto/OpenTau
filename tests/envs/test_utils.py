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

import gymnasium as gym
import pytest

from opentau.envs.utils import are_all_envs_same_type, check_env_attributes_and_types


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
