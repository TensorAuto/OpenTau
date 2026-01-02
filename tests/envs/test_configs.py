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

from dataclasses import dataclass

import pytest

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.envs.configs import EnvConfig


class TestEnvConfig:
    """Test cases for EnvConfig base class"""

    def test_env_config_abstract_methods(self):
        """Test that EnvConfig is abstract and cannot be instantiated directly"""
        with pytest.raises(TypeError):
            EnvConfig()

    def test_env_config_registry(self):
        """Test that EnvConfig is a choice registry"""
        # Check that it has the registry functionality
        assert hasattr(EnvConfig, "get_choice_name")
        assert hasattr(EnvConfig, "register_subclass")

    def test_env_config_type_property(self):
        """Test that type property works correctly"""

        # Create a concrete subclass for testing with unique name
        @EnvConfig.register_subclass("test_env_type")
        @dataclass
        class TestEnvConfigType(EnvConfig):
            import_name: str = "test.module"
            make_id: str = "TestEnv"

            @property
            def gym_kwargs(self) -> dict:
                return {"test": "value"}

        config = TestEnvConfigType()
        assert config.type == "test_env_type"

    def test_env_config_default_values(self):
        """Test default values for EnvConfig fields"""

        @EnvConfig.register_subclass("test_env_defaults")
        @dataclass
        class TestEnvConfigDefaults(EnvConfig):
            import_name: str = "test.module"
            make_id: str = "TestEnv"

            @property
            def gym_kwargs(self) -> dict:
                return {"test": "value"}

        config = TestEnvConfigDefaults()

        # Test default values
        assert config.task is None
        assert config.fps == 30
        assert config.features == {}
        assert config.features_map == {}

    def test_env_config_custom_values(self):
        """Test custom values for EnvConfig fields"""

        @EnvConfig.register_subclass("test_env_custom")
        @dataclass
        class TestEnvConfigCustom(EnvConfig):
            import_name: str = "test.module"
            make_id: str = "TestEnv"

            @property
            def gym_kwargs(self) -> dict:
                return {"test": "value"}

        features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        }
        features_map = {"camera0": "image", "state": "state"}

        config = TestEnvConfigCustom(task="test_task", fps=60, features=features, features_map=features_map)

        assert config.task == "test_task"
        assert config.fps == 60
        assert config.features == features
        assert config.features_map == features_map


class TestEnvConfigRegistry:
    """Test cases for EnvConfig choice registry functionality"""

    def test_register_subclass_decorator(self):
        """Test that @EnvConfig.register_subclass decorator works"""

        # Create a new subclass with unique name
        @EnvConfig.register_subclass("test_env_registry")
        @dataclass
        class TestEnvConfigRegistry(EnvConfig):
            import_name: str = "test.module"
            make_id: str = "TestEnv"

            @property
            def gym_kwargs(self) -> dict:
                return {"test": "value"}

        # Check that it's registered
        assert "test_env_registry" in EnvConfig._choice_registry
        assert EnvConfig._choice_registry["test_env_registry"] is TestEnvConfigRegistry

        # Check that it can be instantiated
        config = TestEnvConfigRegistry()
        assert config.type == "test_env_registry"

    def test_multiple_subclass_registration(self):
        """Test registering multiple subclasses"""

        # Register first subclass with unique name
        @EnvConfig.register_subclass("test_env1")
        @dataclass
        class TestEnv1Config(EnvConfig):
            import_name: str = "env1.module"
            make_id: str = "Env1"

            @property
            def gym_kwargs(self) -> dict:
                return {"env1": "value"}

        # Register second subclass with unique name
        @EnvConfig.register_subclass("test_env2")
        @dataclass
        class TestEnv2Config(EnvConfig):
            import_name: str = "env2.module"
            make_id: str = "Env2"

            @property
            def gym_kwargs(self) -> dict:
                return {"env2": "value"}

        # Check both are registered
        assert "test_env1" in EnvConfig._choice_registry
        assert "test_env2" in EnvConfig._choice_registry

        # Check they can be instantiated
        config1 = TestEnv1Config()
        config2 = TestEnv2Config()

        assert config1.type == "test_env1"
        assert config2.type == "test_env2"

    def test_get_all_choices(self):
        """Test get_known_choices method"""
        choices = EnvConfig.get_known_choices()

        assert isinstance(choices, dict)

        # Should be able to get all registered choices
        for choice in choices:
            assert choice in EnvConfig._choice_registry
