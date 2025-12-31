#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from src.opentau.envs.configs import EnvConfig, MetaworldEnv
from src.opentau.configs.types import FeatureType, PolicyFeature


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


class TestMetaworldEnv:
    """Test cases for MetaworldEnv class"""

    def test_metaworld_env_default_values(self):
        """Test default values for MetaworldEnv"""
        config = MetaworldEnv()

        assert config.import_name == "src.opentau.envs.metaworld"
        assert config.make_id == "Metaworld"
        assert config.task == "Meta-World/MT1"
        assert config.env_name == "button-press-v3"
        assert config.fps == 10
        assert config.render_mode == "rgb_array"
        assert config.episode_length == 200

    def test_metaworld_env_custom_values(self):
        """Test custom values for MetaworldEnv"""
        config = MetaworldEnv(env_name="peg-insert-v3", fps=20, render_mode="human", episode_length=100)

        assert config.env_name == "peg-insert-v3"
        assert config.fps == 20
        assert config.render_mode == "human"
        assert config.episode_length == 100

        # Check that defaults are still set
        assert config.import_name == "src.opentau.envs.metaworld"
        assert config.make_id == "Metaworld"
        assert config.task == "Meta-World/MT1"

    def test_metaworld_env_type(self):
        """Test that MetaworldEnv has correct type"""
        config = MetaworldEnv()
        assert config.type == "metaworld"

    def test_metaworld_env_gym_kwargs(self):
        """Test gym_kwargs property"""
        config = MetaworldEnv(env_name="button-press-v3", render_mode="rgb_array", episode_length=50)

        gym_kwargs = config.gym_kwargs

        expected_kwargs = {
            "task": "Meta-World/MT1",
            "env_name": "button-press-v3",
            "render_mode": "rgb_array",
            "max_episode_steps": 50,
        }

        assert gym_kwargs == expected_kwargs

    def test_metaworld_env_gym_kwargs_custom_values(self):
        """Test gym_kwargs property with custom values"""
        config = MetaworldEnv(env_name="peg-insert-v3", render_mode="human", episode_length=200)

        gym_kwargs = config.gym_kwargs

        expected_kwargs = {
            "task": "Meta-World/MT1",
            "env_name": "peg-insert-v3",
            "render_mode": "human",
            "max_episode_steps": 200,
        }

        assert gym_kwargs == expected_kwargs

    def test_metaworld_env_inheritance(self):
        """Test that MetaworldEnv inherits from EnvConfig"""
        config = MetaworldEnv()

        # Check inheritance
        assert isinstance(config, EnvConfig)

        # Check that it has all the base class attributes
        assert hasattr(config, "import_name")
        assert hasattr(config, "make_id")
        assert hasattr(config, "task")
        assert hasattr(config, "fps")
        assert hasattr(config, "features")
        assert hasattr(config, "features_map")
        assert hasattr(config, "type")
        assert hasattr(config, "gym_kwargs")

    def test_metaworld_env_registration(self):
        """Test that MetaworldEnv is properly registered"""
        # Check that it's registered in the choice registry
        assert "metaworld" in EnvConfig._choice_registry

        # Check that we can get the class from the registry
        registered_class = EnvConfig._choice_registry["metaworld"]
        assert registered_class is MetaworldEnv

    def test_metaworld_env_all_parameters(self):
        """Test MetaworldEnv with all possible parameters"""
        features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        }
        features_map = {"camera0": "image", "state": "state"}

        config = MetaworldEnv(
            import_name="custom.metaworld",
            make_id="CustomMetaworld",
            task="Meta-World/MT10",
            env_name="door-open-v3",
            fps=15,
            render_mode="rgb_array",
            episode_length=300,
            features=features,
            features_map=features_map,
        )

        assert config.import_name == "custom.metaworld"
        assert config.make_id == "CustomMetaworld"
        assert config.task == "Meta-World/MT10"
        assert config.env_name == "door-open-v3"
        assert config.fps == 15
        assert config.render_mode == "rgb_array"
        assert config.episode_length == 300
        assert config.features == features
        assert config.features_map == features_map

    def test_metaworld_env_gym_kwargs_immutability(self):
        """Test that gym_kwargs returns a new dict each time"""
        config = MetaworldEnv()

        kwargs1 = config.gym_kwargs
        kwargs2 = config.gym_kwargs

        # Should be equal but not the same object
        assert kwargs1 == kwargs2
        assert kwargs1 is not kwargs2

        # Modifying one shouldn't affect the other
        kwargs1["test"] = "value"
        assert "test" not in kwargs2

    def test_metaworld_env_gym_kwargs_maps_episode_length(self):
        """Test that episode_length is mapped to max_episode_steps in gym_kwargs"""
        config = MetaworldEnv(episode_length=150)

        gym_kwargs = config.gym_kwargs

        assert "max_episode_steps" in gym_kwargs
        assert gym_kwargs["max_episode_steps"] == 150
        assert "episode_length" not in gym_kwargs

    def test_metaworld_env_gym_kwargs_includes_all_relevant_params(self):
        """Test that gym_kwargs includes all relevant parameters"""
        config = MetaworldEnv(env_name="test-env-v3", render_mode="test_mode", episode_length=999)

        gym_kwargs = config.gym_kwargs

        # Should include all four parameters
        assert len(gym_kwargs) == 4
        assert gym_kwargs["task"] == "Meta-World/MT1"
        assert gym_kwargs["env_name"] == "test-env-v3"
        assert gym_kwargs["render_mode"] == "test_mode"
        assert gym_kwargs["max_episode_steps"] == 999

    def test_metaworld_env_dataclass_functionality(self):
        """Test that MetaworldEnv works as a dataclass"""
        config1 = MetaworldEnv(env_name="env1")
        config2 = MetaworldEnv(env_name="env2")
        config3 = MetaworldEnv(env_name="env1")

        # Test equality
        assert config1 != config2
        assert config1 == config3

        # Test that it can be pickled (basic dataclass functionality)
        import pickle

        pickled = pickle.dumps(config1)
        unpickled = pickle.loads(pickled)
        assert unpickled == config1

    def test_metaworld_env_choice_registry_integration(self):
        """Test that MetaworldEnv integrates properly with choice registry"""
        # Test that we can get all registered choices
        choices = EnvConfig.get_known_choices()
        assert "metaworld" in choices

        # Test that we can get the choice name
        config = MetaworldEnv()
        assert config.get_choice_name(config.__class__) == "metaworld"

        # Test that the type property uses the choice name
        assert config.type == "metaworld"


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
        assert "metaworld" in EnvConfig._choice_registry  # Original should still be there

        # Check they can be instantiated
        config1 = TestEnv1Config()
        config2 = TestEnv2Config()

        assert config1.type == "test_env1"
        assert config2.type == "test_env2"

    def test_get_all_choices(self):
        """Test get_known_choices method"""
        choices = EnvConfig.get_known_choices()

        # Should include metaworld at minimum
        assert "metaworld" in choices
        assert isinstance(choices, dict)

        # Should be able to get all registered choices
        for choice in choices:
            assert choice in EnvConfig._choice_registry

    def test_get_choice_name(self):
        """Test get_choice_name method"""
        config = MetaworldEnv()
        choice_name = config.get_choice_name(config.__class__)

        assert choice_name == "metaworld"
        assert isinstance(choice_name, str)
