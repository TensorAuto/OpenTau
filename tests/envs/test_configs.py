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
from opentau.envs.configs import EnvConfig, EnvMetadataConfig


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


class TestEnvMetadataConfig:
    """Pin the validation contract for the optional pi07 metadata fields
    attached to ``EnvConfig.metadata``. The defaults (all ``None``) must
    stay frozen because eval rollouts that don't set them rely on the
    policy's ``prepare_metadata`` pad path for backwards compatibility.
    """

    def test_all_none_default_passes(self):
        cfg = EnvMetadataConfig()
        assert cfg.speed is None
        assert cfg.quality is None
        assert cfg.mistake is None
        assert cfg.robot_type is None
        assert cfg.control_mode is None

    @pytest.mark.parametrize("mode", ["joint", "ee"])
    @pytest.mark.parametrize("speed", [0, 10, 50, 90, 100])
    def test_valid_values_pass(self, mode, speed):
        EnvMetadataConfig(
            speed=speed,
            quality=3,
            mistake=False,
            robot_type="UR5",
            control_mode=mode,
        )

    @pytest.mark.parametrize("bad_speed", [-10, -1, 1, 5, 9, 11, 15, 105, 110, 1000, 1001])
    def test_speed_must_be_in_zero_to_100_step_10(self, bad_speed):
        with pytest.raises(
            ValueError,
            match=r"speed must be a non-negative multiple of 10 in \[0, 100\]",
        ):
            EnvMetadataConfig(speed=bad_speed)

    def test_speed_rejects_bool(self):
        with pytest.raises(TypeError, match=r"speed must be int"):
            EnvMetadataConfig(speed=True)

    def test_speed_rejects_float(self):
        with pytest.raises(TypeError, match=r"speed must be int"):
            EnvMetadataConfig(speed=50.0)

    @pytest.mark.parametrize("bad_quality", [0, 6, -1, 100])
    def test_quality_out_of_range(self, bad_quality):
        with pytest.raises(ValueError, match=r"quality must be in \[1, 5\]"):
            EnvMetadataConfig(quality=bad_quality)

    def test_quality_rejects_bool(self):
        with pytest.raises(TypeError, match=r"quality must be int"):
            EnvMetadataConfig(quality=True)

    def test_mistake_rejects_non_bool(self):
        with pytest.raises(TypeError, match=r"mistake must be bool"):
            EnvMetadataConfig(mistake=1)

    def test_robot_type_rejects_empty(self):
        with pytest.raises(ValueError, match=r"robot_type must be a non-empty string"):
            EnvMetadataConfig(robot_type="")

    def test_robot_type_rejects_non_str(self):
        with pytest.raises(TypeError, match=r"robot_type must be str"):
            EnvMetadataConfig(robot_type=42)

    @pytest.mark.parametrize("bad_mode", ["joint_position", "EE", "Joint", "cartesian", ""])
    def test_control_mode_must_be_joint_or_ee(self, bad_mode):
        with pytest.raises(ValueError, match=r"control_mode must be one of"):
            EnvMetadataConfig(control_mode=bad_mode)


class TestEnvConfigMetadataField:
    """Verify the ``metadata`` field is wired onto every concrete
    ``EnvConfig`` subclass via the abstract base, with the all-``None``
    default that preserves prior eval behavior.
    """

    def test_concrete_subclass_has_metadata_default(self):
        @EnvConfig.register_subclass("test_env_metadata_default")
        @dataclass
        class TestEnvConfigWithMetadata(EnvConfig):
            @property
            def gym_kwargs(self) -> dict:
                return {}

        config = TestEnvConfigWithMetadata()
        assert isinstance(config.metadata, EnvMetadataConfig)
        assert config.metadata.speed is None
        assert config.metadata.control_mode is None
