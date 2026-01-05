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
"""Type definitions for configuration classes.

This module provides type definitions used across configuration classes, including
enumerations for feature types and normalization modes, as well as protocol
definitions and dataclasses for policy features.
"""

# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class FeatureType(str, Enum):
    """Enumeration of feature types used in policy configurations."""

    STATE = "STATE"
    """Robot state features."""
    VISUAL = "VISUAL"
    """Visual/image features."""
    ENV = "ENV"
    """Environment state features."""
    ACTION = "ACTION"
    """Action features."""


class NormalizationMode(str, Enum):
    """Enumeration of normalization modes for features."""

    MIN_MAX = "MIN_MAX"
    """Normalize using min-max scaling."""
    MEAN_STD = "MEAN_STD"
    """Normalize using mean and standard deviation."""
    IDENTITY = "IDENTITY"
    """No normalization (identity transformation)."""


class DictLike(Protocol):
    """Protocol for dictionary-like objects that support item access.

    This protocol defines the interface for objects that can be accessed
    using dictionary-style indexing with the `[]` operator.
    """

    def __getitem__(self, key: Any) -> Any: ...


@dataclass
class PolicyFeature:
    """Configuration for a policy feature.

    This class describes a single feature used by a policy, including its
    type and shape information.

    Args:
        type: The type of feature (STATE, VISUAL, ENV, or ACTION).
        shape: The shape of the feature as a tuple.
    """

    type: FeatureType
    shape: tuple
