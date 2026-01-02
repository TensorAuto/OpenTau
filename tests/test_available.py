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


import opentau
from opentau.policies.pi0.modeling_pi0 import PI0Policy
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.value.modeling_value import ValueFunction


def test_available_policies():
    """
    This test verifies that the class attribute `name` for all policies is
    consistent with those listed in `src/opentau/__init__.py`.
    """
    policy_classes = [
        PI0Policy,
        ValueFunction,
        PI05Policy,
    ]
    policies = [pol_cls.name for pol_cls in policy_classes]
    assert set(policies) == set(opentau.available_policies), policies


def test_print():
    print(opentau.available_envs)
    print(opentau.available_tasks_per_env)
    print(opentau.available_datasets)
    print(opentau.available_datasets_per_env)
    print(opentau.available_real_world_datasets)
    print(opentau.available_policies)
    print(opentau.available_policies_per_env)
