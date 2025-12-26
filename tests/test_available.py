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


import lerobot
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.tau0.modeling_tau0 import TAU0Policy
from lerobot.common.policies.value.modeling_value import ValueFunction


def test_available_policies():
    """
    This test verifies that the class attribute `name` for all policies is
    consistent with those listed in `lerobot/__init__.py`.
    """
    policy_classes = [
        TAU0Policy,
        PI0Policy,
        ValueFunction,
    ]
    policies = [pol_cls.name for pol_cls in policy_classes]
    assert set(policies) == set(lerobot.available_policies), policies


def test_print():
    print(lerobot.available_envs)
    print(lerobot.available_tasks_per_env)
    print(lerobot.available_datasets)
    print(lerobot.available_datasets_per_env)
    print(lerobot.available_real_world_datasets)
    print(lerobot.available_policies)
    print(lerobot.available_policies_per_env)
