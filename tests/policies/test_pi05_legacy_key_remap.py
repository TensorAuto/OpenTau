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

"""Legacy `normalize_actions.*` keys must remap to `normalize_discrete_actions.*`.

Checkpoints saved before the discrete-action normalizer was renamed (e.g.
``TensorAuto/tPi0.5-libero``, saved 2026-06) carry ``normalize_actions.buffer_actions.{min,max}``.
Current pi05 has no module of that name, so without a remap those stats load as
*unexpected keys*, the model's ``normalize_discrete_actions`` buffers stay at
their ``+inf`` placeholder on the stat-less eval path, and ``make_policy``'s
``_check_norm_stats_loaded`` rejects the checkpoint. The remap lives in
``PI05Policy._fix_pytorch_state_dict_keys``; these tests call that method
directly (with a stub ``self``, which the touched branches never read) so they
run without building a 3.4B-parameter model.
"""

from types import SimpleNamespace

import pytest
import torch

from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.pi05_mem.modeling_pi05 import PI05MemPolicy

# Both copies of the hand-duplicated method must carry the remap: pi05's (which
# pi05_ttt inherits) and pi05_mem's. Sweeping only one is exactly the
# miss-by-omission CLAUDE.md rule 6 documents.
POLICY_CLASSES = [PI05Policy, PI05MemPolicy]


def _run_fix(policy_cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invokes the real (unbound) key-fixing method with a minimal stub self/config."""
    # `self` is only consulted by the adaRMS branches (via `self.model...`); give it
    # a shape those branches can read without triggering the skip.
    stub_self = SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                gemma_expert=SimpleNamespace(config=SimpleNamespace(use_adarms=False))
            )
        )
    )
    stub_config = SimpleNamespace(state_type="continuous")
    return policy_cls._fix_pytorch_state_dict_keys(stub_self, state_dict, stub_config)


@pytest.mark.parametrize("policy_cls", POLICY_CLASSES)
def test_legacy_discrete_normalizer_keys_are_remapped(policy_cls):
    fixed = _run_fix(
        policy_cls,
        {
            "normalize_actions.buffer_actions.min": torch.zeros(32),
            "normalize_actions.buffer_actions.max": torch.ones(32),
        },
    )
    assert set(fixed) == {
        "normalize_discrete_actions.buffer_actions.min",
        "normalize_discrete_actions.buffer_actions.max",
    }


@pytest.mark.parametrize("policy_cls", POLICY_CLASSES)
def test_current_discrete_normalizer_keys_pass_through_unchanged(policy_cls):
    keys = {
        "normalize_discrete_actions.buffer_actions.min": torch.zeros(32),
        "normalize_discrete_actions.buffer_actions.max": torch.ones(32),
        "normalize_inputs.buffer_state.min": torch.zeros(32),
        "normalize_targets.buffer_actions.mean": torch.zeros(32),
        "unnormalize_outputs.buffer_actions.std": torch.ones(32),
    }
    fixed = _run_fix(policy_cls, dict(keys))
    assert set(fixed) == set(keys)


@pytest.mark.parametrize("policy_cls", POLICY_CLASSES)
def test_remap_hits_only_the_module_prefix(policy_cls):
    # A key merely *containing* the substring elsewhere must not be rewritten:
    # startswith-anchored, first occurrence only.
    weird = "model.some_module.normalize_actions.weight"
    fixed = _run_fix(policy_cls, {weird: torch.zeros(1)})
    assert set(fixed) == {weird}
