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

"""Invariants specific to this fork's pi05_ttt port.

Two things this repo needs that upstream does not:

1. ``ttt_enabled`` -- the master off switch, so one config file can A/B TTT
   against stock pi05 and a run can be reverted without editing the codebase.
2. The local teacher-faithful input transforms (``state_keep_dims``,
   ``use_delta_joint_actions``) must reach the TTT sequence path. They used to
   live inline in ``PI05Policy.forward``, which ``PI05TTTPolicy`` overrides
   completely -- so a TTT run would have trained on absolute actions and a full
   state while its checkpoint and its own rollouts used deltas and a sliced
   state. Silent, and worth a test that fails loudly instead.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching, PI05TTTPolicy


class TestTTTEnabledSwitch:
    """``ttt_enabled=False`` must reduce this policy to stock pi05."""

    def test_default_is_enabled(self):
        assert PI05TTTConfig().ttt_enabled is True

    def test_disabled_refuses_train_ttt_only(self):
        with pytest.raises(ValueError, match="nothing trainable"):
            PI05TTTConfig(ttt_enabled=False, train_ttt_only=True)

    def test_disabled_coerces_the_inert_sequence_knobs(self):
        """One boolean must be enough; these two are meaningless without memory."""
        config = PI05TTTConfig(
            ttt_enabled=False, train_ttt_only=False, sequence_length=4, tbptt_segment_length=4
        )
        assert config.sequence_length == 1
        assert config.tbptt_segment_length == 1

    def test_disabled_config_is_constructible(self):
        config = PI05TTTConfig(ttt_enabled=False, train_ttt_only=False)
        assert config.ttt_enabled is False
        assert config.sequence_length == 1

    def test_attach_ttt_layers_is_a_noop_when_disabled(self):
        """No ``ttt`` submodule anywhere, so the hook in the expert cannot fire."""
        stub = SimpleNamespace(config=PI05TTTConfig(ttt_enabled=False, train_ttt_only=False))
        # Unbound call: the disabled branch returns before touching any module.
        assert PI05TTTFlowMatching._attach_ttt_layers(stub) is None
        assert not hasattr(stub, "paligemma_with_expert")

    def test_ttt_parameters_is_empty_when_disabled(self):
        stub = SimpleNamespace(config=PI05TTTConfig(ttt_enabled=False, train_ttt_only=False))
        assert PI05TTTFlowMatching.ttt_parameters(stub) == []

    def test_forward_delegates_to_pi05_when_disabled(self):
        """The disabled branch must call ``PI05Policy.forward``, not the sequence path."""
        source = inspect.getsource(PI05TTTPolicy.forward)
        tree = ast.parse(source.lstrip())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "PI05Policy"
        ]
        assert calls, "ttt_enabled=False must delegate to PI05Policy.forward"

    def test_rollout_state_contract_is_dropped_when_disabled(self):
        """Class default is True; the disabled __init__ shadows it per instance."""
        assert PI05TTTPolicy.carries_rollout_state is True
        source = inspect.getsource(PI05TTTPolicy.__init__)
        assert "self.carries_rollout_state = False" in source


class TestTeacherFaithfulTransformsReachTheSequencePath:
    """The local ``state_keep_dims`` / ``use_delta_joint_actions`` options."""

    @staticmethod
    def _batch():
        state = torch.arange(2 * 13, dtype=torch.float32).reshape(2, 13)
        actions = torch.ones(2, 3, 13)
        return {"state": state, "actions": actions}

    def test_state_keep_dims_zeroes_the_tail(self):
        policy = SimpleNamespace(
            config=SimpleNamespace(state_keep_dims=8, use_delta_joint_actions=False)
        )
        out = PI05Policy._apply_teacher_faithful_transforms(policy, self._batch())
        assert torch.all(out["state"][:, 8:] == 0)
        assert torch.all(out["state"][:, :8] == self._batch()["state"][:, :8])

    def test_delta_joint_actions_subtracts_chunk_start_state(self):
        policy = SimpleNamespace(
            config=SimpleNamespace(state_keep_dims=None, use_delta_joint_actions=True)
        )
        batch = self._batch()
        out = PI05Policy._apply_teacher_faithful_transforms(policy, batch)
        expected = batch["actions"][..., :7] - batch["state"][:, None, :7]
        assert torch.allclose(out["actions"][..., :7], expected)
        # gripper and padding dims stay absolute
        assert torch.allclose(out["actions"][..., 7:], batch["actions"][..., 7:])

    def test_no_options_returns_the_batch_untouched(self):
        policy = SimpleNamespace(
            config=SimpleNamespace(state_keep_dims=None, use_delta_joint_actions=False)
        )
        batch = self._batch()
        assert PI05Policy._apply_teacher_faithful_transforms(policy, batch) is batch

    @pytest.mark.parametrize(
        "func",
        [PI05Policy.forward, PI05Policy.sample_actions, PI05TTTPolicy.forward],
        ids=["pi05.forward", "pi05.sample_actions", "pi05_ttt.forward"],
    )
    def test_every_training_and_inference_path_calls_the_helper(self, func):
        """A path that skips this silently trains or serves a different objective."""
        source = inspect.getsource(func)
        assert "_apply_teacher_faithful_transforms(" in source, (
            f"{func.__qualname__} does not apply the teacher-faithful transforms"
        )

    def test_helper_runs_before_normalization(self):
        """Order matters: the transforms are defined on raw, unnormalized units."""
        source = inspect.getsource(PI05TTTPolicy.forward)
        assert source.index("_apply_teacher_faithful_transforms(") < source.index(
            "self.normalize_inputs("
        )
