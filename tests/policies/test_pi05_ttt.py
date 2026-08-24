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

"""CPU tests for the pi05_ttt policy's TTT layer, config and sequence helpers.

Everything here runs without a GPU and without touching the Hugging Face Hub:
constructing a full ``PI05TTTPolicy`` needs the PaliGemma tokenizer and the FAST
action tokenizer, so the end-to-end tests live in ``test_pi05_ttt_gpu.py``.

The layer tests run in float64 on purpose. The TTT update is an explicit
gradient step written in closed form, and its two halves — every token's output
and the end-of-mini-batch fast weights — have to agree algebraically. In float32
an error in that algebra hides under the tolerance you need for the noise; in
float64 the equivalences below hold to ~1e-12, so they actually pin the math.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTPolicy
from opentau.policies.ttt_layer import (
    TanhGate,
    TTTFastWeights,
    TTTMLPLayer,
    TTTSequenceState,
    _layer_norm_forward,
    _rotary_1d,
)

WIDTH = 32
NUM_HEADS = 4
MINI_BATCH = 6


def _layer(**kwargs) -> TTTMLPLayer:
    """Builds a small float64 TTT layer for numerical tests.

    Args:
        **kwargs: Overrides forwarded to :class:`TTTMLPLayer`.

    Returns:
        A layer with float64 parameters.
    """
    torch.manual_seed(0)
    layer = TTTMLPLayer(width=WIDTH, num_heads=NUM_HEADS, mlp_hidden_multiplier=2, **kwargs)
    return layer.to(torch.float64)


class TestTTTMLPLayerMath:
    """Pins the inner update's algebra."""

    def test_output_equals_fast_model_after_the_update(self):
        """Every token's output must be the fast model *after* the mini-batch's step.

        This is the paper's "update then apply", and it is the property the dual
        form buys: rather than looping token by token, the layer computes the
        outputs and the post-update weights with matmuls. If the two halves
        disagree, the layer is not computing the recurrence it claims to.
        """
        layer = _layer()
        x = torch.randn(2, MINI_BATCH, WIDTH, dtype=torch.float64)
        out, fast_weights = layer(x, mini_batch_size=MINI_BATCH)

        # Re-derive the output by hand: project, normalize, rotate, then apply
        # the returned (post-update) fast weights.
        xq = F.normalize(rearrange(layer.wq(x), "b l (h d) -> b l h d", h=NUM_HEADS), p=2, dim=-1)
        xq = _rotary_1d(xq, torch.arange(MINI_BATCH), layer.rope_theta)
        xq = rearrange(xq, "b c h d -> b h c d")
        z1 = xq @ fast_weights.w1 + fast_weights.b1
        z2 = F.gelu(z1, approximate="tanh") @ fast_weights.w2 + fast_weights.b2
        ln_weight = rearrange(layer.ttt_norm_weight, "h d -> h 1 d")
        ln_bias = rearrange(layer.ttt_norm_bias, "h d -> h 1 d")
        expected = xq + _layer_norm_forward(z2, ln_weight, ln_bias)
        expected = layer.wo(layer.post_norm(rearrange(expected, "b h c d -> b c (h d)")))

        torch.testing.assert_close(out, expected, atol=1e-10, rtol=0)

    def test_layer_is_recurrent_across_timesteps(self):
        """A change at timestep 0 must move timestep 1's output.

        Without this the layer could be a per-timestep function and every other
        test here would still pass — the whole point is that it carries state.
        """
        layer = _layer()
        first = torch.randn(1, 2 * MINI_BATCH, WIDTH, dtype=torch.float64)
        second = first.clone()
        second[:, :MINI_BATCH] = torch.randn(1, MINI_BATCH, WIDTH, dtype=torch.float64)

        out_first, _ = layer(first, mini_batch_size=MINI_BATCH)
        out_second, _ = layer(second, mini_batch_size=MINI_BATCH)

        assert not torch.allclose(out_first[:, MINI_BATCH:], out_second[:, MINI_BATCH:], atol=1e-8), (
            "timestep 1's output ignored timestep 0 — the layer is not recurrent"
        )

    def test_rejects_sequence_not_divisible_by_mini_batch(self):
        layer = _layer()
        with pytest.raises(ValueError, match="multiple of mini_batch_size"):
            layer(torch.randn(1, MINI_BATCH + 1, WIDTH, dtype=torch.float64), mini_batch_size=MINI_BATCH)

    def test_rejects_width_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            TTTMLPLayer(width=30, num_heads=4)

    def test_rejects_odd_head_dim(self):
        with pytest.raises(ValueError, match="even for rotary"):
            TTTMLPLayer(width=12, num_heads=4)

    def test_promotes_bfloat16_but_preserves_input_dtype(self):
        """bf16 input is computed in float32 and returned as bf16.

        The inner loop divides by a standard deviation thousands of times; doing
        that in bf16 drifts. Returning float32 instead would silently change the
        dtype of every downstream activation, so the promotion has to be
        internal only.
        """
        layer = TTTMLPLayer(width=WIDTH, num_heads=NUM_HEADS, mlp_hidden_multiplier=2)
        x = torch.randn(1, MINI_BATCH, WIDTH, dtype=torch.bfloat16)
        out, fast_weights = layer.to(torch.bfloat16)(x, mini_batch_size=MINI_BATCH)
        assert out.dtype == torch.bfloat16
        assert fast_weights.w1.dtype == torch.float32


class TestTBPTT:
    """Pins the carry-and-detach behaviour truncated BPTT depends on."""

    def test_segmented_run_matches_single_run(self):
        """Two segments carrying fast weights must equal one undivided run.

        TBPTT is only allowed to change *where gradients stop*, never the
        forward values. If this drifts, the memory a segmented training run
        builds is not the memory inference will reproduce.
        """
        layer = _layer()
        sequence = torch.randn(2, 4 * MINI_BATCH, WIDTH, dtype=torch.float64)

        full_out, full_weights = layer(sequence, mini_batch_size=MINI_BATCH)
        first_out, first_weights = layer(sequence[:, : 2 * MINI_BATCH], mini_batch_size=MINI_BATCH)
        second_out, second_weights = layer(
            sequence[:, 2 * MINI_BATCH :],
            mini_batch_size=MINI_BATCH,
            fast_weights=first_weights,
            position_offset=2 * MINI_BATCH,
        )

        torch.testing.assert_close(torch.cat([first_out, second_out], dim=1), full_out, atol=1e-10, rtol=0)
        torch.testing.assert_close(second_weights.w1, full_weights.w1, atol=1e-10, rtol=0)

    def test_position_offset_is_load_bearing(self):
        """Resuming without the offset must give a different answer.

        Mutation-killing counterpart to the test above: if the offset were
        ignored, that test would pass anyway on a layer whose RoPE silently
        restarted at zero every segment.
        """
        layer = _layer()
        sequence = torch.randn(2, 4 * MINI_BATCH, WIDTH, dtype=torch.float64)
        _, first_weights = layer(sequence[:, : 2 * MINI_BATCH], mini_batch_size=MINI_BATCH)

        with_offset, _ = layer(
            sequence[:, 2 * MINI_BATCH :],
            mini_batch_size=MINI_BATCH,
            fast_weights=first_weights,
            position_offset=2 * MINI_BATCH,
        )
        without_offset, _ = layer(
            sequence[:, 2 * MINI_BATCH :],
            mini_batch_size=MINI_BATCH,
            fast_weights=first_weights,
            position_offset=0,
        )
        assert not torch.allclose(with_offset, without_offset, atol=1e-8)

    def test_w0_receives_gradient_from_the_first_segment(self):
        """``W_0`` must train, or the "learned initialization" is a random draw.

        The paper meta-learns ``W_0`` through the inner update. The first TBPTT
        segment is the only one whose update originates directly from it, so
        that path is the only thing keeping it from being frozen noise.
        """
        layer = _layer()
        sequence = torch.randn(2, 2 * MINI_BATCH, WIDTH, dtype=torch.float64)

        layer.zero_grad()
        out, _ = layer(sequence, mini_batch_size=MINI_BATCH)
        out.sum().backward()

        assert layer.w1_init.grad is not None
        assert layer.w1_init.grad.abs().sum() > 0

    def test_detached_carry_truncates_the_gradient(self):
        """After a detach, a later segment must give ``W_0`` no gradient.

        This is the "truncated" half of TBPTT. Its counterpart above shows the
        gradient path exists at all, so together they pin that the detach is
        what cuts it — not a missing connection.
        """
        layer = _layer()
        sequence = torch.randn(2, 4 * MINI_BATCH, WIDTH, dtype=torch.float64)

        layer.zero_grad()
        _, first_weights = layer(sequence[:, : 2 * MINI_BATCH], mini_batch_size=MINI_BATCH)
        second_out, _ = layer(
            sequence[:, 2 * MINI_BATCH :],
            mini_batch_size=MINI_BATCH,
            fast_weights=first_weights.detach(),
            position_offset=2 * MINI_BATCH,
        )
        second_out.sum().backward()

        assert layer.w1_init.grad is None or layer.w1_init.grad.abs().sum() == 0

    def test_detach_preserves_values(self):
        layer = _layer()
        _, weights = layer(torch.randn(1, MINI_BATCH, WIDTH, dtype=torch.float64), mini_batch_size=MINI_BATCH)
        detached = weights.detach()
        torch.testing.assert_close(detached.w1, weights.w1, atol=0, rtol=0)
        assert not detached.w1.requires_grad

    def test_checkpointed_scan_matches_plain_scan(self):
        """Gradient checkpointing must only trade compute, never change values."""
        plain = _layer()
        checkpointed = _layer(scan_checkpoint_group_size=2)
        checkpointed.load_state_dict(plain.state_dict())

        sequence = torch.randn(2, 4 * MINI_BATCH, WIDTH, dtype=torch.float64)
        plain_out, _ = plain(sequence, mini_batch_size=MINI_BATCH)
        checkpointed_out, _ = checkpointed(sequence, mini_batch_size=MINI_BATCH)
        torch.testing.assert_close(checkpointed_out, plain_out, atol=1e-10, rtol=0)


class TestTanhGate:
    """Pins the property that makes bolting TTT onto a pretrained policy safe."""

    def test_default_init_is_nearly_a_no_op(self):
        gate = TanhGate(WIDTH, init_value=0.001)
        attention = torch.randn(2, 5, WIDTH)
        ttt = torch.randn(2, 5, WIDTH)
        blended = gate(attention, ttt)
        # tanh(0.001) ~= 0.001, so the perturbation is bounded by 0.001 * |ttt|.
        assert (blended - attention).abs().max() <= 0.0011 * ttt.abs().max()

    def test_zero_init_is_exactly_a_no_op(self):
        gate = TanhGate(WIDTH, init_value=0.0)
        attention = torch.randn(2, 5, WIDTH)
        blended = gate(attention, torch.randn(2, 5, WIDTH))
        torch.testing.assert_close(blended, attention, atol=0, rtol=0)

    def test_alpha_is_learnable_and_per_channel(self):
        gate = TanhGate(WIDTH, init_value=0.001)
        assert gate.alpha.requires_grad
        assert gate.alpha.shape == (WIDTH,)

    def test_open_gate_actually_mixes(self):
        """A trained-open gate must pass the TTT branch through.

        Without this, a gate stuck at zero would satisfy every other test in
        this class.
        """
        gate = TanhGate(WIDTH, init_value=5.0)
        attention = torch.zeros(1, 1, WIDTH)
        ttt = torch.ones(1, 1, WIDTH)
        assert gate(attention, ttt).abs().min() > 0.99


class TestPI05TTTConfig:
    """Validation on the fields the policy adds."""

    def test_defaults_match_the_paper(self):
        config = PI05TTTConfig()
        assert config.n_register_tokens == 16
        assert config.ttt_base_lr == pytest.approx(0.1)
        assert config.ttt_rope_theta == pytest.approx(10000.0)
        assert config.ttt_gate_init == pytest.approx(0.001)
        assert config.train_ttt_only is True

    def test_inherits_pi05_defaults(self):
        config = PI05TTTConfig()
        assert config.chunk_size == 50
        assert config.proj_width == 1024

    def test_mini_batch_size_is_registers_plus_chunk(self):
        config = PI05TTTConfig(n_register_tokens=16, chunk_size=50)
        assert config.n_expert_tokens_per_timestep == 66

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_register_tokens", -1),
            ("ttt_num_heads", 0),
            ("ttt_mlp_hidden_multiplier", 0),
            ("ttt_base_lr", 0.0),
            ("ttt_rope_theta", 0.0),
            ("ttt_scan_checkpoint_group_size", -1),
            ("sequence_length", 0),
            ("tbptt_segment_length", 0),
        ],
    )
    def test_rejects_out_of_range_fields(self, field, value):
        with pytest.raises(ValueError, match=field):
            PI05TTTConfig(**{field: value})

    def test_rejects_segment_length_that_does_not_divide_sequence_length(self):
        with pytest.raises(ValueError, match="must divide"):
            PI05TTTConfig(sequence_length=10, tbptt_segment_length=4)

    def test_registered_under_its_policy_type(self):
        from opentau.policies.factory import get_policy_class, make_policy_config

        assert isinstance(make_policy_config("pi05_ttt"), PI05TTTConfig)
        assert get_policy_class("pi05_ttt") is PI05TTTPolicy
        assert PI05TTTPolicy.name == "pi05_ttt"


class TestSequenceBatchHandling:
    """Pins the ``(B, T)`` folding the sequence path depends on."""

    def test_sequence_length_detects_a_trajectory_batch(self):
        assert PI05TTTPolicy._sequence_length({"actions": torch.zeros(2, 5, 50, 32)}) == 5

    def test_sequence_length_is_none_for_a_flat_batch(self):
        assert PI05TTTPolicy._sequence_length({"actions": torch.zeros(2, 50, 32)}) is None
        assert PI05TTTPolicy._sequence_length({}) is None

    def test_per_timestep_tensors_are_flattened_batch_major(self):
        """Row order must be ``(batch, timestep)``.

        The TTT hook unfolds this axis with the inverse ``rearrange``. If the
        order were timestep-major instead, the memory would interleave
        trajectories and still produce plausible-looking losses.
        """
        batch = {"actions": rearrange(torch.arange(2 * 3 * 4).float(), "(b t x) -> b t x", b=2, t=3)}
        flat = PI05TTTPolicy._flatten_sequence_batch(batch, batch_size=2, num_timesteps=3)
        assert flat["actions"].shape == (6, 4)
        # Row 1 must be trajectory 0's timestep 1, not trajectory 1's timestep 0.
        torch.testing.assert_close(flat["actions"][1], batch["actions"][0, 1])
        torch.testing.assert_close(flat["actions"][3], batch["actions"][1, 0])

    def test_trajectory_level_tensors_are_repeated(self):
        batch = {
            "actions": torch.zeros(2, 3, 50, 32),
            "dataset_index": torch.tensor([7, 9]),
        }
        flat = PI05TTTPolicy._flatten_sequence_batch(batch, batch_size=2, num_timesteps=3)
        assert flat["dataset_index"].tolist() == [7, 7, 7, 9, 9, 9]

    def test_trajectory_level_lists_are_repeated(self):
        batch = {"actions": torch.zeros(2, 3, 50, 32), "task": ["fold towel", "stack cups"]}
        flat = PI05TTTPolicy._flatten_sequence_batch(batch, batch_size=2, num_timesteps=3)
        assert flat["task"] == ["fold towel"] * 3 + ["stack cups"] * 3

    def test_loss_mask_keeps_its_sequence_shape(self):
        batch = {"actions": torch.zeros(2, 3, 50, 32), "loss_mask": torch.ones(2, 3, dtype=torch.bool)}
        flat = PI05TTTPolicy._flatten_sequence_batch(batch, batch_size=2, num_timesteps=3)
        assert flat["loss_mask"].shape == (2, 3)


class TestSegmentRows:
    """Pins the index arithmetic that slices a timestep window out of a flat batch."""

    def test_selects_the_expected_rows(self):
        from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching

        rows = PI05TTTFlowMatching._segment_rows(
            batch_size=2, num_timesteps=4, start=2, stop=4, device=torch.device("cpu")
        )
        # trajectory 0 timesteps 2,3 -> rows 2,3; trajectory 1 -> rows 6,7
        assert rows.tolist() == [2, 3, 6, 7]

    def test_covers_every_row_exactly_once_across_segments(self):
        from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching

        seen = []
        for start in range(0, 4, 2):
            seen.extend(
                PI05TTTFlowMatching._segment_rows(
                    batch_size=3,
                    num_timesteps=4,
                    start=start,
                    stop=start + 2,
                    device=torch.device("cpu"),
                ).tolist()
            )
        assert sorted(seen) == list(range(12))


class TestTTTSequenceState:
    def test_defaults_are_empty_and_independent(self):
        first = TTTSequenceState(num_timesteps=2)
        second = TTTSequenceState(num_timesteps=2)
        first.outgoing[0] = TTTFastWeights(
            w1=torch.zeros(1), b1=torch.zeros(1), w2=torch.zeros(1), b2=torch.zeros(1)
        )
        assert second.outgoing == {}, "default_factory leaked a shared dict between instances"

    def test_missing_incoming_key_means_start_from_w0(self):
        layer = _layer()
        state = TTTSequenceState(num_timesteps=1)
        x = torch.randn(1, MINI_BATCH, WIDTH, dtype=torch.float64)
        from_w0, _ = layer(x, mini_batch_size=MINI_BATCH, fast_weights=None)
        from_missing, _ = layer(x, mini_batch_size=MINI_BATCH, fast_weights=state.incoming.get(0))
        torch.testing.assert_close(from_missing, from_w0, atol=0, rtol=0)
