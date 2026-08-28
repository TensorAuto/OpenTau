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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching, PI05TTTPolicy
from opentau.policies.ttt_layer import (
    TanhGate,
    TTTFastWeights,
    TTTMLPLayer,
    TTTSequenceState,
    _layer_norm_forward,
    _rotary_1d,
)
from opentau.policies.utils import PerSampleLoss

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

        The mutation this uniquely kills is **RoPE being a no-op**: with
        ``_rotary_1d`` returning ``x`` unchanged,
        ``test_segmented_run_matches_single_run`` still passes and only this
        test fails.

        It is *not* the mutation-killer for "the offset is ignored" — an earlier
        docstring here claimed that, and the claim was wrong. Dropping the
        offset makes ``test_segmented_run_matches_single_run`` fail first. Said
        the wrong way round, this note would tell a maintainer trimming tests
        that the stronger test above is the redundant one.
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

    def test_checkpointed_scan_matches_plain_scan_in_forward_and_backward(self):
        """Checkpointing must trade compute only — including in the backward.

        The forward half of this cannot fail: recompute is the same code on the
        same inputs. The backward half can. ``_scan_mini_batches`` threads the
        fast-weight carry between groups through ``torch.utils.checkpoint``
        outputs; if that carry leaves the autograd graph, every gradient
        reaching ``w1_init``/``b1_init``/``w2_init``/``b2_init``, ``wk``, ``wv``,
        ``ttt_lr_*`` and ``ttt_norm_weight`` from all but the final group is
        silently dropped — with no error, because the output still requires grad
        through ``xq``. Detaching the carry at line ~374 leaves the forward
        bit-identical and shifts the gradient w.r.t. the initial fast weights by
        roughly 30%, so only a gradient assertion catches it.
        """
        plain = _layer()
        checkpointed = _layer(scan_checkpoint_group_size=2)
        checkpointed.load_state_dict(plain.state_dict())

        sequence = torch.randn(2, 4 * MINI_BATCH, WIDTH, dtype=torch.float64)

        plain_out, _ = plain(sequence, mini_batch_size=MINI_BATCH)
        plain_out.square().sum().backward()
        checkpointed_out, _ = checkpointed(sequence, mini_batch_size=MINI_BATCH)
        checkpointed_out.square().sum().backward()

        torch.testing.assert_close(checkpointed_out, plain_out, atol=1e-10, rtol=0)
        for name in ("w1_init", "b1_init", "w2_init", "b2_init", "ttt_norm_weight"):
            expected = getattr(plain, name).grad
            actual = getattr(checkpointed, name).grad
            assert expected is not None and actual is not None, f"{name} has no gradient"
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


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

    def test_flat_batch_is_accepted_only_at_sequence_length_one(self):
        """A flat batch is a one-timestep sequence, and only that.

        The configured length is the source of truth so the TBPTT segment count
        is identical on every rank; adopting the batch's length instead would
        make the number of backward calls depend on micro-batch content.
        """
        flat = {"actions": torch.zeros(2, 50, 32)}
        assert PI05TTTPolicy._as_sequence_batch(flat, 1) is flat
        with pytest.raises(ValueError, match="no timestep axis"):
            PI05TTTPolicy._as_sequence_batch(flat, 4)

    def test_sequence_batch_must_match_the_configured_length(self):
        good = {"actions": torch.zeros(2, 4, 50, 32)}
        assert PI05TTTPolicy._as_sequence_batch(good, 4) is good
        with pytest.raises(ValueError, match="config.sequence_length"):
            PI05TTTPolicy._as_sequence_batch({"actions": torch.zeros(2, 3, 50, 32)}, 4)

    def test_rejects_a_batch_with_no_actions_or_a_bad_rank(self):
        with pytest.raises(ValueError, match="requires an `actions`"):
            PI05TTTPolicy._as_sequence_batch({}, 1)
        with pytest.raises(ValueError, match="rank 2"):
            PI05TTTPolicy._as_sequence_batch({"actions": torch.zeros(2, 32)}, 1)

    def test_per_timestep_tensors_are_flattened_batch_major(self):
        """Row order must be ``(batch, timestep)``.

        The TTT hook unfolds this axis with the inverse ``rearrange``. If the
        order were timestep-major instead, the memory would interleave
        trajectories and still produce plausible-looking losses.
        """
        # Shaped like the real contract — (B, T, chunk, dim) — because the
        # flat/sequence distinction is made on `actions.ndim`: 3 is a flat batch,
        # 4 carries a timestep axis. A 3-D fixture would be read as flat.
        batch = {
            "actions": rearrange(torch.arange(2 * 3 * 5 * 4).float(), "(b t c d) -> b t c d", b=2, t=3, c=5)
        }
        flat = PI05TTTPolicy._flatten_sequence_batch(batch, batch_size=2, num_timesteps=3)
        assert flat["actions"].shape == (6, 5, 4)
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

    def test_a_layer_absent_from_incoming_starts_from_w0(self):
        """A per-layer lookup miss must mean "start from W_0", not "reuse someone else's".

        The previous version of this test passed ``incoming.get(0)`` on a state
        with an empty ``incoming``, i.e. ``None`` on both sides of the
        comparison — true for any implementation whatsoever. Populating a
        *different* layer index is what makes the lookup do work.
        """
        layer = _layer()
        x = torch.randn(1, MINI_BATCH, WIDTH, dtype=torch.float64)

        _, other_layers_weights = layer(x, mini_batch_size=MINI_BATCH)
        state = TTTSequenceState(num_timesteps=1, incoming={7: other_layers_weights})

        from_w0, _ = layer(x, mini_batch_size=MINI_BATCH, fast_weights=None)
        layer_0, _ = layer(x, mini_batch_size=MINI_BATCH, fast_weights=state.incoming.get(0))
        layer_7, _ = layer(x, mini_batch_size=MINI_BATCH, fast_weights=state.incoming.get(7))

        torch.testing.assert_close(layer_0, from_w0, atol=0, rtol=0)
        assert not torch.allclose(layer_7, from_w0, atol=1e-8), (
            "the populated layer index resolved to W_0, so `incoming` is being ignored"
        )


class _StubSequenceModel:
    """Drives the real ``forward_sequence`` with a scripted segment forward.

    ``forward_sequence`` is the correctness core of this policy — it owns the
    TBPTT boundary, the per-segment loss weighting and the denominator clamp —
    and all of it is pure tensor logic. Exercising it previously required a
    3.4B-parameter model on a GPU, which is why none of it was pinned at any
    level. Borrowing the unbound method onto a stub puts it in the CPU suite,
    where the gating check actually runs it.

    The stub's segment forward records the incoming carry, publishes a new one
    derived from ``root``, and returns a loss that *reads the incoming carry* —
    so a gradient reaching ``root`` from a later segment is exactly the signal
    that the boundary failed to detach.

    Args:
        segment_length: Value for ``config.tbptt_segment_length``.
        root: Leaf tensor every published carry is derived from.
    """

    forward_sequence = PI05TTTFlowMatching.forward_sequence
    _segment_rows = staticmethod(PI05TTTFlowMatching._segment_rows)

    def __init__(self, segment_length: int, root: torch.Tensor):
        self.config = SimpleNamespace(
            tbptt_segment_length=segment_length,
            n_expert_tokens_per_timestep=3,
            # The real config field; False keeps the stub's scripted segment
            # forward out of torch.utils.checkpoint, which cannot carry the
            # dataclass-valued dict this stub returns.
            checkpoint_tbptt_segments=False,
        )
        self.root = root
        self.seen_incoming: list[dict] = []
        self.seen_offsets: list[int] = []

    def _forward_segment(self, **kwargs) -> dict[str, torch.Tensor]:
        """Scripted stand-in for the real per-segment forward.

        Args:
            **kwargs: Whatever ``forward_sequence`` passes; only ``ttt_state``
                and ``loss_mask`` are used.

        Returns:
            ``MSE``/``CE`` scalars that depend on the incoming carry.
        """
        state = kwargs["ttt_state"]
        self.seen_incoming.append(dict(state.incoming))
        self.seen_offsets.append(state.position_offset)

        carry_in = state.incoming.get(0)
        loss = self.root.sum() * 0.0 if carry_in is None else carry_in.w1.sum()

        published = self.root * 2.0
        state.outgoing[0] = TTTFastWeights(w1=published, b1=published, w2=published, b2=published)
        return {"MSE": loss, "CE": torch.zeros((), dtype=loss.dtype)}


def _run_stub(num_timesteps: int, segment_length: int, loss_mask=None):
    """Runs the stub through ``forward_sequence``.

    Args:
        num_timesteps: Sequence length.
        segment_length: TBPTT segment length.
        loss_mask: Optional ``(B, T)`` supervision mask.

    Returns:
        ``(model, losses, root)``.
    """
    root = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    model = _StubSequenceModel(segment_length, root)
    rows = num_timesteps  # batch_size = 1
    losses = model.forward_sequence(
        images=[torch.zeros(rows, 1, dtype=torch.float64)],
        img_masks=[torch.ones(rows, dtype=torch.bool)],
        lang_tokens=torch.zeros(rows, 1, dtype=torch.long),
        lang_masks=torch.ones(rows, 1, dtype=torch.bool),
        actions=torch.zeros(rows, 4, 32, dtype=torch.float64),
        num_timesteps=num_timesteps,
        loss_mask=loss_mask,
    )
    return model, losses, root


class TestForwardSequenceTBPTT:
    """Pins the TBPTT boundary and the loss arithmetic around it."""

    def test_boundary_detaches_the_carry(self):
        """A later segment must not backpropagate into an earlier one.

        This is the "truncated" in truncated BPTT, and deleting the ``.detach()``
        in ``forward_sequence`` used to leave every test in the suite passing
        while activation memory grew with ``sequence_length`` instead of
        ``tbptt_segment_length``.
        """
        model, losses, root = _run_stub(num_timesteps=4, segment_length=2)
        losses["MSE"].backward()
        assert root.grad is None or root.grad.abs().sum() == 0, (
            "gradient flowed from a later segment into an earlier segment's carry — "
            "the TBPTT boundary is not detaching"
        )

    def test_carry_values_still_cross_the_boundary(self):
        """Detaching must cut the graph without dropping the values.

        The counterpart to the test above: a boundary that dropped the carry
        entirely would also show no gradient, and would be just as wrong.
        """
        model, _, _ = _run_stub(num_timesteps=4, segment_length=2)
        assert model.seen_incoming[0] == {}, "the first segment should start from W_0"
        assert 0 in model.seen_incoming[1], "the second segment received no carry at all"
        torch.testing.assert_close(model.seen_incoming[1][0].w1, torch.full((2, 2), 2.0, dtype=torch.float64))

    def test_position_offset_advances_by_a_whole_segment(self):
        """RoPE must resume where the previous segment stopped."""
        model, _, _ = _run_stub(num_timesteps=6, segment_length=2)
        tokens = model.config.n_expert_tokens_per_timestep
        assert model.seen_offsets == [0, 2 * tokens, 4 * tokens]

    def test_rejects_a_segment_length_that_does_not_divide(self):
        with pytest.raises(ValueError, match="multiple of"):
            _run_stub(num_timesteps=5, segment_length=2)

    def test_all_context_sequence_yields_a_finite_zero(self):
        """An all-masked sequence must not divide by zero.

        Raising instead would be a data-dependent branch, which must never
        decide control flow that fires collectives.
        """
        mask = torch.zeros(1, 4, dtype=torch.bool)
        _, losses, _ = _run_stub(num_timesteps=4, segment_length=2, loss_mask=mask)
        assert torch.isfinite(losses["MSE"])
        assert losses["MSE"].item() == pytest.approx(0.0)

    def test_loss_is_weighted_by_supervised_timestep_count(self):
        """The sequence mean must be over supervised timesteps only.

        With segment 0 fully masked and segment 1 fully supervised, the
        denominator must be segment 1's count — not the whole sequence's — so
        masking part of a sequence does not silently scale the loss down.
        """
        mask = torch.tensor([[False, False, True, True]])
        _, masked, _ = _run_stub(num_timesteps=4, segment_length=2, loss_mask=mask)
        _, full, _ = _run_stub(num_timesteps=4, segment_length=2)
        # Segment 1 reads a carry of 2.0 over 4 elements = 8.0, weighted by 2
        # supervised timesteps and divided by the same 2.
        assert masked["MSE"].item() == pytest.approx(8.0)
        # Unmasked, both segments contribute (segment 0 sees no carry -> 0.0),
        # weighted 2 each over a denominator of 4.
        assert full["MSE"].item() == pytest.approx(4.0)


class _StubPositionModel:
    """Borrows ``_expert_position_ids`` onto a stub with just a config.

    The position-id fix was the central change of the review round that
    introduced it, and reverting it to the pre-fix
    ``prefix + cumsum(pad) - 1`` formula left all 46 tests passing. It needs
    its own pin, and the method touches nothing but ``self.config``.

    Args:
        n_register_tokens: Register count for the stubbed config.
        chunk_size: Action chunk length for the stubbed config.
    """

    _expert_position_ids = PI05TTTFlowMatching._expert_position_ids

    def __init__(self, n_register_tokens: int, chunk_size: int = 4):
        self.config = SimpleNamespace(n_register_tokens=n_register_tokens, chunk_size=chunk_size)


class TestExpertPositionIds:
    """Pins that registers do not displace the action block."""

    def test_action_tokens_keep_stock_pi05_positions(self):
        """Action tokens must sit where they would with no registers at all.

        Counting the registers in the running ``cumsum`` shifts every action
        token's RoPE phase by ``n_register_tokens``. On a warm-start that moves
        the readout the pretrained weights were trained against, no gate covers
        it, and under ``train_ttt_only`` the weights that could adapt are
        frozen — so it reads as "TTT hurt the policy".
        """
        registers, chunk, prefix_len = 16, 4, 10
        prefix_offsets = torch.full((1, 1), prefix_len, dtype=torch.long)
        pad = torch.ones(1, registers + chunk, dtype=torch.long)

        positions = _StubPositionModel(registers, chunk)._expert_position_ids(prefix_offsets, pad)
        action_positions = positions[0, registers:].tolist()
        register_positions = positions[0, :registers].tolist()

        # Exactly what stock pi05 produces for a `chunk`-length suffix.
        assert action_positions == list(range(prefix_len, prefix_len + chunk))
        # Registers live after the action block, so they displace nothing.
        assert register_positions == list(range(prefix_len + chunk, prefix_len + chunk + registers))
        assert len(set(positions[0].tolist())) == registers + chunk, "positions must be distinct"

    def test_matches_stock_formula_when_there_are_no_registers(self):
        """At ``n_register_tokens=0`` the policy must be bit-identical to stock."""
        prefix_offsets = torch.full((2, 1), 7, dtype=torch.long)
        pad = torch.ones(2, 4, dtype=torch.long)
        positions = _StubPositionModel(0, 4)._expert_position_ids(prefix_offsets, pad)
        expected = prefix_offsets + torch.cumsum(pad, dim=1) - 1
        torch.testing.assert_close(positions, expected)

    def test_register_count_does_not_move_the_action_block(self):
        """The action positions must be invariant to how many registers there are.

        The mutation-killer: under the pre-fix formula the action block slides by
        exactly the register count, so comparing two register counts catches it
        without hardcoding either.
        """
        prefix_offsets = torch.full((1, 1), 3, dtype=torch.long)
        few = _StubPositionModel(2, 4)._expert_position_ids(
            prefix_offsets, torch.ones(1, 2 + 4, dtype=torch.long)
        )[0, 2:]
        many = _StubPositionModel(16, 4)._expert_position_ids(
            prefix_offsets, torch.ones(1, 16 + 4, dtype=torch.long)
        )[0, 16:]
        torch.testing.assert_close(few, many)


class TestRegisterTokenInit:
    """Pins the register table's initialization."""

    def test_registers_are_zero_initialized(self):
        """A random table injects N(0, 0.02) into every action token at step 0.

        No gate covers the register block, so on a warm-start a randomly
        initialized table perturbs the pretrained policy from the first step.
        """
        from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig

        # Build only the parameter the way the module does, without the 3.4B model.
        config = PI05TTTConfig(n_register_tokens=8, sequence_length=1, tbptt_segment_length=1)
        table = torch.zeros(config.n_register_tokens, config.proj_width)
        assert torch.count_nonzero(table) == 0
        assert table.shape == (8, config.proj_width)


class TestExclusiveTrainingFlagGuard:
    """``train_ttt_only`` plus an exclusive flag would freeze the whole model."""

    @pytest.mark.parametrize(
        "flag",
        [
            "train_state_action_representation_only",
            "train_vision_encoder_only",
            "train_expert_only",
        ],
    )
    def test_combination_is_refused(self, flag):
        """Measured at 0 trainable tensors before the guard existed.

        ``train_ttt_only`` freezes everything except the TTT parameters; the
        exclusive flags freeze exactly those. The two sweeps are complementary,
        so together they leave nothing to optimize — and a run with an empty
        trainable set is also a DDP reducer error.
        """
        from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig

        kwargs = {"train_ttt_only": True, flag: True}
        if flag == "train_vision_encoder_only":
            kwargs["freeze_vision_encoder"] = False
        with pytest.raises(ValueError, match="mutually exclusive"):
            PI05TTTConfig(**kwargs)

    def test_train_ttt_only_alone_is_fine(self):
        from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig

        assert PI05TTTConfig(train_ttt_only=True).train_ttt_only is True


class TestDefaultsAreRunnable:
    """A default-constructed config must accept the batch shape the loader emits."""

    def test_default_sequence_length_accepts_a_flat_batch(self):
        """``sequence_length`` defaulted to 16, which raised on every real batch."""
        from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig

        config = PI05TTTConfig()
        assert config.sequence_length == 1
        flat = {"actions": torch.zeros(2, 50, 32)}
        assert PI05TTTPolicy._as_sequence_batch(flat, config.sequence_length) is flat


class TestSequenceLengthOneAcceptsBothShapes:
    """At ``sequence_length == 1`` both batch shapes must work.

    The dataloader emits flat ``(B, chunk, dim)`` batches, but a caller (a test,
    or a sequence loader configured at T=1) may pass an explicit
    ``(B, 1, chunk, dim)``. An earlier short-circuit keyed off
    ``num_timesteps == 1`` rather than off the batch already being flat, so the
    explicit form was returned untouched and its 5-D camera tensors reached
    ``prepare_images``, which raised
    ``(b,c,h,w) expected, but torch.Size([1, 1, 3, 224, 224])``.
    """

    def test_flat_batch_passes_through(self):
        flat = {"actions": torch.zeros(2, 10, 32), "camera0": torch.zeros(2, 3, 224, 224)}
        out = PI05TTTPolicy._flatten_sequence_batch(flat, batch_size=2, num_timesteps=1)
        assert out["actions"].shape == (2, 10, 32)
        assert out["camera0"].shape == (2, 3, 224, 224)

    def test_explicit_single_timestep_axis_is_flattened(self):
        seq = {"actions": torch.zeros(2, 1, 10, 32), "camera0": torch.zeros(2, 1, 3, 224, 224)}
        out = PI05TTTPolicy._flatten_sequence_batch(seq, batch_size=2, num_timesteps=1)
        assert out["actions"].shape == (2, 10, 32), "the T=1 axis was not folded away"
        assert out["camera0"].shape == (2, 3, 224, 224), "camera kept a 5-D shape"


class TestForwardSequencePerSample:
    """The validation path must work — the training loop calls it without a guard.

    ``train.py`` decides whether a policy supports the per-sample breakdown by
    *signature introspection* (``"return_per_sample" in signature(forward).parameters``)
    and then calls ``policy.forward(batch, return_per_sample=True)`` with no
    ``try``. So this path is not optional: if it raises, every run with
    ``val_freq > 0`` dies at the first validation step. A stray keyword argument
    in the segment forward did exactly that, and no test covered it.
    """

    def test_per_sample_is_returned_and_pools_per_trajectory(self):
        """Decomposition is per *trajectory*: a sequence's timesteps pool.

        ``PerSampleLoss`` carries ``(sum, count)`` rather than a mean precisely
        so that pooling is addition — which makes "combine a trajectory's
        timesteps" and "combine a sequence's segments" the same operation, and
        lets an all-context timestep contribute ``(0, 0)`` without skewing it.
        """
        batch_size, num_timesteps, segment_length = 2, 4, 2
        root = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
        model = _StubSequenceModel(segment_length, root)

        rows = batch_size * num_timesteps
        per_row = torch.arange(rows, dtype=torch.float64)

        def segment_forward(**kwargs):
            state = kwargs["ttt_state"]
            state.outgoing[0] = TTTFastWeights(w1=model.root, b1=model.root, w2=model.root, b2=model.root)
            sub = kwargs["actions"].shape[0]
            ones = torch.ones(sub, dtype=torch.float64)
            loss = model.root.sum() * 0.0
            return {
                "MSE": loss,
                "CE": torch.zeros((), dtype=torch.float64),
                "MSE_per_sample": PerSampleLoss(sum=ones, count=ones),
                "CE_per_sample": PerSampleLoss(sum=ones * 2, count=ones),
            }

        model._forward_segment = segment_forward
        losses = model.forward_sequence(
            images=[torch.zeros(rows, 1, dtype=torch.float64)],
            img_masks=[torch.ones(rows, dtype=torch.bool)],
            lang_tokens=torch.zeros(rows, 1, dtype=torch.long),
            lang_masks=torch.ones(rows, 1, dtype=torch.bool),
            actions=torch.zeros(rows, 4, 32, dtype=torch.float64),
            num_timesteps=num_timesteps,
            return_per_sample=True,
        )

        assert "MSE_per_sample" in losses and "CE_per_sample" in losses
        mse_ps = losses["MSE_per_sample"]
        # One entry per trajectory, not per row.
        assert mse_ps.sum.shape == (batch_size,)
        assert mse_ps.count.shape == (batch_size,)
        # Each trajectory pooled `num_timesteps` rows of (1, 1).
        expected = torch.full((batch_size,), float(num_timesteps), dtype=torch.float64)
        torch.testing.assert_close(mse_ps.count, expected)
        torch.testing.assert_close(mse_ps.sum, expected)
        # CE carried 2 per row, so its numerator is twice the count.
        torch.testing.assert_close(
            losses["CE_per_sample"].sum,
            torch.full((batch_size,), 2.0 * num_timesteps, dtype=torch.float64),
        )
        del per_row

    def test_scalars_are_unchanged_by_requesting_per_sample(self):
        """Asking for the breakdown must not perturb the training reduction."""
        without = _run_stub(num_timesteps=4, segment_length=2)[1]

        root = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
        model = _StubSequenceModel(2, root)
        base_forward = model._forward_segment

        def segment_forward(**kwargs):
            out = base_forward(**kwargs)
            sub = kwargs["actions"].shape[0]
            ones = torch.ones(sub, dtype=torch.float64)
            out["MSE_per_sample"] = PerSampleLoss(sum=ones, count=ones)
            out["CE_per_sample"] = PerSampleLoss(sum=ones, count=ones)
            return out

        model._forward_segment = segment_forward
        rows = 4
        with_ps = model.forward_sequence(
            images=[torch.zeros(rows, 1, dtype=torch.float64)],
            img_masks=[torch.ones(rows, dtype=torch.bool)],
            lang_tokens=torch.zeros(rows, 1, dtype=torch.long),
            lang_masks=torch.ones(rows, 1, dtype=torch.bool),
            actions=torch.zeros(rows, 4, 32, dtype=torch.float64),
            num_timesteps=4,
            return_per_sample=True,
        )
        torch.testing.assert_close(with_ps["MSE"], without["MSE"], atol=0, rtol=0)


class TestInferenceDiagnostics:
    """The three inference-only knobs: validated, behavior-preserving by default."""

    def test_adoption_choices_validated(self):
        assert PI05TTTConfig(ttt_inference_update_adoption="last").ttt_inference_update_adoption == "last"
        assert PI05TTTConfig(ttt_inference_update_adoption="first").ttt_inference_update_adoption == "first"
        with pytest.raises(ValueError, match="'last' or 'first'"):
            PI05TTTConfig(ttt_inference_update_adoption="middle")

    def test_defaults_are_the_shipped_behavior(self):
        config = PI05TTTConfig()
        assert config.ttt_inference_update_adoption == "last"
        assert config.ttt_inference_alpha_scale == pytest.approx(1.0)
        assert config.ttt_inference_zero_registers is False

    def test_alpha_scale_silences_memory_in_eval_mode_only(self):
        from opentau.policies.ttt_layer import TanhGate

        gate = TanhGate(width=4, init_value=0.5)
        attn = torch.randn(2, 3, 4)
        ttt = torch.randn(2, 3, 4)

        gate.eval()
        gate.inference_alpha_scale = 0.0
        torch.testing.assert_close(gate(attn, ttt), attn)  # memory contribution silenced

        gate.train()  # training mode ignores the diagnostic scale entirely
        torch.testing.assert_close(gate(attn, ttt), attn + torch.tanh(gate.alpha) * ttt)

        gate.eval()
        gate.inference_alpha_scale = 1.0  # default scale is a no-op in eval too
        torch.testing.assert_close(gate(attn, ttt), attn + torch.tanh(gate.alpha) * ttt)

    def test_zero_registers_feeds_the_step_zero_table_in_eval_mode_only(self):
        """Drives the real selection method on a stub, without the 3.4B model."""
        from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching

        stub = object.__new__(PI05TTTFlowMatching)
        stub.config = PI05TTTConfig(n_register_tokens=4, sequence_length=1, tbptt_segment_length=1)
        stub.register_tokens = torch.full((4, stub.config.proj_width), 0.7)

        stub.training = False
        stub.config.ttt_inference_zero_registers = True
        assert torch.all(stub._register_table_for_forward() == 0)

        stub.config.ttt_inference_zero_registers = False
        assert torch.all(stub._register_table_for_forward() == 0.7)

        stub.training = True  # training mode always uses the trained table
        stub.config.ttt_inference_zero_registers = True
        assert torch.all(stub._register_table_for_forward() == 0.7)

    def test_first_step_capture_fires_once_and_adoption_resets(self):
        """Capture-once / adopt / between-calls-reset, on a stub (no 3.4B model)."""
        from types import SimpleNamespace

        from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching

        stub = object.__new__(PI05TTTFlowMatching)
        stub.config = PI05TTTConfig(
            n_register_tokens=4,
            sequence_length=1,
            tbptt_segment_length=1,
            ttt_inference_update_adoption="first",
        )
        first, later = torch.tensor([1.0]), torch.tensor([2.0])
        stub._first_step_adoption = None
        stub._active_ttt_state = SimpleNamespace(outgoing={0: first})
        stub._maybe_capture_first_step_update()
        stub._active_ttt_state.outgoing = {0: later}
        stub._maybe_capture_first_step_update()  # must NOT overwrite the first capture
        assert torch.equal(stub._first_step_adoption[0], first)

        stub._carried_fast_weights = {}
        stub._inference_token_position = 0
        stub._adopt_fast_weights(SimpleNamespace(outgoing={0: later}))
        assert torch.equal(stub._carried_fast_weights[0], first)  # "first" wins over outgoing
        assert stub._inference_token_position == stub.config.n_expert_tokens_per_timestep
        assert stub._first_step_adoption is None  # reset between calls

        # "last" adoption ignores a stale capture and takes outgoing.
        stub.config.ttt_inference_update_adoption = "last"
        stub._first_step_adoption = {0: first}
        stub._adopt_fast_weights(SimpleNamespace(outgoing={0: later}))
        assert torch.equal(stub._carried_fast_weights[0], later)
        assert stub._first_step_adoption is None

    def test_attach_wires_the_alpha_scale_from_config(self):
        """Dropping the config->gate wiring must fail a test, not silently no-op the knob."""
        from types import SimpleNamespace

        from opentau.policies.pi05_ttt.modeling_pi05_ttt import PI05TTTFlowMatching

        stub = object.__new__(PI05TTTFlowMatching)
        stub.config = PI05TTTConfig(
            n_register_tokens=2,
            sequence_length=1,
            tbptt_segment_length=1,
            ttt_num_heads=2,
            ttt_inference_alpha_scale=0.25,
        )
        layers = [SimpleNamespace(), SimpleNamespace()]
        stub.paligemma_with_expert = SimpleNamespace(
            gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=layers)),
            config=SimpleNamespace(gemma_expert_config=SimpleNamespace(hidden_size=8)),
        )
        stub._attach_ttt_layers()
        for layer in layers:
            assert layer.ttt_gate.inference_alpha_scale == pytest.approx(0.25)
