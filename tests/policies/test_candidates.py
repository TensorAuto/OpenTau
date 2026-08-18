#!/usr/bin/env python

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

"""CPU coverage for the best-of-N primitives in ``opentau.policies.candidates``.

The sampler-side wiring is pinned separately; these tests pin the module itself, and
specifically the properties whose violation is invisible in a shape assertion: the ``(b n)``
fan-out *ordering* (a transposed convention has identical shapes and only misattributes
candidates once ``B > 1``), the ``.contiguous()`` that stops the expanded KV cache from
aliasing at ``B == 1``, the selector's NaN-safety, the medoid's row mask, and the arming
precedence that decides whether best-of-N runs at all.

Built on lightweight shells with no real weights, the same way ``test_pi05_accel.py`` is.
"""

import pytest
import torch
from torch import nn

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.candidates import (
    MEDOID,
    ActionChunkCritic,
    MedoidCritic,
    attach_critic,
    collapse_candidates,
    configure_candidates,
    expand_candidates,
    expand_kv_cache,
    load_critic,
    refuse_candidates,
    select_candidate,
)
from opentau.policies.normalize import Unnormalize

CPU = torch.device("cpu")
ACTION_DIM = 4
CHUNK = 4


# --------------------------------------------------------------------------------------
# Expansion / collapse helpers.
# --------------------------------------------------------------------------------------


def test_expansion_is_candidate_major():
    """``(b n)`` means source row ``i`` occupies rows ``[i*n, (i+1)*n)``, contiguously.

    Distinct per-row values are the whole point: a shape-only assertion passes just as
    happily under the transposed ``(n b)`` convention, which is the bug being pinned — it
    silently pairs candidate ``j`` of observation ``i`` with observation ``j``'s prefix.
    """
    source = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    expanded = expand_candidates(source, 2)

    assert expanded.shape == (6, 2)
    assert expanded.tolist() == [
        [10.0, 11.0],
        [10.0, 11.0],
        [20.0, 21.0],
        [20.0, 21.0],
        [30.0, 31.0],
        [30.0, 31.0],
    ]


def test_expansion_materializes_instead_of_aliasing_at_batch_one():
    """``B == 1`` is the entire serving path, and it is exactly where ``einops.repeat``
    hands back a stride-0 alias over the caller's storage. Both halves are asserted: the raw
    repeat *would* have aliased, and the helper does not. Without the first half the pin does
    not depend on the property it names — it would pass on a plain ``torch.clone`` too.
    """
    from einops import repeat

    source = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    aliased = repeat(source, "b ... -> (b n) ...", n=3)
    assert aliased.stride()[0] == 0, "einops.repeat broadcasts rather than copying at B == 1"
    assert aliased.data_ptr() == source.data_ptr(), "...over the caller's own storage"

    expanded = expand_candidates(source, 3)

    assert expanded.stride()[0] != 0
    assert expanded.data_ptr() != source.data_ptr()
    assert expanded.is_contiguous()
    assert torch.equal(expanded, aliased), "materializing must not change the values"


def test_collapse_round_trips_the_expansion():
    source = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

    collapsed = collapse_candidates(expand_candidates(source, 4), 4)

    assert collapsed.shape == (3, 4, 2, 2)
    for candidate in range(4):
        assert torch.equal(collapsed[:, candidate], source)


def test_kv_cache_expansion_walks_the_dict_and_leaves_it_alone():
    """The cache is a dict keyed by layer index, not a list — iterating it the wrong way
    yields integers. The caller's cache is reused after this call (under ``predict_response``
    the autoregressive loop just wrote it), so it must come back untouched.
    """
    key = torch.tensor([[[1.0, 2.0]]])
    value = torch.tensor([[[3.0, 4.0]]])
    cache = {0: {"key_states": key, "value_states": value}}

    expanded = expand_kv_cache(cache, 3)

    assert isinstance(expanded, dict)
    assert list(expanded.keys()) == [0]
    assert expanded[0]["key_states"].shape == (3, 1, 2)
    assert expanded[0]["value_states"].shape == (3, 1, 2)
    assert torch.equal(expanded[0]["key_states"], key.expand(3, 1, 2))

    assert expanded is not cache and expanded[0] is not cache[0]
    assert cache[0]["key_states"] is key and key.shape == (1, 1, 2)


# --------------------------------------------------------------------------------------
# `select_candidate` — total, NaN-safe, lowest-index ties.
# --------------------------------------------------------------------------------------


def test_a_nan_score_never_wins():
    """Raw ``torch.argmax([1.0, nan, 3.0])`` returns 1 — the NaN — so the selector has to
    replace non-finite entries before reducing, or a critic that produces one NaN hands the
    robot that candidate."""
    assert select_candidate(torch.tensor([[1.0, float("nan"), 3.0]])).tolist() == [2]


def test_negative_infinity_behaves_like_nan():
    assert select_candidate(torch.tensor([[1.0, float("-inf"), 3.0]])).tolist() == [2]
    assert select_candidate(torch.tensor([[float("-inf"), 2.0, float("-inf")]])).tolist() == [1]


def test_an_all_non_finite_row_falls_back_to_candidate_zero_without_raising():
    """``select_action`` runs on every rank and is followed by gather collectives, so a
    data-dependent raise here aborts one rank and blocks the rest at NCCL forever."""
    assert select_candidate(torch.tensor([[float("nan"), float("nan")]])).tolist() == [0]
    assert select_candidate(torch.tensor([[float("-inf"), float("-inf")]])).tolist() == [0]


def test_ties_resolve_to_the_lowest_index():
    """Load-bearing: candidate 0 carries the noise draw the ``n_candidates == 1`` path would
    have taken, so a degenerate critic degenerates to legacy behaviour."""
    assert select_candidate(torch.tensor([[2.0, 2.0, 1.0]])).tolist() == [0]
    assert select_candidate(torch.tensor([[1.0, 5.0, 5.0]])).tolist() == [1]


def test_a_degenerate_row_does_not_contaminate_its_neighbours():
    """The fallback is per row. A batched server sees healthy and degenerate observations in
    one call, and a whole-batch fallback would silently disable best-of-N for all of them."""
    scores = torch.tensor([[float("nan"), float("nan"), float("nan")], [1.0, 7.0, 3.0]])

    assert select_candidate(scores).tolist() == [0, 1]


# --------------------------------------------------------------------------------------
# `MedoidCritic` — the parameter-free reference selector.
# --------------------------------------------------------------------------------------


def _medoid(std, *, max_action_dim=None):
    """A ``MedoidCritic`` over a one-head ``Unnormalize`` with the given per-dim ``std``."""
    dim = len(std)
    features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(dim,))}
    unnormalize = Unnormalize(
        features,
        {"ACTION": NormalizationMode.MEAN_STD},
        per_dataset_stats=[{"actions": {"mean": torch.zeros(dim), "std": torch.tensor(std)}}],
        dataset_names=["ds0"],
    )
    return MedoidCritic(unnormalize, max_action_dim=max_action_dim or dim)


def _score(critic, candidates, *, row_mask=None):
    chunk = candidates.shape[2]
    if row_mask is None:
        row_mask = torch.ones(1, chunk, dtype=torch.bool)
    return critic.score_chunks(
        {}, candidates, row_mask=row_mask, dataset_index=torch.zeros(1, dtype=torch.long)
    )


def _chunks(rows_per_candidate, *, dim=2):
    """``(1, N, chunk, dim)`` where candidate ``i`` has constant value ``rows[i][r]`` on row
    ``r``."""
    return torch.tensor([[[[v] * dim for v in rows] for rows in rows_per_candidate]])


def test_medoid_satisfies_the_critic_protocol():
    assert isinstance(_medoid([1.0, 1.0]), ActionChunkCritic)


def test_medoid_refuses_two_candidates_because_every_score_ties():
    """At N == 2 there is one pairwise distance and both candidates carry it, so selection
    always collapses to candidate 0 — best-of-N that silently is not. Constructed at exactly
    N == 2, the boundary the check guards."""
    candidates = _chunks([[0.0, 0.0], [5.0, 5.0]])
    assert candidates.shape[1] == 2

    with pytest.raises(ValueError, match="ties"):
        _score(_medoid([1.0, 1.0]), candidates)


def test_medoid_ranks_the_outlier_last_and_a_clustered_candidate_first():
    """Candidate 0 sits far from the other two, which sit close together.

    The outlier is placed at index 0 deliberately: a critic that emitted constant scores, or
    one whose sign was flipped, would also land on index 0 via the tie fallback, and this
    test would not notice. So the assertions pin all three of the winner, the loser, and
    what a flipped sign would have chosen.
    """
    critic = _medoid([1.0, 1.0])
    candidates = _chunks([[10.0, 10.0], [0.0, 0.0], [0.25, 0.25]])

    scores = _score(critic, candidates)

    assert scores.shape == (1, 3)
    assert select_candidate(scores).tolist() == [2]
    assert int(scores.argmin(dim=1)) == 0, "the outlier must score lowest"
    # Sign check: negate the scores and the outlier wins, which is what a `+total` medoid
    # would do on every observation.
    assert int(torch.argmax(-scores, dim=1)) == 0


def test_medoid_honours_the_row_mask():
    """Candidates 0 and 1 are identical on the executed rows and differ only outside them,
    so they must score identically. Un-masking the same tensors separates them, which is what
    makes this a pin on the mask rather than on the arithmetic.
    """
    critic = _medoid([1.0, 1.0])
    # Rows 0-1 executed, rows 2-3 not. Candidate 1 differs from candidate 0 only on 2-3.
    candidates = _chunks([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 7.0, 7.0], [1.0, 1.0, 0.0, 0.0]])
    executed = torch.tensor([[True, True, False, False]])

    masked = _score(critic, candidates, row_mask=executed)
    unmasked = _score(critic, candidates)

    assert masked[0, 0].item() == masked[0, 1].item()
    assert unmasked[0, 0].item() != unmasked[0, 1].item()


def test_medoid_drops_the_degenerate_action_dims():
    """The pad tail is unsupervised network output. Two candidates differing only there are
    the same chunk as far as the robot is concerned, and must not be ranked apart."""
    critic = _medoid([1.0, 0.0])  # dim 1 is zero-variance, i.e. padding
    candidates = torch.tensor([[[[0.0, 0.0]], [[0.0, 9.0]], [[3.0, 0.0]]]])

    scores = _score(critic, candidates)

    assert scores[0, 0].item() == scores[0, 1].item()


# --------------------------------------------------------------------------------------
# `configure_candidates` — the single arming knob.
# --------------------------------------------------------------------------------------


class _FakeConfig:
    type = "pi05"
    chunk_size = CHUNK
    max_action_dim = ACTION_DIM
    action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))

    def __init__(self, n_candidates=1, action_chunk_critic_path=None):
        self.n_candidates = n_candidates
        self.action_chunk_critic_path = action_chunk_critic_path


class _FakeCfg:
    def __init__(self, **kwargs):
        self.policy = _FakeConfig(**kwargs)


class _FakePolicy(nn.Module):
    """A wired policy shell: it exposes ``n_candidates`` and a real ``Unnormalize``."""

    def __init__(self, *, wired=True):
        super().__init__()
        self.config = _FakeConfig()
        self.weight = nn.Parameter(torch.zeros(1))
        stats = [{"actions": {"mean": torch.zeros(ACTION_DIM), "std": torch.tensor([1.0, 2.0, 0.0, 0.0])}}]
        features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
        self.unnormalize_outputs = Unnormalize(
            features,
            {"ACTION": NormalizationMode.MEAN_STD},
            per_dataset_stats=stats,
            dataset_names=["ds0"],
        )
        if wired:
            self.n_candidates = 1


class _ShapeBrokenCritic:
    """Protocol-shaped but returns ``(B,)`` instead of ``(B, N)``."""

    def score_chunks(self, batch, candidates, *, row_mask, dataset_index):
        return torch.zeros(candidates.shape[0])


def test_configure_writes_one_even_when_nothing_asks_for_candidates():
    """The N == 1 write is not redundant: returning early would leave whatever a previous
    call put there, so a policy re-configured for a second deployment would keep serving
    best-of-N nobody asked for."""
    policy = _FakePolicy()
    policy.n_candidates = 4  # stale value from a previous arming

    assert configure_candidates(policy, _FakeCfg(n_candidates=1), device=CPU) == 1
    assert policy.n_candidates == 1


def test_override_of_one_beats_a_config_asking_for_four():
    """The actual conflict, both directions. ``override=1`` is a caller deliberately
    *disabling* a config-enabled N, so the precedence has to be ``is None`` rather than
    ``or``-truthiness — an ``or`` chain falls straight through a 1 and arms best-of-N against
    the caller's explicit instruction.
    """
    policy = _FakePolicy()
    cfg = _FakeCfg(n_candidates=4, action_chunk_critic_path=MEDOID)

    assert configure_candidates(policy, cfg, device=CPU, override=1) == 1
    assert policy.n_candidates == 1
    assert getattr(policy, "_action_chunk_critic", None) is None, "no critic may be loaded at N == 1"

    # ...and the other side of the same conflict: an override above 1 beats a config of 1.
    disabled = _FakeCfg(n_candidates=1, action_chunk_critic_path=MEDOID)
    assert configure_candidates(policy, disabled, device=CPU, override=3) == 3
    assert policy.n_candidates == 3
    assert isinstance(policy._action_chunk_critic, MedoidCritic)


def test_configure_refuses_a_policy_family_that_is_not_wired():
    """pi0, pi07, cosmos3 and friends accept an unread attribute and then never fan out —
    the silent no-op an operator would read as 'the critic never fires'."""
    policy = _FakePolicy(wired=False)
    assert not hasattr(policy, "n_candidates")

    with pytest.raises(ValueError, match="not wired"):
        configure_candidates(policy, _FakeCfg(n_candidates=4, action_chunk_critic_path=MEDOID), device=CPU)


def test_configure_refuses_candidates_without_a_critic():
    """There is deliberately no implicit fallback selector: best-of-N choosing on no signal
    would look like it works."""
    with pytest.raises(ValueError, match="action_chunk_critic_path"):
        configure_candidates(_FakePolicy(), _FakeCfg(n_candidates=4), device=CPU)


def test_configure_rejects_a_non_critic():
    with pytest.raises(TypeError, match="score_chunks"):
        configure_candidates(_FakePolicy(), _FakeCfg(n_candidates=4), device=CPU, critic=object())


def test_configure_catches_a_wrong_return_shape_at_startup():
    """``runtime_checkable`` ``isinstance`` checks method *presence* only — never the return
    shape — so a critic scoring ``(B,)`` passes the protocol test and would otherwise fail on
    the first robot request instead of at boot."""
    with pytest.raises(TypeError, match=r"\(B, N\)"):
        configure_candidates(_FakePolicy(), _FakeCfg(n_candidates=4), device=CPU, critic=_ShapeBrokenCritic())


def test_configure_reads_the_real_policy_config_fields():
    """Draccus JSON is the repo's primary interface, so the fields have to be real dataclass
    members — and off by default, since a checkpoint's own config must not self-arm."""
    from opentau.policies.pi05.configuration_pi05 import PI05Config

    cfg = _FakeCfg()
    cfg.policy = PI05Config()

    assert cfg.policy.n_candidates == 1
    assert cfg.policy.action_chunk_critic_path is None

    policy = _FakePolicy()
    cfg.policy.n_candidates = 3
    cfg.policy.action_chunk_critic_path = MEDOID
    assert configure_candidates(policy, cfg, device=CPU) == 3


# --------------------------------------------------------------------------------------
# `attach_critic` / `load_critic` / `refuse_candidates`.
# --------------------------------------------------------------------------------------


class _ParametricCritic(nn.Module):
    """A critic with weights of its own — the case where registration actually shows up."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)

    def score_chunks(self, batch, candidates, *, row_mask, dataset_index):
        return torch.zeros(candidates.shape[:2])


def test_attach_critic_keeps_it_out_of_the_state_dict_and_the_optimizer():
    """Plain assignment routes through ``nn.Module.__setattr__``, which registers the critic
    in ``_modules`` — writing its tensors into every checkpoint and handing its parameters to
    ``get_optim_params``. Both halves are asserted, so the pin depends on the property it
    names rather than on the critic happening to be parameter-free.
    """
    policy = _FakePolicy()
    before_keys = set(policy.state_dict().keys())
    before_params = len(list(policy.parameters()))

    attach_critic(policy, _ParametricCritic())

    assert policy._action_chunk_critic is not None
    assert set(policy.state_dict().keys()) == before_keys
    assert len(list(policy.parameters())) == before_params
    assert "_action_chunk_critic" not in policy._modules

    # The same assignment done plainly does change both.
    contrast = _FakePolicy()
    contrast._action_chunk_critic = _ParametricCritic()
    assert set(contrast.state_dict().keys()) != before_keys
    assert len(list(contrast.parameters())) > before_params


def test_load_critic_resolves_the_builtin_selector_and_nothing_else():
    policy = _FakePolicy()

    assert isinstance(load_critic(MEDOID, policy=policy), MedoidCritic)

    with pytest.raises(NotImplementedError, match=MEDOID):
        load_critic("TensorAuto/some-learned-critic", policy=policy)
    with pytest.raises(NotImplementedError):
        load_critic(None, policy=policy)


def test_refuse_candidates_fires_only_above_one():
    """Entry points that cannot honour best-of-N must say so. Silently ignoring the field is
    the failure mode: the operator sets it, sees no error, and believes it is running."""
    with pytest.raises(ValueError, match="batched eval"):
        refuse_candidates(_FakeCfg(n_candidates=2), reason="batched eval memory")

    refuse_candidates(_FakeCfg(n_candidates=1), reason="batched eval memory")
    refuse_candidates(_FakeCfg(), reason="batched eval memory")
    # A config predating the field must not trip it either.
    refuse_candidates(object(), reason="batched eval memory")
