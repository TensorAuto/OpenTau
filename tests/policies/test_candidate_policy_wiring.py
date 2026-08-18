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

"""CPU coverage for best-of-N candidate sampling wired into the pi05 / pi06 samplers.

``tests/policies/test_candidates.py`` pins the helpers in
:mod:`opentau.policies.candidates` in isolation. These tests pin the *wiring* — the two
policy families that PR 1 actually threads ``n_candidates`` through — and they are written
around the three ways this feature goes wrong silently:

1. **It stops being free.** The whole premise is that one VLM prefix pass serves all N
   candidates, so ``test_prefix_pass_runs_once_regardless_of_n_candidates`` counts the
   backbone calls. Nothing else in the suite would notice a per-candidate prefix pass: the
   numbers would be identical and only the latency would move.
2. **A legacy user feels it anyway.** ``n_candidates`` defaults to 1 and every config that
   travels with a checkpoint carries the key, so the N==1 path must be untouched — no
   candidate helper called, the same single noise draw of the same shape, and a 3-D return.
3. **The fan-out is transposed.** ``(b n)`` versus ``(n b)`` is invisible in every shape,
   and at ``B == 1`` — the only batch serving produces — it is invisible in the values too.
   So the equivalence test is parametrized over ``B`` in ``{1, 2, 3}`` with distinct
   per-observation inputs, which is the only shape where the ordering can be observed.

Built on the same lightweight ``object.__new__`` shell the accel tests use
(``tests/policies/test_pi05_accel.py``), so no PaliGemma / Gemma 3 weights are loaded: the
backbone is a stub that returns a KV cache carrying a per-observation signature, and the
denoise step is a closed-form nonlinear function of ``x_t`` and that signature.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn

import opentau
import opentau.policies.pi05.modeling_pi05 as pi05_modeling
import opentau.policies.pi06.modeling_pi06 as pi06_modeling
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.candidates import collapse_candidates, configure_candidates
from opentau.policies.normalize import Unnormalize
from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi06.configuration_pi06 import PI06Config
from opentau.utils.random_utils import set_seed

CHUNK = 4
ACTION_DIM = 4
MAX_ACTION_DIM = 6
MAX_STATE_DIM = 8
NUM_STEPS = 4
PREFIX_TOKENS = 3
HIDDEN = 8
HEAD_DIM = 2
CPU = torch.device("cpu")

#: The three module-level names the samplers import from `candidates`. Monkeypatching must
#: target these per-module bindings, never the definitions in `candidates` itself — both
#: modules do `from opentau.policies.candidates import ...`, so patching the source leaves
#: the already-bound name untouched and a "was it called?" test passes vacuously.
CANDIDATE_HELPERS = ("expand_candidates", "expand_kv_cache", "select_candidate")


@dataclass(frozen=True)
class Family:
    """One policy family wired for best-of-N in PR 1."""

    name: str
    module: ModuleType
    rel_path: str
    config_cls: type
    sampler_cls: type
    policy_cls: type
    backbone_attr: str
    #: pi05's ``embed_prefix`` returns a 4th (adarms) element and its inner ``sample_actions``
    #: takes ``state=``; pi06's do neither.
    pi05_shaped: bool


FAMILIES = [
    Family(
        name="pi05",
        module=pi05_modeling,
        rel_path="pi05/modeling_pi05.py",
        config_cls=PI05Config,
        sampler_cls=pi05_modeling.PI05FlowMatching,
        policy_cls=pi05_modeling.PI05Policy,
        backbone_attr="paligemma_with_expert",
        pi05_shaped=True,
    ),
    Family(
        name="pi06",
        module=pi06_modeling,
        rel_path="pi06/modeling_pi06.py",
        config_cls=PI06Config,
        sampler_cls=pi06_modeling.PI06FlowMatching,
        policy_cls=pi06_modeling.PI06Policy,
        backbone_attr="gemma3_with_expert",
        pi05_shaped=False,
    ),
]

IDS = [family.name for family in FAMILIES]


@pytest.fixture(params=FAMILIES, ids=IDS)
def family(request) -> Family:
    return request.param


# --------------------------------------------------------------------------------------
# Shells: a sampler and a policy wrapper with no pretrained weights behind them.
# --------------------------------------------------------------------------------------


def _config(family: Family, **overrides):
    kwargs = {
        "n_obs_steps": 1,
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "max_delay": 0,
        "num_steps": NUM_STEPS,
        "max_state_dim": MAX_STATE_DIM,
        "max_action_dim": MAX_ACTION_DIM,
        "predict_response": False,
        "output_features": {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    }
    kwargs.update(overrides)
    return family.config_cls(**kwargs)


class _CountingBackbone(nn.Module):
    """Stand-in for the VLM that separates the prefix pass from the denoise passes.

    The counters are the point: the fused path's entire justification is that
    ``fill_kv_cache=True`` runs once per *observation* no matter how many candidates
    follow, and no assertion on the returned actions can tell the two apart.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prefix_fill_calls = 0
        self.denoise_calls = 0

    def forward(self, *, inputs_embeds, fill_kv_cache=False, **kwargs):
        embs = inputs_embeds[0]
        if not fill_kv_cache:
            self.denoise_calls += 1
            return (None, None), None

        self.prefix_fill_calls += 1
        # One scalar per observation, carried into the cache so a denoise step can read back
        # *which* observation its row was conditioned on. A transposed fan-out hands candidate
        # rows the wrong signature, which is the only way the ordering becomes observable.
        signature = reduce(embs, "b t h -> b", "mean")
        cache = {
            layer: {
                name: repeat(
                    signature * (layer + 1), "b -> b t h d", t=PREFIX_TOKENS, h=1, d=HEAD_DIM
                ).contiguous()
                for name in ("key_states", "value_states")
            }
            # Keyed by layer index, matching the real cache: `expand_kv_cache` iterates
            # `.items()`, and a list comprehension over it would yield bare ints.
            for layer in range(2)
        }
        return (torch.zeros(embs.shape[0], 1, HIDDEN), None), cache


def _flow_matching(family: Family, cfg) -> nn.Module:
    """An inner sampler whose ``denoise_step`` is a closed-form velocity field.

    Nonlinear in ``x_t`` so two noise draws take genuinely different trajectories (a linear
    field would make every candidate a scaled copy and hide an ordering bug), and scaled by
    the per-observation signature so conditioning errors surface as value differences.
    """
    fm = object.__new__(family.sampler_cls)
    nn.Module.__init__(fm)
    fm.config = cfg
    backbone = _CountingBackbone()
    setattr(fm, family.backbone_attr, backbone)

    def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
        embs = repeat(lang_tokens.to(torch.float32), "b t -> b t h", h=HIDDEN).contiguous()
        att_masks = torch.zeros(lang_tokens.shape[0], PREFIX_TOKENS, dtype=torch.long)
        if family.pi05_shaped:
            return embs, lang_masks, att_masks, None
        return embs, lang_masks, att_masks

    def denoise_step(prefix_pad_masks, past_key_values, x_t, time):
        backbone.forward(inputs_embeds=[x_t, None], fill_kv_cache=False)
        conditioning = past_key_values[0]["key_states"][:, 0, 0, 0]
        pad = reduce(prefix_pad_masks.to(torch.float32), "r t -> r", "sum")
        scale = rearrange(conditioning * pad, "r -> r 1 1")
        step = reduce(time.to(torch.float32), "b c -> b 1 1", "mean")
        return torch.tanh(x_t * scale) + step

    fm.embed_prefix = embed_prefix
    fm.denoise_step = denoise_step
    return fm


def _backbone(fm, family: Family) -> _CountingBackbone:
    return getattr(fm, family.backbone_attr)


def _lang_tokens(bsize: int) -> Tensor:
    """One distinct token value per observation, so each gets its own cache signature."""
    return repeat(torch.arange(1, bsize + 1), "b -> b t", t=PREFIX_TOKENS).contiguous()


def _action_prefix(bsize: int) -> Tensor:
    """A per-observation constant prefix, distinct across rows."""
    return repeat(
        torch.arange(1, bsize + 1, dtype=torch.float32), "b -> b c d", c=CHUNK, d=MAX_ACTION_DIM
    ).contiguous()


def _sample(
    fm,
    family: Family,
    *,
    lang_tokens: Tensor,
    action_prefix: Tensor | None = None,
    noise: Tensor | None = None,
    delay: int = 0,
    n_candidates: int = 1,
    accel=None,
) -> Tensor:
    """Call the inner sampler, absorbing pi05's extra ``state`` parameter."""
    if action_prefix is None:
        action_prefix = torch.zeros(lang_tokens.shape[0], CHUNK, MAX_ACTION_DIM)
    extra = {"state": None} if family.pi05_shaped else {}
    return family.sampler_cls.sample_actions(
        fm,
        [],
        [],
        lang_tokens,
        torch.ones_like(lang_tokens, dtype=torch.bool),
        action_prefix,
        torch.tensor(delay),
        noise=noise,
        n_candidates=n_candidates,
        accel=accel,
        **extra,
    )


class _RiggedCritic:
    """A critic that always prefers a fixed candidate index, and counts its calls.

    ``MedoidCritic`` refuses N < 3 and its ranking depends on the sampled chunks, neither of
    which suits a wiring test — what these tests need is a *known* answer so the selected row
    can be identified exactly.
    """

    def __init__(self, pick: int = 0, *, tie: bool = False) -> None:
        self.pick = pick
        self.tie = tie
        self.calls = 0

    def score_chunks(self, batch, candidates, *, row_mask, dataset_index):
        self.calls += 1
        bsize, n_cand = candidates.shape[:2]
        if self.tie:
            return torch.zeros(bsize, n_cand)
        scores = torch.full((bsize, n_cand), -1.0)
        scores[:, self.pick % n_cand] = 1.0
        return scores


def _unnormalize() -> Unnormalize:
    """A real ``Unnormalize`` with identity statistics.

    Real rather than stubbed because ``make_meter`` derives the accel dim mask from its
    buffers, and identity so the sampler's raw output survives to the critic unchanged and
    can be compared directly. ``ACTION_DIM < MAX_ACTION_DIM`` reproduces the padded tail.
    """
    features = {"actions": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    stats = [{"actions": {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)}}]
    return Unnormalize(
        features,
        {"ACTION": NormalizationMode.MEAN_STD},
        per_dataset_stats=stats,
        dataset_names=["ds0"],
        eps=1e-6,
    )


def _policy(family: Family, cfg=None, *, accel_prefix: int | None = None):
    """A policy wrapper shell whose ``sample_actions`` / ``select_action`` are the real ones.

    ``n_candidates = 1`` and ``_action_chunk_critic = None`` mirror what ``__init__`` sets;
    that mirroring is what ``test_wrapper_declares_n_candidates_as_one_in_init`` pins against
    the source, so the two cannot drift.
    """
    cfg = cfg if cfg is not None else _config(family)
    policy = object.__new__(family.policy_cls)
    nn.Module.__init__(policy)
    policy.config = cfg
    policy.model = _flow_matching(family, cfg)
    policy.unnormalize_outputs = _unnormalize()
    # `build_provenance` reads the velocity dtype off `next(policy.parameters())`; a shell
    # built from buffers alone has none.
    policy._dtype_probe = nn.Parameter(torch.zeros(1))
    policy.eval = lambda: None  # bypass nn.Module.eval (no policy __init__ was run)
    policy._resolve_dataset_index = lambda batch: torch.zeros(batch["state"].shape[0], dtype=torch.long)
    policy.normalize_inputs = lambda batch, index: dict(batch)
    policy.normalize_targets = lambda batch, index: dict(batch)
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_language = lambda batch: (
        batch["lang_tokens"],
        torch.ones_like(batch["lang_tokens"], dtype=torch.bool),
    )
    policy.prepare_state = lambda batch: batch["state"]
    policy.accel_prefix = accel_prefix
    policy.n_candidates = 1
    policy._action_chunk_critic = None
    family.policy_cls.reset(policy)
    return policy


def _arm(policy, n: int, critic: _RiggedCritic) -> None:
    """Arm best-of-N the way a serving entry point does, then discard the smoke-test call."""
    configure_candidates(policy, policy.config, device=CPU, override=n, critic=critic)
    critic.calls = 0


def _batch(bsize: int = 1) -> dict[str, Tensor]:
    return {
        "state": torch.zeros(bsize, MAX_STATE_DIM),
        "lang_tokens": _lang_tokens(bsize),
    }


def _noise(rows: int, seed: int = 0) -> Tensor:
    generator = torch.Generator(device=CPU).manual_seed(seed)
    return torch.randn(rows, CHUNK, MAX_ACTION_DIM, generator=generator)


# --------------------------------------------------------------------------------------
# N == 1 invariance: the "legacy user feels nothing" contract.
# --------------------------------------------------------------------------------------


def _raiser(name: str):
    def boom(*args, **kwargs):
        raise RuntimeError(f"candidate helper {name} was called")

    return boom


def test_n1_never_enters_the_candidate_path(family, monkeypatch):
    """At N==1 not one candidate helper may run — and each one must really run at N>1.

    The second half is what stops this from passing vacuously. Patching
    ``opentau.policies.candidates.expand_candidates`` (the definition site) would leave the
    sampler's already-bound ``from ... import`` name pointing at the original, so the N==1
    half would pass no matter what the sampler does; the N>1 half fails loudly in that case,
    which is the only way to know the patch targets are the bindings the code actually uses.
    """
    for name in CANDIDATE_HELPERS:
        monkeypatch.setattr(family.module, name, _raiser(name))

    _policy(family).sample_actions(_batch())  # must not raise

    monkeypatch.undo()
    for name in CANDIDATE_HELPERS:
        with monkeypatch.context() as patched:
            patched.setattr(family.module, name, _raiser(name))
            policy = _policy(family)
            _arm(policy, 4, _RiggedCritic(pick=1))
            with pytest.raises(RuntimeError, match=name):
                policy.sample_actions(_batch())


def test_n1_noise_shape_and_rng_draw_are_unchanged(family):
    """``bsize * n_candidates`` must fold to ``bsize``: one draw, the same shape, same stream.

    A second draw, or a draw of a different size, silently re-orders the global RNG for every
    seeded rollout that follows — a reproducibility break with no error and no shape change.
    """
    cfg = _config(family)
    bsize = 2
    fm = _flow_matching(family, cfg)

    draws: list[tuple[tuple[int, ...], Tensor]] = []
    real_sample_noise = family.sampler_cls.sample_noise

    def recording_sample_noise(shape, device):
        drawn = real_sample_noise(fm, shape, device)
        draws.append((tuple(shape), drawn))
        return drawn

    fm.sample_noise = recording_sample_noise

    # What the pre-change sampler drew: one `torch.normal` of exactly this shape, taken as
    # the first consumption of the stream.
    torch.manual_seed(1234)
    expected = torch.normal(
        mean=0.0, std=1.0, size=(bsize, CHUNK, MAX_ACTION_DIM), dtype=torch.float32, device=CPU
    )

    torch.manual_seed(1234)
    _sample(fm, family, lang_tokens=_lang_tokens(bsize), n_candidates=1)

    assert [shape for shape, _ in draws] == [(bsize, CHUNK, MAX_ACTION_DIM)]
    assert torch.equal(draws[0][1], expected)


@pytest.mark.parametrize("n_candidates", [1, 2, 4])
def test_n1_returns_the_same_shape_as_before(family, n_candidates):
    """The candidate axis must never reach a caller.

    ``select_action`` immediately does ``rearrange(actions, "b c d -> c b d")`` and extends a
    deque with the result, so a 4-D return would not raise — it would quietly queue one
    action per candidate and hand the robot the wrong tensor.
    """
    policy = _policy(family)
    if n_candidates > 1:
        _arm(policy, n_candidates, _RiggedCritic(pick=1))

    actions = policy.sample_actions(_batch(bsize=1))

    assert actions.shape == (1, CHUNK, ACTION_DIM)


# --------------------------------------------------------------------------------------
# Candidate correctness.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bsize", [1, 2, 3])
def test_candidate_i_of_a_fused_run_equals_a_standalone_run_with_that_noise(family, bsize):
    """Every fused candidate row must equal the standalone run that noise row would produce.

    This is the correctness claim the whole feature rests on: sharing one prefix pass and one
    KV cache across N candidates changes nothing about what each candidate *is*.

    ``B == 1`` is the only batch shape serving produces and the only one where
    ``einops.repeat`` aliases (stride 0), so it is worth its own case; ``B >= 2`` with
    distinct per-observation language tokens is the case that catches a transposed ``(n b)``
    fan-out, which is invisible at ``B == 1`` and invisible in every shape.

    Bit-identity is asserted because this stub runs float32 on CPU. It would NOT hold in
    bf16 on GPU: the same noise row decoded at a different batch size differs by ~5e-3 there
    (1-2 ULP of bf16, from batch-shape-dependent cuBLAS kernel selection), on unmodified code
    as well. Any cross-batch-shape assertion on real hardware needs a ULP-scale tolerance.
    """
    cfg = _config(family)
    n = 3
    lang_tokens = _lang_tokens(bsize)
    action_prefix = _action_prefix(bsize)
    noise = _noise(bsize * n)

    fused = _sample(
        _flow_matching(family, cfg),
        family,
        lang_tokens=lang_tokens,
        action_prefix=action_prefix,
        noise=noise.clone(),
        n_candidates=n,
    )
    assert fused.shape == (bsize * n, CHUNK, MAX_ACTION_DIM)

    for b in range(bsize):
        for j in range(n):
            row = b * n + j
            standalone = _sample(
                _flow_matching(family, cfg),
                family,
                lang_tokens=lang_tokens[b : b + 1],
                action_prefix=action_prefix[b : b + 1],
                noise=noise[row : row + 1].clone(),
                n_candidates=1,
            )
            assert torch.equal(fused[row : row + 1], standalone), (
                f"fused row {row} (observation {b}, candidate {j}) diverged from its standalone run"
            )

    # The candidates must actually differ, or the equality above would hold under any
    # ordering at all.
    grid = collapse_candidates(fused, n)
    assert not torch.equal(grid[0, 0], grid[0, 1])


@pytest.mark.parametrize("n_candidates", [1, 4])
def test_prefix_pass_runs_once_regardless_of_n_candidates(family, n_candidates):
    """One VLM prefix pass per observation, N or not — the claim that makes this cheap.

    Nothing else in the suite exercises it: a naive implementation that replicates the
    observation across the batch and pays N prefix passes returns exactly the same actions
    and differs only in wall clock and peak memory (measured at N=8: 2.5x slower, and out of
    memory at N=32 where the fused path is not).
    """
    cfg = _config(family)
    fm = _flow_matching(family, cfg)

    _sample(
        fm,
        family,
        lang_tokens=_lang_tokens(1),
        noise=_noise(n_candidates),
        n_candidates=n_candidates,
    )

    assert _backbone(fm, family).prefix_fill_calls == 1
    assert _backbone(fm, family).denoise_calls == cfg.num_steps


def test_frozen_prefix_rows_are_identical_across_candidates(family):
    """Real-time-chunking rows are committed actions: every candidate must carry them byte-identically.

    They are frozen by ``torch.where(prefix_mask, action_prefix, x_t)`` inside the loop, so
    the property holds only if ``action_prefix`` was fanned out in the same ``(b n)`` order
    as the noise. A transposed expansion gives observation 0's candidates observation 1's
    committed actions — the robot would then execute a discontinuity at every re-plan.
    """
    cfg = _config(family, max_delay=2)
    bsize, n, delay = 2, 3, 2
    action_prefix = _action_prefix(bsize)

    fused = _sample(
        _flow_matching(family, cfg),
        family,
        lang_tokens=_lang_tokens(bsize),
        action_prefix=action_prefix,
        noise=_noise(bsize * n),
        delay=delay,
        n_candidates=n,
    )
    grid = collapse_candidates(fused, n)

    for b in range(bsize):
        for j in range(n):
            assert torch.equal(grid[b, j, :delay], action_prefix[b, :delay]), (
                f"observation {b} candidate {j} does not carry its own frozen prefix"
            )

    # Neither half of the assertion above is vacuous: the two observations carry *different*
    # prefixes, and the un-frozen rows do differ between candidates.
    assert not torch.equal(grid[0, 0, :delay], grid[1, 0, :delay])
    assert not torch.equal(grid[0, 0, delay:], grid[0, 1, delay:])


# --------------------------------------------------------------------------------------
# Policy wiring.
# --------------------------------------------------------------------------------------


def test_n_gt_1_without_a_critic_raises(family):
    """Best-of-N with nothing to choose with must fail loudly, naming the way in.

    Reachable only by hand-setting the attribute (``configure_candidates`` loads the critic
    before it writes ``n_candidates``), which is exactly why the message has to point at the
    supported route rather than describe the internal state.
    """
    policy = _policy(family)
    policy.n_candidates = 3

    with pytest.raises(ValueError, match="configure_candidates"):
        policy.sample_actions(_batch())


def test_a_saved_n4_config_does_not_self_arm_an_unconfigured_policy(family):
    """A checkpoint's ``config.json`` carries ``n_candidates``; loading it must not arm anything.

    ``n_candidates`` is serialized with the config, so a checkpoint fine-tuned from a
    best-of-N deployment travels with ``n_candidates: 4``. In-training validation and
    ``benchmark_inference`` build policies through ``make_policy``, which never calls
    ``configure_candidates`` — if the sampler read the config directly, those paths would
    silently fan out 4x, quadrupling the Euler loop's activation memory mid-training.
    """
    cfg = _config(family, n_candidates=4, action_chunk_critic_path="medoid")
    policy = _policy(family, cfg)
    critic = _RiggedCritic(pick=1)
    # Attached but never armed: only `configure_candidates` writes `n_candidates`.
    policy._action_chunk_critic = critic

    actions = policy.sample_actions(_batch(bsize=1))

    assert policy.config.n_candidates == 4, "the config really did ask for best-of-N"
    assert policy.n_candidates == 1
    assert critic.calls == 0
    assert actions.shape == (1, CHUNK, ACTION_DIM)


def test_wrapper_declares_n_candidates_as_one_in_init(family):
    """``__init__`` must set ``self.n_candidates = 1``, not read it off the config.

    Declaring it there is what makes ``configure_candidates``' ``hasattr`` probe meaningful —
    that probe is how every un-wired policy family refuses ``n_candidates > 1`` — and pinning
    the literal ``1`` is what the shells in this module mirror.
    """
    init = _method(_tree(family.rel_path), family.policy_cls.__name__, "__init__")
    assigned = _assignments_to(init, "n_candidates")

    assert assigned, f"{family.policy_cls.__name__}.__init__ must declare `self.n_candidates`"
    assert all(isinstance(node.value, ast.Constant) and node.value.value == 1 for node in assigned), (
        f"{family.policy_cls.__name__}.__init__ must initialize `self.n_candidates` to the "
        "literal 1; reading it from `config` would let a checkpoint self-arm best-of-N"
    )

    critic = _assignments_to(init, "_action_chunk_critic")
    assert critic and all(
        isinstance(node.value, ast.Constant) and node.value.value is None for node in critic
    ), f"{family.policy_cls.__name__}.__init__ must initialize `self._action_chunk_critic` to None"


@pytest.mark.parametrize("tie", [False, True], ids=["distinct-scores", "tied-scores"])
def test_selection_is_identical_across_simulated_process_indices(family, monkeypatch, tie):
    """The chosen candidate index must be a function of the scores alone, never of the RNG.

    ``set_seed(seed, accelerator=...)`` offsets the global torch RNG by
    ``process_index * 12345`` on purpose, so anything drawn from it is rank-dependent by
    construction. ``select_action`` runs on every rank and is followed by gather collectives,
    so a rank picking a different candidate desyncs the emitted actions with nothing raising.

    The tied case is the one that could break: it is where a tie-break implemented with
    ``torch.randint`` / ``torch.multinomial`` would diverge, while the lowest-index rule
    cannot. Noise is passed explicitly so the candidates themselves are identical across the
    simulated ranks and the selection is the only variable.
    """
    picked: list[list[int]] = []
    real_select = family.module.select_candidate

    def recording_select(scores):
        chosen = real_select(scores)
        picked.append(chosen.tolist())
        return chosen

    monkeypatch.setattr(family.module, "select_candidate", recording_select)

    noise = _noise(4)
    for process_index in range(4):
        set_seed(1000, accelerator=MagicMock(process_index=process_index, num_processes=4))
        policy = _policy(family)
        _arm(policy, 4, _RiggedCritic(pick=2, tie=tie))
        policy.sample_actions(_batch(bsize=1), noise=noise.clone())

    assert len(picked) == 4
    assert all(chosen == picked[0] for chosen in picked), f"selection differs across ranks: {picked}"
    # Candidate 0 carries the draw the N==1 path would have taken, so a degenerate critic
    # must degenerate to legacy behaviour rather than to something arbitrary.
    assert picked[0] == ([0] if tie else [2])


def test_last_accel_stays_length_b_and_equals_the_selected_candidate(family):
    """``last_accel`` must stay ``B``-long and describe the chunk that was actually emitted.

    The gRPC server reads ``float(policy.last_accel[0])`` with no width check, so a ``B * N``
    list would not raise — it would publish candidate 0's uncertainty alongside whatever
    chunk the critic picked, a wrong number attributed to the emitted action.
    """
    policy = _policy(family, accel_prefix=NUM_STEPS)
    pick = 2
    _arm(policy, 4, _RiggedCritic(pick=pick))

    policy.sample_actions(_batch(bsize=1), noise=_noise(4))

    assert len(policy.last_accel) == 1
    assert len(policy.last_accel_candidates) == 1
    assert len(policy.last_accel_candidates[0]) == 4
    assert policy.last_accel[0] == policy.last_accel_candidates[0][pick]
    # Not vacuous: the candidates score differently, so picking the wrong row would show.
    assert len(set(policy.last_accel_candidates[0])) > 1


def test_last_candidate_state_is_cleared_on_a_queue_pop_step(family):
    """A queue-pop step must read ``None``, exactly as ``last_accel`` already does.

    One ``sample_actions`` feeds ``n_action_steps`` env steps. Without the clear, a consumer
    logging the candidate scores every step records the same row ``n_action_steps`` times and
    reads it as ``n_action_steps`` independent best-of-N decisions.
    """
    policy = _policy(family, accel_prefix=NUM_STEPS)
    _arm(policy, 4, _RiggedCritic(pick=1))
    batch = _batch(bsize=1)

    policy.select_action(batch)  # re-plans: fills the queue and publishes
    assert policy.last_candidate_scores is not None
    assert policy.last_accel_candidates is not None

    policy.select_action(batch)  # pops from the still-full queue: nothing was re-planned
    assert policy.last_candidate_scores is None
    assert policy.last_accel_candidates is None
    assert policy.last_accel is None


def test_provenance_tuples_stay_row_aligned_at_n_gt_1(family):
    """The two per-sample tuples must both be ``B``-long, and the count must be recorded.

    ``build_provenance`` derives ``num_scored_dims`` from the meter (``B * N`` rows here)
    while taking ``dataset_index`` from its argument, so the two halves are trivially easy to
    slice to different widths — and a consumer zipping them would then attribute every score
    to the wrong sample. ``n_candidates`` belongs in ``COMPARABLE_FIELDS`` because best-of-N conditions the
    emitted chunk on a critic: a threshold calibrated at N=1 does not transfer.
    """
    bsize, n = 2, 3
    policy = _policy(family, accel_prefix=NUM_STEPS)
    _arm(policy, n, _RiggedCritic(pick=1))

    policy.sample_actions(_batch(bsize=bsize), noise=_noise(bsize * n))

    provenance = policy.last_accel_provenance
    assert len(provenance.num_scored_dims) == bsize
    assert len(provenance.dataset_index) == bsize
    assert provenance.n_candidates == n
    assert len(policy.last_accel) == bsize
    assert provenance.num_scored_dims == (ACTION_DIM,) * bsize


def test_b_row_noise_with_n_gt_1_raises_naming_both_shapes(family):
    """A ``B``-row noise at N>1 makes every candidate identical — best-of-N that silently is not.

    It would not raise on its own: the frozen-prefix ``torch.where`` broadcasts a 1-row
    ``x_t`` against the expanded ``action_prefix``, so the run completes and returns N copies
    of one chunk for the critic to "choose" between. The message must name both the expected
    row count and the shape it got, since the caller's bug is one or the other.
    """
    cfg = _config(family)
    bsize, n = 2, 4

    with pytest.raises(ValueError, match=rf"{bsize * n} rows"):
        _sample(
            _flow_matching(family, cfg),
            family,
            lang_tokens=_lang_tokens(bsize),
            noise=_noise(bsize),
            n_candidates=n,
        )

    with pytest.raises(ValueError, match=rf"\({bsize}, {CHUNK}, {MAX_ACTION_DIM}\)"):
        _sample(
            _flow_matching(family, cfg),
            family,
            lang_tokens=_lang_tokens(bsize),
            noise=_noise(bsize),
            n_candidates=n,
        )


def test_b_row_noise_at_n1_keeps_todays_behaviour(family):
    """The guard is deliberately NOT extended to N==1, mismatched rows included.

    Today a 1-row noise against a 2-row observation broadcasts silently. That is not obviously
    good, but the ONNX export wrapper and ``diagnose_accel`` pass explicit noise through this
    path, and widening the check into an unconditional assert would change N==1 behaviour —
    the one thing this feature promises not to touch. Fixing it is a separate decision.
    """
    cfg = _config(family)
    bsize = 2

    actions = _sample(
        _flow_matching(family, cfg),
        family,
        lang_tokens=_lang_tokens(bsize),
        noise=_noise(1),
        n_candidates=1,
    )

    assert actions.shape == (bsize, CHUNK, MAX_ACTION_DIM)


# --------------------------------------------------------------------------------------
# AST helpers, shared with the structural pins above.
# --------------------------------------------------------------------------------------

_POLICIES_ROOT = Path(opentau.__file__).parent / "policies"


def _tree(rel_path: str) -> ast.Module:
    return ast.parse((_POLICIES_ROOT / rel_path).read_text())


def _method(tree: ast.Module, class_name: str, method: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == method:
                    return child
    raise AssertionError(f"{class_name}.{method} not found")


def _assignments_to(node: ast.AST, attr: str) -> list[ast.Assign | ast.AnnAssign]:
    """Every ``self.<attr> = ...`` (annotated or not) inside ``node``."""
    return [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign | ast.AnnAssign)
        for target in ([stmt.target] if isinstance(stmt, ast.AnnAssign) else stmt.targets)
        if isinstance(target, ast.Attribute) and target.attr == attr
    ]


def test_every_wrapper_declaring_n_candidates_threads_it_into_its_sampler():
    """A wrapper that exposes the attribute must actually pass it down.

    ``configure_candidates`` refuses an un-wired family by probing
    ``hasattr(policy, "n_candidates")``, so declaring the attribute is what tells it a family
    is ready. A wrapper that declares it and then never forwards it slips past that probe and
    produces the exact failure the probe exists to prevent: the operator sets
    ``n_candidates=4``, nothing errors, and one chunk is sampled forever.

    Derived by scanning the tree rather than from a hand-kept list, so a family wired in a
    later PR is covered the moment it lands (CLAUDE.md rule 5: two hand-maintained lists fail
    open).
    """
    offenders = []
    scanned = set()
    for path in _POLICIES_ROOT.rglob("modeling_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            init = next(
                (c for c in node.body if isinstance(c, ast.FunctionDef) and c.name == "__init__"), None
            )
            if init is None or not _assignments_to(init, "n_candidates"):
                continue
            scanned.add(node.name)
            sampler = next(
                (c for c in node.body if isinstance(c, ast.FunctionDef) and c.name == "sample_actions"),
                None,
            )
            forwards = sampler is not None and any(
                any(kw.arg == "n_candidates" for kw in call.keywords)
                for call in ast.walk(sampler)
                if isinstance(call, ast.Call)
            )
            if not forwards:
                offenders.append(f"{path.relative_to(_POLICIES_ROOT)}::{node.name}")

    # A scan that matches nothing passes for free, so anchor it on the two families PR 1
    # wires; if the AST shapes above stop matching, this fails instead of going quiet.
    assert {"PI05Policy", "PI06Policy"} <= scanned, f"the scan matched nothing useful: {sorted(scanned)}"
    assert not offenders, (
        "These policy wrappers declare `self.n_candidates` — which is how "
        "`configure_candidates` decides a family is wired — but never forward it to their "
        f"inner sampler, so best-of-N would silently sample one chunk: {offenders}"
    )
