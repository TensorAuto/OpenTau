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

"""Pairing contract for one-shot in-context imitation.

Every property here is one the training run cannot check for itself. A pair with
the wrong mask boundary, or with the target named in the prompt, produces a run
that converges and reports a good number while proving nothing — which is the
same failure mode the alpha=0 ablation had, caught only because something
constructed the breaking case first.

CPU-only: the base dataset is stubbed, so none of this touches the Hub.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from opentau.datasets.paired_sequence import PairedSequenceDataset

T = 4  # timesteps per half
EPISODES = (10, 11, 12)  # default stub episodes for one pairing key


class _StubBase:
    """Minimal stand-in for ``LeRobotDataset``.

    Emits one distinguishable ``T``-timestep window per episode so a
    concatenation can be checked against its inputs.

    Args:
        episodes: Episode indices to serve.
        scenes: Optional ``{episode: scene_id}``.
        ragged: Episode whose window is one timestep short, to exercise the
            mismatch guard.
    """

    def __init__(self, episodes, scenes=None, ragged=None, timestep_pad=False):
        self._episodes = list(episodes)
        self._ragged = ragged
        self._timestep_pad = timestep_pad
        rows = {e: i for i, e in enumerate(self._episodes)}
        self.epi2idx = rows
        self.episode_data_index = {
            "from": torch.tensor([rows[e] for e in self._episodes]),
            "to": torch.tensor([rows[e] + 1 for e in self._episodes]),
        }
        self.meta = SimpleNamespace(
            episodes={e: ({"scene_id": scenes[e]} if scenes else {}) for e in self._episodes}
        )
        self._by_row = {rows[e]: e for e in self._episodes}

    def __getitem__(self, row):
        e = self._by_row[row]
        n = T - 1 if e == self._ragged else T
        item = {
            "state": torch.full((n, 3), float(e)),
            "camera0": torch.full((n, 3, 2, 2), float(e)),
            "loss_mask": torch.ones(n, dtype=torch.bool),
            "prompt": f"open the {'left' if e % 2 else 'right'} drawer",
            "episode_index": torch.tensor(e),
            "dataset_index": torch.tensor(0),
        }
        if self._timestep_pad:
            item["timestep_is_pad"] = torch.zeros(n, dtype=torch.bool)
        return item

    def __len__(self):
        return len(self._episodes)


def _make(episodes=EPISODES, scenes=None, ragged=None, timestep_pad=False, **kw):
    base = _StubBase(episodes, scenes, ragged, timestep_pad)
    return PairedSequenceDataset(
        base=base,
        pairing_keys={"OpenDrawer__left": list(episodes)},
        prompts={"OpenDrawer__left": "Open the drawer."},
        samples_per_epoch=64,
        **kw,
    )


class TestConcatenation:
    def test_sequence_is_both_halves(self):
        s = _make()[0]
        assert s["state"].shape[0] == 2 * T
        assert s["camera0"].shape[0] == 2 * T

    def test_halves_come_from_different_episodes(self):
        """The whole method collapses if a pair is one episode twice."""
        s = _make()[0]
        first, second = s["state"][0, 0].item(), s["state"][T, 0].item()
        assert first != second

    def test_a_precedes_b(self):
        """Order matters: the demonstration must be the part that is masked."""
        ds = _make()
        key, demos, b = ds._draw(0)
        (a,) = demos  # n_demos=1
        s = ds[0]
        assert s["state"][0, 0].item() == float(a)
        assert s["state"][T, 0].item() == float(b)


class TestLossMask:
    def test_boundary_is_exact(self):
        """Off by one here silently supervises a demo frame, or drops a real one."""
        m = _make()[0]["loss_mask"]
        assert m.shape == (2 * T,)
        assert not m[:T].any(), "demonstration half must carry no target"
        assert m[T:].all(), "rollout half must be fully supervised"

    def test_mask_overrides_whatever_the_base_emitted(self):
        """The stub returns all-True; the pair mask must win."""
        assert not _make()[0]["loss_mask"][0]


class TestPromptIsStripped:
    def test_prompt_is_the_ambiguous_one_on_both_halves(self):
        """The breaking case: leave the base prompt and the demo is redundant.

        The stub's episodes say "open the left drawer" / "open the right
        drawer". If either survives into the sample, the instruction names the
        target and the model never needs the demonstration.
        """
        s = _make()[0]
        assert s["prompt"] == "Open the drawer."
        assert "left" not in s["prompt"] and "right" not in s["prompt"]


class TestSameKey:
    def test_pair_never_crosses_keys(self):
        """A cross-key pair teaches the demo to predict the wrong thing."""
        base = _StubBase([1, 2, 3, 4])
        ds = PairedSequenceDataset(
            base=base,
            pairing_keys={"left": [1, 2], "right": [3, 4]},
            prompts={"left": "Open the drawer.", "right": "Open the drawer."},
            samples_per_epoch=64,
        )
        for i in range(40):
            key, demos, b = ds._draw(i)
            (a,) = demos  # n_demos=1
            assert {a, b} <= set(ds.pairing_keys[key])


class TestDeterminism:
    def test_same_index_same_pair(self):
        """Ranks must agree, and a resume must reproduce the run."""
        assert _make()._draw(7) == _make()._draw(7)

    def test_different_indices_differ(self):
        draws = {(tuple(d), b) for _, d, b in (_make()._draw(i) for i in range(40))}
        assert len(draws) > 1, "sampler is returning one pair for every index"


class TestSceneConstraint:
    def test_same_scene_pairs_are_avoided(self):
        """Two episodes of one scene can be solved by copying A's motion."""
        ds = _make(episodes=(1, 2, 3, 4), scenes={1: "k1", 2: "k1", 3: "k2", 4: "k3"})
        same = sum(1 for i in range(60) if ds._scene_of(ds._draw(i)[1][0]) == ds._scene_of(ds._draw(i)[2]))
        assert same == 0

    def test_missing_scene_metadata_does_not_reject_everything(self):
        """Absent scene ids must degrade to a no-op, not an empty epoch."""
        ds = _make(episodes=(1, 2, 3))
        assert ds[0]["state"].shape[0] == 2 * T


class TestGuards:
    def test_thin_key_is_refused(self):
        with pytest.raises(ValueError, match="too thin"):
            PairedSequenceDataset(base=_StubBase([1]), pairing_keys={"k": [1]}, prompts={"k": "x"})

    def test_missing_prompt_is_refused(self):
        """Without this the base prompt leaks the answer silently."""
        with pytest.raises(ValueError, match="no ambiguous prompt"):
            PairedSequenceDataset(base=_StubBase([1, 2]), pairing_keys={"k": [1, 2]}, prompts={})

    def test_ragged_halves_raise(self):
        """A short half would slide the mask against the sequence."""
        ds = _make(episodes=(10, 11, 12), ragged=12)
        with pytest.raises(ValueError, match="disagree on timestep count"):
            for i in range(40):
                ds[i]

    def test_episode_absent_from_base_is_named(self):
        with pytest.raises(KeyError, match="diverged"):
            PairedSequenceDataset(base=_StubBase([1, 2]), pairing_keys={"k": [1, 2, 99]}, prompts={"k": "x"})

    def test_non_sequence_base_is_named(self):
        with pytest.raises(ValueError, match="sequence_length"):
            PairedSequenceDataset._timesteps({"prompt": "x"})


class TestScalarPassthrough:
    def test_scalars_are_not_concatenated(self):
        """``dataset_index`` routes normalization; a doubled one would misroute."""
        s = _make()[0]
        assert s["dataset_index"].ndim == 0


class TestNonSequenceBaseIsRejected:
    """A `sequence_length <= 1` base has no timestep axis to concatenate along.

    The earlier guard read the timestep count from whichever of
    `loss_mask`/`state`/`actions`/`camera0` came first. At `sequence_length == 1`
    the base emits no `loss_mask` and `state` is `(max_state_dim,)`, so it
    returned the *feature* dim: both halves agreed, nothing raised, and the pair
    was concatenated along the feature axis. These pin the two independent places
    that now stop it -- the loader itself, and the config before a run starts.
    """

    class _Seq1Base:
        """A base dataset at `sequence_length == 1`: no `loss_mask`, flat `state`."""

        def __init__(self, episodes):
            self._episodes = list(episodes)
            rows = {e: i for i, e in enumerate(self._episodes)}
            self.epi2idx = rows
            self.episode_data_index = {
                "from": torch.tensor([rows[e] for e in self._episodes]),
                "to": torch.tensor([rows[e] + 1 for e in self._episodes]),
            }
            self.meta = SimpleNamespace(episodes={e: {} for e in self._episodes})
            self._by_row = {v: k for k, v in rows.items()}

        def __len__(self):
            return len(self._episodes)

        def __getitem__(self, row):
            e = self._by_row[row]
            return {
                "state": torch.full((32,), float(e)),  # (max_state_dim,) -- no time axis
                "actions": torch.full((50, 7), float(e)),
                "prompt": "x",
                "episode_index": torch.tensor(e),
                "dataset_index": torch.tensor(0),
            }

    def test_loader_raises_instead_of_corrupting(self):
        ds = PairedSequenceDataset(
            base=self._Seq1Base(EPISODES),
            pairing_keys={"k": list(EPISODES)},
            prompts={"k": None},
            samples_per_epoch=4,
            seed=0,
        )
        with pytest.raises(ValueError, match="no `loss_mask`"):
            ds[0]

    def test_validate_rejects_unit_sequence_length(self, tmp_path):
        """The config path must reject it too.

        A policy with no `sequence_length` field skips the doubled-length check,
        so without this the corrupted run starts and converges on nonsense.
        """
        cfg = TestPairingIsActuallyWired._cfg(policy_seq=2, mixture_seq=1, pair=True, tmp_path=tmp_path)
        with pytest.raises(ValueError, match="sequence_length=1"):
            cfg.validate()


class TestOnlyTemporalKeysConcatenate:
    """Fixed-size tensors must not be doubled when their leading dim equals T.

    `img_is_pad` is `(num_cams,)` and a `subgoal` image is `(3, H, W)`, so the
    old "leading dim == timestep count" test doubled both at `sequence_length`
    3 with 3 cameras -- silently, since collation still succeeded.
    """

    class _CollidingBase(_StubBase):
        """Adds fixed-size keys whose leading dim happens to equal ``T``."""

        def __getitem__(self, row):
            item = super().__getitem__(row)
            item["img_is_pad"] = torch.zeros(T, dtype=torch.bool)  # (num_cams,) == T
            item["subgoal0"] = torch.zeros(T, 8, 8)  # (3, H, W) with T == 3
            return item

    @classmethod
    def _base_with_collisions(cls):
        return cls._CollidingBase(EPISODES)

    def test_colliding_fixed_size_keys_are_not_doubled(self):
        ds = PairedSequenceDataset(
            base=self._base_with_collisions(),
            pairing_keys={"k": list(EPISODES)},
            prompts={"k": None},
            samples_per_epoch=4,
            seed=0,
        )
        sample = ds[0]
        assert sample["state"].shape[0] == 2 * T, "temporal keys must still concatenate"
        assert sample["img_is_pad"].shape[0] == T, "img_is_pad is per-camera, not per-timestep"
        assert sample["subgoal0"].shape[0] == T, "a subgoal image has no timestep axis"

    def test_collision_is_reported_once(self, caplog):
        ds = PairedSequenceDataset(
            base=self._base_with_collisions(),
            pairing_keys={"k": list(EPISODES)},
            prompts={"k": None},
            samples_per_epoch=4,
            seed=0,
        )
        with caplog.at_level(logging.WARNING):
            ds[0]
            ds[1]
        assert caplog.text.count("temporal allowlist") == 2, "one warning per colliding key, once each"
        assert "img_is_pad" in caplog.text and "subgoal0" in caplog.text


class TestMultipleDemonstrations:
    """``n_demos`` places more than one demonstration in context.

    The one-shot recipe concatenated one demo with one rollout. ``n_demos``
    generalises that to N demos, which is what makes a 2-shot or 3-shot
    *training* run possible -- and the eval harness can then prime with the same
    number, so train and test finally agree on how much context the policy sees.
    """

    WIDE = (10, 11, 12, 13, 14)

    def _ds(self, n_demos, episodes=None, **kw):
        eps = list(episodes or self.WIDE)
        return PairedSequenceDataset(
            base=_StubBase(eps),
            pairing_keys={"k": eps},
            prompts={"k": None},
            samples_per_epoch=16,
            seed=0,
            n_demos=n_demos,
            **kw,
        )

    @pytest.mark.parametrize("n_demos", [1, 2, 3])
    def test_sequence_scales_with_demo_count(self, n_demos):
        """The sample is ``(n_demos + 1)`` segments, not always two."""
        sample = self._ds(n_demos)[0]
        assert sample["state"].shape[0] == (n_demos + 1) * T

    @pytest.mark.parametrize("n_demos", [1, 2, 3])
    def test_only_the_rollout_is_supervised(self, n_demos):
        """Every demonstration is masked; exactly one rollout carries the loss.

        The whole method rests on this: if a demo were supervised the model could
        lower the loss by copying it rather than by absorbing it.
        """
        lm = self._ds(n_demos)[0]["loss_mask"]
        assert int((~lm).sum()) == n_demos * T, "demo timesteps must be masked"
        assert int(lm.sum()) == T, "exactly one rollout's worth may be supervised"
        assert not lm[: n_demos * T].any(), "mask is not contiguous demos-then-rollout"
        assert lm[n_demos * T :].all()

    @pytest.mark.parametrize("n_demos", [1, 2, 3])
    def test_every_episode_in_the_tuple_is_distinct(self, n_demos):
        """A repeated episode would make one demonstration redundant."""
        ds = self._ds(n_demos)
        for i in range(16):
            _, demos, rollout = ds._draw(i)
            assert len(demos) == n_demos
            assert len(set(demos) | {rollout}) == n_demos + 1

    def test_n_demos_one_is_unchanged(self):
        """The default must reproduce the pair behaviour bit-for-bit.

        Runs made before n-tuples existed have to stay reproducible, so the
        ``n_demos == 1`` draw deliberately keeps the original sampling path.
        """
        ds = self._ds(1)
        for i in (0, 3, 9):
            _, demos, rollout = ds._draw(i)
            assert len(demos) == 1 and demos[0] != rollout
        assert torch.equal(self._ds(1)[5]["state"], self._ds(1)[5]["state"])

    def test_thin_key_is_rejected_with_the_count_it_needs(self):
        """Three episodes cannot supply two demos plus a distinct rollout."""
        with pytest.raises(ValueError, match=r">=4 episodes for 3 demo"):
            self._ds(3, episodes=(10, 11, 12))

    def test_zero_demos_is_rejected(self):
        with pytest.raises(ValueError, match="n_demos must be >= 1"):
            self._ds(0)


class TestPairingIsActuallyWired:
    """The pairing must reach ``train.py``, not just exist as a class.

    The first smoke run failed on a config guard, and satisfying that guard
    naively would have started a run that trained on single windows with no
    pairing at all — converging, looking healthy, and proving nothing. These
    pin the two places that connect the loader to the training path.
    """

    @staticmethod
    def _cfg(policy_seq, mixture_seq, pair, val_freq=0, tmp_path=None, prompt="Open the drawer."):
        """Builds a real ``TrainPipelineConfig`` so ``validate()`` runs for real."""
        from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
        from opentau.configs.train import TrainPipelineConfig
        from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig

        return TrainPipelineConfig(
            dataset_mixture=DatasetMixtureConfig(
                datasets=[
                    DatasetConfig(
                        repo_id="mock",
                        root="/tmp/mock",
                        episodes=list(EPISODES),
                        ambiguous_prompt=prompt,
                    )
                ],
                weights=[1.0],
                action_freq=30.0,
                sequence_length=mixture_seq,
                pair_episodes=pair,
            ),
            policy=PI05TTTConfig(sequence_length=policy_seq, tbptt_segment_length=policy_seq),
            output_dir=str(tmp_path),
            job_name="test_run",
            seed=42,
            batch_size=8,
            val_freq=val_freq,
            use_policy_training_preset=True,
        )

    def test_validate_accepts_doubled_policy_length_when_pairing(self, tmp_path):
        """Pairing on, policy sized for the concatenation: must pass."""
        self._cfg(policy_seq=2 * T, mixture_seq=T, pair=True, tmp_path=tmp_path).validate()

    def test_validate_rejects_undoubled_policy_length(self, tmp_path):
        """The breaking case: pairing on, policy still expecting one half."""
        cfg = self._cfg(policy_seq=T, mixture_seq=T, pair=True, tmp_path=tmp_path)
        with pytest.raises(ValueError, match="pair_episodes with n_demos"):
            cfg.validate()

    def test_validate_leaves_unpaired_configs_alone(self, tmp_path):
        """The doubling must not leak into configs that do not pair."""
        self._cfg(policy_seq=T, mixture_seq=T, pair=False, tmp_path=tmp_path).validate()
        cfg = self._cfg(policy_seq=2 * T, mixture_seq=T, pair=False, tmp_path=tmp_path)
        with pytest.raises(ValueError, match="!= emitted timesteps"):
            cfg.validate()

    def test_validate_scales_policy_length_by_demo_count(self, tmp_path):
        """`validate()` must expect `(n_demos + 1) x sequence_length`, not 2x.

        Without this a correct 2-shot config is rejected at parse time, and --
        worse -- a config still sized for one demo is silently accepted.
        """
        from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
        from opentau.configs.train import TrainPipelineConfig
        from opentau.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig

        def build(policy_seq, n_demos):
            return TrainPipelineConfig(
                dataset_mixture=DatasetMixtureConfig(
                    datasets=[
                        DatasetConfig(
                            repo_id="mock",
                            root="/tmp/mock",
                            episodes=list(range(n_demos + 1)),
                            ambiguous_prompt="x",
                        )
                    ],
                    weights=[1.0],
                    action_freq=30.0,
                    sequence_length=T,
                    pair_episodes=True,
                    n_demos=n_demos,
                ),
                policy=PI05TTTConfig(sequence_length=policy_seq, tbptt_segment_length=T),
                output_dir=str(tmp_path),
                job_name="t",
                seed=42,
                batch_size=8,
                use_policy_training_preset=True,
            )

        build(policy_seq=3 * T, n_demos=2).validate()  # 2 demos + rollout
        build(policy_seq=4 * T, n_demos=3).validate()  # 3 demos + rollout
        with pytest.raises(ValueError, match="pair_episodes with n_demos=2"):
            build(policy_seq=2 * T, n_demos=2).validate()  # still sized for one demo

    def test_validate_rejects_val_freq_with_pairing(self, tmp_path):
        """`val_freq` and pairing are incompatible, and silently so without this.

        `random_split` partitions frames, not episodes, so both halves keep every
        episode: a paired val subset would draw its pairs from episodes whose
        frames are in the training half. It also cannot be constructed, since the
        `Subset` exposes none of the episode-level attributes the loader indexes
        with. Rejecting at config time beats an AttributeError after the model has
        loaded -- or, worse, a val curve that means nothing.
        """
        cfg = self._cfg(policy_seq=2 * T, mixture_seq=T, pair=True, val_freq=100, tmp_path=tmp_path)
        with pytest.raises(ValueError, match="pair_episodes is on with val_freq"):
            cfg.validate()

        # ...and stays out of the way when pairing is off.
        self._cfg(policy_seq=T, mixture_seq=T, pair=False, val_freq=100, tmp_path=tmp_path).validate()

    def test_factory_emits_paired_samples(self, tmp_path, monkeypatch):
        """`make_dataset_mixture` must hand the mixture a *paired* dataset.

        Runs the real factory path with only the base-dataset build and the
        metadata-heavy mixture constructor stubbed, then inspects what the
        mixture was actually handed: a `PairedSequenceDataset` emitting
        `2 * sequence_length` timesteps with the ambiguous prompt applied. A
        source-level check that `_maybe_pair` is *called* would still pass if it
        returned the dataset unwrapped.
        """
        from opentau.datasets import factory

        cfg = self._cfg(policy_seq=2 * T, mixture_seq=T, pair=True, tmp_path=tmp_path)
        handed = {}

        monkeypatch.setattr(factory, "make_dataset", lambda dcfg, c, **kw: _StubBase(EPISODES))
        monkeypatch.setattr(factory, "_validate_metadata_requirements", lambda *a, **k: None)
        monkeypatch.setattr(
            factory,
            "WeightedDatasetMixture",
            lambda cfg_, datasets, weights, freq: handed.setdefault("datasets", datasets),
        )

        factory.make_dataset_mixture(cfg)
        wrapped = handed["datasets"][0]
        assert isinstance(wrapped, PairedSequenceDataset), (
            f"factory handed the mixture a {type(wrapped).__name__}; training would "
            "run on single unpaired windows from a paired config"
        )
        sample = wrapped[0]
        assert sample["state"].shape[0] == 2 * T, "paired sample is not the concatenation"
        assert sample["prompt"] == "Open the drawer."

    def test_empty_ambiguous_prompt_is_treated_as_absent(self, tmp_path, monkeypatch):
        """`ambiguous_prompt: ""` must mean "no override", consistently in both layers.

        The factory's falsy check already read it as absent and warned, but it
        passed the empty string down to a loader gating on `is not None` -- which
        blanked the prompt instead of leaving the rollout's in place, erasing the
        task identity in exactly the cross-task case the warning describes.
        """
        from opentau.datasets import factory

        cfg = self._cfg(policy_seq=2 * T, mixture_seq=T, pair=True, tmp_path=tmp_path, prompt="")
        handed = {}
        monkeypatch.setattr(factory, "make_dataset", lambda dcfg, c, **kw: _StubBase(EPISODES))
        monkeypatch.setattr(factory, "_validate_metadata_requirements", lambda *a, **k: None)
        monkeypatch.setattr(
            factory,
            "WeightedDatasetMixture",
            lambda cfg_, datasets, weights, freq: handed.setdefault("datasets", datasets),
        )

        factory.make_dataset_mixture(cfg)
        prompt = handed["datasets"][0][0]["prompt"]
        assert prompt, "an empty ambiguous_prompt blanked the sample's instruction"
        assert "drawer" in prompt

    def test_missing_ambiguous_prompt_warns_but_is_allowed(self, caplog):
        """No prompt is legitimate for cross-task transfer, so warn rather than refuse.

        Within-task ambiguity needs the prompt overwritten or the demonstration is
        redundant. Cross-task transfer needs it left alone or the task identity is
        erased. The loader cannot tell those apart, so it says so loudly and lets
        the caller own the choice.
        """
        from types import SimpleNamespace

        from opentau.datasets.factory import _maybe_pair

        ds_cfg = SimpleNamespace(repo_id="r", episodes=[1, 2], ambiguous_prompt=None)
        cfg = SimpleNamespace(dataset_mixture=SimpleNamespace(pair_episodes=True, n_demos=1), seed=0)
        # Proceeds past the prompt check and fails later on the stub dataset,
        # which is the point: the prompt no longer blocks it.
        with caplog.at_level(logging.WARNING), pytest.raises(TypeError):
            _maybe_pair(SimpleNamespace(episodes=[1, 2]), ds_cfg, cfg)
        assert "ambiguous_prompt" in caplog.text
        assert "cross-task transfer" in caplog.text

    def test_pairing_off_is_a_passthrough(self):
        from types import SimpleNamespace

        from opentau.datasets.factory import _maybe_pair

        sentinel = SimpleNamespace(episodes=[1, 2])
        cfg = SimpleNamespace(dataset_mixture=SimpleNamespace(pair_episodes=False), seed=0)
        assert _maybe_pair(sentinel, SimpleNamespace(), cfg) is sentinel


class TestZeroShotProb:
    """`zero_shot_prob` blanks a fraction of demonstrations so one run trains both modes."""

    def test_default_is_inert(self):
        """At the default 0.0 nothing is blanked, so existing runs are unaffected."""
        d = _make(timestep_pad=True)
        assert d.zero_shot_prob == 0.0
        assert not any(d._is_zero_shot(i) for i in range(500))

    def test_pair_draw_is_unperturbed_by_the_mode_draw(self):
        """The mode is drawn off a separate RNG stream, so pair draws are unchanged.

        This is what keeps runs made before this feature reproducible: enabling it
        must not shift which episodes a given index pairs.
        """
        off = _make(timestep_pad=True, zero_shot_prob=0.0)
        on = _make(timestep_pad=True, zero_shot_prob=0.5)
        assert [off._draw(i) for i in range(200)] == [on._draw(i) for i in range(200)]

    def test_rate_tracks_the_configured_probability(self):
        d = _make(timestep_pad=True, zero_shot_prob=0.5)
        hits = sum(d._is_zero_shot(i) for i in range(4000))
        assert 0.45 < hits / 4000 < 0.55

    def test_mode_is_deterministic_per_index(self):
        """Same index, same mode -- so ranks agree and a resume reproduces the run."""
        a = _make(timestep_pad=True, zero_shot_prob=0.5)
        b = _make(timestep_pad=True, zero_shot_prob=0.5)
        assert [a._is_zero_shot(i) for i in range(300)] == [b._is_zero_shot(i) for i in range(300)]

    def test_blanked_sample_zeroes_demo_half_and_marks_it_padded(self):
        """The demonstration half is erased AND padded; the rollout half survives."""
        d = _make(timestep_pad=True, zero_shot_prob=1.0)
        s = d[0]
        cam, tp, lm = s["camera0"], s["timestep_is_pad"], s["loss_mask"]
        assert torch.all(cam[:T] == 0), "demonstration frames were not blanked"
        assert not torch.all(cam[T:] == 0), "the rollout half must keep its content"
        assert torch.all(tp[:T]), "demo half must be marked padded so the TTT inner update skips it"
        assert not torch.any(lm[:T]), "loss_mask across the demonstration must stay False"

    def test_kept_sample_retains_a_real_demonstration(self):
        """A non-blanked sample is untouched -- otherwise 1-shot training is destroyed."""
        d = _make(timestep_pad=True, zero_shot_prob=0.0)
        s = d[0]
        assert not torch.all(s["camera0"][:T] == 0)
        assert not torch.any(s["timestep_is_pad"][:T])

    def test_rejects_an_out_of_range_probability(self):
        with pytest.raises(ValueError, match="zero_shot_prob"):
            _make(timestep_pad=True, zero_shot_prob=1.5)

    def test_refuses_to_blank_without_timestep_is_pad(self):
        """Without the pad flag the TTT inner loop would train on the blank frames.

        That is worse than either mode, so the loader refuses rather than degrading
        quietly -- and refuses at CONSTRUCTION, not on the first blanked sample,
        which at 0 < p < 1 would surface inside a dataloader worker mid-run.
        """
        with pytest.raises(ValueError, match="timestep_is_pad"):
            _make(timestep_pad=False, zero_shot_prob=1.0)

    def test_mode_seed_cannot_collide_with_the_pair_seed_at_seed_zero(self):
        """The mode stream must not be the ``seed * k + index`` stream at seed 0.

        The factory passes ``cfg.seed or 0``, so ``seed == 0`` is the default, and
        there every ``seed * multiplier + index`` collapses to plain ``index`` --
        the mode would then come off the same engine state as the episode choice,
        correlating which episodes a sample pairs with whether it is blanked.

        Pinned against the actual draw rather than the formula: re-deriving the
        arithmetic from copied multipliers would compare constants to themselves
        and still pass if the offset were dropped (CLAUDE.md rule 8). The marginal
        rate would not catch it either -- a collision changes correlation, not the
        fraction blanked.
        """
        d = _make(timestep_pad=True, zero_shot_prob=0.5, seed=0)

        # What `_is_zero_shot` WOULD produce if its seed collapsed to bare `index`,
        # i.e. the pre-offset implementation this guards against.
        collided = [
            bool(torch.rand((1,), generator=torch.Generator().manual_seed(i)).item() < 0.5)
            for i in range(256)
        ]
        actual = [d._is_zero_shot(i) for i in range(256)]
        assert actual != collided, "mode stream collapsed onto the pair-draw seed at seed 0"

    def test_mode_seed_offset_survives_torchs_32_bit_seed_truncation(self):
        """The offset must be < 2**32, or torch discards it and the fix is a no-op.

        Torch's CPU generator consumes only the low 32 bits of a seed, so e.g.
        ``manual_seed(i)`` and ``manual_seed(i + 2**40)`` yield the identical
        stream. Pinned on the generator itself rather than on the constant, so a
        revert to a too-large offset fails here.
        """
        from opentau.datasets.paired_sequence import _MODE_SEED_OFFSET

        def first(seed):
            return torch.rand((1,), generator=torch.Generator().manual_seed(seed)).item()

        assert first(7) != first(7 + _MODE_SEED_OFFSET), "offset discarded by seed truncation"

    def test_mode_seed_cannot_collide_for_any_seed(self):
        """Odd offset vs an even multiplier difference -- unsatisfiable.

        A collision needs ``seed * difference + offset = 0 (mod 2**32)``. With the
        difference even, that first term is even for every seed, so an odd offset
        admits no solution. The difference is derived from the source constants
        rather than copied, so changing a multiplier to one whose difference is odd
        fails here instead of passing against stale prose.
        """
        from opentau.datasets.paired_sequence import (
            _MODE_SEED_MULTIPLIER,
            _MODE_SEED_OFFSET,
            _PAIR_SEED_MULTIPLIER,
        )

        # Derived from the source constants, not a copy of their difference, so
        # changing either multiplier to one whose difference is odd fails here
        # rather than passing against a stale number.
        difference = _MODE_SEED_MULTIPLIER - _PAIR_SEED_MULTIPLIER
        assert _MODE_SEED_OFFSET % 2 == 1, "an even offset admits a colliding seed"
        assert difference % 2 == 0, "an odd multiplier difference voids the oddness argument"
        assert not [s for s in range(20000) if (s * difference + _MODE_SEED_OFFSET) % 2**32 == 0]


class TestZeroShotProbIsActuallyWired:
    """``zero_shot_prob`` must reach the dataset, not just exist on the config.

    Every ``TestZeroShotProb`` case builds ``PairedSequenceDataset`` directly, so
    dropping the kwarg in the factory would leave a configured value silently
    inert -- the fail-open class ``TestPairingIsActuallyWired`` exists to catch.
    """

    def test_factory_passes_zero_shot_prob_through(self, monkeypatch):
        from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
        from opentau.datasets import factory as ds_factory

        seen = {}

        class _Spy(PairedSequenceDataset):
            def __init__(self, *a, **kw):
                seen.update(kw)
                super().__init__(*a, **kw)

        monkeypatch.setattr("opentau.datasets.paired_sequence.PairedSequenceDataset", _Spy)

        base = _StubBase(EPISODES, timestep_pad=True)
        dataset_cfg = DatasetConfig(
            repo_id="mock", root="/tmp/mock", episodes=list(EPISODES), ambiguous_prompt="Open the drawer."
        )
        cfg = SimpleNamespace(
            dataset_mixture=DatasetMixtureConfig(
                datasets=[dataset_cfg], weights=[1.0], pair_episodes=True, n_demos=1, zero_shot_prob=0.25
            ),
            seed=7,
        )
        ds_factory._maybe_pair(base, dataset_cfg, cfg)
        assert seen.get("zero_shot_prob") == 0.25, "the factory dropped zero_shot_prob"


class TestZeroShotPromptStaysInformative:
    """A blanked sample keeps a task-identifying prompt.

    ``ambiguous_prompt`` exists because the demonstration carries the answer. With
    the demonstration erased, overwriting the prompt too would leave the sample
    with neither and the target unlearnable.
    """

    def test_blanked_sample_keeps_the_rollouts_own_prompt(self):
        d = _make(timestep_pad=True, zero_shot_prob=1.0)
        assert d[0]["prompt"] != "Open the drawer."

    def test_kept_sample_still_gets_the_ambiguous_prompt(self):
        d = _make(timestep_pad=True, zero_shot_prob=0.0)
        assert d[0]["prompt"] == "Open the drawer."
