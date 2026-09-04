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

    def __init__(self, episodes, scenes=None, ragged=None):
        self._episodes = list(episodes)
        self._ragged = ragged
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
        return {
            "state": torch.full((n, 3), float(e)),
            "camera0": torch.full((n, 3, 2, 2), float(e)),
            "loss_mask": torch.ones(n, dtype=torch.bool),
            "prompt": f"open the {'left' if e % 2 else 'right'} drawer",
            "episode_index": torch.tensor(e),
            "dataset_index": torch.tensor(0),
        }

    def __len__(self):
        return len(self._episodes)


def _make(episodes=EPISODES, scenes=None, ragged=None, **kw):
    base = _StubBase(episodes, scenes, ragged)
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
        key, a, b = ds._draw(0)
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
            key, a, b = ds._draw(i)
            assert {a, b} <= set(ds.pairing_keys[key])


class TestDeterminism:
    def test_same_index_same_pair(self):
        """Ranks must agree, and a resume must reproduce the run."""
        assert _make()._draw(7) == _make()._draw(7)

    def test_different_indices_differ(self):
        draws = {_make()._draw(i)[1:] for i in range(40)}
        assert len(draws) > 1, "sampler is returning one pair for every index"


class TestSceneConstraint:
    def test_same_scene_pairs_are_avoided(self):
        """Two episodes of one scene can be solved by copying A's motion."""
        ds = _make(episodes=(1, 2, 3, 4), scenes={1: "k1", 2: "k1", 3: "k2", 4: "k3"})
        same = sum(1 for i in range(60) if ds._scene_of(ds._draw(i)[1]) == ds._scene_of(ds._draw(i)[2]))
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
        with pytest.raises(ValueError, match="pair_episodes doubles"):
            cfg.validate()

    def test_validate_leaves_unpaired_configs_alone(self, tmp_path):
        """The doubling must not leak into configs that do not pair."""
        self._cfg(policy_seq=T, mixture_seq=T, pair=False, tmp_path=tmp_path).validate()
        cfg = self._cfg(policy_seq=2 * T, mixture_seq=T, pair=False, tmp_path=tmp_path)
        with pytest.raises(ValueError, match="!= emitted timesteps"):
            cfg.validate()

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
        cfg = SimpleNamespace(dataset_mixture=SimpleNamespace(pair_episodes=True), seed=0)
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
