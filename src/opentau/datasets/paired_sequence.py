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

"""Two episodes of one task, joined into a single supervised sequence.

This is the data side of one-shot in-context imitation. A sample is episode A
(the demonstration) followed by episode B (the rollout), concatenated along the
timestep axis, with ``loss_mask`` zero across A and one across B. A's timesteps
still update the TTT fast weights; only B's carry an imitation target. The
gradient on B can therefore only fall by having absorbed something from A.

**Nothing is written to disk.** The pair exists as a tensor for one training
step. Materializing ~146k pairs at 196 timesteps and 3 cameras would be on the
order of 100 TB, and would freeze one arbitrary draw of a pair space the sampler
otherwise re-draws every epoch.

**Why this cannot be a config flag on the existing loader.**
:class:`~opentau.datasets.lerobot_dataset.LeRobotDataset` builds a window from
``delta_timestamps``, which clamps at episode boundaries by design — the padding
behaviour that ``_reshape_to_sequence`` reports through ``*_is_pad``. A pair
spans two *different* episodes, so it cannot be expressed as one window however
the offsets are written. This class calls the base dataset twice instead, which
keeps every downstream guarantee (image standardization, action padding, the
sequence reshape) rather than re-deriving them.

**Why the prompt is overwritten.** The pairing key is the thing the prompt must
hide. For ``OpenDrawer`` the episodes say "open the left drawer" / "open the
right drawer"; the sample says "Open the drawer." on *both* halves. If the
instruction named the target, the demonstration would be redundant and the model
would learn to ignore it — the failure this whole design exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

from opentau.datasets.lerobot_dataset import LeRobotDataset


class PairedSequenceDataset(Dataset):
    """Draws two episodes of one pairing key and returns them as one sequence.

    Args:
        base: Dataset to read windows from. Its ``sequence_length`` must be the
            per-half length, i.e. half the sequence the policy will see.
        pairing_keys: ``{key: [episode_index, ...]}``. A pair is always drawn
            within a single key, so the key is exactly what the demonstration
            conveys and the prompt withholds.
        prompts: ``{key: instruction}`` with the varying slot removed.
        samples_per_epoch: Length reported to the sampler. Pairs are drawn on
            demand, so this sets epoch size rather than the size of anything
            stored.
        seed: Base seed for pair draws. Combined with the sample index so a
            given index yields the same pair on every rank and every epoch,
            which keeps distributed runs and resumes reproducible.
        forbid_same_scene: Reject a draw whose two episodes share a scene id, so
            the pair cannot be solved by copying A's motion frame-for-frame.

    Raises:
        ValueError: If a key has fewer than two episodes, or ``prompts`` is
            missing a key.
    """

    #: Keys carrying no timestep axis; taken from B, the supervised half.
    _SCALAR_PASSTHROUGH = frozenset(
        {
            "dataset_index",
            "dataset_repo_id",
            "robot_type",
            "control_mode",
            "episode_index",
            "frame_index",
            "real_action_dim",
        }
    )

    def __init__(
        self,
        base: LeRobotDataset,
        pairing_keys: dict[str, list[int]],
        prompts: dict[str, str],
        samples_per_epoch: int = 100_000,
        seed: int = 0,
        forbid_same_scene: bool = True,
    ) -> None:
        thin = {k: len(v) for k, v in pairing_keys.items() if len(v) < 2}
        if thin:
            raise ValueError(f"pairing keys need >=2 episodes to pair; too thin: {thin}")
        missing = sorted(set(pairing_keys) - set(prompts))
        if missing:
            raise ValueError(f"no ambiguous prompt supplied for keys: {missing}")

        self.base = base
        self.keys = sorted(pairing_keys)
        self.pairing_keys = {k: list(v) for k, v in pairing_keys.items()}
        self.prompts = prompts
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.forbid_same_scene = forbid_same_scene

        # Row index of each episode's final frame. The base loader anchors a
        # window at its last timestep, so anchoring here makes each half cover
        # its episode end-to-end rather than an arbitrary interior slice.
        self._last_row = self._build_last_row_index()

        # A scene filter that can never be satisfied is worse than none: every
        # draw falls through to the fallback. Say so, loudly, at construction.
        if self.forbid_same_scene:
            degenerate = [
                k for k, eps in self.pairing_keys.items() if len({self._scene_of(e) for e in eps}) <= 1
            ]
            if degenerate:
                logging.warning(
                    "forbid_same_scene is on, but %d/%d pairing key(s) have a single "
                    "scene id across every episode, so the filter can never be "
                    "satisfied and each draw falls back to a random distinct pair: %s. "
                    "The scene constraint is doing nothing for these keys — set "
                    "forbid_same_scene=False, or supply a real per-episode scene id.",
                    len(degenerate),
                    len(self.pairing_keys),
                    degenerate[:6],
                )

        # The pair count logged below is an UPPER BOUND. A filter or fallback bug
        # can silently collapse the draws onto a handful of pairs while that
        # number stays reassuringly large -- which is exactly what happened
        # (plan doc 11.30: two runs trained on one pair per key). Measure what
        # `_draw` actually produces.
        probe = min(1000, max(200, 8 * len(self.keys)))
        drawn = {self._draw(i) for i in range(probe)}
        reachable = min(probe, sum(len(v) * (len(v) - 1) for v in self.pairing_keys.values()))
        if len(drawn) < 0.25 * reachable:
            logging.error(
                "PAIR DIVERSITY COLLAPSE: %d probe draws produced only %d distinct "
                "(key, demo, rollout) triples, against %d reachable. The model will "
                "see far fewer examples than the pair count below implies. Check "
                "forbid_same_scene and the _draw fallback before training.",
                probe,
                len(drawn),
                reachable,
            )
        else:
            logging.info("pair diversity ok: %d distinct triples from %d probe draws", len(drawn), probe)

        total = sum(len(v) * (len(v) - 1) for v in self.pairing_keys.values())
        logging.info(
            "PairedSequenceDataset: %d keys, %d episodes, %d ordered pairs available, %d samples/epoch",
            len(self.keys),
            sum(len(v) for v in self.pairing_keys.values()),
            total,
            samples_per_epoch,
        )

    def _build_last_row_index(self) -> dict[int, int]:
        """Maps episode index to the row index of its last frame.

        Returns:
            ``{episode_index: row}``.

        Raises:
            RuntimeError: If the base dataset has not built its episode index.
            KeyError: If a requested episode is absent from the base dataset.
        """
        if self.base.episode_data_index is None or self.base.epi2idx is None:
            raise RuntimeError("base dataset has no episode_data_index; construct it before wrapping")
        out: dict[int, int] = {}
        for eps in self.pairing_keys.values():
            for e in eps:
                if e not in self.base.epi2idx:
                    raise KeyError(
                        f"episode {e} is in the manifest but not in the base dataset — "
                        "the config's `episodes` list and the manifest have diverged"
                    )
                out[e] = int(self.base.episode_data_index["to"][self.base.epi2idx[e]].item()) - 1
        return out

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _draw(self, index: int) -> tuple[str, int, int]:
        """Chooses a key and two distinct episodes for one sample.

        Derived from ``seed`` and ``index`` alone — never from global RNG state —
        so every rank draws the same pair for the same index and a resume
        reproduces the run.

        Args:
            index: Sample index.

        Returns:
            ``(key, episode_a, episode_b)``.
        """
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        key = self.keys[int(torch.randint(len(self.keys), (1,), generator=g))]
        eps = self.pairing_keys[key]

        for _ in range(16):
            i, j = torch.randint(len(eps), (2,), generator=g).tolist()
            if i == j:
                continue
            a, b = eps[i], eps[j]
            if not self.forbid_same_scene or self._scene_of(a) != self._scene_of(b):
                return key, a, b

        # The scene constraint is a quality filter; distinctness is the actual
        # invariant. Fall back to a random DISTINCT pair rather than a fixed
        # one -- when every episode of a key shares a scene id the loop above
        # can never succeed, and a deterministic fallback then collapses the
        # whole key onto a single pair for every sample. That is silent: the
        # logged "N ordered pairs available" stays large while the model sees
        # one example, which looks exactly like a very fast-learning run until
        # rollout success falls off a cliff.
        i = int(torch.randint(len(eps), (1,), generator=g))
        j = (i + 1 + int(torch.randint(len(eps) - 1, (1,), generator=g))) % len(eps)
        return key, eps[i], eps[j]

    def _scene_of(self, episode: int) -> Any:
        """Scene identifier for an episode, or the episode itself if unavailable.

        Args:
            episode: Episode index.

        Returns:
            A hashable scene id. Falls back to the episode index, which makes
            the same-scene filter a no-op rather than silently rejecting every
            pair when the metadata lacks a scene field.
        """
        info = getattr(self.base.meta, "episodes", {}).get(episode, {})
        for field in ("scene_id", "scene", "source_prefix"):
            if field in info:
                return info[field]
        return episode

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Builds one demonstration+rollout sample.

        Args:
            index: Sample index.

        Returns:
            The concatenated sample. Timestep-axis tensors are joined A-then-B;
            ``loss_mask`` is False across A and True across B; ``prompt`` is the
            key's ambiguous instruction on both halves.

        Raises:
            ValueError: If the two halves disagree on their timestep count,
                which would silently misalign the mask against the sequence.
        """
        key, ep_a, ep_b = self._draw(index)
        a = self.base[self._last_row[ep_a]]
        b = self.base[self._last_row[ep_b]]

        t_a = self._timesteps(a)
        t_b = self._timesteps(b)
        if t_a != t_b:
            raise ValueError(
                f"halves disagree on timestep count ({t_a} vs {t_b}) for key {key}; "
                "the base dataset's sequence_length must be fixed across episodes"
            )

        out: dict[str, Any] = {}
        for k, vb in b.items():
            va = a.get(k)
            if (
                k not in self._SCALAR_PASSTHROUGH
                and isinstance(vb, torch.Tensor)
                and isinstance(va, torch.Tensor)
                and vb.ndim >= 1
                and va.shape[0] == t_a
                and vb.shape[0] == t_b
            ):
                out[k] = torch.cat([va, vb], dim=0)
            else:
                out[k] = vb

        out["loss_mask"] = torch.cat([torch.zeros(t_a, dtype=torch.bool), torch.ones(t_b, dtype=torch.bool)])
        # Both halves, not just B: the demonstration must not be labelled with
        # the answer either.
        #
        # A ``None`` prompt means the caller deliberately declined to overwrite
        # it -- cross-task transfer, where the instruction names the task and
        # the demonstration supplies the unfamiliar object's handling. Keeping
        # B's own prompt is then correct; blanking it would erase the task
        # identity the model needs.
        if self.prompts[key] is not None:
            out["prompt"] = self.prompts[key]
        out["pairing_key"] = key
        return out

    def __getattr__(self, name: str) -> Any:
        """Delegates anything unknown to the wrapped dataset.

        ``WeightedDatasetMixture`` and ``_TaggedDataset`` reach for ``meta``,
        ``repo_id``, ``shallow_copy_with_dropout`` and friends. Forwarding makes
        this a transparent wrapper rather than something every consumer has to
        be taught about.

        Only called for attributes not found normally, so it cannot shadow this
        class's own behaviour.

        Args:
            name: Attribute name.

        Returns:
            The attribute from the wrapped dataset.

        Raises:
            AttributeError: If the wrapped dataset lacks it too.
        """
        # `object.__getattribute__` avoids recursing through this hook while
        # `self.base` is still being set in __init__.
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError as exc:  # pragma: no cover - construction order
            raise AttributeError(name) from exc
        return getattr(base, name)

    @staticmethod
    def _timesteps(item: dict[str, Any]) -> int:
        """Reads the timestep count from a base-dataset sample.

        Args:
            item: One sample from the base dataset.

        Returns:
            Number of timesteps.

        Raises:
            ValueError: If no key carries a usable timestep axis.
        """
        for k in ("loss_mask", "state", "actions", "camera0"):
            v = item.get(k)
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])
        raise ValueError(
            "cannot determine the timestep count; the base dataset is not emitting "
            "sequences (is dataset_mixture.sequence_length set above 1?)"
        )
