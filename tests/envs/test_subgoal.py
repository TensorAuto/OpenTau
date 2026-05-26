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

"""Tests for the eval-time subgoal image generator and add_subgoal_images.

These pin the contract between
:class:`opentau.envs.subgoal.LiberoLastFrameSubgoalGenerator` and
:meth:`opentau.policies.pi07.low_level.modeling_pi07_low_level.PI07LowLevelPolicy.prepare_subgoal_images`:
the model derives ``subgoal{k}`` keys from ``config.image_features``,
expects each as ``(B, 3, H, W)`` floats in ``[0, 1]``, and reads a single
``subgoal_is_pad`` ``(B,)`` bool. Drift in any of those shapes / dtypes
silently changes the prefix the model sees at eval.

The mocked-dataset tests cover the generator wiring without touching the
network or any real video files. A separate ``@pytest.mark.slow`` block
exercises the real ``TensorAuto/libero`` repo end-to-end; it self-skips
when the dataset metadata cannot be reached so the CPU CI subset stays
green on hosts without cluster cache.
"""

import random
from unittest.mock import MagicMock, patch

import pytest
import torch

from opentau.envs.subgoal import LiberoLastFrameSubgoalGenerator
from opentau.envs.utils import add_subgoal_images

# Two real LIBERO task strings from the libero_spatial suite — picked so
# the mocked metadata covers > 1 language and the integration test has a
# realistic prompt to look up.
_PROMPT_A = "pick up the alphabet soup and place it in the basket"
_PROMPT_B = "pick up the cream cheese and place it in the basket"
_UNKNOWN_PROMPT = "do something that does not exist in the dataset"


def _build_fake_meta(
    *,
    fps: int = 20,
    episode_lengths: dict[int, int] | None = None,
    episode_tasks: dict[int, list[str]] | None = None,
    video_keys: tuple[str, ...] = ("image", "wrist_image"),
):
    """Construct a duck-typed ``LeRobotDatasetMetadata`` for tests.

    Avoids the real ``__init__``'s metadata download by stubbing only the
    attributes :class:`LiberoLastFrameSubgoalGenerator` actually reads:
    ``episodes``, ``video_keys``, ``image_keys``, ``fps``, ``repo_id``,
    ``revision``, ``root``, and ``get_video_file_path``.
    """
    episode_lengths = episode_lengths or {0: 50, 1: 75, 2: 100}
    episode_tasks = episode_tasks or {0: [_PROMPT_A], 1: [_PROMPT_A], 2: [_PROMPT_B]}

    meta = MagicMock()
    meta.repo_id = "TensorAuto/libero"
    meta.revision = "v2.1"
    meta.root = MagicMock()  # any non-None Path-like is fine since _resolve_video_path is mocked
    meta.fps = fps
    meta.video_keys = list(video_keys)
    meta.image_keys = []
    meta.episodes = {
        ep_idx: {"episode_index": ep_idx, "length": episode_lengths[ep_idx], "tasks": episode_tasks[ep_idx]}
        for ep_idx in episode_lengths
    }
    # ``get_video_file_path`` is only used inside ``_resolve_video_path``,
    # which we patch out in the tests, but expose a benign stub so any
    # accidental call surfaces as a clear error rather than an
    # ``AttributeError``.
    meta.get_video_file_path = MagicMock(side_effect=lambda ep_idx, key: f"videos/ep{ep_idx}_{key}.mp4")
    return meta


def _make_generator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolution: tuple[int, int] = (32, 32),
    num_cams: int = 2,
    fake_meta=None,
    decoded_frame_value: float = 0.5,
    seed: int | None = None,
) -> LiberoLastFrameSubgoalGenerator:
    """Build a generator with patched metadata + a stub video decoder.

    Uses ``monkeypatch`` (pytest-managed) so the patches stay live for the
    duration of the test — both ``__init__`` (reading metadata) and any
    later ``__call__`` (decoding a frame) need them active. The decoder
    returns a constant 256×256 RGB tensor so we can verify
    ``resize_with_pad`` lands at the requested ``resolution`` and the
    pixel range survives normalization.
    """
    fake_meta = fake_meta or _build_fake_meta()
    fake_frame = torch.full((1, 3, 256, 256), decoded_frame_value, dtype=torch.float32)

    monkeypatch.setattr("opentau.envs.subgoal.LeRobotDatasetMetadata", lambda *args, **kwargs: fake_meta)
    monkeypatch.setattr("opentau.envs.subgoal.decode_video_frames", lambda *args, **kwargs: fake_frame)

    gen = LiberoLastFrameSubgoalGenerator(resolution=resolution, num_cams=num_cams, seed=seed)
    # Stub out _resolve_video_path on the live instance so the decoded-
    # path code never touches hf_hub_download.
    gen._resolve_video_path = MagicMock(return_value="<fake-path>")
    return gen


def _make_obs(batch_size: int, *, device: str = "cpu") -> dict:
    """Minimal observation post `preprocess_observation` + `add_envs_task`.

    The generator only reads ``observation["state"]`` for batch size /
    device routing — the prompt list is fed directly to
    ``start_episode``. We don't synthesize ``camera{k}`` etc. because the
    generator never reads them.
    """
    return {"state": torch.zeros(batch_size, 8, device=device, dtype=torch.float32)}


class TestLiberoLastFrameSubgoalGeneratorInit:
    """Pin the language→episode index construction and camera resolution."""

    def test_builds_lang_to_episodes_from_meta(self, monkeypatch):
        gen = _make_generator(monkeypatch)

        # Episode 0 and 1 both list PROMPT_A; episode 2 lists PROMPT_B.
        assert sorted(gen.lang_to_episodes[_PROMPT_A]) == [0, 1]
        assert gen.lang_to_episodes[_PROMPT_B] == [2]
        assert _UNKNOWN_PROMPT not in gen.lang_to_episodes

    def test_known_languages_sorted(self, monkeypatch):
        gen = _make_generator(monkeypatch)
        langs = gen.known_languages()
        assert langs == sorted(langs)
        assert _PROMPT_A in langs and _PROMPT_B in langs

    def test_camera_keys_match_data_features_name_mapping(self, monkeypatch):
        """``camera{k}`` → raw key resolution comes from
        ``DATA_FEATURES_NAME_MAPPING["TensorAuto/libero"]`` (image / wrist_image).
        """
        gen = _make_generator(monkeypatch)
        assert gen.camera_keys == {0: "image", 1: "wrist_image"}

    def test_unknown_repo_id_raises(self, monkeypatch):
        """An unmapped repo_id has no ``camera{k}`` resolution path; the
        generator surfaces that as a ``KeyError`` at init rather than
        silently emitting empty subgoals.
        """
        monkeypatch.setattr(
            "opentau.envs.subgoal.LeRobotDatasetMetadata",
            lambda *args, **kwargs: _build_fake_meta(),
        )
        with pytest.raises(KeyError, match="DATA_FEATURES_NAME_MAPPING"):
            LiberoLastFrameSubgoalGenerator(repo_id="not/in-mapping")

    def test_num_cams_caps_at_dataset_cameras(self, monkeypatch):
        """If ``num_cams`` exceeds what the dataset exposes, the generator
        only serves what it has (here: 2 cameras). The policy fills the
        rest via its ``-1`` placeholder path.
        """
        gen = _make_generator(monkeypatch, num_cams=4)
        assert gen.camera_keys == {0: "image", 1: "wrist_image"}


class TestStartEpisode:
    """Pin the random-pick + missing-prompt behaviour of start_episode."""

    def test_picks_one_episode_per_env(self, monkeypatch):
        gen = _make_generator(monkeypatch)
        random.seed(0)
        gen.start_episode([_PROMPT_A, _PROMPT_A, _PROMPT_B])
        chosen = gen._local.chosen_episodes
        assert len(chosen) == 3
        assert chosen[0] in {0, 1}
        assert chosen[1] in {0, 1}
        assert chosen[2] == 2

    def test_missing_prompt_raises_value_error(self, monkeypatch):
        gen = _make_generator(monkeypatch)
        with pytest.raises(ValueError, match=_UNKNOWN_PROMPT):
            gen.start_episode([_UNKNOWN_PROMPT])

    def test_deterministic_with_seeded_random(self, monkeypatch):
        """Two generators with the same ``seed`` produce the same per-rollout
        picks — locks in the per-instance ``random.Random`` so the global
        ``random`` state cannot foil reproducibility-by-seed expectations.
        """
        gen_a = _make_generator(monkeypatch, seed=123)
        gen_a.start_episode([_PROMPT_A, _PROMPT_A, _PROMPT_A])
        first = list(gen_a._local.chosen_episodes)

        gen_b = _make_generator(monkeypatch, seed=123)
        gen_b.start_episode([_PROMPT_A, _PROMPT_A, _PROMPT_A])
        second = list(gen_b._local.chosen_episodes)

        assert first == second

    def test_global_random_state_does_not_affect_picks(self, monkeypatch):
        """Per-instance RNG isolates picks from the process-global
        ``random`` — mutating the global state between picks must not
        change the per-rollout selection.
        """
        gen = _make_generator(monkeypatch, seed=7)
        gen.start_episode([_PROMPT_A, _PROMPT_A])
        first = list(gen._local.chosen_episodes)

        # Burn the global RNG. With a per-instance RNG, this is invisible
        # to the generator; without it, the second pick would diverge.
        for _ in range(100):
            random.random()

        gen2 = _make_generator(monkeypatch, seed=7)
        gen2.start_episode([_PROMPT_A, _PROMPT_A])
        second = list(gen2._local.chosen_episodes)

        assert first == second


class TestCall:
    """Shape / dtype / value contract surfaced to the policy."""

    def test_returns_subgoal_keys_for_each_camera(self, monkeypatch):
        gen = _make_generator(monkeypatch, resolution=(32, 32))
        random.seed(0)
        obs = _make_obs(batch_size=2)
        gen.start_episode([_PROMPT_A, _PROMPT_B])

        out = gen(obs)
        # Two cameras → subgoal0 and subgoal1; plus the single
        # subgoal_is_pad bool.
        assert set(out) == {"subgoal0", "subgoal1", "subgoal_is_pad"}

    def test_subgoal_tensor_shape_and_value_range(self, monkeypatch):
        """Shape is ``(B, 3, H, W)`` at ``resolution`` (after resize_with_pad
        from the synthetic 256×256 source) and values stay in ``[0, 1]``.
        """
        gen = _make_generator(monkeypatch, resolution=(48, 64), decoded_frame_value=0.5)
        random.seed(0)
        obs = _make_obs(batch_size=3)
        gen.start_episode([_PROMPT_A] * 3)

        out = gen(obs)
        for k in (0, 1):
            t = out[f"subgoal{k}"]
            assert t.shape == (3, 3, 48, 64), f"subgoal{k} shape: {tuple(t.shape)}"
            assert float(t.min()) >= 0.0
            assert float(t.max()) <= 1.0

    def test_subgoal_is_pad_all_false(self, monkeypatch):
        gen = _make_generator(monkeypatch)
        random.seed(0)
        obs = _make_obs(batch_size=2)
        gen.start_episode([_PROMPT_A, _PROMPT_B])
        out = gen(obs)

        assert out["subgoal_is_pad"].dtype == torch.bool
        assert out["subgoal_is_pad"].shape == (2,)
        assert not out["subgoal_is_pad"].any()

    def test_dtype_matches_observation_state(self, monkeypatch):
        """Subgoal tensors land on the same device + floating dtype as
        ``observation["state"]`` so downstream concat / cast in
        ``prepare_subgoal_images`` doesn't trigger a dtype promotion.
        """
        gen = _make_generator(monkeypatch)
        random.seed(0)
        obs = {"state": torch.zeros(2, 8, dtype=torch.bfloat16)}
        gen.start_episode([_PROMPT_A, _PROMPT_B])
        out = gen(obs)

        for k in (0, 1):
            assert out[f"subgoal{k}"].dtype == torch.bfloat16
        assert out["subgoal_is_pad"].device.type == "cpu"

    def test_caches_decoded_frames(self, monkeypatch):
        """Calling ``__call__`` twice in the same rollout (with the same
        chosen episodes) only loads each episode's frames once. The
        episode-level cache batches all cameras together so an
        image-dtype dataset (cameras share a parquet) doesn't pay per-
        camera I/O.
        """
        gen = _make_generator(monkeypatch)
        random.seed(0)
        obs = _make_obs(batch_size=2)
        gen.start_episode([_PROMPT_A, _PROMPT_B])

        with patch.object(gen, "_load_episode_frames", wraps=gen._load_episode_frames) as load_spy:
            gen(obs)
            gen(obs)
            # 2 envs → 2 episodes, one load per episode; the second
            # __call__ should be served entirely from cache.
            assert load_spy.call_count == 2

    def test_call_without_start_episode_raises(self, monkeypatch):
        gen = _make_generator(monkeypatch)
        obs = _make_obs(batch_size=1)
        with pytest.raises(RuntimeError, match="start_episode"):
            gen(obs)

    def test_batch_mismatch_raises(self, monkeypatch):
        """If env.num_envs changes between ``start_episode`` and ``__call__``
        the generator surfaces a clear error instead of silently
        broadcasting / slicing.
        """
        gen = _make_generator(monkeypatch)
        random.seed(0)
        gen.start_episode([_PROMPT_A, _PROMPT_B])  # picked for B=2
        obs = _make_obs(batch_size=4)  # mismatching B
        with pytest.raises(ValueError, match="batch size"):
            gen(obs)

    def test_image_dtype_loads_via_parquet_path(self, monkeypatch):
        """When the dataset exposes cameras as image-dtype (``TensorAuto/libero``
        is the real-world case — both cameras stored per-frame in parquet),
        the generator hits the ``load_dataset`` + parquet path and not the
        video decoder. All image cameras for one episode share a single
        parquet load.
        """
        meta = _build_fake_meta(video_keys=())  # no video cameras
        meta.image_keys = ["image", "wrist_image"]
        meta.get_data_file_path = MagicMock(side_effect=lambda ep_idx: f"data/episode_{ep_idx:06d}.parquet")

        fake_row = {
            "image": torch.full((3, 256, 256), 0.3, dtype=torch.float32),
            "wrist_image": torch.full((3, 256, 256), 0.7, dtype=torch.float32),
        }
        # ``load_dataset(...).set_transform(...); ds[idx]`` is what the
        # generator does. Mock both calls with a small in-memory stand-in.
        fake_ds = MagicMock()
        fake_ds.__getitem__ = MagicMock(return_value=fake_row)

        monkeypatch.setattr("opentau.envs.subgoal.LeRobotDatasetMetadata", lambda *args, **kwargs: meta)
        monkeypatch.setattr("opentau.envs.subgoal.load_dataset", lambda *args, **kwargs: fake_ds)

        gen = LiberoLastFrameSubgoalGenerator(resolution=(64, 64), num_cams=2)
        gen._resolve_data_path = MagicMock(return_value="<fake-parquet>")

        random.seed(0)
        gen.start_episode([_PROMPT_A])
        out = gen(_make_obs(batch_size=1))

        assert out["subgoal0"].shape == (1, 3, 64, 64)
        assert out["subgoal1"].shape == (1, 3, 64, 64)
        # ``set_transform`` should fire (the parquet rows are PIL until
        # transformed) and ``__getitem__`` exactly once (cameras share the
        # row).
        fake_ds.set_transform.assert_called_once()
        fake_ds.__getitem__.assert_called_once()


class TestAddSubgoalImages:
    """add_subgoal_images is a 3-line wiring helper; pin its contract."""

    def test_noop_when_generator_none(self):
        obs = _make_obs(batch_size=2)
        original_keys = set(obs.keys())
        out = add_subgoal_images(obs, None)
        assert out is obs
        assert set(out.keys()) == original_keys

    def test_merges_generator_output(self, monkeypatch):
        gen = _make_generator(monkeypatch)
        random.seed(0)
        obs = _make_obs(batch_size=2)
        gen.start_episode([_PROMPT_A, _PROMPT_B])
        original_keys = set(obs.keys())

        out = add_subgoal_images(obs, gen)
        assert out is obs  # mutated in place
        new_keys = set(out.keys()) - original_keys
        assert new_keys == {"subgoal0", "subgoal1", "subgoal_is_pad"}


# ---------------------------------------------------------------------------
# Integration test against the real `TensorAuto/libero` dataset.
# Marked `slow` so the default `pytest -m "not gpu"` CPU CI subset still
# runs it, but a local dev iteration can skip via `-m "not slow"`.
# Self-skips if metadata cannot be loaded (no network / no cache).
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestRealDatasetIntegration:
    @pytest.fixture(scope="class")
    def generator(self):
        from opentau.envs.subgoal import LeRobotDatasetMetadata

        try:
            LeRobotDatasetMetadata(repo_id="TensorAuto/libero")
        except Exception as e:
            pytest.skip(f"TensorAuto/libero metadata unavailable: {e}")
        return LiberoLastFrameSubgoalGenerator(
            repo_id="TensorAuto/libero",
            resolution=(64, 64),
            num_cams=2,
        )

    def test_known_languages_cover_libero_suites(self, generator):
        """The 20fps v2.1 relabel should expose at least the 40 standard
        LIBERO task strings (spatial + object + goal + 10 = 40). Asserting
        a loose lower bound keeps this from flaking if the dataset grows.
        """
        langs = generator.known_languages()
        assert len(langs) >= 40, f"expected ≥40 task languages, got {len(langs)}"

    def test_call_returns_real_last_frame(self, generator):
        """End-to-end: pick the first known language, sample an episode,
        decode its last frame, return correctly shaped subgoal tensors.
        """
        prompt = generator.known_languages()[0]
        obs = _make_obs(batch_size=1)
        generator.start_episode([prompt])

        out = generator(obs)
        assert set(out) == {"subgoal0", "subgoal1", "subgoal_is_pad"}
        assert out["subgoal0"].shape == (1, 3, 64, 64)
        assert out["subgoal1"].shape == (1, 3, 64, 64)
        assert float(out["subgoal0"].min()) >= 0.0
        assert float(out["subgoal0"].max()) <= 1.0
