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

"""Eval-time subgoal image generators.

The pi07 low-level and pi07-paligemma low-level policies are trained with
subgoal images keyed as ``subgoal{k}`` (one per camera in
``config.image_features``) plus a single ``subgoal_is_pad`` bool. At eval
time these keys are produced by a :class:`SubgoalImageGenerator` that
inspects the observation dict (post ``preprocess_observation`` +
``add_envs_task`` + ``add_eval_metadata``) and returns the matching batch
keys. :func:`opentau.envs.utils.add_subgoal_images` is the wiring helper
that calls the generator and merges its output into the observation.

The only concrete generator implemented today is
:class:`LiberoLastFrameSubgoalGenerator`, which serves the
``TensorAuto/libero`` 20fps v2.1 relabel: it looks up the env's task
language (one of the 40 LIBERO task strings) in the dataset, samples a
random matching episode at the start of each rollout, and returns that
episode's last frame from each camera as the subgoal.
"""

from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch import Tensor

from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.standard_data_format_mapping import DATA_FEATURES_NAME_MAPPING
from opentau.datasets.utils import hf_transform_to_torch
from opentau.datasets.video_utils import decode_video_frames


class SubgoalImageGenerator(Protocol):
    """Eval-time contract for producing subgoal images.

    Generators are constructed once at the top of
    :func:`opentau.scripts.eval.eval` and plumbed through to
    :func:`opentau.scripts.eval.rollout`, which calls
    :meth:`start_episode` once at the start of each rollout (matching the
    "once per episode reset" cadence) and then :meth:`__call__` on every
    step. Implementations are expected to memoise the per-rollout choice
    inside :meth:`start_episode` so the per-step call is cheap.
    """

    def start_episode(self, prompts: list[str]) -> None:
        """Re-sample the per-env subgoal source for a fresh rollout.

        Called once at the top of :func:`opentau.scripts.eval.rollout`
        after ``env.reset()`` and ``add_envs_task`` have populated
        ``observation["prompt"]``.

        Args:
            prompts: Task language string per env in the vector, length
                ``env.num_envs``.
        """
        ...

    def __call__(self, observation: dict[str, Any]) -> dict[str, Tensor]:
        """Return the subgoal batch keys for the current observation.

        Args:
            observation: Observation dict after ``preprocess_observation``,
                ``add_envs_task``, and ``add_eval_metadata`` — contains
                ``camera{k}``, ``state``, ``img_is_pad``, ``prompt``, and
                any pi07 metadata keys.

        Returns:
            Dict with ``subgoal{k}`` ``(B, 3, H, W)`` floats in ``[0, 1]``
            for each camera the generator can serve, plus
            ``subgoal_is_pad`` ``(B,)`` bool. Cameras the generator does
            not have data for are left out; the policy's
            ``prepare_subgoal_images`` fills missing slots with ``-1``
            placeholders and a ``mask=False``.
        """
        ...


class LiberoLastFrameSubgoalGenerator:
    """Naive LIBERO subgoal generator backed by ``TensorAuto/libero``.

    For each env in the rollout, looks up the task language string (one
    of the 40 LIBERO subtask strings) in the dataset's episode index,
    samples a random matching episode at the start of the rollout, and
    returns that episode's *last* frame from each video camera as the
    subgoal. Camera images are resized with aspect-ratio padding to match
    ``resolution`` (the same shape ``preprocess_observation`` produces for
    the live ``camera{k}`` keys).

    The chosen episodes are pinned by :meth:`start_episode` and reused
    across every :meth:`__call__` in the rollout — fresh draws happen on
    the next ``rollout()`` call, which corresponds to the next
    ``env.reset()``. If a prompt has no matching episode in the dataset,
    :meth:`start_episode` raises ``ValueError``.

    Today only ``repo_id="TensorAuto/libero"`` is exercised; other
    LeRobot v2.1 repos with the same camera-name convention should work
    but are not tested.

    Train↔eval distribution shift (intentional, naive baseline):
        Training samples a *within-episode* future frame
        (``BaseDataset._sample_subgoal_frame``: uniform ``[t, t+4s]`` or
        segment end). This generator picks a *different* episode (matched
        only by task language) and always returns its last frame. That's
        a different conditional distribution from training; the policy
        was not conditioned on a foreign trajectory's terminus. This is
        deliberate as a first baseline — see the PR that introduced this
        module for the rationale. Follow-up improvements (uniform
        within-episode sampling, or sampling from the rollout's own
        replay) would close most of the gap.

    Memory:
        The per-(episode, camera) frame cache is unbounded — it grows
        monotonically across rollouts and survives every mid-training
        eval block. Bounded in practice by the source dataset size
        (~660 MB worst case for full LIBERO at the default resolution),
        but a larger source would need an LRU. TODO if/when this
        generator gets pointed at a non-LIBERO repo.
    """

    def __init__(
        self,
        repo_id: str = "TensorAuto/libero",
        root: str | Path | None = None,
        resolution: tuple[int, int] = (256, 256),
        num_cams: int = 2,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        seed: int | None = None,
    ):
        """Load dataset metadata and build the task-language episode index.

        Args:
            repo_id: HuggingFace dataset repo to source subgoal frames
                from. Must be a v2.1 LeRobot dataset registered in
                :data:`DATA_FEATURES_NAME_MAPPING`.
            root: Optional local cache root for the dataset. Defaults to
                the OpenTau-managed location used by ``LeRobotDataset``.
            resolution: Target ``(H, W)`` for the returned subgoal
                images, matching ``cfg.resolution``.
            num_cams: Number of camera slots to populate. Capped by the
                cameras the source dataset actually exposes (typically 2
                for LIBERO).
            tolerance_s: Timestamp tolerance for video frame decoding.
            revision: Optional explicit dataset revision (e.g. ``"v2.1"``,
                a branch, or a commit SHA). ``None`` falls back to
                ``CODEBASE_VERSION`` inside
                :class:`LeRobotDatasetMetadata` — currently ``v2.1``,
                matching ``TensorAuto/libero``. Pin this when the
                codebase and dataset versions might drift apart and you
                need the eval-time subgoals to come from a specific
                relabel pass.
            seed: Seed for the per-instance ``random.Random`` used to
                pick episodes in :meth:`start_episode`. ``None`` falls
                back to a default-constructed ``Random`` (entropy-seeded
                per process). Threading ``cfg.seed`` here is what makes
                episode picks reproducible across runs; the process-
                global ``random`` is not used so concurrent
                ``eval_policy_all`` workers don't race on shared RNG
                state.
        """
        self.repo_id = repo_id
        self.resolution = resolution
        self.tolerance_s = tolerance_s

        # `LeRobotDatasetMetadata.__init__` downloads only the `meta/`
        # subdir on cache miss — the video files themselves are pulled
        # lazily by `_decode_last_frame` on first use.
        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)

        if repo_id not in DATA_FEATURES_NAME_MAPPING:
            raise KeyError(
                f"LiberoLastFrameSubgoalGenerator: repo_id {repo_id!r} has no entry in "
                f"DATA_FEATURES_NAME_MAPPING; cannot resolve camera{{k}} -> raw key."
            )
        # TODO: thread control_mode and use `resolve_feature_mapping(repo_id, control_mode)`
        # for dual-split repos; cameras are identical across control modes so the plain
        # repo_id entry is correct here today.
        name_map = DATA_FEATURES_NAME_MAPPING[repo_id]

        # Resolve camera slots to the raw feature keys the dataset
        # exposes, cap by `num_cams`, drop any slot whose raw key is not
        # present in the dataset's video_keys / image_keys. Slots beyond
        # this list won't get a `subgoal{k}` and the model fills them
        # with -1 placeholders.
        camera_keys: dict[int, str] = {}
        for k in range(num_cams):
            raw_key = name_map.get(f"camera{k}")
            if raw_key is None:
                continue
            if raw_key in self.meta.video_keys or raw_key in self.meta.image_keys:
                camera_keys[k] = raw_key
        if not camera_keys:
            raise ValueError(
                f"No subgoal cameras resolved for {repo_id!r}: name_map="
                f"{name_map}, video_keys={self.meta.video_keys}, image_keys={self.meta.image_keys}."
            )
        self.camera_keys = camera_keys

        self.lang_to_episodes: dict[str, list[int]] = {}
        for ep_idx, info in self.meta.episodes.items():
            for task_str in info.get("tasks", []):
                self.lang_to_episodes.setdefault(task_str, []).append(ep_idx)

        # The frame cache is shared across rollouts (read-mostly; the
        # CPython GIL makes the `setdefault` + assignment safe enough — a
        # racing-thread cache miss at worst duplicates a video decode).
        # `chosen_episodes` lives on threadlocal storage so concurrent
        # `eval_policy_all` workers (when `max_parallel_tasks > 1`) don't
        # clobber each other's per-rollout pick.
        self._frame_cache: dict[int, dict[str, Tensor]] = {}
        self._local = threading.local()
        # Per-instance RNG (not the process-global `random`) so a fixed
        # `seed` reproduces the per-rollout episode picks even when
        # other code paths consume the global RNG between rollouts.
        # Threadlocal `chosen_episodes` already isolates the *outcome*
        # of each pick from concurrent threads; for bit-identical picks
        # under `max_parallel_tasks > 1` you'd additionally need per-
        # thread RNGs (not done here — single-threaded eval is the
        # common case and the LiberoEnv default).
        self._rng = random.Random(seed)

    @property
    def fps(self) -> int:
        return self.meta.fps

    def known_languages(self) -> list[str]:
        return sorted(self.lang_to_episodes)

    def start_episode(self, prompts: list[str]) -> None:
        """Sample one random matching episode per env, store for the rollout.

        Raises:
            ValueError: If any prompt has no matching episode in
                ``lang_to_episodes``. The error names the prompt and the
                count of known languages.
        """
        chosen: list[int] = []
        for prompt in prompts:
            candidates = self.lang_to_episodes.get(prompt)
            if not candidates:
                raise ValueError(
                    f"LiberoLastFrameSubgoalGenerator: no episodes in {self.repo_id!r} match "
                    f"prompt {prompt!r} (dataset has {len(self.lang_to_episodes)} known task "
                    f"languages)."
                )
            chosen.append(self._rng.choice(candidates))
        self._local.chosen_episodes = chosen

    def __call__(self, observation: dict[str, Any]) -> dict[str, Tensor]:
        """Return the cached subgoal batch keys.

        :meth:`start_episode` must have been called first to populate the
        per-env episode choice.
        """
        chosen = getattr(self._local, "chosen_episodes", None)
        if chosen is None:
            raise RuntimeError(
                "LiberoLastFrameSubgoalGenerator: start_episode(prompts) must be called "
                "before __call__(observation). Did the rollout skip the per-episode setup?"
            )
        state = observation["state"]
        if state.shape[0] != len(chosen):
            raise ValueError(
                f"LiberoLastFrameSubgoalGenerator: observation batch size {state.shape[0]} "
                f"does not match the {len(chosen)} episodes chosen at start_episode(); "
                f"was the env batch resized mid-rollout?"
            )
        device = state.device
        dtype = state.dtype if state.dtype.is_floating_point else torch.bfloat16

        out: dict[str, Tensor] = {}
        for k, raw_key in self.camera_keys.items():
            per_env = [self._cached_frame(ep_idx, raw_key) for ep_idx in chosen]
            stacked = torch.stack(per_env, dim=0).to(device=device, dtype=dtype)
            out[f"subgoal{k}"] = stacked
        out["subgoal_is_pad"] = torch.zeros(state.shape[0], dtype=torch.bool, device=device)
        return out

    def _cached_frame(self, ep_idx: int, raw_key: str) -> Tensor:
        """Per-(episode, camera) lookup with episode-level batch decode.

        Cache misses load *every* camera of the episode in one shot so an
        image-dtype dataset (where all cameras share the same parquet
        file) avoids loading the parquet once per camera.
        """
        if ep_idx not in self._frame_cache:
            self._frame_cache[ep_idx] = self._load_episode_frames(ep_idx)
        return self._frame_cache[ep_idx][raw_key]

    def _load_episode_frames(self, ep_idx: int) -> dict[str, Tensor]:
        """Decode the last frame of ``ep_idx`` for every camera in ``camera_keys``.

        Returns a dict ``{raw_key: (3, *resolution) float tensor in [0, 1]}``
        — same convention as the training-time ``_emit_optional_keys`` path
        so the model sees identically prepared subgoals at train and eval.

        Video cameras decode one frame each via ``decode_video_frames``.
        Image cameras share a single parquet load so a multi-image-camera
        dataset (e.g. ``TensorAuto/libero`` with ``image`` + ``wrist_image``)
        only pays the parquet I/O once per episode.
        """
        info = self.meta.episodes[ep_idx]
        length = int(info["length"])
        if length <= 0:
            raise ValueError(
                f"LiberoLastFrameSubgoalGenerator: episode {ep_idx} in {self.repo_id!r} has "
                f"non-positive length {length!r}."
            )
        last_idx_in_ep = length - 1

        video_keys = [rk for rk in self.camera_keys.values() if rk in self.meta.video_keys]
        image_keys = [rk for rk in self.camera_keys.values() if rk in self.meta.image_keys]

        out: dict[str, Tensor] = {}

        if video_keys:
            last_ts = last_idx_in_ep / self.fps
            for raw_key in video_keys:
                local_path = self._resolve_video_path(ep_idx, raw_key)
                frames = decode_video_frames(local_path, [last_ts], self.tolerance_s)
                out[raw_key] = self._finalize_frame(frames[0], ep_idx, raw_key)

        if image_keys:
            local_path = self._resolve_data_path(ep_idx)
            ds = load_dataset("parquet", data_files=str(local_path), split="train")
            ds.set_transform(hf_transform_to_torch)
            row = ds[last_idx_in_ep]
            for raw_key in image_keys:
                out[raw_key] = self._finalize_frame(row[raw_key], ep_idx, raw_key)

        return out

    def _finalize_frame(self, frame: Tensor, ep_idx: int, raw_key: str) -> Tensor:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(
                f"LiberoLastFrameSubgoalGenerator: expected decoded frame shape (3, H, W), "
                f"got {tuple(frame.shape)} for episode {ep_idx} key {raw_key!r}."
            )
        return _resize_with_pad(frame, self.resolution[1], self.resolution[0])

    def _resolve_video_path(self, ep_idx: int, raw_key: str) -> Path:
        local_path = self.meta.root / self.meta.get_video_file_path(ep_idx, raw_key)
        if not local_path.is_file():
            hf_hub_download(
                repo_id=self.meta.repo_id,
                filename=str(self.meta.get_video_file_path(ep_idx, raw_key)),
                repo_type="dataset",
                revision=self.meta.revision,
                local_dir=self.meta.root,
            )
        return local_path

    def _resolve_data_path(self, ep_idx: int) -> Path:
        local_path = self.meta.root / self.meta.get_data_file_path(ep_idx)
        if not local_path.is_file():
            hf_hub_download(
                repo_id=self.meta.repo_id,
                filename=str(self.meta.get_data_file_path(ep_idx)),
                repo_type="dataset",
                revision=self.meta.revision,
                local_dir=self.meta.root,
            )
        return local_path


def _resize_with_pad(img: Tensor, width: int, height: int, pad_value: float = 0.0) -> Tensor:
    """Aspect-preserving resize + top/left padding to ``(height, width)``.

    Mirrors :meth:`BaseDataset.resize_with_pad` exactly — kept as a free
    function here so the generator does not need to construct a full
    dataset to apply the transform.
    """
    batched = img.ndim == 4
    if not batched:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"Expected (C,H,W) or (T,C,H,W), got shape {tuple(img.shape)}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_h = max(0, height - resized_height)
    pad_w = max(0, width - resized_width)
    padded = F.pad(resized, (pad_w, 0, pad_h, 0), value=pad_value)
    return padded.squeeze(0) if not batched else padded
