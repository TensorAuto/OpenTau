# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

r"""Environment wrapper for RoboCasa365 kitchen tasks.

Ported from upstream LeRobot's ``lerobot/envs/robocasa.py`` and reshaped to
OpenTau's LIBERO conventions: cameras are remapped to ``camera0``/``camera1``/...
so :func:`opentau.envs.utils.preprocess_observation` and the policy's
``num_cams`` zero-fill path consume them exactly like LIBERO, and the vec-env
builder shards tasks across accelerator ranks so distributed eval and the
``_rank{N}``-strip uniqueness assumption in
:func:`opentau.scripts.eval.collect_grid_summary_videos` both hold.

The underlying simulator (``robocasa`` / ``robosuite`` 1.5) is imported lazily
inside :meth:`RoboCasaEnv._ensure_env` and :func:`_resolve_tasks`, so importing
this module (e.g. in the CPU test suite) never requires the sim to be installed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import socket
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile

import einops
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from opentau.constants import HF_OPENTAU_HOME
from opentau.utils.accelerate_utils import acc_print, get_proc_accelerator
from opentau.utils.io_utils import silence_output_unless_error

# Flat action/state vector dimensions for the PandaOmron mobile manipulator
# (RoboCasa365's default robot).
OBS_STATE_DIM = 16  # base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)
ACTION_DIM = 12  # base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default PandaOmron cameras (raw RoboCasa names). The wrapper remaps these to
# ``camera0``/``camera1``/``camera2`` so the OpenTau policy input structure
# matches LIBERO's (``camera{i}`` keys + ``img_is_pad``), independent of how
# many real cameras a given config renders.
DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
]

# Object-mesh registries to sample from. RoboCasa's upstream default is
# ("objaverse", "lightwheel"), but the objaverse pack is huge (~30GB) and most
# setups only download the lightwheel pack (`--type objs_lw` in
# `download_kitchen_assets`). When a sampled object category has zero candidates
# in every registry, robocasa crashes with `ValueError: Probabilities contain
# NaN`. Restricting to registries that are actually on disk avoids that.
DEFAULT_OBJ_REGISTRIES: tuple[str, ...] = ("lightwheel",)

# Task-group shortcuts accepted as ``env.task``. A group name expands to the
# upstream RoboCasa task list and auto-sets the dataset split; individual task
# names (optionally comma-separated) take precedence and only match exactly.
_TASK_GROUP_SPLITS = {
    "atomic_seen": "target",
    "composite_seen": "target",
    "composite_unseen": "target",
    "pretrain50": "pretrain",
    "pretrain100": "pretrain",
    "pretrain200": "pretrain",
    "pretrain300": "pretrain",
}


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings.

    Local copy of the LIBERO helper so importing this module never pulls in
    ``opentau.envs.libero`` (which imports the ``libero`` package at top level).
    """
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _default_camera_name_mapping(camera_names: Sequence[str]) -> dict[str, list[str]]:
    """Map each raw RoboCasa camera to a positional ``camera{i}`` key.

    Mirrors ``LiberoEnv``'s ``camera_name_mapping``: the first rendered camera
    becomes ``camera0``, the second ``camera1``, and so on, so the policy
    receives a consistent ``camera{i}`` structure regardless of the raw names.
    """
    return {cam: [f"camera{i}"] for i, cam in enumerate(camera_names)}


ROBOCASA_ASSETS_ROOT_ENV = "ROBOCASA_ASSETS_ROOT"


def _resolve_robocasa_assets_root() -> Path:
    r"""Directory where RoboCasa kitchen assets live, *outside* the ephemeral uv venv.

    Resolution order: the ``ROBOCASA_ASSETS_ROOT`` env var, else a stable default under
    OpenTau's HuggingFace cache (``HF_OPENTAU_HOME/robocasa/assets``). Deterministic in
    every process — the main process and each ``AsyncVectorEnv`` spawn worker (which
    inherits the env var) — so the per-process loader redirect and the one-time download
    always agree on the same path.
    """
    env = os.environ.get(ROBOCASA_ASSETS_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    return HF_OPENTAU_HOME / "robocasa" / "assets"


def _internal_robocasa_pkg_dir() -> str:
    r"""Realpath of OpenTau's *internal* ``opentau/scripts/robocasa`` package.

    That package (the inference WebSocket server) shares the bare name ``robocasa``
    with the external sim. When ``train.py`` / ``eval.py`` run as scripts — which is
    how ``accelerate launch`` and the ``opentau-*`` console entry points invoke them —
    Python puts ``opentau/scripts/`` on ``sys.path[0]``, so ``import robocasa`` /
    ``find_spec("robocasa")`` resolve to this empty package instead of the sim. The
    helpers below use this path to skip the shadow.
    """
    return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "robocasa"))


@contextlib.contextmanager
def _robocasa_unshadowed():
    r"""Temporarily drop the ``sys.path`` entry that shadows the external robocasa
    package, and purge an already-imported shadow, so ``import robocasa`` /
    ``find_spec("robocasa")`` resolve to the installed sim. Restores ``sys.path`` on exit.
    """
    internal = _internal_robocasa_pkg_dir()
    # If the internal package was already imported under the bare name, evict it
    # (and its submodules) so resolution can fall through to the real sim.
    existing = sys.modules.get("robocasa")
    if existing is not None:
        mod_file = getattr(existing, "__file__", "") or ""
        if os.path.realpath(os.path.dirname(mod_file)) == internal:
            for name in [m for m in sys.modules if m == "robocasa" or m.startswith("robocasa.")]:
                del sys.modules[name]
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if os.path.realpath(os.path.join(p, "robocasa")) != internal]
    try:
        yield
    finally:
        sys.path[:] = saved_path


def _robocasa_pkg_assets_dir() -> Path:
    r"""Locate ``<robocasa-pkg>/models/assets`` *without importing* robocasa.

    Importing robocasa eagerly builds its object registry by scanning this directory at
    import time (``kitchen_object_utils`` walks ``assets/objects`` to populate every
    category's ``mjcf_paths``). So relocation has to be a symlink that is already in place
    *before* the first import — which means locating the package via its import spec, not
    via ``robocasa.__path__`` (that would require importing it).
    """
    # Skip OpenTau's internal ``opentau/scripts/robocasa`` shadow, else ``find_spec``
    # returns it and we look for assets in the wrong directory (see _internal_robocasa_pkg_dir).
    with _robocasa_unshadowed():
        spec = importlib.util.find_spec("robocasa")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("robocasa is not installed; cannot locate its assets dir.")
    return Path(next(iter(spec.submodule_search_locations))) / "models" / "assets"


def _direct_download_url(shared_url: str) -> str:
    r"""Convert a Box shared link into a direct-download ``.zip`` URL.

    Reimplements ``download_kitchen_assets._get_direct_download_url`` so we never import
    that module — importing it imports robocasa, which would build the object registry
    against the not-yet-relocated assets dir.
    """
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.zip"


# Per-attempt socket timeout for asset downloads: a wedged connection (no data, no error)
# raises socket.timeout (an OSError) and falls into the retry loop instead of hanging.
_DOWNLOAD_TIMEOUT_S = 120


def _download_and_extract_zip(url: str, dest: Path, retries: int = 3) -> None:
    r"""Download ``url`` (a ``.zip``) and extract it so its top-level dir lands at ``dest``.

    Mirrors robocasa's own downloader (extract into ``dest.parent``; the zip's single
    top-level directory matches ``dest.name``, and the download is retried a few times —
    each attempt bounded by a socket timeout so a hung connection still fails over) but
    without importing anything from robocasa. ``extractall`` is not atomic, but the caller
    only writes the pack's done-marker after this returns cleanly, so a failed/partial
    extract is simply retried and overwritten on the next attempt/run.
    """
    download_dir = dest.parent
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / f"{dest.name}.zip"
    last_err: Exception | None = None
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_DOWNLOAD_TIMEOUT_S)
    try:
        for attempt in range(1, retries + 1):
            try:
                urlretrieve(url, filename=str(zip_path))  # noqa: S310  # nosec B310 - trusted Box URL
                with ZipFile(zip_path) as zf:
                    zf.extractall(path=str(download_dir))  # noqa: S202  # nosec B202 - trusted archive
                return
            except (OSError, BadZipFile) as err:
                last_err = err
                acc_print(f"[opentau] RoboCasa asset download attempt {attempt}/{retries} failed: {err}")
            finally:
                if zip_path.exists():
                    zip_path.unlink()
        raise RuntimeError(
            f"Failed to download RoboCasa assets from {url} after {retries} attempts"
        ) from last_err
    finally:
        socket.setdefaulttimeout(old_timeout)


def _load_box_links(pkg_assets: Path, external_root: Path) -> dict:
    r"""Load robocasa's ``box_links_assets.json`` (the per-pack download URLs)."""
    for base in (external_root, pkg_assets):
        path = base / "box_links" / "box_links_assets.json"
        if path.is_file():
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"box_links_assets.json not found under {external_root} or {pkg_assets}.")


def _symlink_pkg_assets_to(pkg_assets: Path, external_root: Path) -> None:
    r"""Make ``pkg_assets`` (``<venv>/robocasa/models/assets``) a symlink to ``external_root``.

    This is what relocates assets out of the ephemeral venv: robocasa keeps reading its
    hardcoded ``models/assets`` path, but that path now points at the external store, so
    even the import-time object-registry scan sees the downloaded packs. Idempotent.
    """
    target = external_root.resolve()
    if pkg_assets.is_symlink():
        if pkg_assets.resolve() == target:
            return
        pkg_assets.unlink()
    elif pkg_assets.exists():
        shutil.rmtree(pkg_assets)
    pkg_assets.parent.mkdir(parents=True, exist_ok=True)
    pkg_assets.symlink_to(target, target_is_directory=True)


def _maybe_relink_robocasa_assets() -> None:
    r"""Point robocasa's assets dir at the external store, *if* that store is ready.

    Runs in every process right before robocasa is imported (notably ``AsyncVectorEnv``
    spawn workers, which import robocasa fresh and must see the symlink the main process
    created). It only acts once the store has been seeded by ``_ensure_robocasa_assets``,
    so a plain install with no relocation keeps its wheel-bundled assets untouched.
    """
    external_root = _resolve_robocasa_assets_root()
    if not (external_root / ".opentau_seeded").exists():
        return
    _symlink_pkg_assets_to(_robocasa_pkg_assets_dir(), external_root)


def _import_robocasa_with_version_shim() -> None:
    """Import the top-level ``robocasa`` package despite its over-strict pins.

    robocasa 1.0.1 hard-asserts ``mujoco.__version__ == "3.3.1"`` and
    ``numpy.__version__ in ["2.2.5"]`` at import time. OpenTau shares LIBERO's
    stack (robosuite 1.5.2 + ``mujoco>=3.3.5`` + ``numpy>=2.2.6``) — and robosuite
    1.5.2 itself only needs ``mujoco>=3.3.0`` / ``numpy>=1.13.3`` — so those
    equality pins are too strict to satisfy here. Spoof the two version strings
    only for the one-time package import, then restore them. A robocasa packaging
    fork that drops the asserts makes this shim unnecessary; until then it lets
    the integration run on a stock ``pip install robocasa``. No-op once imported.

    Also skips OpenTau's internal ``opentau/scripts/robocasa`` package, which would
    otherwise shadow the sim under the script-launched entry points (see
    :func:`_internal_robocasa_pkg_dir`).
    """
    # Already imported as the *real* sim -> nothing to do. (A shadow import is
    # purged inside ``_robocasa_unshadowed`` so the real package can load.)
    existing = sys.modules.get("robocasa")
    if existing is not None:
        mod_file = getattr(existing, "__file__", "") or ""
        if os.path.realpath(os.path.dirname(mod_file)) != _internal_robocasa_pkg_dir():
            return
    import mujoco
    import numpy

    saved = (mujoco.__version__, numpy.__version__)
    try:
        mujoco.__version__ = "3.3.1"
        numpy.__version__ = "2.2.5"
        # Relocate robocasa's assets dir to the external store (a symlink) BEFORE importing
        # it: the import eagerly builds its object registry by scanning models/assets, so
        # the symlink must already be in place or the registry comes up empty (every object
        # category gets zero candidates -> "Probabilities contain NaN" at sampling time).
        _maybe_relink_robocasa_assets()
        with _robocasa_unshadowed():
            import robocasa  # noqa: F401  (runs robocasa's version asserts once)
    finally:
        mujoco.__version__, numpy.__version__ = saved


def _resolve_tasks(task: str) -> tuple[list[str], str | None]:
    """Resolve an ``env.task`` value to ``(task_names, split_override)``.

    If ``task`` is a known task-group name (e.g. ``atomic_seen``,
    ``pretrain100``), expand it via
    ``robocasa.utils.dataset_registry.{TARGET,PRETRAINING}_TASKS`` and return the
    matching split. Otherwise treat ``task`` as a single task or comma-separated
    list and leave the split untouched (``None``). The ``robocasa`` import is
    deferred to here so only group expansion requires the sim package.
    """
    key = task.strip()
    if key in _TASK_GROUP_SPLITS:
        _import_robocasa_with_version_shim()
        from robocasa.utils.dataset_registry import PRETRAINING_TASKS, TARGET_TASKS

        combined = {**TARGET_TASKS, **PRETRAINING_TASKS}
        if key not in combined:
            raise ValueError(
                f"Task group '{key}' is not available in this version of robocasa. "
                f"Known groups: {sorted(combined.keys())}."
            )
        return list(combined[key]), _TASK_GROUP_SPLITS[key]

    names = [t.strip() for t in task.split(",") if t.strip()]
    if not names:
        raise ValueError("`task` must contain at least one RoboCasa task name.")
    return names, None


# RoboCasa ships its assets as separately-downloaded packs. Every kitchen scene needs the
# base textures (plain + AI-generated wall/floor textures) and lightwheel fixtures; object
# meshes depend on which registries the env samples from (``obj_registries``). Keys below
# are the registry names accepted in ``RoboCasaEnv.obj_registries``; values map to packs.
_BASE_ASSET_PACKS: tuple[str, ...] = ("textures", "tex_generative", "fixtures_lw")
_REGISTRY_TO_PACK: dict[str, str] = {
    "lightwheel": "objs_lw",
    "objaverse": "objs_objaverse",
    "aigen": "objs_aigen",
    "aigen_objs": "objs_aigen",
}
# pack name -> (key into robocasa's ``BOX_LINKS``, destination subdir under assets_root)
_PACK_DEST: dict[str, tuple[str, str]] = {
    "textures": ("textures", "textures"),
    "tex_generative": ("generative_textures", "generative_textures"),
    "fixtures_lw": ("fixtures_lightwheel", "fixtures"),
    "objs_lw": ("objects_lightwheel", "objects/lightwheel"),
    "objs_objaverse": ("objaverse", "objects/objaverse"),
    "objs_aigen": ("aigen_objs", "objects/aigen_objs"),
}


def _needed_asset_packs(obj_registries: Sequence[str]) -> list[str]:
    r"""Asset packs required for ``obj_registries``: the base packs + per-registry objects."""
    packs = list(_BASE_ASSET_PACKS)
    for reg in obj_registries:
        pack = _REGISTRY_TO_PACK.get(reg)
        if pack is not None and pack not in packs:
            packs.append(pack)
    return packs


def _ensure_robocasa_assets(assets_root: Path, obj_registries: Sequence[str]) -> None:
    r"""Download the asset packs ``obj_registries`` needs into ``assets_root`` and relocate.

    One-time, idempotent, and safe under distributed eval: only the main (or solo) process
    seeds / downloads / symlinks, and **every** rank meets at a barrier afterwards (so
    callers must invoke this before any per-rank early return, or non-main ranks desync at
    NCCL).

    robocasa ships some assets inside the wheel (fixture / scene / arena XML) that no
    download pack contains, so the external store is first seeded from the installed
    package, then the heavy packs (object meshes, textures, fixture meshes) are downloaded
    into it, and finally the venv's ``models/assets`` is replaced by a symlink to the
    store. Nothing here imports robocasa — that would build its object registry against the
    not-yet-relocated assets dir — so the package is located via its import spec instead.
    Per-pack marker files make re-runs (and the mixed ``fixtures`` dir, which holds both
    shipped XML and downloaded meshes) skip cleanly.

    Note: only the *global* main process downloads, so under multi-node eval ``assets_root``
    must be on storage shared by all nodes. The default (under ``HF_OPENTAU_HOME``) is fine
    for single-node eval; for a node-local ``HF_OPENTAU_HOME``, point ``ROBOCASA_ASSETS_ROOT``
    at a shared path so non-zero nodes find the seeded store.
    """
    pkg_assets = _robocasa_pkg_assets_dir()
    needed = _needed_asset_packs(obj_registries)

    acc = get_proc_accelerator()
    distributed = acc is not None and acc.num_processes > 1
    is_main_or_solo = (not distributed) or acc.is_main_process
    try:
        if is_main_or_solo:
            assets_root.mkdir(parents=True, exist_ok=True)
            # 1) Seed the wheel-shipped stubs (XML not in any pack) from the real venv dir.
            seed_marker = assets_root / ".opentau_seeded"
            if not pkg_assets.is_symlink() and pkg_assets.exists() and not seed_marker.exists():
                shutil.copytree(pkg_assets, assets_root, dirs_exist_ok=True)
                seed_marker.touch()
            # 2) Download each missing pack into the external store.
            box_links = _load_box_links(pkg_assets, assets_root)
            for pack in needed:
                pack_marker = assets_root / f".opentau_pack_{pack}.done"
                if pack_marker.exists():
                    continue
                box_key, subdir = _PACK_DEST[pack]
                dest = assets_root / subdir
                acc_print(f"[opentau] downloading RoboCasa asset pack '{pack}' -> {dest}")
                _download_and_extract_zip(_direct_download_url(box_links[box_key]), dest)
                pack_marker.touch()
            # 3) Relocate: point the venv's assets dir at the populated external store so
            #    robocasa's import-time registry scan reads it.
            if seed_marker.exists():
                _symlink_pkg_assets_to(pkg_assets, assets_root)
        if distributed:
            acc.wait_for_everyone()
    except Exception:
        # Keep ranks in lockstep even if the main process fails here, so non-main ranks
        # don't hang at the next collective; the exception still aborts the main process.
        if distributed:
            acc.wait_for_everyone()
        raise


def convert_action(flat_action: np.ndarray) -> dict[str, Any]:
    """Split a flat ``(12,)`` action vector into a RoboCasa action dict.

    Layout: base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1).
    """
    return {
        "action.base_motion": flat_action[0:4],
        "action.control_mode": flat_action[4:5],
        "action.end_effector_position": flat_action[5:8],
        "action.end_effector_rotation": flat_action[8:11],
        "action.gripper_close": flat_action[11:12],
    }


class RoboCasaEnv(gym.Env):
    r"""Gym wrapper for RoboCasa365 kitchen environments.

    Wraps ``RoboCasaGymEnv`` from the ``robocasa`` package and converts its
    dict-based observations/actions into the flat arrays OpenTau expects. Raw
    camera frames are remapped to ``camera{i}`` keys (see
    ``camera_name_mapping``) so the policy input structure matches LIBERO.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task: str,
        camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 512,
        visualization_height: int = 512,
        split: str | None = None,
        episode_length: int | None = None,
        obj_registries: Sequence[str] = DEFAULT_OBJ_REGISTRIES,
        episode_index: int = 0,
        camera_name_mapping: dict[str, list[str]] | None = None,
    ):
        r"""Initialize the RoboCasaEnv.

        Args:
            task: RoboCasa task name (e.g. ``"CloseFridge"``).
            camera_name: Raw RoboCasa camera name(s); comma-separated string or
                sequence. Both count and order are driven by this value.
            obs_type: ``"pixels"`` or ``"pixels_agent_pos"``.
            render_mode: Rendering mode for the environment.
            observation_width: Width of observation images.
            observation_height: Height of observation images.
            visualization_width: Width of visualization frames.
            visualization_height: Height of visualization frames.
            split: RoboCasa dataset split (``None``/``"all"``/``"pretrain"``/``"target"``).
            episode_length: Max steps per episode (``_max_episode_steps``); defaults to 1000.
            obj_registries: Object-mesh registries to sample assets from.
            episode_index: Per-worker index (``0..n_envs-1``) used to spread the
                ``reset`` seed so each sub-env explores a distinct layout.
            camera_name_mapping: Optional mapping from raw camera names to
                positional ``camera{i}`` keys; defaults to first→``camera0``, etc.
        """
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.split = split
        self.obj_registries = tuple(obj_registries)
        self.episode_index = int(episode_index)

        self.camera_name = _parse_camera_names(camera_name)
        if camera_name_mapping is None:
            camera_name_mapping = _default_camera_name_mapping(self.camera_name)
        self.camera_name_mapping = camera_name_mapping
        for cam in self.camera_name_mapping:
            assert not isinstance(self.camera_name_mapping[cam], str), (
                "camera_name_mapping values must be lists of strings; "
                f"got string {self.camera_name_mapping[cam]} for {cam} instead"
            )

        self._max_episode_steps = episode_length if episode_length is not None else 1000

        # Deferred — created on first reset()/render()/step() inside the worker
        # subprocess to avoid inheriting a stale GPU/EGL context across fork().
        self._env: Any = None
        self.task_description = ""
        # Latest per-camera frames ({camera0,camera1,...}), cached by
        # _format_raw_obs so render() can composite all cameras for eval videos.
        self._render_pixels: dict[str, np.ndarray] | None = None

        images = {}
        for cam in self.camera_name:
            for mapped_cam in self.camera_name_mapping[cam]:
                images[mapped_cam] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                )

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(OBS_STATE_DIM,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type '{self.obs_type}'. Use 'pixels' or 'pixels_agent_pos'.")

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def _ensure_env(self) -> None:
        r"""Create the underlying ``RoboCasaGymEnv`` on first use.

        Called inside the worker subprocess (after fork/spawn) so each worker
        gets its own clean rendering context rather than inheriting a stale one
        from the parent (which crashes with ``AsyncVectorEnv``).
        """
        if self._env is not None:
            return
        # Mute the noisy one-per-worker robosuite/robocasa import + construction
        # banner (private-macro / mink / mimicgen / controller-config lines): with
        # AsyncVectorEnv there is one copy per spawn worker, i.e. n_envs * world_size
        # duplicates flooding the log on every eval. Captured output is replayed to
        # stderr only if construction raises, so a real failure stays debuggable.
        with silence_output_unless_error(label=f"task={self.task} idx={self.episode_index}"):
            _import_robocasa_with_version_shim()
            from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

            # RoboCasaGymEnv defaults split="test", which create_env rejects (only
            # None/"all"/"pretrain"/"target" are valid). Always pass a valid value.
            self._env = RoboCasaGymEnv(
                env_name=self.task,
                camera_widths=self.observation_width,
                camera_heights=self.observation_height,
                split=self.split if self.split is not None else "all",
                obj_registries=self.obj_registries,
            )

            ep_meta = self._env.env.get_ep_meta()
            self.task_description = ep_meta.get("lang", self.task)

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a ``RoboCasaGymEnv`` observation dict to OpenTau format."""
        # RoboCasaGymEnv emits camera frames under "video.<cam>".
        images: dict[str, np.ndarray] = {}
        for cam in self.camera_name:
            key = f"video.{cam}"
            if key not in raw_obs:
                continue
            frame = raw_obs[key]
            for mapped_cam in self.camera_name_mapping[cam]:
                images[mapped_cam] = frame

        # Cache for render(): the underlying RoboCasaGymEnv.render() is single-cam,
        # so we composite these frames ourselves to show every camera in eval videos.
        self._render_pixels = images

        if self.obs_type == "pixels":
            return {"pixels": images}

        # `state.*` keys come from PandaOmronKeyConverter inside the wrapper.
        agent_pos = np.concatenate(
            [
                raw_obs.get("state.base_position", np.zeros(3)),
                raw_obs.get("state.base_rotation", np.zeros(4)),
                raw_obs.get("state.end_effector_position_relative", np.zeros(3)),
                raw_obs.get("state.end_effector_rotation_relative", np.zeros(4)),
                raw_obs.get("state.gripper_qpos", np.zeros(2)),
            ],
            axis=-1,
        ).astype(np.float32)

        return {"pixels": images, "agent_pos": agent_pos}

    def render(self) -> np.ndarray:
        r"""Render an RGB array for video recording: all configured cameras
        concatenated side-by-side (left→right in ``camera_name`` order), so eval
        rollout videos show every camera the policy sees rather than just one."""
        self._ensure_env()
        assert self._env is not None
        if self._render_pixels:
            frames = list(self._render_pixels.values())  # each (H, W, 3), camera_name order
            return einops.rearrange(np.stack(frames), "n h w c -> h (n w) c")
        # No observation cached yet (render before the first reset/step).
        return self._env.render()

    def reset(self, seed=None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Reset the environment, deriving a per-worker seed from ``episode_index``."""
        self._ensure_env()
        assert self._env is not None
        super().reset(seed=seed)
        # Spread the seed across workers so n_envs factories don't all roll the
        # same scene: shift an explicit seed by episode_index; with no seed fall
        # back to episode_index so each worker is still distinct.
        worker_seed = seed + self.episode_index if seed is not None else self.episode_index
        raw_obs, _ = self._env.reset(seed=worker_seed)

        ep_meta = self._env.env.get_ep_meta()
        self.task_description = ep_meta.get("lang", self.task)

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        r"""Take a step; remaps RoboCasa's ``info["success"]`` to ``is_success``."""
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        # Policies may emit a padded action wider than ACTION_DIM; keep the
        # leading RoboCasa dims (mirrors LiberoEnv).
        if len(action) > ACTION_DIM:
            action = action[:ACTION_DIM]

        action_dict = convert_action(action)
        raw_obs, reward, done, truncated, info = self._env.step(action_dict)

        # RoboCasa reports success under "success"; OpenTau's rollout reads
        # "is_success". Bridge the two here.
        is_success = bool(info.get("success", False))
        terminated = done or is_success
        info.update({"task": self.task, "done": done, "is_success": is_success})

        observation = self._format_raw_obs(raw_obs)
        if terminated:
            # Auto-reset on terminal so the next rollout step starts a fresh
            # episode (the rollout masks post-done data). Mirrors LiberoEnv.step;
            # the task / done / is_success keys set above are what eval consumes.
            self.reset()

        return observation, reward, terminated, truncated, info

    def close(self):
        r"""Close the environment and release any resources."""
        if self._env is not None:
            self._env.close()


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    camera_names: list[str],
    obs_type: str,
    render_mode: str,
    observation_width: int,
    observation_height: int,
    visualization_width: int,
    visualization_height: int,
    split: str | None,
    episode_length: int | None,
    obj_registries: Sequence[str],
) -> list[Callable[[], RoboCasaEnv]]:
    """Build ``n_envs`` factory callables for a single task.

    Each factory carries a distinct ``episode_index`` (``0..n_envs-1``) so
    ``RoboCasaEnv.reset()`` derives a per-worker seed from the rollout seed.
    """

    def _make_env(episode_index: int) -> RoboCasaEnv:
        return RoboCasaEnv(
            task=task,
            camera_name=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            split=split,
            episode_length=episode_length,
            obj_registries=obj_registries,
            episode_index=episode_index,
        )

    return [partial(_make_env, i) for i in range(n_envs)]


# main API entry point
def create_robocasa_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = ",".join(DEFAULT_CAMERAS),
    env_cls: type[gym.vector.SyncVectorEnv] | type[gym.vector.AsyncVectorEnv] | None = None,
    episode_length: int | None = None,
    obj_registries: Sequence[str] = DEFAULT_OBJ_REGISTRIES,
    assets_root: str | None = None,
    auto_download_assets: bool = True,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    r"""Create vectorized RoboCasa365 environments with a consistent return shape.

    Returns:
        ``dict[task_name][0] -> vec_env`` (``env_cls([...])`` with ``n_envs``
        factories). Each distinct task is its own group, so eval reports a
        per-task ``Success/{task}`` and a per-task ``Eval Videos/{task}_0`` grid.

    ``task`` can be a single task name (``CloseFridge``), a comma-separated list
    (``CloseFridge,PickPlaceCoffee``), or a benchmark-group shortcut
    (``atomic_seen``/``composite_seen``/``composite_unseen``/``pretrain50``…),
    which auto-expands and auto-sets the dataset ``split``.

    When run under an accelerator with multiple processes, tasks are sharded
    round-robin (``idx % num_processes == process_index``) so each rank evaluates
    a disjoint subset — matching LIBERO and keeping the ``_rank{N}``-strip
    uniqueness assumption in ``collect_grid_summary_videos`` valid.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    # Relocate RoboCasa assets out of the (ephemeral) uv venv and make the packs the
    # requested registries need available. Resolve + export the root, then run the download
    # BEFORE `_resolve_tasks` (or any other) robocasa import, so the assets-dir symlink is
    # already in place when robocasa builds its object registry at import. Propagate the
    # root via os.environ so AsyncVectorEnv spawn workers relink to the same store. The
    # download is rank-0 gated with an internal barrier, run before the per-rank task
    # sharding / empty-rank early return below so every rank reaches that barrier.
    resolved_assets_root = (
        _resolve_robocasa_assets_root() if assets_root is None else Path(assets_root).expanduser()
    )
    os.environ[ROBOCASA_ASSETS_ROOT_ENV] = str(resolved_assets_root)
    if auto_download_assets:
        _ensure_robocasa_assets(resolved_assets_root, obj_registries)

    gym_kwargs = dict(gym_kwargs or {})
    obs_type = gym_kwargs.pop("obs_type", "pixels_agent_pos")
    render_mode = gym_kwargs.pop("render_mode", "rgb_array")
    observation_width = gym_kwargs.pop("observation_width", 256)
    observation_height = gym_kwargs.pop("observation_height", 256)
    visualization_width = gym_kwargs.pop("visualization_width", 512)
    visualization_height = gym_kwargs.pop("visualization_height", 512)
    split = gym_kwargs.pop("split", None)

    camera_names = _parse_camera_names(camera_name)
    task_names, group_split = _resolve_tasks(str(task))
    if group_split is not None and split is None:
        split = group_split

    # Shard tasks across accelerator ranks (round-robin), so distributed eval
    # spreads tasks disjointly and per-task video keys stay unique after the
    # `_rank{N}` strip in `collect_grid_summary_videos`.
    accelerator = get_proc_accelerator()
    if accelerator is not None:
        task_names = [
            t
            for idx, t in enumerate(task_names)
            if idx % accelerator.num_processes == accelerator.process_index
        ]
        acc_print(
            f"Creating RoboCasa envs | tasks={task_names} | split={split} | "
            f"n_envs(per task)={n_envs} | rank={accelerator.process_index}"
        )
    else:
        acc_print(f"Creating RoboCasa envs | tasks={task_names} | split={split} | n_envs(per task)={n_envs}")

    # No tasks on this rank (more ranks than tasks) → empty dict, like LIBERO.
    if not task_names:
        acc_print("No RoboCasa tasks assigned to this rank, returning empty dict.")
        return {}

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for task_name in task_names:
        fns = _make_env_fns(
            task=task_name,
            n_envs=n_envs,
            camera_names=camera_names,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            split=split,
            episode_length=episode_length,
            obj_registries=obj_registries,
        )
        out[task_name][0] = env_cls(fns)
        acc_print(f"Built vec env | task={task_name} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {name: dict(task_map) for name, task_map in out.items()}
