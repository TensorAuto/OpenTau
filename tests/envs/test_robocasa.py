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

r"""CPU-only tests for the RoboCasa env integration.

These never import the ``robocasa`` / ``robosuite`` sim packages: the wrapper
defers those imports into ``_ensure_env`` / ``_resolve_tasks``, so config
registration, the factory dispatch, per-rank task sharding, and the pure
action/camera helpers are all exercisable without a GPU or the sim installed.
Real sim rollouts are validated separately on a CUDA box.
"""

import contextlib
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opentau.configs.train import TrainPipelineConfig
from opentau.configs.types import FeatureType
from opentau.constants import ACTION, HF_OPENTAU_HOME, OBS_IMAGES, OBS_STATE
from opentau.envs.configs import EnvConfig, RoboCasaEnv
from opentau.envs.factory import make_env_config, make_envs
from opentau.envs.robocasa import (
    ACTION_DIM,
    OBS_STATE_DIM,
    ROBOCASA_ASSETS_ROOT_ENV,
    _default_camera_name_mapping,
    _direct_download_url,
    _ensure_robocasa_assets,
    _import_robocasa_with_version_shim,
    _internal_robocasa_pkg_dir,
    _maybe_relink_robocasa_assets,
    _needed_asset_packs,
    _official_task_horizon,
    _parse_camera_names,
    _resolve_robocasa_assets_root,
    _resolve_split,
    _resolve_tasks,
    _robocasa_pkg_assets_dir,
    _robocasa_unshadowed,
    _symlink_pkg_assets_to,
    convert_action,
    create_robocasa_envs,
)


class TestRoboCasaConfig:
    """Config registration, defaults, and feature wiring."""

    def test_registered_in_choice_registry(self):
        assert "robocasa" in EnvConfig._choice_registry
        assert EnvConfig._choice_registry["robocasa"] is RoboCasaEnv
        assert RoboCasaEnv().type == "robocasa"

    def test_default_values(self):
        cfg = RoboCasaEnv()
        assert cfg.task == "CloseFridge"
        assert cfg.fps == 20
        assert cfg.episode_length == 1000
        assert cfg.obs_type == "pixels_agent_pos"
        assert cfg.observation_height == 256
        assert cfg.observation_width == 256
        assert cfg.split is None
        assert cfg.obj_registries == ["lightwheel"]
        assert len(_parse_camera_names(cfg.camera_name)) == 3

    def test_action_and_state_features(self):
        cfg = RoboCasaEnv()
        assert cfg.features["action"].type == FeatureType.ACTION
        assert cfg.features["action"].shape == (12,)
        # pixels_agent_pos adds a 16-D proprio state.
        assert cfg.features["agent_pos"].type == FeatureType.STATE
        assert cfg.features["agent_pos"].shape == (16,)
        assert cfg.features_map["action"] == ACTION
        assert cfg.features_map["agent_pos"] == OBS_STATE

    def test_camera_features_map_uses_image_convention(self):
        """The 3 raw cameras map to OpenTau's image/image2/image3 keys."""
        cfg = RoboCasaEnv()
        cams = _parse_camera_names(cfg.camera_name)
        assert cfg.features_map[f"pixels/{cams[0]}"] == f"{OBS_IMAGES}.image"
        assert cfg.features_map[f"pixels/{cams[1]}"] == f"{OBS_IMAGES}.image2"
        assert cfg.features_map[f"pixels/{cams[2]}"] == f"{OBS_IMAGES}.image3"
        for cam in cams:
            assert cfg.features[f"pixels/{cam}"].type == FeatureType.VISUAL
            assert cfg.features[f"pixels/{cam}"].shape == (256, 256, 3)

    def test_pixels_only_omits_agent_pos(self):
        cfg = RoboCasaEnv(obs_type="pixels")
        assert "agent_pos" not in cfg.features

    @pytest.mark.parametrize("bad_fps", [0, -5])
    def test_rejects_nonpositive_fps(self, bad_fps):
        with pytest.raises(ValueError, match="must be positive"):
            RoboCasaEnv(fps=bad_fps)

    def test_rejects_unsupported_obs_type(self):
        with pytest.raises(ValueError, match="Unsupported obs_type"):
            RoboCasaEnv(obs_type="state")

    def test_gym_kwargs_carries_obs_params_and_split(self):
        cfg = RoboCasaEnv(split="pretrain")
        kwargs = cfg.gym_kwargs
        assert kwargs["obs_type"] == "pixels_agent_pos"
        assert kwargs["observation_height"] == 256
        assert kwargs["observation_width"] == 256
        assert kwargs["visualization_height"] == 512
        assert kwargs["split"] == "pretrain"

    def test_gym_kwargs_omits_split_when_none(self):
        assert "split" not in RoboCasaEnv().gym_kwargs


class TestMakeEnvConfigRoboCasa:
    """``make_env_config`` dispatch."""

    def test_make_env_config_robocasa(self):
        cfg = make_env_config("robocasa")
        assert isinstance(cfg, RoboCasaEnv)
        assert cfg.task == "CloseFridge"

    def test_make_env_config_robocasa_with_overrides(self):
        cfg = make_env_config("robocasa", task="OpenDrawer", split="target")
        assert cfg.task == "OpenDrawer"
        assert cfg.split == "target"


class TestPureHelpers:
    """Helpers that never touch the sim."""

    def test_convert_action_layout(self):
        flat = np.arange(ACTION_DIM, dtype=np.float32)
        out = convert_action(flat)
        np.testing.assert_array_equal(out["action.base_motion"], flat[0:4])
        np.testing.assert_array_equal(out["action.control_mode"], flat[4:5])
        np.testing.assert_array_equal(out["action.end_effector_position"], flat[5:8])
        np.testing.assert_array_equal(out["action.end_effector_rotation"], flat[8:11])
        np.testing.assert_array_equal(out["action.gripper_close"], flat[11:12])

    def test_parse_camera_names(self):
        assert _parse_camera_names("a, b ,c") == ["a", "b", "c"]
        assert _parse_camera_names(["x", "y"]) == ["x", "y"]
        with pytest.raises(ValueError):
            _parse_camera_names(" , ")

    def test_default_camera_name_mapping_is_positional(self):
        mapping = _default_camera_name_mapping(["cam_a", "cam_b", "cam_c"])
        assert mapping == {"cam_a": ["camera0"], "cam_b": ["camera1"], "cam_c": ["camera2"]}

    def test_resolve_single_and_comma_tasks_no_split(self):
        # Concrete task names never import robocasa and leave split untouched.
        assert _resolve_tasks("CloseFridge") == (["CloseFridge"], None)
        names, split = _resolve_tasks("CloseFridge, PickPlaceCoffee")
        assert names == ["CloseFridge", "PickPlaceCoffee"]
        assert split is None

    def test_resolve_empty_task_raises(self):
        with pytest.raises(ValueError, match="at least one RoboCasa task"):
            _resolve_tasks("  ,  ")

    def test_constants(self):
        assert ACTION_DIM == 12
        assert OBS_STATE_DIM == 16

    def test_version_shim_is_noop_when_robocasa_already_imported(self):
        # When robocasa is already in sys.modules the shim must return early
        # without importing mujoco/numpy, so CPU-only runs (no sim) never touch
        # those modules. Inject a placeholder and assert it returns cleanly.
        import sys

        had = "robocasa" in sys.modules
        saved = sys.modules.get("robocasa")
        sys.modules["robocasa"] = object()
        try:
            _import_robocasa_with_version_shim()  # must not raise / import anything
        finally:
            if had:
                sys.modules["robocasa"] = saved
            else:
                sys.modules.pop("robocasa", None)


@contextlib.contextmanager
def _fake_dataset_registry(atomic: dict, composite: dict):
    """Inject a fake ``robocasa.utils.dataset_registry`` for the duration of the
    block so ``_official_task_horizon`` resolves against it without the real sim.

    The version shim is patched to a no-op (it would otherwise import mujoco/numpy
    and assert versions); only the registry module is faked, mirroring how the
    other tests stub ``robocasa`` in ``sys.modules``.
    """
    import sys
    import types

    mod = types.ModuleType("robocasa.utils.dataset_registry")
    mod.ATOMIC_TASK_DATASETS = atomic
    mod.COMPOSITE_TASK_DATASETS = composite
    saved = sys.modules.get("robocasa.utils.dataset_registry")
    sys.modules["robocasa.utils.dataset_registry"] = mod
    try:
        with patch("opentau.envs.robocasa._import_robocasa_with_version_shim"):
            yield
    finally:
        if saved is not None:
            sys.modules["robocasa.utils.dataset_registry"] = saved
        else:
            sys.modules.pop("robocasa.utils.dataset_registry", None)


@contextlib.contextmanager
def _fake_task_group_registry(target: dict, pretraining: dict):
    """Inject a fake ``robocasa.utils.dataset_registry`` exposing ``TARGET_TASKS``
    / ``PRETRAINING_TASKS`` so ``_resolve_tasks`` can expand group shortcuts
    without the real sim (mirrors ``_fake_dataset_registry``)."""
    import sys
    import types

    mod = types.ModuleType("robocasa.utils.dataset_registry")
    mod.TARGET_TASKS = target
    mod.PRETRAINING_TASKS = pretraining
    saved = sys.modules.get("robocasa.utils.dataset_registry")
    sys.modules["robocasa.utils.dataset_registry"] = mod
    try:
        with patch("opentau.envs.robocasa._import_robocasa_with_version_shim"):
            yield
    finally:
        if saved is not None:
            sys.modules["robocasa.utils.dataset_registry"] = saved
        else:
            sys.modules.pop("robocasa.utils.dataset_registry", None)


class TestResolveSplitAndTaskGroups:
    """Split resolution + task-group → split mapping.

    Pins that every benchmark-group shortcut resolves to the ``pretrain`` split
    and that concrete single-task configs default to ``pretrain`` too, so an
    accidental revert of either default fails CI instead of passing silently.
    """

    _TARGET_GROUPS = ("atomic_seen", "composite_seen", "composite_unseen")
    _PRETRAIN_GROUPS = ("pretrain50", "pretrain100", "pretrain200", "pretrain300")

    def test_all_task_groups_resolve_to_pretrain_split(self):
        target = {g: [f"{g}Task"] for g in self._TARGET_GROUPS}
        pretraining = {g: [f"{g}Task"] for g in self._PRETRAIN_GROUPS}
        with _fake_task_group_registry(target, pretraining):
            for group in self._TARGET_GROUPS + self._PRETRAIN_GROUPS:
                names, split = _resolve_tasks(group)
                assert names == [f"{group}Task"]
                assert split == "pretrain", f"{group} must resolve to the pretrain split"

    def test_resolve_split_defaults_unset_to_pretrain(self):
        # Concrete / comma-separated task names carry no group split -> pretrain.
        assert _resolve_split(None, None) == "pretrain"

    def test_resolve_split_uses_group_split_when_no_explicit(self):
        assert _resolve_split(None, "pretrain") == "pretrain"
        assert _resolve_split(None, "target") == "target"

    def test_resolve_split_explicit_overrides_group_and_default(self):
        assert _resolve_split("all", None) == "all"
        assert _resolve_split("target", "pretrain") == "target"


class TestOfficialTaskHorizon:
    """``_official_task_horizon`` reads per-task horizons from the registry."""

    def test_known_atomic_task_returns_registry_horizon(self):
        with _fake_dataset_registry({"CloseFridge": {"horizon": 900}}, {}):
            assert _official_task_horizon("CloseFridge") == 900

    def test_known_composite_task_returns_registry_horizon(self):
        with _fake_dataset_registry({}, {"ArrangeVegetables": {"horizon": 1200}}):
            assert _official_task_horizon("ArrangeVegetables") == 1200

    def test_horizon_is_coerced_to_int(self):
        with _fake_dataset_registry({"CloseFridge": {"horizon": 450.0}}, {}):
            result = _official_task_horizon("CloseFridge")
        assert result == 450 and isinstance(result, int)

    def test_unknown_task_returns_none(self):
        with _fake_dataset_registry({"CloseFridge": {"horizon": 900}}, {}):
            assert _official_task_horizon("NoSuchTask") is None

    def test_entry_without_horizon_returns_none(self):
        with _fake_dataset_registry({"WeirdTask": {"name": "x"}}, {}):
            assert _official_task_horizon("WeirdTask") is None

    def test_import_failure_returns_none(self):
        # robocasa not installed -> the shim raises -> caught -> None (1000-step
        # fallback). This is the path the new diagnostic log guards.
        with patch(
            "opentau.envs.robocasa._import_robocasa_with_version_shim",
            side_effect=ImportError("no robocasa"),
        ):
            assert _official_task_horizon("CloseFridge") is None


class TestRobocasaUnshadow:
    """The import-shadow guard.

    OpenTau's internal ``opentau/scripts/robocasa`` package (the inference
    WebSocket server) shares the bare name ``robocasa`` with the external sim.
    Under the script-launched entry points (``accelerate launch`` / the
    ``opentau-*`` console scripts) Python puts ``opentau/scripts`` on
    ``sys.path[0]``, so ``import robocasa`` / ``find_spec("robocasa")`` resolve
    to that empty package instead of the sim. These tests pin that
    ``_robocasa_unshadowed`` drops the shadowing path entry and evicts an
    already-imported shadow — without the real sim installed.
    """

    def test_internal_pkg_dir_resolves_to_scripts_robocasa(self):
        internal = _internal_robocasa_pkg_dir()
        assert os.path.isdir(internal)
        assert internal.endswith(os.path.join("opentau", "scripts", "robocasa"))
        # Already a realpath, so the comparisons inside the helper are stable.
        assert internal == os.path.realpath(internal)

    def test_unshadow_drops_shadowing_path_entry_and_restores(self):
        import importlib.util
        import sys

        internal = _internal_robocasa_pkg_dir()
        scripts_dir = os.path.dirname(internal)  # .../opentau/scripts

        saved_path = list(sys.path)
        sys.path.insert(0, scripts_dir)  # mimic train.py being launched as a script
        try:
            # The shadow is reachable before unshadowing...
            spec = importlib.util.find_spec("robocasa")
            assert spec is not None and spec.origin is not None
            assert os.path.realpath(os.path.dirname(spec.origin)) == internal

            before = list(sys.path)
            with _robocasa_unshadowed():
                # ...and no remaining sys.path entry resolves `robocasa` to it.
                assert all(os.path.realpath(os.path.join(p, "robocasa")) != internal for p in sys.path)
            # sys.path is restored verbatim on exit.
            assert sys.path == before
        finally:
            sys.path[:] = saved_path

    def test_unshadow_evicts_already_imported_shadow(self):
        import sys
        import types

        internal = _internal_robocasa_pkg_dir()
        had = "robocasa" in sys.modules
        saved = sys.modules.get("robocasa")
        saved_sub = sys.modules.get("robocasa.server")

        # Fake a shadow import: a module whose file lives in the internal package,
        # plus a submodule, exactly as importing the WebSocket server would leave them.
        shadow = types.ModuleType("robocasa")
        shadow.__file__ = os.path.join(internal, "__init__.py")
        sys.modules["robocasa"] = shadow
        sys.modules["robocasa.server"] = types.ModuleType("robocasa.server")
        try:
            with _robocasa_unshadowed():
                # The shadow and its submodules are evicted so resolution falls
                # through to the real sim.
                assert "robocasa" not in sys.modules
                assert "robocasa.server" not in sys.modules
        finally:
            sys.modules.pop("robocasa", None)
            sys.modules.pop("robocasa.server", None)
            if had:
                sys.modules["robocasa"] = saved
                if saved_sub is not None:
                    sys.modules["robocasa.server"] = saved_sub

    def test_unshadow_leaves_real_robocasa_import_untouched(self):
        """A ``robocasa`` already imported as the *real* sim (file outside the
        internal package) must not be evicted.
        """
        import sys
        import types

        had = "robocasa" in sys.modules
        saved = sys.modules.get("robocasa")

        real = types.ModuleType("robocasa")
        real.__file__ = os.path.join(os.sep, "opt", "site-packages", "robocasa", "__init__.py")
        sys.modules["robocasa"] = real
        try:
            with _robocasa_unshadowed():
                assert sys.modules.get("robocasa") is real
        finally:
            if had:
                sys.modules["robocasa"] = saved
            else:
                sys.modules.pop("robocasa", None)


def _mock_accelerator(num_processes: int, process_index: int) -> Mock:
    acc = Mock()
    acc.num_processes = num_processes
    acc.process_index = process_index
    return acc


class TestCreateRoboCasaEnvs:
    """``create_robocasa_envs`` return shape and per-rank task sharding.

    ``env_cls`` is mocked so the env factories are never invoked — no
    ``RoboCasaEnv`` is constructed and the sim is never imported. Asset
    auto-download is disabled here (its own tests cover it) so these stay sim-free.
    """

    def test_returns_one_vec_env_per_task(self):
        sentinel = Mock(name="vec_env")
        env_cls = Mock(return_value=sentinel)
        with patch("opentau.envs.robocasa.get_proc_accelerator", return_value=None):
            out = create_robocasa_envs(task="A,B", n_envs=3, env_cls=env_cls, auto_download_assets=False)
        assert set(out.keys()) == {"A", "B"}
        assert out["A"][0] is sentinel and out["B"][0] is sentinel
        # Each task built one vec env from exactly n_envs factory callables.
        assert env_cls.call_count == 2
        fns = env_cls.call_args[0][0]
        assert len(fns) == 3
        assert all(callable(f) for f in fns)

    @pytest.mark.parametrize(
        ("process_index", "expected"),
        [(0, {"A", "C"}), (1, {"B", "D"})],
    )
    def test_round_robin_task_sharding(self, process_index, expected):
        env_cls = Mock(return_value=Mock())
        acc = _mock_accelerator(num_processes=2, process_index=process_index)
        with patch("opentau.envs.robocasa.get_proc_accelerator", return_value=acc):
            out = create_robocasa_envs(task="A,B,C,D", n_envs=1, env_cls=env_cls, auto_download_assets=False)
        assert set(out.keys()) == expected

    def test_more_ranks_than_tasks_returns_empty(self):
        env_cls = Mock(return_value=Mock())
        acc = _mock_accelerator(num_processes=4, process_index=2)
        with patch("opentau.envs.robocasa.get_proc_accelerator", return_value=acc):
            out = create_robocasa_envs(task="A", n_envs=1, env_cls=env_cls, auto_download_assets=False)
        assert out == {}
        env_cls.assert_not_called()

    def test_rejects_bad_n_envs(self):
        # n_envs is validated before any accelerator call, so no patch needed.
        with pytest.raises(ValueError, match="positive int"):
            create_robocasa_envs(task="A", n_envs=0, env_cls=Mock())

    def test_rejects_non_callable_env_cls(self):
        with pytest.raises(ValueError, match="env_cls must be a callable"):
            create_robocasa_envs(task="A", n_envs=1, env_cls=None)

    def test_episode_length_none_threads_official_horizon_into_env_fns(self):
        # episode_length=None -> each task's official horizon is resolved and
        # threaded into _make_env_fns (the per-task duration the eval runs for).
        env_cls = Mock(return_value=Mock())
        with (
            patch("opentau.envs.robocasa.get_proc_accelerator", return_value=None),
            patch("opentau.envs.robocasa._official_task_horizon", return_value=777) as horizon,
            patch("opentau.envs.robocasa._make_env_fns", return_value=[lambda: None]) as make_fns,
        ):
            create_robocasa_envs(
                task="CloseFridge",
                n_envs=1,
                env_cls=env_cls,
                episode_length=None,
                auto_download_assets=False,
            )
        horizon.assert_called_once_with("CloseFridge")
        assert make_fns.call_args.kwargs["episode_length"] == 777

    def test_explicit_episode_length_skips_official_horizon(self):
        # An explicit int forces one global cap and never consults the registry.
        env_cls = Mock(return_value=Mock())
        with (
            patch("opentau.envs.robocasa.get_proc_accelerator", return_value=None),
            patch("opentau.envs.robocasa._official_task_horizon") as horizon,
            patch("opentau.envs.robocasa._make_env_fns", return_value=[lambda: None]) as make_fns,
        ):
            create_robocasa_envs(
                task="CloseFridge",
                n_envs=1,
                env_cls=env_cls,
                episode_length=450,
                auto_download_assets=False,
            )
        horizon.assert_not_called()
        assert make_fns.call_args.kwargs["episode_length"] == 450


class TestMakeEnvsDispatch:
    """``make_envs`` routes RoboCasa configs to ``create_robocasa_envs``."""

    @pytest.fixture
    def mock_train_cfg(self):
        return Mock(spec=TrainPipelineConfig)

    def test_make_envs_dispatches_to_create_robocasa_envs(self, mock_train_cfg):
        expected = {"CloseFridge": {0: Mock()}}
        with patch("opentau.envs.robocasa.create_robocasa_envs", return_value=expected) as mock_create:
            result = make_envs(RoboCasaEnv(), mock_train_cfg, n_envs=2, use_async_envs=False)

        assert result is expected
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["task"] == "CloseFridge"
        assert kwargs["n_envs"] == 2
        assert kwargs["episode_length"] == 1000
        assert kwargs["obj_registries"] == ("lightwheel",)
        assert kwargs["assets_root"] is None
        assert kwargs["auto_download_assets"] is True
        # SyncVectorEnv path: env_cls is the class itself.
        import gymnasium as gym

        assert kwargs["env_cls"] is gym.vector.SyncVectorEnv


def _fake_robocasa_assets(monkeypatch, tmp_path: Path) -> tuple[Path, list[dict]]:
    """Point ``_ensure_robocasa_assets`` at a throwaway pkg dir with a stubbed downloader.

    Mirrors a real robocasa install enough to exercise seed -> download -> symlink without
    the sim or the network: ``_robocasa_pkg_assets_dir`` resolves to a fake ``models/assets``
    dir holding the box-links JSON + a shipped stub, and ``_download_and_extract_zip`` just
    records its calls and creates the destination. Returns ``(pkg_assets, recorded_calls)``.
    """
    pkg_assets = tmp_path / "pkg" / "models" / "assets"
    (pkg_assets / "box_links").mkdir(parents=True)
    (pkg_assets / "box_links" / "box_links_assets.json").write_text(
        json.dumps(
            {
                "textures": "https://x.box.com/s/tex",
                "generative_textures": "https://x.box.com/s/gentex",
                "fixtures_lightwheel": "https://x.box.com/s/fix",
                "objects_lightwheel": "https://x.box.com/s/objlw",
                "objaverse": "https://x.box.com/s/obja",
                "aigen_objs": "https://x.box.com/s/aig",
            }
        )
    )
    (pkg_assets / "shipped_stub.xml").write_text("<mujoco/>")

    calls: list[dict] = []

    def _fake_download(url, dest):
        calls.append({"url": url, "dest": str(dest)})
        Path(dest).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("opentau.envs.robocasa._robocasa_pkg_assets_dir", lambda: pkg_assets)
    monkeypatch.setattr("opentau.envs.robocasa._download_and_extract_zip", _fake_download)
    return pkg_assets, calls


class TestResolveAssetsRoot:
    """``_resolve_robocasa_assets_root`` env-var resolution + default location."""

    def test_env_var_takes_precedence(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ROBOCASA_ASSETS_ROOT_ENV, str(tmp_path))
        assert _resolve_robocasa_assets_root() == tmp_path

    def test_default_under_hf_opentau_home(self, monkeypatch):
        monkeypatch.delenv(ROBOCASA_ASSETS_ROOT_ENV, raising=False)
        assert _resolve_robocasa_assets_root() == HF_OPENTAU_HOME / "robocasa" / "assets"

    def test_env_var_is_expanduser(self, monkeypatch):
        monkeypatch.setenv(ROBOCASA_ASSETS_ROOT_ENV, "~/robocasa_assets_xyz")
        out = _resolve_robocasa_assets_root()
        assert "~" not in str(out)
        assert str(out).endswith("robocasa_assets_xyz")


class TestNeededAssetPacks:
    """``_needed_asset_packs`` maps obj_registries to downloader pack names."""

    def test_lightwheel_default(self):
        assert _needed_asset_packs(["lightwheel"]) == ["textures", "tex_generative", "fixtures_lw", "objs_lw"]

    def test_objaverse_appends_pack(self):
        assert _needed_asset_packs(["lightwheel", "objaverse"]) == [
            "textures",
            "tex_generative",
            "fixtures_lw",
            "objs_lw",
            "objs_objaverse",
        ]

    def test_unknown_registry_ignored(self):
        assert _needed_asset_packs(["nonsuch"]) == ["textures", "tex_generative", "fixtures_lw"]

    def test_duplicates_collapsed(self):
        assert _needed_asset_packs(["lightwheel", "lightwheel"]) == [
            "textures",
            "tex_generative",
            "fixtures_lw",
            "objs_lw",
        ]


class TestDirectDownloadUrl:
    """``_direct_download_url`` rewrites a Box share link to a direct-download ``.zip``."""

    def test_rewrites_box_share_url(self):
        assert (
            _direct_download_url("https://utexas.box.com/s/abc123")
            == "https://utexas.box.com/shared/static/abc123.zip"
        )

    def test_tolerates_trailing_slash(self):
        assert (
            _direct_download_url("https://utexas.box.com/s/xyz789/")
            == "https://utexas.box.com/shared/static/xyz789.zip"
        )


class TestEnsureRoboCasaAssets:
    """``_ensure_robocasa_assets`` seeds stubs, downloads packs, symlinks, and is rank-safe."""

    def test_seeds_downloads_and_symlinks_when_absent(self, monkeypatch, tmp_path):
        pkg_assets, calls = _fake_robocasa_assets(monkeypatch, tmp_path)
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: None)
        root = tmp_path / "external"

        _ensure_robocasa_assets(root, ["lightwheel"])

        # wheel-shipped stub seeded into the external store + seed marker written
        assert (root / "shipped_stub.xml").is_file()
        assert (root / ".opentau_seeded").is_file()
        # lightwheel -> textures, tex_generative, fixtures_lw, objs_lw downloaded + marked
        assert len(calls) == 4
        for pack in ("textures", "tex_generative", "fixtures_lw", "objs_lw"):
            assert (root / f".opentau_pack_{pack}.done").is_file()
        # relocation: the venv assets dir is now a symlink to the external store
        assert pkg_assets.is_symlink()
        assert pkg_assets.resolve() == root.resolve()

    def test_skips_packs_with_existing_markers(self, monkeypatch, tmp_path):
        pkg_assets, calls = _fake_robocasa_assets(monkeypatch, tmp_path)
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: None)
        root = tmp_path / "external"
        root.mkdir()
        (root / ".opentau_seeded").touch()
        for pack in ("textures", "tex_generative", "fixtures_lw", "objs_lw"):
            (root / f".opentau_pack_{pack}.done").touch()

        _ensure_robocasa_assets(root, ["lightwheel"])

        assert calls == []  # everything already present, nothing re-downloaded
        assert pkg_assets.is_symlink()  # still (re)linked to the store

    def test_non_main_rank_does_nothing_but_barriers(self, monkeypatch, tmp_path):
        pkg_assets, calls = _fake_robocasa_assets(monkeypatch, tmp_path)
        acc = Mock()
        acc.num_processes = 2
        acc.is_main_process = False
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: acc)
        root = tmp_path / "external"

        _ensure_robocasa_assets(root, ["lightwheel"])

        assert calls == []
        assert not root.exists()  # non-main rank does not seed/download/mkdir
        assert not pkg_assets.is_symlink()  # nor relink
        acc.wait_for_everyone.assert_called_once()

    def test_main_rank_distributed_downloads_and_barriers(self, monkeypatch, tmp_path):
        pkg_assets, calls = _fake_robocasa_assets(monkeypatch, tmp_path)
        acc = Mock()
        acc.num_processes = 2
        acc.is_main_process = True
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: acc)
        root = tmp_path / "external"

        _ensure_robocasa_assets(root, ["lightwheel"])

        assert len(calls) == 4
        assert pkg_assets.is_symlink()
        acc.wait_for_everyone.assert_called_once()


class TestSymlinkAndRelink:
    """``_symlink_pkg_assets_to`` / ``_maybe_relink_robocasa_assets`` relocation behaviour."""

    def test_symlink_replaces_real_dir(self, tmp_path):
        pkg_assets = tmp_path / "pkg" / "assets"
        pkg_assets.mkdir(parents=True)
        (pkg_assets / "stub.txt").write_text("x")
        external = tmp_path / "external"
        external.mkdir()

        _symlink_pkg_assets_to(pkg_assets, external)

        assert pkg_assets.is_symlink()
        assert pkg_assets.resolve() == external.resolve()

    def test_symlink_is_idempotent(self, tmp_path):
        pkg_assets = tmp_path / "pkg" / "assets"
        pkg_assets.mkdir(parents=True)
        external = tmp_path / "external"
        external.mkdir()

        _symlink_pkg_assets_to(pkg_assets, external)
        _symlink_pkg_assets_to(pkg_assets, external)  # second call is a no-op

        assert pkg_assets.is_symlink()
        assert pkg_assets.resolve() == external.resolve()

    def test_maybe_relink_noop_when_store_not_seeded(self, monkeypatch, tmp_path):
        pkg_assets = tmp_path / "pkg" / "assets"
        pkg_assets.mkdir(parents=True)
        external = tmp_path / "external"  # never created / not seeded
        monkeypatch.setattr("opentau.envs.robocasa._resolve_robocasa_assets_root", lambda: external)
        monkeypatch.setattr("opentau.envs.robocasa._robocasa_pkg_assets_dir", lambda: pkg_assets)

        _maybe_relink_robocasa_assets()

        assert not pkg_assets.is_symlink()  # bundled assets left untouched

    def test_maybe_relink_symlinks_when_seeded(self, monkeypatch, tmp_path):
        pkg_assets = tmp_path / "pkg" / "assets"
        pkg_assets.mkdir(parents=True)
        external = tmp_path / "external"
        external.mkdir()
        (external / ".opentau_seeded").touch()
        monkeypatch.setattr("opentau.envs.robocasa._resolve_robocasa_assets_root", lambda: external)
        monkeypatch.setattr("opentau.envs.robocasa._robocasa_pkg_assets_dir", lambda: pkg_assets)

        _maybe_relink_robocasa_assets()

        assert pkg_assets.is_symlink()
        assert pkg_assets.resolve() == external.resolve()


class TestCreateRoboCasaEnvsAssets:
    """``create_robocasa_envs`` exports the assets root and gates the download."""

    def test_sets_env_var_and_calls_ensure(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ROBOCASA_ASSETS_ROOT_ENV, raising=False)
        ensure = Mock()
        monkeypatch.setattr("opentau.envs.robocasa._ensure_robocasa_assets", ensure)
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: None)
        env_cls = Mock(return_value=Mock(name="vec_env"))

        create_robocasa_envs(task="CloseFridge", n_envs=1, env_cls=env_cls, assets_root=str(tmp_path))

        assert os.environ[ROBOCASA_ASSETS_ROOT_ENV] == str(tmp_path)
        ensure.assert_called_once()
        assert ensure.call_args.args[0] == tmp_path

    def test_auto_download_false_skips_ensure_but_still_sets_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ROBOCASA_ASSETS_ROOT_ENV, raising=False)
        ensure = Mock()
        monkeypatch.setattr("opentau.envs.robocasa._ensure_robocasa_assets", ensure)
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: None)
        env_cls = Mock(return_value=Mock())

        create_robocasa_envs(
            task="CloseFridge",
            n_envs=1,
            env_cls=env_cls,
            assets_root=str(tmp_path),
            auto_download_assets=False,
        )

        ensure.assert_not_called()
        assert os.environ[ROBOCASA_ASSETS_ROOT_ENV] == str(tmp_path)

    def test_default_assets_root_uses_resolver(self, monkeypatch):
        monkeypatch.delenv(ROBOCASA_ASSETS_ROOT_ENV, raising=False)
        monkeypatch.setattr("opentau.envs.robocasa._ensure_robocasa_assets", Mock())
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: None)
        env_cls = Mock(return_value=Mock())

        create_robocasa_envs(task="CloseFridge", n_envs=1, env_cls=env_cls)

        assert os.environ[ROBOCASA_ASSETS_ROOT_ENV] == str(HF_OPENTAU_HOME / "robocasa" / "assets")

    def test_ensure_runs_before_empty_rank_return(self, monkeypatch):
        # A rank with no assigned tasks still runs `_ensure_robocasa_assets` (and thus its
        # barrier) before the empty-rank early return — otherwise ranks desync at NCCL.
        ensure = Mock()
        monkeypatch.setattr("opentau.envs.robocasa._ensure_robocasa_assets", ensure)
        acc = _mock_accelerator(num_processes=4, process_index=2)
        monkeypatch.setattr("opentau.envs.robocasa.get_proc_accelerator", lambda: acc)
        env_cls = Mock(return_value=Mock())

        out = create_robocasa_envs(task="A", n_envs=1, env_cls=env_cls)

        assert out == {}  # this rank got no task
        ensure.assert_called_once()  # but the asset/barrier step still ran


class TestRenderMultiCamera:
    """``render()`` composites every configured camera side-by-side for eval videos.

    CPU-only: the gym env is built without the sim (``_ensure_env`` returns early
    once ``_env`` is set), and the per-camera frames ``render`` reads are stubbed
    directly via ``_render_pixels``.
    """

    @staticmethod
    def _env_with_pixels(pixels: dict | None):
        from opentau.envs.robocasa import RoboCasaEnv as RoboCasaGymEnv

        env = RoboCasaGymEnv(task="CloseFridge")
        env._env = Mock()  # non-None so _ensure_env() is a no-op (no sim build)
        env._render_pixels = pixels
        return env

    def test_concatenates_all_cameras_left_to_right(self):
        env = self._env_with_pixels(
            {
                "camera0": np.full((4, 5, 3), 1, np.uint8),
                "camera1": np.full((4, 5, 3), 2, np.uint8),
                "camera2": np.full((4, 5, 3), 3, np.uint8),
            }
        )
        frame = env.render()
        # 3 cameras (5 px wide each) tiled along width, in camera_name order.
        assert frame.shape == (4, 15, 3)
        assert (frame[:, 0:5] == 1).all()
        assert (frame[:, 5:10] == 2).all()
        assert (frame[:, 10:15] == 3).all()

    def test_single_camera_is_not_widened(self):
        env = self._env_with_pixels({"camera0": np.full((4, 5, 3), 7, np.uint8)})
        frame = env.render()
        assert frame.shape == (4, 5, 3)
        assert (frame == 7).all()

    def test_falls_back_to_env_render_before_first_observation(self):
        env = self._env_with_pixels(None)  # no frames cached yet
        sentinel = np.zeros((2, 2, 3), np.uint8)
        env._env.render.return_value = sentinel
        assert env.render() is sentinel


class TestResetSeed:
    """``RoboCasaEnv.reset`` forwards an explicit seed to the sim verbatim.

    Spreading the seed across the ``n_envs`` workers is the caller's job:
    gymnasium's vector envs hand each sub-env a distinct ``seed[i]`` (or ``seed+i``
    for an int) and the eval harness builds an explicit per-worker range. Adding
    ``episode_index`` on top of an already-distinct seed would *double*-shift it and
    make scene seeds collide across rollout batches (so an eval samples fewer
    distinct scenes than ``n_episodes``). Only the unseeded path falls back to
    ``episode_index`` to keep workers distinct.

    CPU-only: ``_env`` is stubbed so ``_ensure_env`` is a no-op and no sim is built.
    """

    @staticmethod
    def _stub_env(episode_index: int):
        from opentau.envs.robocasa import RoboCasaEnv as RoboCasaGymEnv

        env = RoboCasaGymEnv(task="CloseFridge", episode_index=episode_index)
        env._env = Mock()  # non-None so _ensure_env() is a no-op (no sim build)
        env._env.reset.return_value = ({}, {})  # (raw_obs, info); raw_obs unused below
        env._env.env.get_ep_meta.return_value = {"lang": "close the fridge"}
        return env

    @pytest.mark.parametrize("episode_index", [0, 1, 5])
    def test_explicit_seed_passes_through_without_episode_index_shift(self, episode_index):
        env = self._stub_env(episode_index)
        with patch.object(env, "_format_raw_obs", return_value={"pixels": {}}):
            env.reset(seed=100)
        # Seeded with exactly the caller's seed — NOT seed + episode_index.
        env._env.reset.assert_called_once_with(seed=100)

    @pytest.mark.parametrize("episode_index", [0, 1, 5])
    def test_unseeded_reset_falls_back_to_episode_index(self, episode_index):
        env = self._stub_env(episode_index)
        with patch.object(env, "_format_raw_obs", return_value={"pixels": {}}):
            env.reset()  # no seed → fall back to episode_index so workers differ
        env._env.reset.assert_called_once_with(seed=episode_index)

    def test_per_worker_range_yields_contiguous_collision_free_scene_seeds(self):
        # Emulate the eval harness: gymnasium hands sub-env i its own seeds[i] from
        # the per-worker range [S, S+1, ..., S+N-1]. The underlying scene seeds must
        # be exactly those values — contiguous and distinct, no spacing-of-2 gaps
        # that would alias across batches.
        start, n_envs = 100, 4
        scene_seeds = []
        for i in range(n_envs):
            env = self._stub_env(episode_index=i)
            with patch.object(env, "_format_raw_obs", return_value={"pixels": {}}):
                env.reset(seed=start + i)  # seeds[i] as distributed by the vector env
            scene_seeds.append(env._env.reset.call_args.kwargs["seed"])
        assert scene_seeds == [100, 101, 102, 103]
        assert len(set(scene_seeds)) == n_envs  # no collisions


@pytest.mark.gpu
@pytest.mark.slow
def test_robocasa_autodownload_and_rollout_from_relocated_root(monkeypatch):
    """End-to-end on a CUDA box (needs the sim + EGL + network on a cold cache).

    Auto-downloads the lightwheel asset packs into the venv-external assets root, then a
    real ``CloseFridge`` reset+step reads from there — proving both the auto-download and
    the relocation (``robocasa.models.assets_root`` points outside ``site-packages``).
    The packs are cached across runs via the per-pack marker files.
    """
    monkeypatch.setenv("MUJOCO_GL", "egl")
    from opentau.envs.robocasa import RoboCasaEnv as RoboCasaGymEnv

    root = _resolve_robocasa_assets_root()
    monkeypatch.setenv(ROBOCASA_ASSETS_ROOT_ENV, str(root))
    _ensure_robocasa_assets(root, ["lightwheel"])

    # Relocation: the venv assets dir is a symlink into the external (out-of-venv) store.
    pkg_assets = _robocasa_pkg_assets_dir()
    assert pkg_assets.is_symlink()
    assert pkg_assets.resolve() == root.resolve()
    assert "site-packages" not in str(root)
    assert (root / "objects" / "lightwheel").is_dir()

    # robocasa's hardcoded assets_root resolves to the external store via the symlink.
    _import_robocasa_with_version_shim()
    import robocasa.models

    assert Path(robocasa.models.assets_root).resolve() == root.resolve()

    env = RoboCasaGymEnv(task="CloseFridge", observation_width=128, observation_height=128)
    try:
        obs, _info = env.reset(seed=0)
        assert "pixels" in obs
        obs, _rew, _term, _trunc, _info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert "pixels" in obs
    finally:
        env.close()
