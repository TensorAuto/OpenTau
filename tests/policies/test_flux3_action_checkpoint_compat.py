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

"""Loading a released FLUX 3 Action package, and matching the recipe it was trained under.

Every constant here was read off the real released artifacts rather than assumed --
``black-forest-labs/flux-3-action-{droid,so101}``'s ``config.json`` and Black Forest Labs'
published DROID recipe (their ``docs/droid-finetune.md`` and ``configs/droid/train.json``).
That distinction matters: an earlier draft of this port guessed several of these values and
every guess was wrong, in ways that load without erroring.

No weights are downloaded. The values below are the checkpoints' own, transcribed.
"""

import dataclasses

import pytest

from opentau.configs.policies import PreTrainedConfig
from opentau.policies.flux3_action.configuration_flux3_action import (
    DEFAULT_TEXT_FIXED_LENGTH,
    DEFAULT_VIDEO_POSITION_FPS,
    Flux3ActionConfig,
)

#: The fields of ``black-forest-labs/flux-3-action-droid``'s ``config.json`` that this port
#: reads, transcribed. Not the whole file -- the released config also carries bookkeeping
#: this config neither declares nor needs -- but complete for every key the loader touches,
#: and used as the single payload for both the field-level assertions and the end-to-end
#: parse test, so the two cannot describe different packages.
DROID_RELEASED = {
    "type": "flux3",
    "action_modality": "action_prediction_droid",
    "conditioning": "frame",
    "action_representation": "absolute",
    "delta_absolute_dims": [],
    "dtype": "bfloat16",
    "dit_config": None,
    "text_fixed_length": None,
    "video_position_fps": None,
    "chunk_size": 32,
    "n_action_steps": 32,
    "n_obs_steps": 1,
    "fps": 15.0,
    "camera_layout": "droid",
    "camera_keys": [
        "observation.images.wrist_image_left",
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    ],
    "canvas_hw": [544, 736],
    "action_scale": 2.0,
    "gripper_flip_dims": [-1],
    "use_peft": False,
    "use_relative_actions": False,
    "relative_exclude_joints": ["gripper"],
    "packer": None,
    "action_feature_names": None,
    "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"},
    "input_features": {"observation.state": {"type": "STATE", "shape": [8]}},
    "output_features": {"action": {"type": "ACTION", "shape": [8]}},
}

#: Verbatim from ``black-forest-labs/flux-3-action-so101``'s ``config.json``. Kept because
#: it differs from droid on exactly the fields whose mistranslation would be silent.
SO101_RELEASED = {
    "conditioning": "history",
    "action_representation": "delta",
    "delta_absolute_dims": [-1],
    "action_modality": "action",
}


# --------------------------------------------------------------------------- type alias
def test_released_packages_declare_type_flux3():
    """A released package says ``"type": "flux3"``; this repo registers ``flux3_action``.

    Without the alias, ``from_pretrained`` fails with "Couldn't find a choice class for
    'flux3'" and a released checkpoint cannot be loaded at all.
    """
    assert PreTrainedConfig.get_choice_class("flux3") is Flux3ActionConfig
    assert PreTrainedConfig.get_choice_class("flux3_action") is Flux3ActionConfig
    # The alias must not become canonical: decorator order decides which name
    # ``get_choice_name`` returns, and ``cfg.type`` is what gets written into every saved
    # config and read back by ``get_policy_class``. Swapping the two decorators flips this.
    assert Flux3ActionConfig().type == "flux3_action"


def test_field_default_resolves_a_default_factory_field():
    """Directly exercise the factory branch, which no current translation target uses.

    Asserting only that today's targets resolve would pass with the factory branch
    deleted — the mutation that motivated extracting this helper.
    """
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(Flux3ActionConfig)}
    factory = next(f for f in fields.values() if f.default_factory is not dataclasses.MISSING)
    resolved = Flux3ActionConfig._field_default(factory)
    assert resolved is not dataclasses.MISSING, f"{factory.name}'s factory default unresolved"
    assert resolved == factory.default_factory()

    plain = fields["inference_profile"]
    assert Flux3ActionConfig._field_default(plain) == plain.default


def test_a_default_factory_target_would_not_break_the_conflict_guard():
    """The guard resolves defaults, and ``f.default`` is MISSING for factory fields.

    No current translation targets one, but treating the MISSING sentinel as the default
    would make an untouched factory field look explicitly set and raise on every package
    load -- so the resolution is checked directly rather than left to a future accident.
    """
    import dataclasses

    from opentau.policies.flux3_action.configuration_flux3_action import LEGACY_TRANSLATIONS

    fields = {f.name: f for f in dataclasses.fields(Flux3ActionConfig)}
    factory_fields = [n for n, f in fields.items() if f.default_factory is not dataclasses.MISSING]
    assert factory_fields, "expected at least one default_factory field to exercise this"
    # a config built with only a legacy key set must load cleanly despite those fields
    cfg = Flux3ActionConfig(conditioning="history", n_obs_steps=1, gripper_flip_dims=())
    assert cfg.inference_profile == "history"
    # and every canonical target's default must be resolvable, factory or not
    for _legacy, canonical, _m, _c in LEGACY_TRANSLATIONS:
        f = fields[canonical]
        assert f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING, (
            f"{canonical} has no resolvable default"
        )


# --------------------------------------------------------------------------- null fields
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("dit_config", {}),
        ("text_fixed_length", DEFAULT_TEXT_FIXED_LENGTH),
        ("video_position_fps", DEFAULT_VIDEO_POSITION_FPS),
    ],
)
def test_nulls_in_a_released_config_become_their_defaults(field, expected):
    """Released packages send these three as ``null`` meaning "library default".

    Upstream validates all three (a dict to splat, fps > 0, 1 <= length <= 8192) and
    rejects None, so they must be restored rather than forwarded.
    """
    cfg = Flux3ActionConfig(**{field: None})
    assert getattr(cfg, field) == expected


# --------------------------------------------------------------------------- alias translation
def test_droid_package_fields_translate():
    cfg = Flux3ActionConfig(
        conditioning=DROID_RELEASED["conditioning"],
        action_representation=DROID_RELEASED["action_representation"],
        delta_absolute_dims=DROID_RELEASED["delta_absolute_dims"],
        dtype=DROID_RELEASED["dtype"],
    )
    assert cfg.inference_profile == "default"
    assert cfg.action_parameterization == "absolute"
    assert cfg.absolute_action_dims == ()
    assert cfg.torch_dtype == "bfloat16"


def test_so101_package_translates_to_history_and_delta():
    """The case that makes translation load-bearing rather than cosmetic.

    droid is frame/absolute and so101 is history/delta. Dropping these fields as "unknown"
    would load so101 configured for absolute actions against delta-trained weights --
    no error, no warning, just wrong actions.
    """
    cfg = Flux3ActionConfig(
        conditioning=SO101_RELEASED["conditioning"],
        action_representation=SO101_RELEASED["action_representation"],
        delta_absolute_dims=SO101_RELEASED["delta_absolute_dims"],
        n_obs_steps=1,
        gripper_flip_dims=(),
    )
    assert cfg.inference_profile == "history"
    assert cfg.action_parameterization == "joint_delta"
    assert cfg.absolute_action_dims == (-1,)


@pytest.mark.parametrize(
    ("field", "value"),
    [("conditioning", "sideways"), ("action_representation", "relative")],
)
def test_an_unrecognised_translation_value_raises(field, value):
    """A future package spelling must fail loudly, not fall through to a default."""
    with pytest.raises(ValueError, match=field):
        Flux3ActionConfig(**{field: value})


# --------------------------------------------------------------------------- unsupported features
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_peft", True),
        ("use_relative_actions", True),
        ("packer", "some_packer"),
        ("action_feature_names", ["a"]),
    ],
)
def test_features_this_port_does_not_implement_raise(field, value):
    """Declared so a package parses -- but never silently ignored.

    A checkpoint trained with one of these carries behaviour the port would not reproduce,
    and ignoring it produces a model that runs and is quietly wrong.
    """
    with pytest.raises(ValueError, match=field):
        Flux3ActionConfig(**{field: value})


def test_their_no_op_values_are_accepted():
    """Both released packages set these to no-op values; those must load cleanly."""
    cfg = Flux3ActionConfig(
        use_peft=False,
        use_relative_actions=False,
        packer=None,
        action_feature_names=None,
        relative_exclude_joints=["gripper"],
    )
    assert cfg.type == "flux3_action"


# --------------------------------------------------------------------------- widths
def test_state_and_action_widths_match_the_released_checkpoints():
    """The released checkpoints are 8-wide and upstream asserts equality with action_dim.

    OpenTau's usual padding to 32 would make ``_state`` raise on the first real batch.
    """
    cfg = Flux3ActionConfig()
    assert cfg.max_state_dim == 8
    assert cfg.max_action_dim == 8
    assert cfg.action_dim == 8


# --------------------------------------------------------------------------- BFL recipe parity
#: Black Forest Labs' published DROID fine-tuning recipe.
BFL_RECIPE = {
    "optimizer_betas": (0.9, 0.99),
    "optimizer_eps": 1e-8,
    "optimizer_weight_decay": 0.05,
    "optimizer_lr": 1.92e-4,
    "optimizer_lr_heads_multiplier": 5.0,
    "train_timestep_width": 0.75,
    "train_timestep_shift": 42.0,
    "separate_timesteps": False,
    "action_loss_weight": 50.0,
    "video_loss_weight": 1.0,
    "loss_reduction": "joint_tokens",
    "caption_dropout": 0.1,
    "action_scale": 2.0,
    "chunk_size": 32,
    "fps": 15.0,
    "action_parameterization": "absolute",
    "action_dim": 8,
}


@pytest.mark.parametrize(("field", "expected"), sorted(BFL_RECIPE.items()))
def test_defaults_match_black_forest_labs_droid_recipe(field, expected):
    """Fine-tuning starts from their weights, so it should start from their recipe.

    Two of these (beta2 and weight decay) were this repo's generic defaults and silently
    disagreed with the recipe the released weights were produced under.
    """
    got = getattr(Flux3ActionConfig(), field)
    assert (tuple(got) if isinstance(got, tuple) else got) == expected


def test_optimizer_preset_carries_the_recipe_through():
    """The values must survive into the optimizer, not just sit on the config."""
    preset = Flux3ActionConfig().get_optimizer_preset()
    assert preset.lr == BFL_RECIPE["optimizer_lr"]
    assert tuple(preset.betas) == BFL_RECIPE["optimizer_betas"]
    assert preset.weight_decay == BFL_RECIPE["optimizer_weight_decay"]


def test_camera_window_matches_the_recipes_frame_count():
    """The recipe says "33 frames at 15 Hz"; the window is derived, so pin the agreement."""
    cfg = Flux3ActionConfig()
    assert len(cfg.camera_delta_indices) == 33
    assert len(cfg.camera_delta_indices) == cfg.to_policy_config().window_frames


# --------------------------------------------------------------------------- checkpoint keys
def test_checkpoint_keys_are_rerooted_under_the_wrapper():
    """A released ``model.safetensors`` is saved from the upstream policy, rooted at ``dit.*``.

    This wrapper nests it as ``self.model``, so without re-rooting every tensor is reported
    both missing and unexpected -- and under the default ``strict=False`` that leaves a
    randomly initialized 7B model that runs and returns plausible-shaped garbage.
    """
    import torch

    from opentau.policies.flux3_action.modeling_flux3_action import Flux3ActionPolicy

    captured = {}

    class _Stub:
        def _promote_legacy_norm_buffers_in_state_dict(self, sd):
            pass

        def load_state_dict(self, sd, strict):
            captured.update(sd)
            return [], []

        def to(self, *a, **k):
            return self

    import safetensors.torch as st

    original = st.load_file
    st.load_file = lambda *a, **k: {"dit.block.weight": torch.zeros(1)}
    try:
        Flux3ActionPolicy._load_as_safetensor(_Stub(), "ignored", "cpu", False)
    finally:
        st.load_file = original

    assert list(captured) == ["model.dit.block.weight"]


def test_rerooting_is_idempotent():
    """An already-wrapped checkpoint (one we saved ourselves) must not gain a second prefix."""
    import torch

    from opentau.policies.flux3_action.modeling_flux3_action import Flux3ActionPolicy

    captured = {}

    class _Stub:
        def _promote_legacy_norm_buffers_in_state_dict(self, sd):
            pass

        def load_state_dict(self, sd, strict):
            captured.update(sd)
            return [], []

        def to(self, *a, **k):
            return self

    import safetensors.torch as st

    original = st.load_file
    st.load_file = lambda *a, **k: {"model.dit.block.weight": torch.zeros(1)}
    try:
        Flux3ActionPolicy._load_as_safetensor(_Stub(), "ignored", "cpu", False)
    finally:
        st.load_file = original

    assert list(captured) == ["model.dit.block.weight"]


def test_every_released_field_is_declared():
    """Draccus rejects unknown keys, so a package fails to parse if we omit one.

    Pinning the real key set means a future package gaining a field fails here, where the
    message is actionable, rather than inside ``from_pretrained``.
    """
    declared = {f.name for f in dataclasses.fields(Flux3ActionConfig)}
    # "type" is consumed by draccus as the choice key, never as a field.
    for key in set(DROID_RELEASED) | set(SO101_RELEASED):
        if key == "type":
            continue
        assert key in declared, f"released packages carry {key!r}, which this config omits"


# --------------------------------------------------------------------------- spelling conflicts
def test_an_explicit_override_is_not_clobbered_by_the_legacy_spelling():
    """Translation overwrites the canonical field, so a real override must not lose silently."""
    with pytest.raises(ValueError, match="torch_dtype"):
        Flux3ActionConfig(dtype="bfloat16", torch_dtype="float32")
    # the canonical value must differ from its own default, or it lands in the
    # documented-undetectable case pinned below
    with pytest.raises(ValueError, match="action_parameterization"):
        Flux3ActionConfig(
            action_representation="absolute",
            action_parameterization="joint_delta",
            gripper_flip_dims=(),
        )


def test_a_package_alone_still_translates_cleanly():
    """The normal load path: only the legacy key present, so the package's value wins."""
    cfg = Flux3ActionConfig(conditioning="history", n_obs_steps=1, gripper_flip_dims=())
    assert cfg.inference_profile == "history"


def test_setting_the_canonical_field_to_its_own_default_is_documented_as_undetectable():
    """The one case the conflict check cannot see, pinned so it stays a known limitation.

    A dataclass cannot distinguish "set to the default" from "not set", so the package
    still wins here. Asserting the actual behaviour keeps this from being rediscovered as
    a bug later.
    """
    cfg = Flux3ActionConfig(
        conditioning="history", inference_profile="default", n_obs_steps=1, gripper_flip_dims=()
    )
    assert cfg.inference_profile == "history"


def test_the_guard_and_the_translation_read_the_same_table():
    """One table drives both, so coverage cannot drift.

    An earlier version listed the translations in three places -- the translation itself,
    the conflict guard, and this test -- and the guard silently missed one, on the field
    this port calls load-bearing. Now a fifth entry added to ``LEGACY_TRANSLATIONS`` is
    automatically translated *and* conflict-checked, and this test asserts the table is
    the only source.
    """
    import inspect

    from opentau.policies.flux3_action.configuration_flux3_action import LEGACY_TRANSLATIONS

    declared = {f.name for f in dataclasses.fields(Flux3ActionConfig)}
    for legacy, canonical, mapping, _coerce in LEGACY_TRANSLATIONS:
        assert legacy in declared, f"{legacy} is translated but not a declared field"
        assert canonical in declared, f"{canonical} is a translation target but not declared"
        if mapping is not None:
            assert mapping, f"{legacy} has an empty value map"

    # both consumers must iterate the table rather than re-listing its contents
    for fn in (Flux3ActionConfig.__post_init__, Flux3ActionConfig._reject_conflicting_spellings):
        assert "LEGACY_TRANSLATIONS" in inspect.getsource(fn), (
            f"{fn.__name__} does not read the shared table, so it can drift from it"
        )


def test_the_table_covers_every_legacy_key_the_released_packages_send():
    """Derived from the real packages' keys, not from what the code happens to handle."""
    from opentau.policies.flux3_action.configuration_flux3_action import LEGACY_TRANSLATIONS

    translated = {legacy for legacy, _c, _m, _co in LEGACY_TRANSLATIONS}
    # these carry meaning and must be translated; the rest are unimplemented-feature flags
    for key in ("conditioning", "action_representation", "delta_absolute_dims", "dtype"):
        assert key in translated, f"released packages send {key!r} but nothing translates it"


def test_delta_absolute_dims_conflict_raises():
    """The specific omission: a sequence-valued translation, compared by value not identity."""
    with pytest.raises(ValueError, match="absolute_action_dims"):
        Flux3ActionConfig(
            action_representation="delta",
            delta_absolute_dims=[-1],
            absolute_action_dims=(3,),
            gripper_flip_dims=(),
        )


def test_delta_absolute_dims_agreeing_is_accepted():
    """Same value in both spellings is not a conflict -- list vs tuple must not fool it."""
    cfg = Flux3ActionConfig(
        action_representation="delta",
        delta_absolute_dims=[-1],
        absolute_action_dims=(-1,),
        gripper_flip_dims=(),
    )
    assert cfg.absolute_action_dims == (-1,)


# --------------------------------------------------------------------------- the real parse path
def test_a_verbatim_released_config_parses_end_to_end(tmp_path):
    """Route the real `config.json` through `from_pretrained`, not the constructor.

    Every other test here builds the config directly, which skips the machinery that
    actually matters when loading a package: draccus dispatching on ``"type": "flux3"``,
    coercing JSON types onto the declared fields, and rejecting unknown keys. Three of the
    six gaps this PR fixes — the unregistered alias, the `null` values, and the nine
    undeclared keys — were invisible to constructor-level tests and only surfaced when a
    real package was loaded.
    """
    import json

    released = dict(DROID_RELEASED)
    (tmp_path / "config.json").write_text(json.dumps(released))

    cfg = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(cfg, Flux3ActionConfig), "the flux3 alias did not dispatch"
    assert cfg.type == "flux3_action", "the alias must not become the canonical name"
    # the nulls were restored rather than forwarded
    assert cfg.dit_config == {}
    assert cfg.text_fixed_length == DEFAULT_TEXT_FIXED_LENGTH
    assert cfg.video_position_fps == DEFAULT_VIDEO_POSITION_FPS
    # the legacy spellings were translated onto the canonical fields and cleared
    assert cfg.inference_profile == "default"
    assert cfg.action_parameterization == "absolute"
    assert cfg.absolute_action_dims == ()
    assert cfg.torch_dtype == "bfloat16"
    assert cfg.conditioning is None and cfg.action_representation is None
    # and the widths the checkpoint expects survived
    assert cfg.max_state_dim == 8 and cfg.max_action_dim == 8


def test_a_released_config_with_an_unsupported_feature_is_refused_at_parse_time(tmp_path):
    """A package using a feature this port lacks must fail on load, not silently later.

    Note draccus replaces the raised message with its own "Couldn't instantiate class ..."
    wrapper, so the explanatory text from ``__post_init__`` does not reach the user through
    this path. Refusing is the property that matters -- the alternative is loading weights
    trained with a behaviour the port would silently skip -- but the diagnostic is worse
    here than when constructing the config directly, which is asserted separately above.
    """
    import json

    released = {"type": "flux3", "conditioning": "frame", "use_relative_actions": True}
    (tmp_path / "config.json").write_text(json.dumps(released))

    with pytest.raises(Exception, match="Flux3ActionConfig"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_the_direct_constructor_still_reports_which_feature_was_refused(tmp_path):
    """The informative message exists; it is draccus that flattens it on the parse path."""
    with pytest.raises(ValueError, match="use_relative_actions"):
        Flux3ActionConfig(use_relative_actions=True)
