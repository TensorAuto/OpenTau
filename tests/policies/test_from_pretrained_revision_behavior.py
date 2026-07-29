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

"""End-to-end ``@revision`` behavior for every policy that overrides from_pretrained.

Complements the structural checks in
``test_from_pretrained_revision_overrides.py`` by actually calling
``from_pretrained`` and asserting what reached the Hub.

These tests are cheap ONLY because every override resolves the weights file
*before* ``cls(config, **kwargs)`` builds the model — otherwise each parameter
here would instantiate a multi-billion-parameter policy. If someone moves model
construction back above weight resolution, these tests go from milliseconds to
minutes (or OOM), which is itself the signal.
"""

import pytest

import opentau.policies.pretrained as pt
from opentau.policies.factory import get_policy_class, make_policy_config
from opentau.policies.pretrained import CheckpointWeightsNotFoundError

# Every policy whose `from_pretrained` resolves weights itself, by its factory
# registration name. `pi0` / `value` / `cosmos3*` delegate to the base class,
# which is covered by tests/policies/test_pretrained_revision.py.
POLICY_TYPES = [
    "pi05",
    "pi05_mem",
    "pi06",
    "pi07_low_level",
    "pi07_high_level",
    "pi07_paligemma_low_level",
    "pi07_paligemma_high_level_planner",
]


class _Recorder:
    """Stand-in for ``hf_hub_download`` that records and then fails the lookup."""

    def __init__(self, exc):
        self.calls: list[dict] = []
        self._exc = exc

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        raise self._exc

    @property
    def weight_calls(self) -> list[dict]:
        return [c for c in self.calls if c.get("filename") == "model.safetensors"]


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_revision_suffix_reaches_the_hub(monkeypatch, policy_type):
    """``<repo>@6000`` loads the ``6000`` tag, not ``main``.

    Before the shared resolver, five of these seven passed
    ``revision=kwargs.get("revision")`` — always ``None`` for a keyword-only
    parameter — so the tag was silently ignored and ``main`` was served instead.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    recorder = _Recorder(RevisionNotFoundError("no such revision"))
    monkeypatch.setattr(pt, "hf_hub_download", recorder)

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)

    with pytest.raises(FileNotFoundError):
        policy_cls.from_pretrained("TensorAuto/dummy@6000", config=config)

    assert recorder.weight_calls, f"{policy_type}: never attempted to fetch model.safetensors"
    call = recorder.weight_calls[-1]
    assert call["repo_id"] == "TensorAuto/dummy"
    assert call["revision"] == "6000"


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_unknown_revision_is_never_downgraded_to_an_untrained_model(monkeypatch, policy_type):
    """A bad tag must raise, not hand back randomly-initialized weights.

    The old behavior logged a warning and returned the freshly-constructed
    model, so a typo'd tag started training from noise and only showed up as a
    strange loss curve hours later.
    """
    from huggingface_hub.errors import RevisionNotFoundError

    monkeypatch.setattr(pt, "hf_hub_download", _Recorder(RevisionNotFoundError("no such revision")))

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)

    with pytest.raises(FileNotFoundError) as excinfo:
        policy_cls.from_pretrained("TensorAuto/dummy@99999", config=config)
    # Must not be the one error the overrides are allowed to swallow.
    assert not isinstance(excinfo.value, CheckpointWeightsNotFoundError)


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_unknown_repo_raises(monkeypatch, policy_type):
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.setattr(pt, "hf_hub_download", _Recorder(RepositoryNotFoundError("nope")))

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)

    with pytest.raises(FileNotFoundError):
        policy_cls.from_pretrained("TensorAuto/does-not-exist", config=config)


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_legacy_checkpoint_resolves_config_version_to_zero(monkeypatch, tmp_path, policy_type):
    """A checkpoint with no ``config_version`` must be read as legacy 0, not current.

    ``config_version`` gates the normalization convention the weights were
    trained under. These overrides never resolved it — and since ``make_policy``
    always passes ``config=``, their config-loading branch never ran — so a
    pre-versioning checkpoint silently got the *current* convention and
    normalized differently than it was trained. Silent: the weights load fine.

    Stops before model construction, so it stays instant.
    """
    ckpt = tmp_path / "legacy_ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    # A pre-versioning checkpoint: config present, no `config_version` key.
    (ckpt / "config.json").write_text('{"type": "%s"}' % policy_type)

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)
    config.config_version = None

    def _stop(*a, **kw):
        raise RuntimeError("stop here — provenance resolved, model not yet built")

    monkeypatch.setattr(f"{policy_cls.__module__}.{policy_cls.__name__}.__init__", _stop)

    with pytest.raises(RuntimeError, match="stop here"):
        policy_cls.from_pretrained(str(ckpt), config=config)

    assert config.config_version == 0, (
        f"{policy_type}: a checkpoint with no config_version must resolve to legacy 0, "
        f"got {config.config_version}"
    )


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_tagged_checkpoint_config_version_is_inherited(monkeypatch, tmp_path, policy_type):
    """A checkpoint that declares its convention keeps it."""
    ckpt = tmp_path / "tagged_ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    (ckpt / "config.json").write_text('{"type": "%s", "config_version": 1}' % policy_type)

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)
    config.config_version = None

    def _stop(*a, **kw):
        raise RuntimeError("stop here")

    monkeypatch.setattr(f"{policy_cls.__module__}.{policy_cls.__name__}.__init__", _stop)

    with pytest.raises(RuntimeError, match="stop here"):
        policy_cls.from_pretrained(str(ckpt), config=config)

    assert config.config_version == 1


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_explicit_config_version_is_never_overwritten(monkeypatch, tmp_path, policy_type):
    """An explicitly-set version is a deliberate escape hatch; the peek must not clobber it."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    (ckpt / "config.json").write_text('{"type": "%s", "config_version": 1}' % policy_type)

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)
    config.config_version = 0  # user pinned the legacy convention on purpose

    def _stop(*a, **kw):
        raise RuntimeError("stop here")

    monkeypatch.setattr(f"{policy_cls.__module__}.{policy_cls.__name__}.__init__", _stop)

    with pytest.raises(RuntimeError, match="stop here"):
        policy_cls.from_pretrained(str(ckpt), config=config)

    assert config.config_version == 0


@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_local_dir_with_at_in_its_name_reaches_the_resolver_unsplit(monkeypatch, tmp_path, policy_type):
    """``/…/run@6000`` is a directory, and each override must pass it through whole.

    Intercepts the resolver inside the policy's own module namespace — that is
    what proves the *override* (not just the resolver) declined to split, and it
    short-circuits before the model is constructed, so this stays instant.
    """
    ckpt = tmp_path / "run@6000"
    ckpt.mkdir()

    policy_cls = get_policy_class(policy_type)
    config = make_policy_config(policy_type)
    module = policy_cls.__module__

    seen: list[tuple] = []

    def _capture(spec, **kwargs):
        seen.append((str(spec), kwargs.get("revision")))
        raise RuntimeError("stop here — resolution reached, model not yet built")

    monkeypatch.setattr(f"{module}.resolve_pretrained_weights_file", _capture)

    with pytest.raises(RuntimeError, match="stop here"):
        policy_cls.from_pretrained(str(ckpt), config=config)

    assert seen == [(str(ckpt), None)], (
        f"{policy_type}: expected the local path to reach the resolver unsplit, got {seen}"
    )
