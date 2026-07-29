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

"""Behavioral tests for checkpoint weight resolution and ``@revision`` support.

Offline: ``hf_hub_download`` is monkeypatched at its import site in
``opentau.policies.pretrained``, the same seam
``tests/policies/test_config_version_norm.py`` uses.
"""

import json

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

import opentau.policies.pretrained as pt
from opentau.policies.pretrained import (
    CheckpointWeightsNotFoundError,
    _peek_config_version,
    resolve_pretrained_weights_file,
)


@pytest.fixture
def recorder(monkeypatch):
    """Capture every ``hf_hub_download`` call and return a dummy local path."""
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return "/tmp/fake/model.safetensors"  # noqa: S108 — never opened

    monkeypatch.setattr(pt, "hf_hub_download", _fake)
    return calls


def _raiser(monkeypatch, exc):
    def _fake(**kwargs):
        raise exc

    monkeypatch.setattr(pt, "hf_hub_download", _fake)


# --------------------------------------------------------------------------- #
# resolve_pretrained_weights_file
# --------------------------------------------------------------------------- #


def test_resolver_splits_repo_and_revision(recorder):
    resolve_pretrained_weights_file("TensorAuto/foo@6000")
    assert recorder[0]["repo_id"] == "TensorAuto/foo"
    assert recorder[0]["revision"] == "6000"
    assert recorder[0]["filename"] == "model.safetensors"


def test_resolver_without_suffix_leaves_revision_unset(recorder):
    resolve_pretrained_weights_file("TensorAuto/foo")
    assert recorder[0]["repo_id"] == "TensorAuto/foo"
    assert recorder[0]["revision"] is None


def test_resolver_forwards_every_download_control_kwarg(recorder):
    """These were all silently dropped by the per-policy ``cached_file`` calls."""
    resolve_pretrained_weights_file(
        "TensorAuto/foo@6000",
        cache_dir="/tmp/cache",  # noqa: S108
        force_download=True,
        resume_download=True,
        proxies={"http": "http://proxy"},
        token="hf_dummy",
        local_files_only=True,
    )
    call = recorder[0]
    assert call["cache_dir"] == "/tmp/cache"  # noqa: S108
    assert call["force_download"] is True
    assert call["resume_download"] is True
    assert call["proxies"] == {"http": "http://proxy"}
    assert call["token"] == "hf_dummy"
    assert call["local_files_only"] is True


def test_resolver_reads_a_local_dir_without_touching_the_hub(tmp_path, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("a local checkpoint directory must not hit the Hub")

    monkeypatch.setattr(pt, "hf_hub_download", _boom)
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert resolve_pretrained_weights_file(str(tmp_path)) == str(tmp_path / "model.safetensors")


def test_resolver_local_dir_with_at_in_the_name_is_not_split(tmp_path, monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("a local checkpoint directory must not hit the Hub")

    monkeypatch.setattr(pt, "hf_hub_download", _boom)
    ckpt = tmp_path / "run@6000"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    assert resolve_pretrained_weights_file(str(ckpt)) == str(ckpt / "model.safetensors")


def test_resolver_local_dir_without_weights_is_the_tolerated_error(tmp_path):
    """The DeepSpeed/ZeRO resume shape — the only failure a policy may swallow."""
    with pytest.raises(CheckpointWeightsNotFoundError, match="convert_checkpoint"):
        resolve_pretrained_weights_file(str(tmp_path))


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RevisionNotFoundError("nope"), "6000"),
        (RepositoryNotFoundError("nope"), "not found"),
        (GatedRepoError("nope"), "not found"),
        (EntryNotFoundError("nope"), "could not be fetched"),
    ],
)
def test_resolver_hub_failures_are_loud_and_never_tolerated(monkeypatch, exc, expected):
    """A Hub source can never legitimately lack weights, so none of these may be swallowed.

    They must also *not* be ``CheckpointWeightsNotFoundError``, which is what the
    policy overrides catch — otherwise a typo'd tag would silently produce a
    randomly-initialized model.
    """
    _raiser(monkeypatch, exc)
    with pytest.raises(FileNotFoundError, match=expected) as excinfo:
        resolve_pretrained_weights_file("TensorAuto/foo@6000")
    assert not isinstance(excinfo.value, CheckpointWeightsNotFoundError)


def test_resolver_revision_error_names_the_tags_page(monkeypatch):
    _raiser(monkeypatch, RevisionNotFoundError("nope"))
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_pretrained_weights_file("TensorAuto/foo@99999")
    message = str(excinfo.value)
    assert "99999" in message
    assert "huggingface.co/TensorAuto/foo/tags" in message
    assert "main" in message


def test_resolver_conflicting_revisions_raise():
    with pytest.raises(ValueError, match="Conflicting revisions"):
        resolve_pretrained_weights_file("TensorAuto/foo@6000", revision="7000")


# --------------------------------------------------------------------------- #
# _peek_config_version
# --------------------------------------------------------------------------- #


def test_peek_config_version_splits_the_revision_suffix(monkeypatch, tmp_path):
    calls: list[dict] = []
    payload = tmp_path / "config.json"
    payload.write_text(json.dumps({"config_version": 1}))

    def _fake(**kwargs):
        calls.append(kwargs)
        return str(payload)

    monkeypatch.setattr(pt, "hf_hub_download", _fake)
    assert _peek_config_version("TensorAuto/foo@6000") == 1
    assert calls[0]["repo_id"] == "TensorAuto/foo"
    assert calls[0]["revision"] == "6000"


def test_peek_config_version_falls_back_to_hf_config_json(monkeypatch, tmp_path):
    """Uploaded checkpoint repos carry `hf_config.json`, not `config.json`."""
    hf_config = tmp_path / "hf_config.json"
    hf_config.write_text(json.dumps({"policy": {"config_version": 1}}))

    def _fake(*, filename, **kwargs):
        if filename != "hf_config.json":
            raise EntryNotFoundError("absent")
        return str(hf_config)

    monkeypatch.setattr(pt, "hf_hub_download", _fake)
    assert _peek_config_version("TensorAuto/foo@6000") == 1
