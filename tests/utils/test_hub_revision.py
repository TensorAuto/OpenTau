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

"""Unit tests for the ``repo_id@revision`` checkpoint-spec parser.

Deliberately a separate file from ``tests/utils/test_hub.py``: that one is
module-level ``pytest.mark.network`` (it round-trips against the live Hub), and
these must run on the gating CPU job. Nothing here touches the network.
"""

from pathlib import Path

import pytest

from opentau.utils.hub import format_repo_revision, split_repo_revision


def test_none_and_empty_pass_through():
    assert split_repo_revision(None) == (None, None)
    assert split_repo_revision(None, "6000") == (None, "6000")
    assert split_repo_revision("") == ("", None)


def test_plain_repo_id_is_unchanged():
    """The 350 legacy ``<runid>-<slug>-<step>`` repos must keep resolving to main."""
    spec = "TensorAuto/bwk9804f-rhp_frombase_pi05_quantile_abs_rtc4-6000"
    assert split_repo_revision(spec) == (spec, None)


def test_repo_at_revision_is_split():
    assert split_repo_revision("TensorAuto/bwk9804f-rhp_frombase_pi05@6000") == (
        "TensorAuto/bwk9804f-rhp_frombase_pi05",
        "6000",
    )


def test_bare_repo_name_without_namespace_is_split():
    assert split_repo_revision("some-model@v1.2") == ("some-model", "v1.2")


def test_revision_may_contain_slashes_and_extra_ats():
    """Only the *head* is slash-limited — a git ref may be ``refs/pr/1``."""
    assert split_repo_revision("TensorAuto/foo@refs/pr/1") == ("TensorAuto/foo", "refs/pr/1")
    assert split_repo_revision("TensorAuto/foo@weird@ref") == ("TensorAuto/foo", "weird@ref")


def test_existing_local_dir_containing_at_is_not_split(tmp_path):
    """A real checkpoint directory may contain '@'; splitting it would be data loss."""
    ckpt = tmp_path / "foo@bar" / "checkpoints" / "001000"
    ckpt.mkdir(parents=True)
    assert split_repo_revision(str(ckpt)) == (str(ckpt), None)


def test_existing_local_file_containing_at_is_not_split(tmp_path):
    """``TrainPipelineConfig.from_pretrained`` accepts a path *to* train_config.json."""
    cfg = tmp_path / "run@6000" / "train_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}")
    assert split_repo_revision(str(cfg)) == (str(cfg), None)


@pytest.mark.parametrize(
    "spec",
    [
        "/fss/outputs/foo@bar/checkpoints/001000",  # moved absolute checkpoint
        "./outputs/foo@bar",
        "../outputs/foo@bar",
        "~/checkpoints/run@1000",
        r"C:\checkpoints\run@1000\model",
        "a/b/c@1000",  # two slashes -> deeper than any repo id
    ],
)
def test_path_shaped_specs_are_never_split(spec):
    """Even when the path does not exist, it must fail as a path, not as a hub id."""
    assert split_repo_revision(spec) == (spec, None)


def test_head_that_exists_locally_is_not_split(tmp_path, monkeypatch):
    """``outputs/run@6000`` where ``outputs/run`` is a real dir is a path, not a tag."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "run").mkdir(parents=True)
    assert split_repo_revision("outputs/run@6000") == ("outputs/run@6000", None)


def test_trailing_at_is_not_split():
    assert split_repo_revision("TensorAuto/foo@") == ("TensorAuto/foo@", None)


def test_accepts_pathlib_input():
    assert split_repo_revision(Path("TensorAuto/foo@6000")) == ("TensorAuto/foo", "6000")


def test_matching_explicit_revision_is_accepted():
    assert split_repo_revision("TensorAuto/foo@6000", "6000") == ("TensorAuto/foo", "6000")


def test_explicit_revision_survives_a_spec_without_a_suffix():
    assert split_repo_revision("TensorAuto/foo", "6000") == ("TensorAuto/foo", "6000")


def test_conflicting_revisions_raise():
    with pytest.raises(ValueError, match="Conflicting revisions"):
        split_repo_revision("TensorAuto/foo@6000", "7000")


@pytest.mark.parametrize(
    "spec",
    [
        "TensorAuto/foo",
        "TensorAuto/foo@6000",
        "TensorAuto/foo@refs/pr/1",
        "/abs/path@1000",
        "",
    ],
)
def test_split_is_idempotent(spec):
    """Pins the re-entrancy contract every nested ``from_pretrained`` relies on.

    An outer loader splits a spec and forwards ``(repo_id, revision)`` to an inner
    one, which splits again. If the second pass changed anything — or raised on the
    now-explicit revision — nested loads would break.
    """
    once = split_repo_revision(spec)
    assert split_repo_revision(*once) == once


@pytest.mark.parametrize("spec", ["TensorAuto/foo", "TensorAuto/foo@6000", "TensorAuto/foo@refs/pr/1"])
def test_format_round_trips_hub_specs(spec):
    assert format_repo_revision(*split_repo_revision(spec)) == spec
