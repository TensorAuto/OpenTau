# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from opentau.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    TRAINING_STEP,
)
from opentau.utils.train_utils import (
    RunningBestTracker,
    find_missing_rng_state_ranks,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_running_best_state,
    load_training_step,
    prune_old_checkpoints,
    reseed_new_ranks_on_resume,
    running_best_is_improvement,
    save_running_best_state,
    save_training_step,
    update_last_checkpoint,
)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_find_missing_rng_state_ranks(tmp_path):
    """find_missing_rng_state_ranks flags ranks whose per-rank RNG file is absent."""
    # Complete checkpoint: one random_states_<i>.pkl per rank -> nothing missing.
    for i in range(8):
        (tmp_path / f"random_states_{i}.pkl").touch()
    assert find_missing_rng_state_ranks(tmp_path, 8) == []

    # The output_dir-divergence failure mode: rank 6 wrote to a sibling dir, so its
    # per-rank file is absent here.
    (tmp_path / "random_states_6.pkl").unlink()
    assert find_missing_rng_state_ranks(tmp_path, 8) == [6]

    # Unrelated files are ignored, and a non-existent directory reports every rank.
    (tmp_path / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").touch()
    assert find_missing_rng_state_ranks(tmp_path, 8) == [6]
    assert find_missing_rng_state_ranks(tmp_path / "does_not_exist", 3) == [0, 1, 2]


def test_get_step_checkpoint_dir():
    """
    Tests if checkpoint directory is returned correctly
    """
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    """
    Tests if training file is saved
    """
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    """
    Tests if step is loaded correctly
    """
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_update_last_checkpoint(tmp_path):
    """
    Tests if checkpoint is stored correctly
    """
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


class TestPruneOldCheckpoints:
    """Test suite for the prune_old_checkpoints function."""

    @patch("opentau.utils.train_utils.shutil.rmtree")
    @patch("opentau.utils.train_utils.logging")
    def test_prune_old_checkpoints_success(self, mock_logging, mock_rmtree):
        """Test successful pruning of old checkpoints."""
        # Mock checkpoint directory structure
        latest_checkpoint_path = "/path/to/checkpoints/000100"

        # Create mock parent directory with multiple checkpoints
        mock_parent_dir = Mock()
        mock_parent_dir.is_dir.return_value = True
        mock_parent_dir.resolve.return_value = Path("/path/to/checkpoints")

        mock_old_checkpoint1 = Mock()
        mock_old_checkpoint1.is_dir.return_value = True
        mock_old_checkpoint1.resolve.return_value = Path("/path/to/checkpoints/000095")
        mock_old_checkpoint1.name = "000095"

        mock_old_checkpoint2 = Mock()
        mock_old_checkpoint2.is_dir.return_value = True
        mock_old_checkpoint2.resolve.return_value = Path("/path/to/checkpoints/000096")
        mock_old_checkpoint2.name = "000096"

        mock_latest_checkpoint_mock = Mock()
        mock_latest_checkpoint_mock.is_dir.return_value = True
        mock_latest_checkpoint_mock.resolve.return_value = Path("/path/to/checkpoints/000100")
        mock_latest_checkpoint_mock.name = "000100"

        mock_file = Mock(name="config.json")
        mock_file.is_dir.return_value = False

        mock_parent_dir.iterdir.return_value = [
            mock_old_checkpoint1,
            mock_old_checkpoint2,
            mock_latest_checkpoint_mock,
            mock_file,
        ]

        # Mock latest checkpoint
        mock_latest_checkpoint = Mock()
        mock_latest_checkpoint.is_dir.return_value = True
        mock_latest_checkpoint.parent = mock_parent_dir
        mock_latest_checkpoint.resolve.return_value = Path("/path/to/checkpoints/000100")
        mock_latest_checkpoint.name = "000100"

        with patch("opentau.utils.train_utils.Path") as mock_path:
            mock_path.return_value.resolve.return_value = mock_latest_checkpoint
            mock_path.return_value.parent = mock_parent_dir

            # Call the function
            prune_old_checkpoints(latest_checkpoint_path)

            # Verify logging calls
            mock_logging.info.assert_any_call(
                "Starting cleanup in '/path/to/checkpoints'. Keeping checkpoint: '000100'"
            )
            mock_logging.info.assert_any_call("Deleting old checkpoint directory: 000095")
            mock_logging.info.assert_any_call("Successfully deleted 000095")
            mock_logging.info.assert_any_call("Deleting old checkpoint directory: 000096")
            mock_logging.info.assert_any_call("Successfully deleted 000096")

            # Verify rmtree was called for old checkpoints only
            assert mock_rmtree.call_count == 2
            mock_rmtree.assert_any_call(mock_old_checkpoint1)
            mock_rmtree.assert_any_call(mock_old_checkpoint2)

    @patch("opentau.utils.train_utils.logging")
    def test_prune_old_checkpoints_parent_dir_not_exists(self, mock_logging):
        """Test behavior when parent directory doesn't exist."""
        latest_checkpoint_path = "/nonexistent/path/checkpoint/000100"

        mock_latest_checkpoint = Mock()
        mock_parent_dir = Mock()
        mock_parent_dir.is_dir.return_value = False
        mock_parent_dir.resolve.return_value = Path("/nonexistent/path/checkpoint")
        mock_latest_checkpoint.parent = mock_parent_dir

        with patch("opentau.utils.train_utils.Path") as mock_path:
            mock_path.return_value.resolve.return_value = mock_latest_checkpoint
            mock_path.return_value.parent = mock_parent_dir

            prune_old_checkpoints(latest_checkpoint_path)

            mock_logging.error.assert_called_once_with(
                "Parent directory '/nonexistent/path/checkpoint' does not exist. Aborting cleanup."
            )

    @patch("opentau.utils.train_utils.logging")
    def test_prune_old_checkpoints_latest_not_directory(self, mock_logging):
        """Test behavior when latest checkpoint is not a directory."""
        latest_checkpoint_path = "/path/to/file.txt"

        mock_latest_checkpoint = Mock()
        mock_parent_dir = Mock()
        mock_parent_dir.is_dir.return_value = True
        mock_latest_checkpoint.is_dir.return_value = False
        mock_latest_checkpoint.parent = mock_parent_dir
        mock_latest_checkpoint.resolve.return_value = Path("/path/to/file.txt")

        with patch("opentau.utils.train_utils.Path") as mock_path:
            mock_path.return_value.resolve.return_value = mock_latest_checkpoint
            mock_path.return_value.parent = mock_parent_dir

            prune_old_checkpoints(latest_checkpoint_path)

            mock_logging.warning.assert_called_once_with(
                "Checkpoint '/path/to/file.txt' is not a valid directory. Aborting cleanup."
            )

    @patch("opentau.utils.train_utils.shutil.rmtree")
    @patch("opentau.utils.train_utils.logging")
    def test_prune_old_checkpoints_no_old_checkpoints(self, mock_logging, mock_rmtree):
        """Test behavior when there are no old checkpoints to delete."""
        latest_checkpoint_path = "/path/to/checkpoints/000100"

        # Mock parent directory with only the latest checkpoint and some files
        mock_parent_dir = Mock()
        mock_parent_dir.is_dir.return_value = True
        mock_parent_dir.resolve.return_value = Path("/path/to/checkpoints")

        mock_latest_checkpoint_mock = Mock(name="000100")
        mock_latest_checkpoint_mock.is_dir.return_value = True
        mock_latest_checkpoint_mock.resolve.return_value = Path("/path/to/checkpoints/000100")

        mock_file1 = Mock(name="config.json")
        mock_file1.is_dir.return_value = False

        mock_file2 = Mock(name="README.md")
        mock_file2.is_dir.return_value = False

        mock_parent_dir.iterdir.return_value = [
            mock_latest_checkpoint_mock,
            mock_file1,
            mock_file2,
        ]

        # Mock latest checkpoint
        mock_latest_checkpoint = Mock()
        mock_latest_checkpoint.is_dir.return_value = True
        mock_latest_checkpoint.parent = mock_parent_dir
        mock_latest_checkpoint.resolve.return_value = Path("/path/to/checkpoints/000100")
        mock_latest_checkpoint.name = "000100"

        with patch("opentau.utils.train_utils.Path") as mock_path:
            mock_path.return_value.resolve.return_value = mock_latest_checkpoint
            mock_path.return_value.parent = mock_parent_dir

            prune_old_checkpoints(latest_checkpoint_path)

            # Verify no deletion occurred
            mock_rmtree.assert_not_called()

            # Verify logging
            mock_logging.info.assert_called_once_with(
                "Starting cleanup in '/path/to/checkpoints'. Keeping checkpoint: '000100'"
            )


def _make_accelerator(num_processes: int, process_index: int, is_main: bool | None = None):
    """Build a stub Accelerator that ``reseed_new_ranks_on_resume`` and ``set_seed`` accept."""
    acc = Mock()
    acc.num_processes = num_processes
    acc.process_index = process_index
    acc.is_main_process = is_main if is_main is not None else (process_index == 0)
    return acc


def _populate_rng_files(checkpoint_dir: Path, count: int) -> None:
    for i in range(count):
        (checkpoint_dir / f"random_states_{i}.pkl").write_bytes(b"\x00")


class TestReseedNewRanksOnResume:
    """Cover the file-counting branches of ``reseed_new_ranks_on_resume``."""

    def test_same_world_size_is_noop(self, tmp_path):
        _populate_rng_files(tmp_path, 4)
        acc = _make_accelerator(num_processes=4, process_index=2)
        with patch("opentau.utils.random_utils.set_seed") as mock_set_seed:
            reseed_new_ranks_on_resume(tmp_path, acc, seed=1234)
        mock_set_seed.assert_not_called()

    def test_scale_up_reseeds_only_new_ranks(self, tmp_path):
        # Saved on 4 ranks, resuming on 8: ranks 4..7 must be reseeded.
        _populate_rng_files(tmp_path, 4)
        with patch("opentau.utils.random_utils.set_seed") as mock_set_seed:
            for rank in range(8):
                acc = _make_accelerator(num_processes=8, process_index=rank)
                reseed_new_ranks_on_resume(tmp_path, acc, seed=42)
        # Only ranks 4..7 should have triggered a reseed.
        assert mock_set_seed.call_count == 4
        reseeded_ranks = sorted(
            call.kwargs["accelerator"].process_index for call in mock_set_seed.call_args_list
        )
        assert reseeded_ranks == [4, 5, 6, 7]

    def test_scale_up_with_seed_none_skips_set_seed(self, tmp_path, caplog):
        _populate_rng_files(tmp_path, 2)
        acc = _make_accelerator(num_processes=4, process_index=3)
        with (
            patch("opentau.utils.random_utils.set_seed") as mock_set_seed,
            caplog.at_level("WARNING"),
        ):
            reseed_new_ranks_on_resume(tmp_path, acc, seed=None)
        mock_set_seed.assert_not_called()
        assert any("cfg.seed is None" in rec.message for rec in caplog.records)

    def test_scale_down_does_not_reseed(self, tmp_path):
        _populate_rng_files(tmp_path, 8)
        acc = _make_accelerator(num_processes=4, process_index=0)
        with patch("opentau.utils.random_utils.set_seed") as mock_set_seed:
            reseed_new_ranks_on_resume(tmp_path, acc, seed=42)
        mock_set_seed.assert_not_called()

    def test_no_rng_files_early_returns_without_zerodiv(self, tmp_path, caplog):
        # Empty checkpoint dir: must not raise ZeroDivisionError, must warn on main only.
        acc_main = _make_accelerator(num_processes=4, process_index=0, is_main=True)
        with (
            patch("opentau.utils.random_utils.set_seed") as mock_set_seed,
            caplog.at_level("WARNING"),
        ):
            reseed_new_ranks_on_resume(tmp_path, acc_main, seed=42)
        mock_set_seed.assert_not_called()
        assert any("No random_states_*.pkl files found" in rec.message for rec in caplog.records)

    def test_top_level_log_gated_on_main_process(self, tmp_path, caplog):
        _populate_rng_files(tmp_path, 2)
        # Non-main rank should not emit the high-level scale-up warning.
        acc_nonmain = _make_accelerator(num_processes=4, process_index=1, is_main=False)
        with patch("opentau.utils.random_utils.set_seed"), caplog.at_level("WARNING"):
            reseed_new_ranks_on_resume(tmp_path, acc_nonmain, seed=42)
        scale_up_msgs = [rec for rec in caplog.records if "Resuming on more processes" in rec.message]
        assert scale_up_msgs == []

    def test_scale_down_log_gated_on_main_process(self, tmp_path, caplog):
        _populate_rng_files(tmp_path, 8)
        acc_nonmain = _make_accelerator(num_processes=4, process_index=2, is_main=False)
        with caplog.at_level("INFO"):
            reseed_new_ranks_on_resume(tmp_path, acc_nonmain, seed=42)
        scale_down_msgs = [rec for rec in caplog.records if "Resuming on fewer processes" in rec.message]
        assert scale_down_msgs == []


# ---------------------------------------------------------------------------
# Running-best checkpoint bookkeeping
# ---------------------------------------------------------------------------


def test_running_best_is_improvement_maximize():
    """Strict improvement when higher is better; equal/worse/None/NaN do not count."""
    assert running_best_is_improvement(60.0, 50.0, higher_is_better=True) is True
    assert running_best_is_improvement(50.0, 50.0, higher_is_better=True) is False
    assert running_best_is_improvement(40.0, 50.0, higher_is_better=True) is False
    assert running_best_is_improvement(0.0, float("-inf"), higher_is_better=True) is True
    assert running_best_is_improvement(None, 50.0, higher_is_better=True) is False
    assert running_best_is_improvement(float("nan"), 50.0, higher_is_better=True) is False


def test_running_best_is_improvement_minimize():
    """Strict improvement when lower is better (validation loss)."""
    assert running_best_is_improvement(0.05, 0.1, higher_is_better=False) is True
    assert running_best_is_improvement(0.1, 0.1, higher_is_better=False) is False
    assert running_best_is_improvement(0.2, 0.1, higher_is_better=False) is False
    assert running_best_is_improvement(1.0, float("inf"), higher_is_better=False) is True
    assert running_best_is_improvement(float("nan"), 0.1, higher_is_better=False) is False


def _mk_tracker(tmp_path, higher_is_better=True, max_count=1):
    return RunningBestTracker(
        output_dir=tmp_path,
        total_steps=1000,
        higher_is_better=higher_is_better,
        max_count=max_count,
    )


def _mk_step_dir(tracker, step):
    """Create (mkdir) the checkpoint dir a tracker would use for a step, and return it."""
    d = tracker.step_dir(step)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_running_best_tracker_first_register_no_eviction(tmp_path):
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=1)
    _mk_step_dir(tracker, 100)
    deleted = tracker.register(100, 50.0, is_regular=False)
    assert deleted == []
    assert tracker.best == 50.0
    assert [e["step"] for e in tracker.steps] == [100]


def test_running_best_tracker_evicts_non_regular(tmp_path):
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=1)
    d100 = _mk_step_dir(tracker, 100)
    tracker.register(100, 50.0, is_regular=False)
    _mk_step_dir(tracker, 200)
    deleted = tracker.register(200, 60.0, is_regular=False)
    assert deleted == [d100]
    assert tracker.best == 60.0
    assert [e["step"] for e in tracker.steps] == [200]


def test_running_best_tracker_keeps_regular_checkpoint(tmp_path):
    """An evicted entry that coincided with a regular checkpoint is NOT deleted."""
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=1)
    d100 = _mk_step_dir(tracker, 100)
    tracker.register(100, 50.0, is_regular=True)  # coincided with a save_freq checkpoint
    _mk_step_dir(tracker, 200)
    deleted = tracker.register(200, 60.0, is_regular=False)
    assert deleted == []  # regular dir left to normal retention
    assert d100.exists()
    assert [e["step"] for e in tracker.steps] == [200]


def test_running_best_tracker_keeps_top_n(tmp_path):
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=3)
    all_deleted = []
    for i, step in enumerate([100, 200, 300, 400, 500]):
        _mk_step_dir(tracker, step)
        all_deleted += tracker.register(step, float(10 * (i + 1)), is_regular=False)
    # Pool holds the 3 most-recent; the 2 oldest non-regular dirs were deleted.
    assert [e["step"] for e in tracker.steps] == [300, 400, 500]
    assert sorted(d.name for d in all_deleted) == [
        get_step_checkpoint_dir(tmp_path, 1000, 100).name,
        get_step_checkpoint_dir(tmp_path, 1000, 200).name,
    ]


def test_running_best_tracker_count_larger_than_found(tmp_path):
    tracker = _mk_tracker(tmp_path, higher_is_better=False, max_count=5)
    _mk_step_dir(tracker, 100)
    assert tracker.register(100, 0.2, is_regular=False) == []
    _mk_step_dir(tracker, 200)
    assert tracker.register(200, 0.1, is_regular=False) == []
    assert {e["step"] for e in tracker.steps} == {100, 200}


def test_running_best_tracker_protected_dirs(tmp_path):
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=3)
    for i, step in enumerate([100, 200]):
        _mk_step_dir(tracker, step)
        tracker.register(step, float(10 * (i + 1)), is_regular=False)
    assert tracker.protected_dirs() == {tracker.step_dir(100), tracker.step_dir(200)}


def test_running_best_state_roundtrip(tmp_path):
    tracker = _mk_tracker(tmp_path, higher_is_better=False, max_count=2)
    _mk_step_dir(tracker, 100)
    tracker.register(100, 0.2, is_regular=True)
    _mk_step_dir(tracker, 200)
    tracker.register(200, 0.1, is_regular=False)
    save_running_best_state(tracker, metric="val_loss")

    loaded = load_running_best_state(tmp_path, total_steps=1000, higher_is_better=False, max_count=2)
    assert loaded.best == 0.1
    assert [e["step"] for e in loaded.steps] == [100, 200]
    assert loaded.steps[0]["is_regular"] is True
    assert loaded.steps[1]["is_regular"] is False


def test_running_best_state_missing_file_is_fresh(tmp_path):
    loaded = load_running_best_state(tmp_path, total_steps=1000, higher_is_better=True, max_count=1)
    assert loaded.best == float("-inf")
    assert loaded.steps == []


def test_running_best_state_self_heals_pruned_dirs(tmp_path):
    """Pool entries whose dir was pruned are dropped on load; the high-water mark survives."""
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=3)
    _mk_step_dir(tracker, 100)
    tracker.register(100, 50.0, is_regular=False)
    _mk_step_dir(tracker, 200)
    tracker.register(200, 60.0, is_regular=False)
    save_running_best_state(tracker, metric="eval_success")

    # Simulate a later prune removing the step-100 dir.
    shutil.rmtree(tracker.step_dir(100))

    loaded = load_running_best_state(tmp_path, total_steps=1000, higher_is_better=True, max_count=3)
    assert [e["step"] for e in loaded.steps] == [200]
    assert loaded.best == 60.0  # high-water mark preserved


def test_running_best_state_direction_mismatch_resets_best(tmp_path):
    """Resuming with a flipped optimization direction discards the stale high-water mark."""
    tracker = _mk_tracker(tmp_path, higher_is_better=True, max_count=3)
    _mk_step_dir(tracker, 100)
    tracker.register(100, 50.0, is_regular=False)
    save_running_best_state(tracker, metric="eval_success")

    # Load as a minimize metric (val_loss): the persisted best (50.0) is uncomparable -> reset.
    loaded = load_running_best_state(tmp_path, total_steps=1000, higher_is_better=False, max_count=3)
    assert loaded.best == float("inf")  # reset to the minimize sentinel
    assert [e["step"] for e in loaded.steps] == [100]  # pool kept


# ---------------------------------------------------------------------------
# prune_old_checkpoints protected_paths
# ---------------------------------------------------------------------------


def test_prune_skips_protected_paths(tmp_path):
    """Protected (running-best) dirs and non-dir files are never deleted."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    latest = ckpts / "000300"
    protected = ckpts / "000150"
    old = ckpts / "000100"
    for d in (latest, protected, old):
        d.mkdir()
    state_file = ckpts / "running_best.json"
    state_file.write_text("{}")

    prune_old_checkpoints(str(latest), protected_paths={protected})

    assert latest.exists()
    assert protected.exists()
    assert state_file.exists()
    assert not old.exists()


def test_prune_protected_none_is_backward_compatible(tmp_path):
    """Default protected_paths=None keeps only the latest dir (existing behavior)."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    latest = ckpts / "000300"
    old = ckpts / "000100"
    for d in (latest, old):
        d.mkdir()

    prune_old_checkpoints(str(latest))

    assert latest.exists()
    assert not old.exists()
