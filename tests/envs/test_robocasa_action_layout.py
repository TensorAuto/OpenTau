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

"""The RoboCasa flat-action layout, pinned against its two independent sources.

``envs/robocasa.py::convert_action`` used to slice a *base-first* layout while RoboCasa's
own converter — and the RoboCasa365 LeRobot datasets every policy trains on — use an
**EE-first** one. The two disagree on where every field lives, and the consequence is
silent: the arm still moves, the episode still runs, the success rate is just far lower.

Two pins, because either alone can rot:

* against ``robocasa.utils.env_utils.convert_action`` when RoboCasa is installed, which is
  the definition the simulator itself is written to; and
* against the *dataset's* recorded column signature, which is what the policy is trained to
  emit and needs no RoboCasa install to check.
"""

import numpy as np
import pytest

from opentau.envs.robocasa import ACTION_DIM, _import_robocasa_with_version_shim, convert_action


def test_convert_action_uses_the_ee_first_layout():
    flat = np.arange(12, dtype=np.float32)
    out = convert_action(flat)
    assert out["action.end_effector_position"].tolist() == [0, 1, 2]
    assert out["action.end_effector_rotation"].tolist() == [3, 4, 5]
    assert out["action.gripper_close"].tolist() == [6]
    assert out["action.base_motion"].tolist() == [7, 8, 9, 10]
    assert out["action.control_mode"].tolist() == [11]
    assert ACTION_DIM == 12


def test_convert_action_matches_robocasas_own_converter():
    """The simulator's definition. Skipped when robocasa is not importable.

    Not ``importorskip``: that only turns ``ImportError`` into a skip, and a stock (non-fork)
    robocasa install fails its import-time ``mujoco`` / ``numpy`` equality asserts with
    ``AssertionError`` -- which is precisely what ``_import_robocasa_with_version_shim``
    exists to get past, so route the import through it and skip on anything it cannot fix.
    """
    try:
        _import_robocasa_with_version_shim()
        from robocasa.utils import env_utils
    except Exception as err:  # not installed, or import-time asserts the shim cannot satisfy
        pytest.skip(f"robocasa is not importable here ({type(err).__name__}: {err})")

    flat = np.arange(12, dtype=np.float32)
    ours = convert_action(flat)
    theirs = env_utils.convert_action(flat)
    assert set(ours) == set(theirs)
    for key in ours:
        assert ours[key].tolist() == np.asarray(theirs[key]).tolist(), key


def test_convert_action_matches_the_datasets_recorded_column_signature():
    """The layout is also readable straight off the training data, with no RoboCasa import.

    Measured over 4000 frames of ``pepijn223/robocasa_pretrain_human300_v4``: exactly one
    column is binary +/-1 (the gripper), exactly four consecutive columns are identically
    zero (base motion, held still in these task classes) and exactly one is constant (the
    control mode). Those three facts pin the whole layout, and they are the reason the
    base-first reading was wrong: it put the "control mode" on a column ranging over
    +/-0.49 and the gripper on one that never opened.
    """
    binary_column = 6
    zero_columns = [7, 8, 9, 10]
    constant_column = 11

    flat = np.arange(12, dtype=np.float32)
    out = convert_action(flat)
    assert out["action.gripper_close"].tolist() == [binary_column]
    assert out["action.base_motion"].tolist() == zero_columns
    assert out["action.control_mode"].tolist() == [constant_column]
    # ...and the remaining six continuous columns are the end-effector command.
    assert out["action.end_effector_position"].tolist() + out["action.end_effector_rotation"].tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
