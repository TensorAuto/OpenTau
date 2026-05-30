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

"""Standard data format mapping for dataset feature names and loss types.

This module provides mappings between standard feature names used internally
by OpenTau and dataset-specific feature names used in various robot learning
and vision-language datasets. It also maps datasets to their appropriate loss
types for training.

The standard format uses canonical names like "camera0", "camera1", "state",
"actions", "prompt", and "response", while different datasets may use
various naming conventions (e.g., "observation.images.image",
"observation.state", "action", "task", etc.). These mappings enable the
codebase to work with multiple datasets without requiring dataset-specific
code paths.

Key Features:
    - Feature name standardization: Maps dataset-specific feature names to
      standard format names for consistent processing across datasets.
    - Multi-camera support: Handles datasets with varying numbers of camera
      views, mapping them to standard camera0, camera1, etc. names.
    - Loss type specification: Maps datasets to appropriate loss functions
      (MSE for continuous actions, CE for discrete classification tasks).

Constants:

    DATA_FEATURES_NAME_MAPPING
        Dictionary mapping dataset repository IDs to feature name dictionaries.
        Each inner dictionary maps standard feature names (keys) to
        dataset-specific feature names (values). Standard feature names include:

        - "camera0", "camera1", ...: Camera/image observations
        - "state": Robot state observations
        - "actions": Action outputs
        - "prompt": Task descriptions or prompts
        - "response": Expected responses or labels

Example:
    Access feature name mapping for a dataset:
        >>> mapping = DATA_FEATURES_NAME_MAPPING["lerobot/aloha_mobile_cabinet"]
        >>> mapping["camera0"]  # Returns "observation.images.cam_right_wrist"
        >>> mapping["actions"]  # Returns "action"

"""

DATA_FEATURES_NAME_MAPPING = {
    "ML-GOD/mt-button-press": {
        "camera0": "observation.image",
        "state": "observation.robot_state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "ML-GOD/libero_spatial_no_noops_1.0.0_lerobot": {
        "camera0": "observation.images.image",
        "camera1": "observation.images.wrist_image",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "ML-GOD/libero": {
        "camera0": "image",
        "camera1": "wrist_image",
        "state": "state",
        "actions": "actions",
        "prompt": "task",
        "response": "response",
    },
    # `TensorAuto/libero` is the corrected re-export of upstream
    # `physical-intelligence/libero`: identical frames/episodes, but the fps
    # label is fixed (10 -> 20 Hz) and the format bumped (v2.0 -> v2.1). The
    # upstream copy carries a *wrong* fps=10 label even though its actions are
    # authored at the sim's native 20 Hz, which silently breaks training/eval
    # frequency alignment. We deliberately do NOT keep a
    # `physical-intelligence/libero` entry here (an unmapped repo_id raises at
    # dataset build) to discourage its use — point configs at `TensorAuto/libero`.
    "TensorAuto/libero": {
        "camera0": "image",
        "camera1": "wrist_image",
        "state": "state",
        "actions": "actions",
        "prompt": "task",
        "response": "response",
    },
    "danaaubakirova/koch_test": {
        "camera0": "observation.images.laptop",
        "camera1": "observation.images.phone",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "lerobot/droid_100": {
        "camera0": "observation.images.exterior_image_1_left",
        "camera1": "observation.images.exterior_image_2_left",
        "camera2": "observation.images.wrist_image_left",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "lerobot/aloha_mobile_cabinet": {
        "camera0": "observation.images.cam_right_wrist",
        "camera1": "observation.images.cam_high",
        "camera2": "observation.images.cam_left_wrist",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "autox/agibot-sample": {
        "camera0": "observation.images.head_left_fisheye",
        "camera1": "observation.images.head_right_fisheye",
        "camera2": "observation.images.top_head",
        "camera3": "observation.images.hand_left",
        "camera4": "observation.images.hand_right",
        "camera5": "observation.images.head_center_fisheye",
        "camera6": "observation.images.back_left_fisheye",
        "camera7": "observation.images.back_right_fisheye",
        "camera8": "observation.images.cam_top_depth",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "bi-so100-block-manipulation": {
        "camera0": "observation.images.top",
        "camera1": "observation.images.main",
        "camera2": "observation.images.cv",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "cube-on-cylinder": {
        "camera0": "observation.images.top",
        "camera1": "observation.images.main",
        "camera2": "observation.images.cv",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "cylinder-on-cube": {
        "camera0": "observation.images.top",
        "camera1": "observation.images.main",
        "camera2": "observation.images.cv",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "l-shape-on-cross-shape": {
        "camera0": "observation.images.top",
        "camera1": "observation.images.main",
        "camera2": "observation.images.cv",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "lerobot/svla_so101_pickplace": {
        "camera0": "observation.images.up",
        "camera1": "observation.images.side",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "lerobot/svla_so100_pickplace": {
        "camera0": "observation.images.top",
        "camera1": "observation.images.wrist",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "lerobot/svla_so100_stacking": {
        "camera0": "observation.images.top",
        "camera1": "observation.images.wrist",
        "state": "observation.state",
        "actions": "action",
        "prompt": "task",
        "response": "response",
    },
    "dummy": {
        "camera0": "image",
        "state": "state",
        "actions": "actions",
        "prompt": "prompt",
        "response": "postfix",
    },
    "vsr": {
        "camera0": "image",
        "state": "state",
        "actions": "actions",
        "prompt": "prompt",
        "response": "postfix",
    },
    "clevr": {
        "camera0": "image",
        "state": "state",
        "actions": "actions",
        "prompt": "prompt",
        "response": "postfix",
    },
    "cocoqa": {
        "camera0": "image",
        "state": "state",
        "actions": "actions",
        "prompt": "prompt",
        "response": "postfix",
    },
    "lerobot_dummy": {
        "camera0": "camera0",
        "state": "state",
        "actions": "actions",
        "prompt": "task",
        "response": "response",
    },
}


# Case-insensitive sentinels that mean "no control-mode discriminator", mirroring
# `opentau.datasets.dataset_mixture._NORM_KEY_MISSING_VALUES` so that the column
# resolution here and the norm-head split there agree on what counts as a real
# control mode.
_MISSING_CONTROL_MODE_VALUES: frozenset[str] = frozenset({"unknown"})


def feature_mapping_key(repo_id: str, control_mode: str | None) -> str:
    """Return the ``DATA_FEATURES_NAME_MAPPING`` key for a dataset entry.

    Two mixture entries can share a ``repo_id`` while declaring different
    ``data_features_name_mapping`` values — e.g. the same robot exposed as
    ``control_mode="joint"`` (``actions -> action_joint``) and
    ``control_mode="ee"`` (``actions -> action_ee``). Keying the global mapping
    by ``repo_id`` alone makes the second registration clobber the first, so the
    "joint" entry would silently read the end-effector column. Disambiguating by
    ``control_mode`` lets both coexist.

    Returns ``"<repo_id>::<control_mode>"`` when ``control_mode`` is a real,
    non-empty label, and the plain ``repo_id`` otherwise (missing / whitespace /
    ``"unknown"``) — so built-in defaults and single-mode datasets keep their
    existing ``repo_id`` keys.
    """
    cm = (control_mode or "").strip()
    if cm and cm.casefold() not in _MISSING_CONTROL_MODE_VALUES:
        return f"{repo_id}::{cm}"
    return repo_id


def resolve_feature_mapping(repo_id: str, control_mode: str | None = None) -> dict[str, str]:
    """Look up a dataset's feature-name mapping, control-mode-aware.

    Prefers the precise ``"<repo_id>::<control_mode>"`` entry and falls back to
    the plain ``repo_id`` entry (built-in defaults / single-mode datasets /
    back-compat). Raises ``KeyError`` if neither is registered.
    """
    mapping = DATA_FEATURES_NAME_MAPPING.get(feature_mapping_key(repo_id, control_mode))
    if mapping is None:
        mapping = DATA_FEATURES_NAME_MAPPING.get(repo_id)
    if mapping is None:
        raise KeyError(
            f"No feature mapping registered for repo_id={repo_id!r} (control_mode={control_mode!r})."
        )
    return mapping
