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
    "physical-intelligence/libero": {
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
        "local_camera0": "observation.images.top",
        "local_camera1": "observation.images.main",
        "local_camera2": "observation.images.cv",
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
        "local_camera0": "observation.images.up",
        "local_camera1": "observation.images.side",
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
    "pixmo": {
        "camera0": "image",
        "state": "state",
        "actions": "actions",
        "prompt": "prompt",
        "response": "postfix",
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

"""
Use "MSE" for mean squared error and "CE" for cross entropy.
Usually robotic data with actions will have an MSE loss while
VQA tasks will have a CE loss.
"""
LOSS_TYPE_MAPPING = {
    "ML-GOD/mt-button-press": "MSE",
    "ML-GOD/libero_spatial_no_noops_1.0.0_lerobot": "MSE",
    "ML-GOD/libero": "MSE",
    "physical-intelligence/libero": "MSE",
    "danaaubakirova/koch_test": "MSE",
    "lerobot/droid_100": "MSE",
    "lerobot/aloha_mobile_cabinet": "MSE",
    "autox/agibot-sample": "MSE",
    "bi-so100-block-manipulation": "MSE",
    "cube-on-cylinder": "MSE",
    "cylinder-on-cube": "MSE",
    "l-shape-on-cross-shape": "MSE",
    "lerobot/svla_so101_pickplace": "MSE",
    "lerobot/svla_so100_pickplace": "MSE",
    "lerobot/svla_so100_stacking": "MSE",
    "pixmo": "CE",
    "dummy": "CE",
    "vsr": "CE",
    "clevr": "CE",
    "cocoqa": "CE",
    "lerobot_dummy": "MSE",
}
