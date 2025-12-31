from collections import OrderedDict

import numpy as np
import pytest
import torch
from einops import rearrange
from robosuite.utils.transform_utils import quat2axisangle

from opentau.utils.libero import (
    LiberoObservationRecorder,
    libero2torch,
    rotate_numpy_image,
    summarize_libero_results,
)
from opentau.configs.libero import LiberoEnvConfig, TrainConfigWithLiberoEval
from tests.utils import generic_equal


def _get_summary(**kwargs):
    default = {
        "success_indices": [],
        "failure_indices": [],
        "crashed_indices": [],
        "success_count": 0,
        "failure_count": 0,
        "crashed_count": 0,
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "crashed_rate": 0.0,
        "steps_taken": [],
        "avg_steps_taken_until_success": None,
    }

    default.update(kwargs)

    return default


H, W, C = 5, 5, 3  # tiny image for testing
HWC = H * W * C
assert 2 * HWC == 2 * H * W * C, "HWC should be equal to 2 * H * W * C for the test images"

ROBOT0_JOINT_POS = np.linspace(0, 1, 7)
ROBOT0_EEF_POS = np.linspace(0, 1, 3)
ROBOT0_EEF_QUAT = np.linspace(0, 1, 4)
ROBOT0_GRIPPER_QPOS = np.array([0.5, -0.5])
AGENTVIEW_IMAGE = np.arange(0, HWC).astype(np.uint8).reshape(H, W, C)
ROBOT0_EYE_IN_HAND_IMAGE = np.arange(HWC, 2 * HWC).astype(np.uint8).reshape(H, W, C)
PROMPT = "pick up the alphabet soup and place it in the basket"
MAX_STATE_DIM = 10
ACTION_CHUNK = 50
NUM_CAMS, ACTION_EXPERT_NUM_CAMS = 2, 2


@pytest.fixture
def obs():
    return OrderedDict(
        {
            "robot0_joint_pos": ROBOT0_JOINT_POS,
            "robot0_eef_pos": ROBOT0_EEF_POS,
            "robot0_eef_quat": ROBOT0_EEF_QUAT,
            "robot0_gripper_qpos": ROBOT0_GRIPPER_QPOS,
            "agentview_image": AGENTVIEW_IMAGE,
            "robot0_eye_in_hand_image": ROBOT0_EYE_IN_HAND_IMAGE,
        }
    )


@pytest.fixture
def expected_torch_input():
    state = np.hstack(
        [ROBOT0_EEF_POS, quat2axisangle(ROBOT0_EEF_QUAT), ROBOT0_GRIPPER_QPOS, np.array([0.0, 0.0])]
    )
    return {
        "camera0": torch.tensor(rotate_numpy_image(AGENTVIEW_IMAGE).copy()),
        "camera1": torch.tensor(rotate_numpy_image(ROBOT0_EYE_IN_HAND_IMAGE).copy()),
        "local_camera0": torch.tensor(rotate_numpy_image(AGENTVIEW_IMAGE).copy()),
        "local_camera1": torch.tensor(rotate_numpy_image(ROBOT0_EYE_IN_HAND_IMAGE).copy()),
        "prompt": PROMPT,
        "state": torch.tensor(state.copy()),
        "img_is_pad": torch.zeros(NUM_CAMS, dtype=torch.bool),
        "local_img_is_pad": torch.zeros(ACTION_EXPERT_NUM_CAMS, dtype=torch.bool),
        "action_is_pad": torch.zeros(ACTION_CHUNK, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    argnames=["results", "expected"],
    argvalues=[
        ([], {"message": "No results to summarize."}),
        (
            [10],
            _get_summary(
                success_indices=[0],
                success_count=1,
                success_rate=1.0,
                steps_taken=[10],
                total_simulations=1,
                avg_steps_taken_until_success=10,
            ),
        ),
        (
            [-1],
            _get_summary(
                failure_indices=[0],
                failure_count=1,
                failure_rate=1.0,
                steps_taken=[-1],
                total_simulations=1,
                avg_steps_taken_until_success=None,
            ),
        ),
        (
            [-2],
            _get_summary(
                crashed_indices=[0],
                crashed_count=1,
                crashed_rate=1.0,
                steps_taken=[-2],
                total_simulations=1,
                avg_steps_taken_until_success=None,
            ),
        ),
        (
            [10, 20, -1, -2],
            _get_summary(
                success_indices=[0, 1],
                failure_indices=[2],
                crashed_indices=[3],
                success_count=2,
                failure_count=1,
                crashed_count=1,
                success_rate=0.5,
                failure_rate=0.25,
                crashed_rate=0.25,
                steps_taken=[10, 20, -1, -2],
                total_simulations=4,
                avg_steps_taken_until_success=15,
            ),
        ),
    ],
)
def test_summarize_libero_results(results, expected):
    summary = summarize_libero_results(results)
    assert summary == expected


@pytest.mark.parametrize(argnames=["height", "width"], argvalues=[(64, 64), (128, 256), (256, 128)])
def test_rotate_numpy_image(height, width):
    img = np.random.randint(0, 256, (height, width, 3))
    img2 = rotate_numpy_image(img)
    img2 = rearrange(img2, "c h w -> h w c")
    # Rotating by 90 degrees twice is equivalent to flipping vertically and horizontally
    img2 = np.flip(np.flip(img2, 0), 1)
    img2 *= 255.0
    assert np.allclose(img, img2)


def test_libero2torch(obs, expected_torch_input, dataset_mixture_config, policy_config):
    cfg = TrainConfigWithLiberoEval(
        dataset_mixture=dataset_mixture_config,
        policy=policy_config,
        batch_size=8,
        libero=LiberoEnvConfig(suite="object", id=0),
        max_state_dim=MAX_STATE_DIM,
        action_chunk=ACTION_CHUNK,
        action_expert_num_cams=ACTION_EXPERT_NUM_CAMS,
        num_cams=NUM_CAMS,
    )
    torch_input = libero2torch(obs, cfg, "cpu", torch.float64)

    assert generic_equal(torch_input, expected_torch_input)


def test_libero_observation_recorder(tmp_path):
    cams = ["c1", "c2"]

    with LiberoObservationRecorder(None, camera_names=cams) as recorder:
        recorder.record({})

    root = tmp_path / "test_recorder"
    with LiberoObservationRecorder(root, camera_names=cams) as recorder:
        for _ in range(10):
            recorder.record(
                {
                    "c1": np.zeros((16, 16, 3), dtype=np.uint8),
                    "c2": np.ones((16, 16, 3), dtype=np.uint8),
                }
            )

    assert (root / "c1.mp4").exists()
    assert (root / "c2.mp4").exists()
