#!/usr/bin/env python

import logging
from collections import deque
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import numpy as np
import onnxruntime as ort
import torch

from src.opentau.policies.factory import get_policy_class
from src.opentau.utils.random_utils import set_seed
from src.opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    init_logging,
)
from src.opentau.configs import parser
from src.opentau.configs.train import TrainPipelineConfig


class ActionInferenceWrapper:
    r"""This wrapper is used to mock the inference behavior of the robot action decoder.
    The main logic is exported to an ONNX script, which is then loaded here.
    NOTE: this is not a subclass of torch.nn.Module and can run independently of the PyTorch framework.
    """

    def __init__(self, sess: ort.InferenceSession):
        self.sess = sess
        self.key_states_cache = None
        self.value_states_cache = None
        self.q = deque()

    def update_vlm_token_cache(
        self, vlm_token_cache: tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor, int]
    ):
        """Update the VLM token cache with new tokens."""
        # prefix_pad_masks, prefix_offsets and num_cross_att_tokens are not used in this wrapper,
        past_key_values, _prefix_pad_masks, _prefix_offsets, _num_cross_att_tokens = vlm_token_cache
        assert len(past_key_values) == 1, (
            f"Expected a single layer of past key values. Got {len(past_key_values)} layers."
        )
        idx = list(past_key_values)[0]
        # TODO try not to cast to np.float32 and keep using torch.bfloat16
        self.key_states_cache = self.torch2np(past_key_values[idx]["key_states"])
        self.value_states_cache = self.torch2np(past_key_values[idx]["value_states"])

    @staticmethod
    def torch2np(tensor: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor to a numpy array."""
        if tensor.dtype is torch.bfloat16:
            # Convert bfloat16 to float32 for numpy compatibility
            tensor = tensor.to(torch.float32)
        return tensor.detach().cpu().numpy()

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self.key_states_cache is None:
            raise ValueError("VLM token cache is not set. Call `update_vlm_token_cache` first.")
        if not self.q:
            observation = dict(observation)  # Ensure we don't modify the original observation
            observation["key_states"] = self.key_states_cache
            observation["value_states"] = self.value_states_cache
            input_feed = {
                input_meta.name: observation[input_meta.name] for input_meta in self.sess.get_inputs()
            }
            (actions,) = self.sess.run(None, input_feed)
            self.q.extend(actions)

        return self.q.popleft()


@parser.wrap()
def inference_main(cfg: TrainPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = auto_torch_device()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # -------------------------------------------------------------------------------------
    # Create the cloud VLM using local checkpoint. The cloud VLM is PyTorch model.
    logging.info("Creating cloud VLM")
    policy_class = get_policy_class(cfg.policy.type)
    cloud_vlm = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    cloud_vlm.set_execution_target("cloud")
    cloud_vlm.to(device=device, dtype=torch.bfloat16)
    cloud_vlm.eval()
    get_cloud_vlm_tokens = attempt_torch_compile(cloud_vlm.get_vlm_tokens, device_hint=device)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Create the robot action decoder using the ONNX artifact, which does not require PyTorch.
    logging.info("Creating robot action decoder")
    onnx_path = Path(cfg.policy.pretrained_path) / "robot_action_decoder.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Could not find ONNX file at {onnx_path}. "
            f"Did you export it using the `export_to_onnx.py` script?"
        )
    onnx_session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"],
    )
    # Log input and output metadata
    for i, input_meta in enumerate(onnx_session.get_inputs()):
        logging.info(f"Input {i}: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
    for i, output_meta in enumerate(onnx_session.get_outputs()):
        logging.info(f"Output {i}: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")
    robot_action_decoder = ActionInferenceWrapper(onnx_session)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Prepare dummy observation for the cloud VLM
    vlm_camera_observation = {
        f"camera{i}": torch.zeros((1, 3, *cfg.resolution), dtype=torch.bfloat16, device=device)
        for i in range(cfg.num_cams)
    }
    vlm_observation = {
        **vlm_camera_observation,
        "prompt": ["Pick up yellow lego block and put it in the bin"],
        "state": torch.zeros((1, cfg.max_state_dim), dtype=torch.bfloat16, device=device),
        "img_is_pad": torch.zeros((1, cfg.num_cams), dtype=torch.bool, device=device),
    }
    # -------------------------------------------------------------------------------------

    with torch.inference_mode():
        for _ in range(1000):
            # -----------------------------------------------------------------------------
            # Fetch VLM tokens from the cloud
            vlm_tokens = get_cloud_vlm_tokens(vlm_observation)
            # -----------------------------------------------------------------------------

            # -----------------------------------------------------------------------------
            # Prepare dummy observation for the robot action decoder
            # The observations should match the ONNX model input requirements and be in numpy format
            local_camera_observation = {
                f"local_camera{i}": np.zeros((1, 3, *cfg.resolution), dtype=np.float32)
                for i in range(cfg.action_expert_num_cams)
            }
            local_observation = {
                **local_camera_observation,
                "state": np.zeros((1, cfg.max_state_dim), dtype=np.float32),
            }
            # -----------------------------------------------------------------------------

            # -----------------------------------------------------------------------------
            # Transport vlm_tokens from the cloud to the robot
            # In a real scenario, this is run in a different thread than select_action over the network
            robot_action_decoder.update_vlm_token_cache(vlm_tokens)
            # Select action using the latest VLM token cache and local observation
            action = robot_action_decoder.select_action(local_observation)
            # -----------------------------------------------------------------------------

            print(f"Output dummy action: {action}")

    logging.info("End of inference")


if __name__ == "__main__":
    init_logging()
    inference_main()
