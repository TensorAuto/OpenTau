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

import logging
from pathlib import Path

import torch

from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.policies.pi0.modeling_pi0 import PI0Policy
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.utils.monkey_patch import (
    torch_cumsum_patch,
    torch_full_patch,
    torch_pow_patch,
)
from opentau.utils.utils import auto_torch_device

# Some patches are necessary only for dynamo export, which has current upstream bugs.
# Nonetheless, we apply them here to ensure future compatibility.
patches = [
    torch_cumsum_patch,  # This is always necessary to load the ONNX artifact without error.
    torch_full_patch,
    torch_pow_patch,
]

KEY_STATES = "key_states"
VALUE_STATES = "value_states"


class InferenceWrapper(torch.nn.Module):
    r"""Helper class to wrap the robot action decoder for ONNX export,
    such that each input tensor is an individual argument to the `forward` method.
    """

    def __init__(
        self,
        decoder: PI0Policy | PI05Policy,
        *,
        prefix_pad_masks: torch.Tensor,
        prefix_offsets: torch.Tensor,
        num_cross_att_tokens: int,
        layer_idx: int,
    ):
        super().__init__()
        self.decoder = decoder
        self.prefix_pad_masks = prefix_pad_masks
        self.prefix_offsets = prefix_offsets
        self.num_cross_att_tokens = num_cross_att_tokens
        self.layer_idx = layer_idx

    def forward(self, key_states, value_states, state):
        vlm_tokens = (
            {
                self.layer_idx: {
                    KEY_STATES: key_states,
                    VALUE_STATES: value_states,
                },
            },
            self.prefix_pad_masks,
            self.prefix_offsets,
            self.num_cross_att_tokens,
        )
        observation = {
            "state": state,
        }
        actions = self.decoder.sample_actions(
            observation,
            vlm_token_cache_override=vlm_tokens,
        )
        return actions


# Get the VLM cache for the dummy observation. This guarantees consistency with post-loading usage.
def get_vlm_cache(cfg: TrainPipelineConfig, device: torch.device, dtype: torch.dtype):
    logging.info("Getting VLM cache...")
    policy_class = get_policy_class(cfg.policy.type)
    cloud_vlm = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    cloud_vlm.set_execution_target("cloud")
    cloud_vlm.to(device=device, dtype=torch.bfloat16)
    cloud_vlm.eval()

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
    cache, prefix_pad_masks, prefix_offsets, num_cross_att_tokens = cloud_vlm.get_vlm_tokens(vlm_observation)
    assert len(cache) == 1, f"Expected only one cache entry for the dummy observation. Got {len(cache)}."
    idx = list(cache)[0]
    return (
        cache[idx][KEY_STATES].to(dtype=dtype),
        cache[idx][VALUE_STATES].to(dtype=dtype),
        prefix_pad_masks,
        prefix_offsets,
        num_cross_att_tokens,
        idx,
    )


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    device = auto_torch_device()
    dtype = torch.float32

    # arguments for the dummy observation
    (
        key_states,
        value_states,
        prefix_pad_masks,
        prefix_offsets,
        num_cross_att_tokens,
        layer_idx,
    ) = get_vlm_cache(cfg, device, dtype)
    state = torch.zeros((1, cfg.max_state_dim), device=device, dtype=dtype)
    args = (key_states, value_states, state)
    logging.info("Generated example args")

    policy_class = get_policy_class(cfg.policy.type)
    robot_action_decoder = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    robot_action_decoder.set_execution_target("robot")
    robot_action_decoder.to(device)
    robot_action_decoder.to(dtype=dtype)
    robot_action_decoder.eval()
    inference_wrapper = InferenceWrapper(
        robot_action_decoder,
        prefix_pad_masks=prefix_pad_masks,
        prefix_offsets=prefix_offsets,
        num_cross_att_tokens=num_cross_att_tokens,
        layer_idx=layer_idx,
    )
    logging.info("Loaded policy")

    logging.info("Applying monkey patches...")
    for patch in patches:
        patch()

    logging.info("Exporting model to ONNX...")
    with torch.inference_mode():
        path = Path(cfg.policy.pretrained_path) / "robot_action_decoder.onnx"
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)  # Should be a no-op
        print("Exporting model to ONNX at path:", path)
        print("Current directory:", Path.cwd())
        print("Trying to write to:", path)
        with open(path, "wb"):
            print("Write permissions check passed for:", path)
        print("Running torch.onnx.export...")
        torch.onnx.export(
            inference_wrapper.eval(),
            args,
            path,
            input_names=[KEY_STATES, VALUE_STATES, "state"],
            output_names=["action_chunk"],
            opset_version=18,
            do_constant_folding=False,  # constant folding causes weird errors (getting dim -1 from a 0-dim scalar) after forward pass succeeds
        )
        logging.info(f"Successfully exported model to '{path}'.")


if __name__ == "__main__":
    main()
