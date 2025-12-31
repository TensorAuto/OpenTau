#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import inspect
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from opentau import available_policies
from opentau.datasets.dataset_mixture import WeightedDatasetMixture
from opentau.datasets.utils import cycle, dataset_to_policy_features
from opentau.policies.factory import (
    get_policy_class,
    make_policy_config,
)
from opentau.policies.normalize import Normalize, Unnormalize
from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.value.reward import (
    calculate_n_step_return,
    calculate_return_bins_with_equal_width,
)
from opentau.utils.utils import create_dummy_observation
from opentau.configs.train import TrainPipelineConfig
from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from tests.artifacts.policies.save_policy_to_safetensors import get_policy_stats
from tests.utils import generic_equal, require_x86_64_kernel


@pytest.mark.slow  # ~ 2 min
@pytest.mark.parametrize("policy_name", ["pi0"])
@pytest.mark.gpu
def test_save_and_load_pretrained(dummy_dataset_metadata, tmp_path, policy_name: str):
    policy_cls = get_policy_class(policy_name)
    policy_cfg = make_policy_config(policy_name)
    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }
    policy = policy_cls(policy_cfg)
    policy = policy.to(device="cuda", dtype=torch.bfloat16)
    save_dir = tmp_path / f"test_save_and_load_pretrained_{policy_cls.__name__}"
    policy.save_pretrained(save_dir, safe_serialization=False)
    loaded_policy = policy_cls.from_pretrained(save_dir, config=policy_cfg).to(dtype=torch.bfloat16)
    torch.testing.assert_close(
        list(policy.parameters()), list(loaded_policy.parameters()), check_device=False
    )


@pytest.mark.parametrize("policy_name", available_policies)
def test_get_policy_and_config_classes(policy_name: str):
    """Check that the correct policy and config classes are returned."""
    policy_cls = get_policy_class(policy_name)
    policy_cfg = make_policy_config(policy_name)
    assert policy_cls.name == policy_name
    assert issubclass(
        policy_cfg.__class__, inspect.signature(policy_cls.__init__).parameters["config"].annotation
    )


@pytest.mark.skip(reason="Slow and we do not really use policy defaults")
@pytest.mark.gpu
@pytest.mark.slow  # ~ 1 min
@pytest.mark.parametrize("policy_name", ["pi0"])
def test_policy_defaults(dummy_dataset_metadata, policy_name: str):
    """Check that the policy can be instantiated with defaults."""
    policy_cls = get_policy_class(policy_name)
    policy_cfg = make_policy_config(policy_name)
    features = dataset_to_policy_features(dummy_dataset_metadata.features)
    policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    policy_cfg.input_features = {
        key: ft for key, ft in features.items() if key not in policy_cfg.output_features
    }
    policy_cls(policy_cfg)


@pytest.mark.parametrize("insert_temporal_dim", [False, True])
def test_normalize(insert_temporal_dim, capsys):
    """
    Test that normalize/unnormalize can run without exceptions when properly set up, and that they raise
    an exception when the forward pass is called without the stats having been provided.

    TODO(rcadene, alexander-soare): This should also test that the normalization / unnormalization works as
    expected.
    """

    input_features = {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 96, 96),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(10,),
        ),
    }
    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(5,),
        ),
    }

    norm_map = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }

    dataset_stats = {
        "observation.image": {
            "mean": torch.randn(3, 1, 1),
            "std": torch.randn(3, 1, 1),
            "min": torch.randn(3, 1, 1),
            "max": torch.randn(3, 1, 1),
        },
        "observation.state": {
            "mean": torch.randn(10),
            "std": torch.randn(10),
            "min": torch.randn(10),
            "max": torch.randn(10),
        },
        "action": {
            "mean": torch.randn(5),
            "std": torch.randn(5),
            "min": torch.randn(5),
            "max": torch.randn(5),
        },
    }

    bsize = 2
    input_batch = {
        "observation.image": torch.randn(bsize, 3, 96, 96),
        "observation.state": torch.randn(bsize, 10),
    }
    output_batch = {
        "action": torch.randn(bsize, 5),
    }

    if insert_temporal_dim:
        tdim = 4

        for key in input_batch:
            # [2,3,96,96] -> [2,tdim,3,96,96]
            input_batch[key] = torch.stack([input_batch[key]] * tdim, dim=1)

        for key in output_batch:
            output_batch[key] = torch.stack([output_batch[key]] * tdim, dim=1)

    # test without stats
    normalize = Normalize(input_features, norm_map, stats=None)
    for ft in input_features:
        assert hasattr(normalize, "buffer_" + ft.replace(".", "_"))

    with pytest.raises(AssertionError):
        normalize(input_batch)

    # test with stats
    normalize = Normalize(input_features, norm_map, stats=dataset_stats)
    normalize(input_batch)

    # test loading pretrained models
    new_normalize = Normalize(input_features, norm_map, stats=None)
    new_normalize.load_state_dict(normalize.state_dict())
    new_normalize(input_batch)

    # test without stats
    unnormalize = Unnormalize(output_features, norm_map, stats=None)
    with pytest.raises(AssertionError):
        unnormalize(output_batch)

    # test with stats
    unnormalize = Unnormalize(output_features, norm_map, stats=dataset_stats)
    for ft in output_features:
        assert hasattr(unnormalize, "buffer_" + ft.replace(".", "_"))

    unnormalize(output_batch)

    # test loading pretrained models
    new_unnormalize = Unnormalize(output_features, norm_map, stats=None)
    new_unnormalize.load_state_dict(unnormalize.state_dict())
    unnormalize(output_batch)

    # check warmings
    normalize({"observation.image": torch.randn(bsize, 3, 96, 96)})
    captured = capsys.readouterr()

    assert "Warning: observation.state was missing from the batch during " in captured.err

    unnormalize({})
    captured = capsys.readouterr()

    assert "Warning: action was missing from the batch during " in captured.err


@pytest.mark.parametrize(
    "ds_repo_id, policy_name, policy_kwargs, file_name_extra",
    [
        pytest.param(
            "lerobot/droid_100",
            "pi0",
            {"chunk_size": 50, "pretrained_path": "lerobot/pi0"},
            "pretrained",
            marks=pytest.mark.skip(reason="Slow"),
        ),
    ],
)
# As artifacts have been generated on an x86_64 kernel, this test won't
# pass if it's run on another platform due to floating point errors
@require_x86_64_kernel
@pytest.mark.gpu
@pytest.mark.slow
def test_backward_compatibility(ds_repo_id: str, policy_name: str, policy_kwargs: dict, file_name_extra: str):
    """
    NOTE: If this test does not pass, and you have intentionally changed something in the policy:
        1. Inspect the differences in policy outputs and make sure you can account for them. Your PR should
           include a report on what changed and how that affected the outputs.
        2. Go to the `if __name__ == "__main__"` block of `tests/scripts/save_policy_to_safetensors.py` and
           add the policies you want to update the test artifacts for.
        3. Run `python tests/scripts/save_policy_to_safetensors.py`. The test artifact
           should be updated.
        4. Check that this test now passes.
        5. Remember to restore `tests/scripts/save_policy_to_safetensors.py` to its original state.
        6. Remember to stage and commit the resulting changes to `tests/artifacts`.
    """
    ds_name = ds_repo_id.split("/")[-1]
    artifact_dir = (
        Path(__file__).parent.parent / "artifacts/policies" / f"{ds_name}_{policy_name}_{file_name_extra}"
    )
    artifact_dir = artifact_dir.resolve()
    saved_output_dict = load_file(artifact_dir / "output_dict.safetensors")
    saved_actions = load_file(artifact_dir / "actions.safetensors")

    output_dict, actions = get_policy_stats(ds_repo_id, policy_name, policy_kwargs)

    for key in saved_output_dict:
        torch.testing.assert_close(output_dict[key], saved_output_dict[key], check_device=False)
    for key in saved_actions:
        rtol, atol = (2e-3, 5e-6) if policy_name == "diffusion" else (None, None)  # HACK
        torch.testing.assert_close(actions[key], saved_actions[key], rtol=rtol, atol=atol, check_device=False)


@pytest.mark.parametrize(
    "success, b, episode_end_idx, max_episode_length, current_idx, c_neg, expected_bin_idx",
    [(True, 200, 401, 600, 200, -1000.0, int(400 / 3)), (False, 200, 401, 600, 200, -1000.0, 0)],
)
def test_return_equal_bin(
    success, b, episode_end_idx, max_episode_length, current_idx, c_neg, expected_bin_idx
):
    bin_idx, _ = calculate_return_bins_with_equal_width(
        success, b, episode_end_idx, max_episode_length, current_idx, c_neg
    )

    assert bin_idx >= 0
    assert bin_idx <= b + 1
    assert bin_idx == expected_bin_idx


@pytest.mark.parametrize(
    "success, n_steps_look_ahead, episode_end_idx, max_episode_length, current_idx, c_neg, expected_return",
    [
        (True, 50, 401, 600, 200, -1000.0, -5 / 60),
        (True, 50, 401, 600, 390, -1000.0, -1 / 60),
        (False, 50, 401, 600, 200, -1000.0, -5 / 60),
        (False, 50, 401, 600, 390, -1000.0, -101 / 60),
    ],
)
def test_return_for_advantage(
    success, n_steps_look_ahead, episode_end_idx, max_episode_length, current_idx, c_neg, expected_return
):
    return_value = calculate_n_step_return(
        success, n_steps_look_ahead, episode_end_idx, max_episode_length, current_idx, c_neg
    )

    assert math.isclose(return_value, expected_return)
