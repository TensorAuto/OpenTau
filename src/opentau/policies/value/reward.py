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


def calculate_return_bins_with_equal_width(
    success: bool,
    b: int,
    episode_end_idx: int,
    reward_normalizer: int,
    current_idx: int,
    c_neg: float = -100.0,
) -> int:
    """
    Defines sparse Reward function for the pi0.6 policy to train value function network.
    Args:
        success: defuines if the episode was successful or failed
        B: number of bins to discretize the reward into , including the special bin 0
        episode_end_idx: index of the end of the episode, exclusive to the last step
        reward_normalizer: maximum length of the episode
        current_idx: current index of the episode
        C_neg: negative reward for failed episodes
    Returns:
        tuple: (reward bin index)
    """

    # calculate the reward for each step ie -1 till the end of episode and exclude the last step
    return_value = current_idx - episode_end_idx + 1
    # add negative reward for last step if episode is a failure, else add nothing for a successful episode
    if not success:
        return_value += c_neg

    # normalize the reward to the range of -1 to 0
    return_normalized = return_value / reward_normalizer
    # mapping normalized reward [-1,0) to bin index [0,b-1]
    bin_idx = int((return_normalized + 1) * (b - 1))
    return bin_idx, return_normalized


def calculate_n_step_return(
    success: bool,
    n_steps_look_ahead: int,
    episode_end_idx: int,
    reward_normalizer: int,
    current_idx: int,
    c_neg: float = -100.0,
) -> int:
    """
    Defines sparse Reward function for the pi0.6 policy to calculate advantage .
    Args:
        success: defuines if the episode was successful or failed
        N_steps_look_ahead: number of steps to look ahead for calculating reward
        episode_end_idx: index of the end of the episode
        reward_normalizer: maximum length of the episode
        current_idx: current index of the episode
        C_neg: negative reward for failed episodes
    Returns:
        tuple: (continuous reward)
    """
    # calculate the reward till the next n_steps_look_ahead steps
    return_value = max(current_idx - episode_end_idx + 1, -1 * n_steps_look_ahead)
    # add negative reward for last step if episode is a failure, else add nothing for a successful episode. also check if
    if not success and current_idx + n_steps_look_ahead >= episode_end_idx:
        return_value += c_neg

    # normalize the reward to the range of -1 to 0
    return_normalized = return_value / reward_normalizer

    return return_normalized
