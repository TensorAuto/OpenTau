def calculate_return_bins_with_equal_width(
    success: bool,
    b: int,
    episode_end_idx: int,
    max_episode_length: int,
    current_idx: int,
    c_neg: float = -100.0,
) -> int:
    """
    Defines sparse Reward function for the pi0.6 policy to train value function network.
    Args:
        success: defuines if the episode was successful or failed
        B: number of bins to discretize the reward into , including the special bin 0
        episode_end_idx: index of the end of the episode, exclusive to the last step
        max_episode_length: maximum length of the episode
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
    return_normalized = return_value / max_episode_length

    # if reward_normalize if small than -1 but it in special bin 0
    if return_normalized < -1:
        return 0, return_normalized
    # otherwise, compute the bin index and add 1 as the bin index should start from 1 and not 0
    else:
        bin_idx = int((return_normalized + 1) * (b - 2))
        return bin_idx + 1, return_normalized


def calculate_return_for_advantage(
    success: bool,
    n_steps_look_ahead: int,
    episode_end_idx: int,
    max_episode_length: int,
    current_idx: int,
    c_neg: float = -100.0,
) -> int:
    """
    Defines sparse Reward function for the pi0.6 policy to calculate advantage .
    Args:
        success: defuines if the episode was successful or failed
        N_steps_look_ahead: number of steps to look ahead for calculating reward
        episode_end_idx: index of the end of the episode
        max_episode_length: maximum length of the episode
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
    return_normalized = return_value / max_episode_length

    return return_normalized
