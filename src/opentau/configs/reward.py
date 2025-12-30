from dataclasses import dataclass


@dataclass
class RewardConfig:
    number_of_bins: int = 201
    C_neg: float = -1000.0
    reward_normalizer: int = 400
    N_steps_look_ahead: int = 50
