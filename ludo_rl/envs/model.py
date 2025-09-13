from __future__ import annotations

from dataclasses import dataclass, field

from ludo.constants import Colors
from rl_base.envs.model import (
    BaseEnvConfig,
    BaseRewardConfig,
    ObservationConfig,
    OpponentCurriculumConfig,
    OpponentsConfig,
)


@dataclass
class RewardConfig(BaseRewardConfig):
    """Improved reward configuration for better PPO learning."""

    # Increase terminal rewards for stronger learning signals
    win: float = 200.0  # Increased from 100.0
    lose: float = -200.0  # Increased from -150.0
    finish_token: float = 25.0  # Increased from 10.0

    # Boost major action rewards
    capture: float = 15.0  # Increased from 8.0
    got_captured: float = -15.0  # Increased from -8.0
    extra_turn: float = 8.0  # Increased from 3.0

    # Significantly increase progress rewards for better learning signal
    progress_scale: float = 0.5  # Increased from 0.05 (10x boost!)
    home_progress_bonus: float = 5.0  # Increased from 2.0
    home_approach_bonus: float = 3.0  # Increased from 1.0

    # Boost strategic positioning rewards
    blocking_bonus: float = 4.0  # Increased from 1.5
    safety_bonus: float = 6.0  # Increased from 2.0

    # Increase exploration incentives
    diversity_bonus: float = 2.0  # Increased from 0.5
    active_token_bonus: float = 0.3  # Increased from 0.1
    inactivity_penalty: float = -0.1  # Increased from -0.02

    # Stronger penalties for illegal actions
    illegal_action: float = -30.0  # Increased from -15.0

    # Enable probabilistic rewards for risk-awareness
    use_probabilistic_rewards: bool = False
    risk_weight: float = 0.3  # Increased from 0.2


@dataclass
class EnvConfig(BaseEnvConfig):
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    opponent_curriculum: OpponentCurriculumConfig = field(
        default_factory=OpponentCurriculumConfig
    )
    # If True, environment will randomize the agent's seat/color each reset
    randomize_agent_seat: bool = True
