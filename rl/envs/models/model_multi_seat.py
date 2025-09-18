from __future__ import annotations

from dataclasses import dataclass, field

from ludo_engine.models import Colors

from .model_base import (
    BaseEnvConfig,
    BaseRewardConfig,
    ObservationConfig,
    OpponentCurriculumConfig,
    OpponentsConfig,
)


@dataclass
class RewardConfig(BaseRewardConfig):
    """Corrected reward configuration for stable PPO learning.

    Values are tuned for stable RL training with proper reward shaping.
    Terminal rewards are moderate to provide clear learning signals.
    Action rewards are balanced to encourage good play without dominating.
    """

    # Terminal rewards (highest priority - sparse but strong signals)
    win: float = 100.0  # Increased for stronger learning signal
    lose: float = -100.0  # Increased for stronger learning signal
    finish_token: float = 10.0  # Increased for stronger learning signal

    # Major action rewards (should encourage key strategic actions)
    capture: float = 8.0  # Capturing opponent tokens - very valuable
    got_captured: float = -8.0  # Being captured - symmetric penalty
    extra_turn: float = 3.0  # Rolling 6 for another turn - useful but not dominant

    # Strategic positioning rewards (moderate shaping)
    home_progress_bonus: float = 2.0  # Progress in home column
    blocking_bonus: float = 1.5  # Creating blocks to protect position
    safety_bonus: float = 2.0  # Moving to safe positions
    home_approach_bonus: float = 1.0  # Approaching home entry

    # Small continuous rewards (minimal shaping to avoid reward hacking)
    progress_scale: float = 0.05  # Increased from 0.01 for more learning signal
    diversity_bonus: float = 0.5  # Encouraging token activation
    active_token_bonus: float = 0.1  # Bonus per active token
    inactivity_penalty: float = -0.02  # Penalty per token stuck at home

    # Strategic milestone bonuses (configurable scaling) - simplified
    home_column_entry_bonus: float = (
        0.5  # Reduced bonus for entering home column approach
    )
    safe_square_bonus: float = 0.1  # Reduced bonus for landing on safe squares
    near_home_bonus: float = 0.2  # Reduced bonus for being very close to home

    # Diversity bonus scaling (to reduce frequency/magnitude)
    diversity_penalty_scale: float = (
        0.05  # Further reduced scale factor for inactivity penalty
    )
    diversity_bonus_scale: float = (
        0.2  # Further reduced scale factor for active token bonus
    )

    # Clear boundaries (strong penalties for invalid actions)
    illegal_action: float = (
        -50.0
    )  # Invalid moves - very strong negative to prevent illegal actions
    time_penalty: float = -0.001  # Reduced time penalty for less negative pressure

    # Advanced features (disabled by default for simplicity)
    use_probabilistic_rewards: bool = False
    risk_weight: float = 0.2


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
