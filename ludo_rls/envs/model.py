from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ludo.constants import Colors


# Reward calculation constants
class SimpleRewardConstants:
    """Minimal constants for reward calculations - let the agent learn strategy."""

    # Core rewards (let agent learn when these matter)
    TOKEN_PROGRESS_REWARD = 1.0  # Increased from 0.1 for stronger learning signal
    CAPTURE_REWARD = 10.0  # Increased from 5.0
    GOT_CAPTURED_PENALTY = -6.0  # Increased from -3.0
    EXTRA_TURN_BONUS = 2.0  # Increased from 1.0
    SAFE_POSITION_BONUS = 1.0  # Increased from 0.5

    # Terminal rewards
    WIN_REWARD = 200.0  # Increased from 100.0
    LOSS_PENALTY = -200.0  # Increased from -100.0

    # Minimal scaling (avoid over-engineering)
    PROGRESS_NONLINEARITY = 1.2  # Slight non-linearity for progress
    HOME_COLUMN_MULTIPLIER = 2.0  # Bonus for home column progress


@dataclass
class RewardConfig:
    """Balanced reward configuration for strategic Ludo play (self-play tuned)."""

    # Major events (highest priority)
    win: float = 30.0  # Game victory
    lose: float = -30.0  # Game loss
    finish_token: float = 8.0  # Finishing tokens
    capture: float = 6.0  # Capturing opponents
    got_captured: float = -6.0  # Own token captured

    # Action rewards (moderate)
    extra_turn: float = 2.0  # Rolling a 6 (useful but not dominant)

    # Strategic positioning (light shaping)
    home_progress_bonus: float = 3.0
    blocking_bonus: float = 2.0
    safety_bonus: float = 2.0
    home_approach_bonus: float = 0.5

    # Small continuous signals
    progress_scale: float = 0.05
    diversity_bonus: float = 0.25

    # Penalties / pacing
    illegal_action: float = -8.0
    illegal_masked_scale: float = 0.25  # Scale when mask auto-corrects
    time_penalty: float = -0.01

    # Risk modulation (unused currently)
    use_probabilistic_rewards: bool = False
    risk_weight: float = 0.3
    penalize_loss: bool = True


@dataclass
class ObservationConfig:
    include_blocking_count: bool = True
    include_turn_index: bool = True
    include_raw_dice: bool = True
    normalize_positions: bool = True


@dataclass
class EnvConfig:
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    seed: Optional[int] = None
    use_action_mask: bool = (
        True  # When True, invalid chosen actions are auto-corrected without large penalty
    )
