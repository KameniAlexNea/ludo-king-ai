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
    """Balanced reward configuration for strategic Ludo play."""

    # Major events (highest priority - should dominate)
    win: float = 50.0  # Game victory
    lose: float = -50.0  # Game loss
    finish_token: float = 15.0  # Finishing tokens (very important)
    capture: float = 8.0  # Capturing opponents (important)
    got_captured: float = -5.0  # Getting captured (bad but recoverable)

    # Action rewards (moderate priority)
    extra_turn: float = 3.0  # Rolling a 6 (useful)

    # Strategic positioning (should encourage good play but not dominate)
    home_progress_bonus: float = 4.0  # Progress in home stretch
    blocking_bonus: float = 2.0  # Creating blocks
    safety_bonus: float = 3.0  # Escaping danger
    home_approach_bonus: float = 1.0  # Approaching home entry

    # Small continuous signals (should barely register)
    progress_scale: float = 0.1  # General progress (minimal)
    diversity_bonus: float = 0.3  # Token diversity (minimal)

    # Penalties (should be clear boundaries)
    illegal_action: float = -10.0  # Invalid moves (strong negative)
    time_penalty: float = -0.01  # Efficiency (tiny)

    # Risk modulation (if you want it)
    use_probabilistic_rewards: bool = False  # Start simple, add complexity later
    risk_weight: float = 0.3  # Conservative if enabled


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
    use_action_mask: bool = True  # When True, invalid chosen actions are auto-corrected without large penalty
