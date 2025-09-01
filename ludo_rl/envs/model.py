from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ludo.constants import Colors


# Reward calculation constants
class SimpleRewardConstants:
    """Constants for reward calculations to avoid magic numbers."""

    # Threat calculation
    MAX_THREAT_DISTANCE = 6
    THREAT_NORMALIZATION_FACTOR = 2.0

    # Progress thresholds
    MIN_PROGRESS_DELTA = 0.01
    SIGNIFICANT_PROGRESS_THRESHOLD = 0.5

    # Threat levels
    LOW_THREAT_THRESHOLD = 0.1
    MEDIUM_THREAT_THRESHOLD = 0.4
    HIGH_THREAT_THRESHOLD = 0.6
    CRITICAL_THREAT_THRESHOLD = 0.7
    URGENT_THREAT_THRESHOLD = 0.8

    # Progress reward scaling
    PROGRESS_REWARD_EXPONENT = 2
    PROGRESS_REWARD_QUADRATIC_SCALE = 1
    PROGRESS_REWARD_CAP_MAX = 5.0
    PROGRESS_REWARD_CAP_MIN = -5.0

    # Safety reward
    SAFETY_MULTIPLIER_BASE = 1.0
    SAFETY_MULTIPLIER_EXPONENT = 1.5
    SAFETY_MULTIPLIER_MAX = 2.0

    # Capture rewards by game phase
    CAPTURE_EARLY = 8.0
    CAPTURE_MID = 12.0
    CAPTURE_LATE = 18.0
    CAPTURE_ENDGAME = 25.0

    # Capture multipliers
    CAPTURE_THREAT_MULTIPLIER_HIGH = 2.0
    CAPTURE_THREAT_MULTIPLIER_MEDIUM = 1.5
    CAPTURE_PROGRESS_EXPONENT = 2

    # Got captured penalties by game phase
    GOT_CAPTURED_EARLY = -3.0
    GOT_CAPTURED_MID = -5.0
    GOT_CAPTURED_LATE = -10.0
    GOT_CAPTURED_ENDGAME = -20.0

    # Home column capture penalty
    HOME_COLUMN_CAPTURE_PENALTY = -15.0
    HOME_COLUMN_CAPTURE_PENALTY_ENDGAME_MULTIPLIER = 3.0

    # Moving rewards by phase
    MOVING_EXIT_HOME_EARLY = 5.0
    MOVING_EXIT_HOME_MID = 3.0
    MOVING_EXIT_HOME_LATE = 2.0
    MOVING_ENTER_HOME_LATE = 8.0
    MOVING_ENTER_HOME_EARLY = 4.0
    MOVING_HOME_PROGRESS_ENDGAME = 2.0
    MOVING_HOME_PROGRESS_EARLY = 1.0
    MOVING_REGULAR = 0.5

    # Moving reward bonuses
    MOVING_SAFE_BONUS = 0.05
    MOVING_PROGRESS_BONUS_SCALE = 0.2
    MOVING_PROGRESS_SIGNIFICANT_MULTIPLIER = 1.5
    MOVING_THREAT_MULTIPLIER = 1.0

    # Finish rewards by phase
    FINISH_EARLY = 20.0
    FINISH_MID = 30.0
    FINISH_LATE = 50.0
    FINISH_ENDGAME = 100.0

    # Finish reward threat multipliers
    FINISH_THREAT_MULTIPLIER_URGENT = 3.0
    FINISH_THREAT_MULTIPLIER_HIGH = 2.0

    # Extra turn threat multiplier
    EXTRA_TURN_THREAT_MULTIPLIER = 1.0

    # Threat level thresholds for rewards
    REWARD_THREAT_THRESHOLD_HIGH = 0.5
    REWARD_THREAT_THRESHOLD_URGENT = 0.8
    REWARD_THREAT_THRESHOLD_MODERATE = 0.6

    # Progress thresholds
    PROGRESS_SIGNIFICANT_THRESHOLD = 0.01
    PROGRESS_SIGNIFICANT_MULTIPLIER = 0.5

    # Opponent threat levels
    OPPONENT_THREAT_FINISHED_1 = 0.5
    OPPONENT_THREAT_HOME_TOKENS_2 = 0.3
    OPPONENT_THREAT_NONE = 0.0

    # Game phase thresholds
    GAME_PHASE_ENDGAME_FINISHED = 2
    GAME_PHASE_LATE_FINISHED = 1
    GAME_PHASE_LATE_HOME_TOKENS = 2

    # Home column scaling
    HOME_PROGRESS_EXPONENT_EARLY = 1.3
    HOME_PROGRESS_EXPONENT_ENDGAME = 1.5
    HOME_PROGRESS_BASE_EARLY = 1.0
    HOME_PROGRESS_BASE_ENDGAME = 2.0

    # Capture bonuses
    HOME_COLUMN_CAPTURE_BASE = 1.5
    HOME_COLUMN_CAPTURE_INCREMENT = 0.3

    # Blocking rewards
    BLOCKING_BASE_REWARD = 2.0
    HOME_ENTRY_BLOCKING_MULTIPLIER = 3.0
    STACK_BLOCKING_MULTIPLIER = 1.5
    BLOCKING_HOME_ENTRY_REWARD = 4.0
    STACKING_REWARD = 1.5
    BLOCKING_ADVANCEMENT_REWARD = 2.0
    BLOCKING_DISTANCE = 1

    # Terminal rewards
    WIN_MARGIN_DOMINANT = 3
    WIN_MARGIN_CLEAR = 2
    LOSS_MARGIN_SHUTOUT = 4
    LOSS_MARGIN_BAD = 3

    WIN_MULTIPLIER_DOMINANT = 2.0
    WIN_MULTIPLIER_CLEAR = 1.5
    LOSS_MULTIPLIER_SHUTOUT = 2.0
    LOSS_MULTIPLIER_BAD = 1.5



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
class OpponentsConfig:
    candidates: List[str] = field(
        default_factory=lambda: [
            "killer",
            "winner",
            "optimist",
            "balanced",
            "defensive",
            "random",
            "cautious",
            "probabilistic",
            "probabilistic_v2",
            "probabilistic_v3",
        ]
    )


@dataclass
class EnvConfig:
    agent_color: str = Colors.RED
    max_turns: int = 1000
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_cfg: ObservationConfig = field(default_factory=ObservationConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    seed: Optional[int] = None
