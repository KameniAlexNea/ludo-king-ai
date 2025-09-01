from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ludo.constants import Colors


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
