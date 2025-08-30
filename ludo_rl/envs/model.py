from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ludo.constants import Colors


@dataclass
class RewardConfig:
    """Simple, effective reward configuration for stable RL training.

    Clear signal differentiation with meaningful magnitudes:
        - Event rewards (5-10) >> step rewards (0.05-0.5)
        - No systematic bias against actions
        - Fast learning through clear gradients
    """

    # Primary events - clear, meaningful signals
    capture: float = 5.0          # Significant positive for captures
    got_captured: float = -3.0    # Significant negative for getting captured
    finish_token: float = 10.0    # Big positive for finishing tokens
    win: float = 50.0             # Huge positive for winning
    lose: float = -50.0           # Huge negative for losing

    # Dense shaping - scaled up to matter
    progress_scale: float = 5.0   # 10x larger than before

    # Remove training killers
    time_penalty: float = 0.0     # REMOVE - was killing learning
    illegal_action: float = -0.1  # Tiny penalty
    extra_turn: float = 1.0       # Small bonus
    blocking_bonus: float = 0.15  # Keep small
    diversity_bonus: float = 2.0  # Moderate bonus for exploration

    # Disable complex probabilistic system (was preventing learning)
    use_probabilistic_rewards: bool = False  # Disable for clear signals
    risk_weight: float = 1.0
    opportunity_weight: float = 0.8
    horizon_turns: int = 3
    discount_lambda: float = 0.85
    opportunity_bonus_scale: float = 0.3
    finishing_probability_weight: float = 0.6


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
