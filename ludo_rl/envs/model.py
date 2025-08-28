from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ludo.constants import Colors


@dataclass
class RewardConfig:
    """Reward shaping configuration (scaled for stable RL training).

    Magnitudes chosen to avoid sparse domination while preserving signal:
        - Win / lose kept an order of magnitude above per-move signals (10 / -10)
        - Progress shaping small dense signal encourages forward motion
        - Capture / finish moderate bonuses; capture symmetrical with loss
        - Illegal action mildly discouraged (mask should usually prevent it)
    """

    # Primary events
    capture: float = 2.0
    got_captured: float = -2.5
    finish_token: float = 5.0
    win: float = 10.0
    lose: float = -10.0

    # Dense shaping (per total normalized progress delta across 4 tokens)
    progress_scale: float = 0.5

    # Miscellaneous
    time_penalty: float = -0.001
    illegal_action: float = -0.05
    extra_turn: float = 0.3
    blocking_bonus: float = 0.15
    diversity_bonus: float = 0.2  # first time a token leaves home

    # Probabilistic reward scaling
    use_probabilistic_rewards: bool = False
    risk_weight: float = 1.0
    horizon_turns: int = 3
    discount_lambda: float = 0.85


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
