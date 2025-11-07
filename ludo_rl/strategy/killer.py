from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class KillerStrategyConfig(BaseStrategyConfig):
    capture_weight: tuple[float, float] = (8.0, 12.0)
    progress_weight: tuple[float, float] = (0.8, 1.4)
    risk_discount: tuple[float, float] = (1.0, 2.0)
    extra_turn_bonus: tuple[float, float] = (3.0, 5.5)
    safe_bonus: tuple[float, float] = (0.5, 1.5)

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        rng = rng or random
        return {
            "capture_weight": rng.uniform(*self.capture_weight),
            "progress_weight": rng.uniform(*self.progress_weight),
            "risk_discount": rng.uniform(*self.risk_discount),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "safe_bonus": rng.uniform(*self.safe_bonus),
        }


@dataclass(slots=True)
class KillerStrategy(BaseStrategy):
    """Aggressively hunts opponent pieces even at higher risk."""

    name: ClassVar[str] = "killer"
    config: ClassVar[KillerStrategyConfig] = KillerStrategyConfig()

    capture_weight: float = 10.0
    progress_weight: float = 1.0
    risk_discount: float = 1.5
    extra_turn_bonus: float = 4.0
    safe_bonus: float = 1.0

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = move.capture_count * self.capture_weight
        score += move.progress * self.progress_weight
        if move.extra_turn:
            score += self.extra_turn_bonus
        if move.enters_safe_zone:
            score += self.safe_bonus

        # Willing to take risks: subtract less than other strategies
        score -= move.risk * self.risk_discount
        # Encourage pushing pieces that are not already home-bound
        score -= move.distance_to_goal * 0.02

        return score
