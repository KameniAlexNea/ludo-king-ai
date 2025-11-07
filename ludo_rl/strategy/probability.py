from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class ProbabilityStrategyConfig(BaseStrategyConfig):
    progress_weight: tuple[float, float] = (1.2, 1.8)
    capture_weight: tuple[float, float] = (5.0, 8.0)
    risk_weight: tuple[float, float] = (3.0, 5.0)
    safe_bonus: tuple[float, float] = (1.5, 3.0)
    extra_turn_bonus: tuple[float, float] = (2.0, 4.0)

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        rng = rng or random
        return {
            "progress_weight": rng.uniform(*self.progress_weight),
            "capture_weight": rng.uniform(*self.capture_weight),
            "risk_weight": rng.uniform(*self.risk_weight),
            "safe_bonus": rng.uniform(*self.safe_bonus),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
        }


@dataclass(slots=True)
class ProbabilityStrategy(BaseStrategy):
    """Balances progress with estimated knockout risk."""

    name: ClassVar[str] = "probability"
    config: ClassVar[ProbabilityStrategyConfig] = ProbabilityStrategyConfig()

    progress_weight: float = 1.5
    capture_weight: float = 6.0
    risk_weight: float = 4.0
    safe_bonus: float = 2.0
    extra_turn_bonus: float = 3.0

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = move.progress * self.progress_weight
        if move.can_capture:
            score += move.capture_count * self.capture_weight
        if move.enters_safe_zone:
            score += self.safe_bonus
        if move.extra_turn:
            score += self.extra_turn_bonus
        score -= move.risk * self.risk_weight
        score -= move.leaving_safe_zone * (self.safe_bonus * 0.5)
        score -= move.distance_to_goal * 0.05
        return score
