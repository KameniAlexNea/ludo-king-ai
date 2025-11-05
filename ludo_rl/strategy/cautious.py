from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class CautiousStrategyConfig(BaseStrategyConfig):
    safe_bonus: tuple[float, float] = (3.0, 6.0)
    blockade_bonus: tuple[float, float] = (2.5, 4.5)
    risk_penalty: tuple[float, float] = (5.0, 7.5)
    leave_safe_penalty: tuple[float, float] = (4.0, 6.5)
    progress_weight: tuple[float, float] = (0.6, 1.2)

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        rng = rng or random
        return {
            "safe_bonus": rng.uniform(*self.safe_bonus),
            "blockade_bonus": rng.uniform(*self.blockade_bonus),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "progress_weight": rng.uniform(*self.progress_weight),
        }


class CautiousStrategy(BaseStrategy):
    """Prioritises safety, safe zones, and avoiding exposure."""

    name = "cautious"
    config: ClassVar[CautiousStrategyConfig] = CautiousStrategyConfig()

    def __init__(
        self,
        safe_bonus: float = 4.0,
        blockade_bonus: float = 3.0,
        risk_penalty: float = 6.0,
        leave_safe_penalty: float = 5.0,
        progress_weight: float = 0.8,
    ) -> None:
        self.safe_bonus = safe_bonus
        self.blockade_bonus = blockade_bonus
        self.risk_penalty = risk_penalty
        self.leave_safe_penalty = leave_safe_penalty
        self.progress_weight = progress_weight

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = move.progress * self.progress_weight
        if move.enters_safe_zone:
            score += self.safe_bonus
        if move.forms_blockade:
            score += self.blockade_bonus
        if move.extra_turn:
            score += 1.0  # small bonus for tempo

        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        # discourage racing ahead when close to opponents
        score -= min(move.distance_to_goal, 20) * 0.05
        return score
