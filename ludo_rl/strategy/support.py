from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class SupportStrategyConfig(BaseStrategyConfig):
    yard_bonus: tuple[float, float] = (3.0, 5.5)
    lag_weight: tuple[float, float] = (0.2, 0.5)
    lead_penalty: tuple[float, float] = (0.15, 0.35)
    safe_zone_bonus: tuple[float, float] = (1.0, 2.5)
    risk_penalty: tuple[float, float] = (0.6, 1.2)

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        rng = rng or random
        return {
            "yard_bonus": rng.uniform(*self.yard_bonus),
            "lag_weight": rng.uniform(*self.lag_weight),
            "lead_penalty": rng.uniform(*self.lead_penalty),
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "risk_penalty": rng.uniform(*self.risk_penalty),
        }


class SupportStrategy(BaseStrategy):
    """Keeps the team's pieces evenly developed instead of racing a single runner."""

    name = "support"
    config: ClassVar[SupportStrategyConfig] = SupportStrategyConfig()

    def __init__(
        self,
        yard_bonus: float = 4.0,
        lag_weight: float = 0.3,
        lead_penalty: float = 0.2,
        safe_zone_bonus: float = 1.5,
        risk_penalty: float = 0.8,
    ) -> None:
        self.yard_bonus = yard_bonus
        self.lag_weight = lag_weight
        self.lead_penalty = lead_penalty
        self.safe_zone_bonus = safe_zone_bonus
        self.risk_penalty = risk_penalty

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        counts = ctx.my_distribution
        positions = np.arange(counts.shape[0])
        total_pieces = counts.sum()
        mean_progress = (
            float(np.dot(positions, counts) / total_pieces) if total_pieces else 0.0
        )

        score = 0.0
        if move.current_pos == 0 and ctx.dice_roll == 6:
            score += self.yard_bonus
        lag_bonus = max(0.0, mean_progress - move.current_pos)
        score += lag_bonus * self.lag_weight

        lead_distance = max(0.0, move.new_pos - mean_progress)
        score -= lead_distance * self.lead_penalty

        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        score -= move.risk * self.risk_penalty

        return score
