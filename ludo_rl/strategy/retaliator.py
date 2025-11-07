from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from .base import BaseStrategy, BaseStrategyConfig
from .features import nearest_opponent_distance, opponent_density_within
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class RetaliatorStrategyConfig(BaseStrategyConfig):
    capture_weight: tuple[float, float] = (7.0, 10.0)
    proximity_radius: tuple[int, int] = (4, 8)
    proximity_weight: tuple[float, float] = (1.0, 2.5)
    density_radius: tuple[int, int] = (2, 5)
    density_weight: tuple[float, float] = (0.4, 0.8)
    risk_penalty: tuple[float, float] = (0.8, 1.4)
    leave_safe_penalty: tuple[float, float] = (0.8, 1.8)
    extra_turn_bonus: tuple[float, float] = (1.5, 3.0)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        return {
            "capture_weight": rng.uniform(*self.capture_weight),
            "proximity_radius": rng.randint(*self.proximity_radius),
            "proximity_weight": rng.uniform(*self.proximity_weight),
            "density_radius": rng.randint(*self.density_radius),
            "density_weight": rng.uniform(*self.density_weight),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
        }


@dataclass(slots=True)
class RetaliatorStrategy(BaseStrategy):
    """Prioritises striking back at nearby opponents while staying battle-ready."""

    name: ClassVar[str] = "retaliator"
    config: ClassVar[RetaliatorStrategyConfig] = RetaliatorStrategyConfig()

    capture_weight: float = 8.0
    proximity_radius: int = 6
    proximity_weight: float = 1.5
    density_radius: int = 3
    density_weight: float = 0.5
    risk_penalty: float = 1.0
    leave_safe_penalty: float = 1.0
    extra_turn_bonus: float = 2.0

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.can_capture:
            score += move.capture_count * self.capture_weight

        distance = nearest_opponent_distance(ctx.board, move.new_pos)
        score += max(0, self.proximity_radius - distance) * self.proximity_weight
        density = opponent_density_within(
            ctx.board, move.new_pos, radius=self.density_radius
        )
        score += density * self.density_weight

        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        if move.extra_turn:
            score += self.extra_turn_bonus

        return score
