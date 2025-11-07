from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from .base import BaseStrategy, BaseStrategyConfig
from .features import opponent_density_within
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class RusherStrategyConfig(BaseStrategyConfig):
    progress_weight: tuple[float, float] = (1.8, 2.5)
    distance_penalty: tuple[float, float] = (0.08, 0.15)
    extra_turn_bonus: tuple[float, float] = (2.0, 4.5)
    risk_penalty: tuple[float, float] = (0.3, 0.8)
    density_radius: tuple[int, int] = (1, 3)
    density_penalty: tuple[float, float] = (0.1, 0.3)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        return {
            "progress_weight": rng.uniform(*self.progress_weight),
            "distance_penalty": rng.uniform(*self.distance_penalty),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "density_radius": rng.randint(*self.density_radius),
            "density_penalty": rng.uniform(*self.density_penalty),
        }


@dataclass(slots=True)
class RusherStrategy(BaseStrategy):
    """Prioritises raw progress toward home, largely ignoring skirmishes."""

    name: ClassVar[str] = "rusher"
    config: ClassVar[RusherStrategyConfig] = RusherStrategyConfig()

    progress_weight: float = 2.0
    distance_penalty: float = 0.1
    extra_turn_bonus: float = 3.0
    risk_penalty: float = 0.5
    density_radius: int = 2
    density_penalty: float = 0.2

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        score += move.progress * self.progress_weight
        score -= move.distance_to_goal * self.distance_penalty
        if move.extra_turn:
            score += self.extra_turn_bonus
        score -= move.risk * self.risk_penalty

        opp_density = opponent_density_within(
            ctx.opponent_distribution, move.new_pos, radius=self.density_radius
        )
        score -= opp_density * self.density_penalty

        return score
