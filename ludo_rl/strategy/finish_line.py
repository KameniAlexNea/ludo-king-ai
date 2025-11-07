from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class FinishLineStrategyConfig(BaseStrategyConfig):
    finish_bonus: tuple[float, float] = (12.0, 18.0)
    distance_weight: tuple[float, float] = (0.4, 0.8)
    progress_weight: tuple[float, float] = (1.0, 1.5)
    safe_zone_bonus: tuple[float, float] = (3.0, 5.5)
    capture_weight: tuple[float, float] = (1.5, 3.5)
    risk_penalty: tuple[float, float] = (0.5, 1.2)
    leave_safe_penalty: tuple[float, float] = (1.0, 3.0)
    safe_threshold: tuple[int, int] = (
        strategy_config.home_start - 4,
        strategy_config.home_start + 4,
    )

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        low, high = self.safe_threshold
        low = max(low, strategy_config.home_start - 6)
        high = min(high, strategy_config.home_finish)
        if high < low:
            high = low
        return {
            "finish_bonus": rng.uniform(*self.finish_bonus),
            "distance_weight": rng.uniform(*self.distance_weight),
            "progress_weight": rng.uniform(*self.progress_weight),
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "capture_weight": rng.uniform(*self.capture_weight),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "safe_threshold": rng.randint(low, high),
        }


@dataclass(slots=True)
class FinishLineStrategy(BaseStrategy):
    """Pushes the leading runner across the finish line as quickly as possible."""

    name: ClassVar[str] = "finish_line"
    config: ClassVar[FinishLineStrategyConfig] = FinishLineStrategyConfig()

    finish_bonus: float = 15.0
    distance_weight: float = 0.6
    progress_weight: float = 1.2
    safe_zone_bonus: float = 4.0
    capture_weight: float = 2.0
    risk_penalty: float = 0.8
    leave_safe_penalty: float = 2.0
    safe_threshold: int = strategy_config.home_start

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.enters_home:
            score += self.finish_bonus

        score -= move.distance_to_goal * self.distance_weight
        score += move.progress * self.progress_weight

        if move.enters_safe_zone and move.new_pos >= self.safe_threshold:
            score += self.safe_zone_bonus
        if move.can_capture:
            score += move.capture_count * self.capture_weight

        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        return score
