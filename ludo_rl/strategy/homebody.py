from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class HomebodyStrategyConfig(BaseStrategyConfig):
    safe_zone_bonus: tuple[float, float] = (3.0, 5.5)
    blockade_bonus: tuple[float, float] = (4.0, 6.5)
    home_bonus: tuple[float, float] = (5.0, 8.0)
    leave_start_penalty: tuple[float, float] = (1.0, 3.5)
    leave_safe_penalty: tuple[float, float] = (3.0, 5.5)
    risk_penalty: tuple[float, float] = (1.0, 2.0)
    distance_penalty: tuple[float, float] = (0.03, 0.08)
    distance_cap: tuple[int, int] = (8, 16)
    near_home_offset: tuple[int, int] = (-8, -4)
    near_home_bonus: tuple[float, float] = (2.0, 3.5)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        offset_low, offset_high = self.near_home_offset
        base = strategy_config.main_track_end
        threshold = base + rng.randint(offset_low, offset_high)
        threshold = max(strategy_config.home_start, min(threshold, base))
        return {
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "blockade_bonus": rng.uniform(*self.blockade_bonus),
            "home_bonus": rng.uniform(*self.home_bonus),
            "leave_start_penalty": rng.uniform(*self.leave_start_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "distance_penalty": rng.uniform(*self.distance_penalty),
            "distance_cap": rng.randint(*self.distance_cap),
            "near_home_threshold": threshold,
            "near_home_bonus": rng.uniform(*self.near_home_bonus),
        }


@dataclass(slots=True)
class HomebodyStrategy(BaseStrategy):
    """Plays defensively, preferring safe zones, blockades, and home progress."""

    name: ClassVar[str] = "homebody"
    config: ClassVar[HomebodyStrategyConfig] = HomebodyStrategyConfig()

    safe_zone_bonus: float = 4.0
    blockade_bonus: float = 5.0
    home_bonus: float = 6.0
    leave_start_penalty: float = 2.0
    leave_safe_penalty: float = 4.0
    risk_penalty: float = 1.5
    distance_penalty: float = 0.05
    distance_cap: int = 10
    near_home_threshold: int = strategy_config.main_track_end - 6
    near_home_bonus: float = 2.5

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        if move.forms_blockade:
            score += self.blockade_bonus
        if move.enters_home:
            score += self.home_bonus
        if move.current_pos == 0 and ctx.dice_roll == 6:
            score -= self.leave_start_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty
        score -= move.risk * self.risk_penalty
        score -= min(move.distance_to_goal, self.distance_cap) * self.distance_penalty

        if move.new_pos >= self.near_home_threshold:
            score += self.near_home_bonus

        return score
