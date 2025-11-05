from __future__ import annotations

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class HomebodyStrategy(BaseStrategy):
    """Plays defensively, preferring safe zones, blockades, and home progress."""

    name = "homebody"

    def __init__(
        self,
        safe_zone_bonus: float = 4.0,
        blockade_bonus: float = 5.0,
        home_bonus: float = 6.0,
        leave_start_penalty: float = 2.0,
        leave_safe_penalty: float = 4.0,
        risk_penalty: float = 1.5,
        distance_penalty: float = 0.05,
        distance_cap: int = 10,
        near_home_threshold: int = strategy_config.main_track_end - 6,
        near_home_bonus: float = 2.5,
    ) -> None:
        self.safe_zone_bonus = safe_zone_bonus
        self.blockade_bonus = blockade_bonus
        self.home_bonus = home_bonus
        self.leave_start_penalty = leave_start_penalty
        self.leave_safe_penalty = leave_safe_penalty
        self.risk_penalty = risk_penalty
        self.distance_penalty = distance_penalty
        self.distance_cap = distance_cap
        self.near_home_threshold = near_home_threshold
        self.near_home_bonus = near_home_bonus

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
