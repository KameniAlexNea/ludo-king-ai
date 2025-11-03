from __future__ import annotations

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class DefensiveStrategy(BaseStrategy):
    """Keeps pieces sheltered and avoids exposing them to captures."""

    name = "defensive"

    def __init__(
        self,
        enter_home_bonus: float = 6.0,
        safe_zone_bonus: float = 5.0,
        blockade_bonus: float = 3.5,
        extra_turn_bonus: float = 2.0,
        risk_penalty: float = 2.0,
        leave_safe_penalty: float = 6.0,
        progress_weight: float = 0.3,
    ) -> None:
        self.enter_home_bonus = enter_home_bonus
        self.safe_zone_bonus = safe_zone_bonus
        self.blockade_bonus = blockade_bonus
        self.extra_turn_bonus = extra_turn_bonus
        self.risk_penalty = risk_penalty
        self.leave_safe_penalty = leave_safe_penalty
        self.progress_weight = progress_weight

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.enters_home:
            score += self.enter_home_bonus
        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        if move.forms_blockade:
            score += self.blockade_bonus
        if move.extra_turn:
            score += self.extra_turn_bonus

        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty
        # Encourage modest progress so pieces do not stall forever
        score += move.progress * self.progress_weight
        return score
