from __future__ import annotations

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class ProbabilityStrategy(BaseStrategy):
    """Balances progress with estimated knockout risk."""

    name = "probability"

    def __init__(
        self,
        progress_weight: float = 1.5,
        capture_weight: float = 6.0,
        risk_weight: float = 4.0,
        safe_bonus: float = 2.0,
        extra_turn_bonus: float = 3.0,
    ) -> None:
        self.progress_weight = progress_weight
        self.capture_weight = capture_weight
        self.risk_weight = risk_weight
        self.safe_bonus = safe_bonus
        self.extra_turn_bonus = extra_turn_bonus

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
