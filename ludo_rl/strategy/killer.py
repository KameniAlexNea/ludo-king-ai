from __future__ import annotations

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class KillerStrategy(BaseStrategy):
    """Aggressively hunts opponent pieces even at higher risk."""

    name = "killer"

    def __init__(
        self,
        capture_weight: float = 10.0,
        progress_weight: float = 1.0,
        risk_discount: float = 1.5,
        extra_turn_bonus: float = 4.0,
        safe_bonus: float = 1.0,
    ) -> None:
        self.capture_weight = capture_weight
        self.progress_weight = progress_weight
        self.risk_discount = risk_discount
        self.extra_turn_bonus = extra_turn_bonus
        self.safe_bonus = safe_bonus

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = move.capture_count * self.capture_weight
        score += move.progress * self.progress_weight
        if move.extra_turn:
            score += self.extra_turn_bonus
        if move.enters_safe_zone:
            score += self.safe_bonus

        # Willing to take risks: subtract less than other strategies
        score -= move.risk * self.risk_discount
        # Encourage pushing pieces that are not already home-bound
        score -= move.distance_to_goal * 0.02

        return score
