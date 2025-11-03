from __future__ import annotations

from typing import Optional

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class CautiousStrategy(BaseStrategy):
    """Prioritises safety, safe zones, and avoiding exposure."""

    name = "cautious"

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

    def select_move(self, ctx: StrategyContext) -> Optional[MoveOption]:
        best: Optional[MoveOption] = None
        best_score = float("-inf")

        for move in ctx.iter_legal():
            score = self._score_move(move)
            if score > best_score or (
                score == best_score and best and move.piece_id < best.piece_id
            ):
                best = move
                best_score = score

        return best

    def _score_move(self, move: MoveOption) -> float:
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
