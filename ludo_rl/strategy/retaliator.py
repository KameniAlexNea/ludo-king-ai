from __future__ import annotations

from typing import Optional

from .base import BaseStrategy
from .features import nearest_opponent_distance, opponent_density_within
from .types import MoveOption, StrategyContext


class RetaliatorStrategy(BaseStrategy):
    """Prioritises striking back at nearby opponents while staying battle-ready."""

    name = "retaliator"

    def __init__(
        self,
        capture_weight: float = 8.0,
        proximity_radius: int = 6,
        proximity_weight: float = 1.5,
        density_radius: int = 3,
        density_weight: float = 0.5,
        risk_penalty: float = 1.0,
        leave_safe_penalty: float = 1.0,
        extra_turn_bonus: float = 2.0,
    ) -> None:
        self.capture_weight = capture_weight
        self.proximity_radius = proximity_radius
        self.proximity_weight = proximity_weight
        self.density_radius = density_radius
        self.density_weight = density_weight
        self.risk_penalty = risk_penalty
        self.leave_safe_penalty = leave_safe_penalty
        self.extra_turn_bonus = extra_turn_bonus

    def select_move(self, ctx: StrategyContext) -> Optional[MoveOption]:
        best: Optional[MoveOption] = None
        best_score = float("-inf")

        for move in ctx.iter_legal():
            score = self._score_move(ctx, move)
            if score > best_score or (
                score == best_score and best and move.piece_id < best.piece_id
            ):
                best = move
                best_score = score

        return best

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
