from __future__ import annotations

import numpy as np
from typing import Optional

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class SupportStrategy(BaseStrategy):
    """Keeps the team's pieces evenly developed instead of racing a single runner."""

    name = "support"

    def __init__(
        self,
        yard_bonus: float = 4.0,
        lag_weight: float = 0.3,
        lead_penalty: float = 0.2,
        safe_zone_bonus: float = 1.5,
        risk_penalty: float = 0.8,
    ) -> None:
        self.yard_bonus = yard_bonus
        self.lag_weight = lag_weight
        self.lead_penalty = lead_penalty
        self.safe_zone_bonus = safe_zone_bonus
        self.risk_penalty = risk_penalty

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
        counts = ctx.my_distribution
        positions = np.arange(counts.shape[0])
        total_pieces = counts.sum()
        mean_progress = (
            float(np.dot(positions, counts) / total_pieces) if total_pieces else 0.0
        )

        score = 0.0
        if move.current_pos == 0 and ctx.dice_roll == 6:
            score += self.yard_bonus
        lag_bonus = max(0.0, mean_progress - move.current_pos)
        score += lag_bonus * self.lag_weight

        lead_distance = max(0.0, move.new_pos - mean_progress)
        score -= lead_distance * self.lead_penalty

        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        score -= move.risk * self.risk_penalty

        return score
