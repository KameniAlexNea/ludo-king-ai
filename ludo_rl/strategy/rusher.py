from __future__ import annotations

from typing import Optional

from .base import BaseStrategy
from .features import opponent_density_within
from .types import MoveOption, StrategyContext


class RusherStrategy(BaseStrategy):
    """Prioritises raw progress toward home, largely ignoring skirmishes."""

    name = "rusher"

    def __init__(
        self,
        progress_weight: float = 2.0,
        distance_penalty: float = 0.1,
        extra_turn_bonus: float = 3.0,
        risk_penalty: float = 0.5,
        density_radius: int = 2,
        density_penalty: float = 0.2,
    ) -> None:
        self.progress_weight = progress_weight
        self.distance_penalty = distance_penalty
        self.extra_turn_bonus = extra_turn_bonus
        self.risk_penalty = risk_penalty
        self.density_radius = density_radius
        self.density_penalty = density_penalty

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
        score += move.progress * self.progress_weight
        score -= move.distance_to_goal * self.distance_penalty
        if move.extra_turn:
            score += self.extra_turn_bonus
        score -= move.risk * self.risk_penalty

        opp_density = opponent_density_within(
            ctx.board, move.new_pos, radius=self.density_radius
        )
        score -= opp_density * self.density_penalty

        return score
