from __future__ import annotations

from .base import BaseStrategy
from .features import nearest_opponent_distance, opponent_density_within
from .types import MoveOption, StrategyContext


class HeatSeekerStrategy(BaseStrategy):
    """Moves toward concentrations of opponents to force engagements."""

    name = "heatseeker"

    def __init__(
        self,
        capture_weight: float = 12.0,
        proximity_radius: int = 6,
        proximity_weight: float = 2.0,
        density_radius: int = 4,
        density_weight: float = 0.7,
        progress_weight: float = 0.5,
        risk_penalty: float = 0.6,
    ) -> None:
        self.capture_weight = capture_weight
        self.proximity_radius = proximity_radius
        self.proximity_weight = proximity_weight
        self.density_radius = density_radius
        self.density_weight = density_weight
        self.progress_weight = progress_weight
        self.risk_penalty = risk_penalty

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        if move.can_capture:
            return move.capture_count * self.capture_weight

        distance = nearest_opponent_distance(ctx.board, move.new_pos)
        density = opponent_density_within(
            ctx.board, move.new_pos, radius=self.density_radius
        )

        score = 0.0
        score += max(0, self.proximity_radius - distance) * self.proximity_weight
        score += density * self.density_weight
        score += move.progress * self.progress_weight
        score -= move.risk * self.risk_penalty

        return score
