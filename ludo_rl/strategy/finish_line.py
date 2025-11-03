from __future__ import annotations

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class FinishLineStrategy(BaseStrategy):
    """Pushes the leading runner across the finish line as quickly as possible."""

    name = "finish_line"

    def __init__(
        self,
        finish_bonus: float = 15.0,
        distance_weight: float = 0.6,
        progress_weight: float = 1.2,
        safe_zone_bonus: float = 4.0,
        capture_weight: float = 2.0,
        risk_penalty: float = 0.8,
        leave_safe_penalty: float = 2.0,
        safe_threshold: int = strategy_config.home_start,
    ) -> None:
        self.finish_bonus = finish_bonus
        self.distance_weight = distance_weight
        self.progress_weight = progress_weight
        self.safe_zone_bonus = safe_zone_bonus
        self.capture_weight = capture_weight
        self.risk_penalty = risk_penalty
        self.leave_safe_penalty = leave_safe_penalty
        self.safe_threshold = safe_threshold

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.enters_home:
            score += self.finish_bonus

        score -= move.distance_to_goal * self.distance_weight
        score += move.progress * self.progress_weight

        if move.enters_safe_zone and move.new_pos >= self.safe_threshold:
            score += self.safe_zone_bonus
        if move.can_capture:
            score += move.capture_count * self.capture_weight

        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        return score
