from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ludo_rl.ludo.config import config

from .base import BaseStrategy
from .types import MoveOption, StrategyContext


class HoarderStrategy(BaseStrategy):
    """Holds chokepoints near the yard and builds blockades to stall opponents."""

    name = "hoarder"

    def __init__(
        self,
        chokepoints: Optional[Sequence[int]] = None,
        blockade_bonus: float = 6.0,
        chokepoint_bonus: float = 4.0,
        safe_zone_bonus: float = 2.5,
        leave_safe_penalty: float = 5.0,
        progress_weight: float = 0.2,
        risk_penalty: float = 1.0,
    ) -> None:
        self.chokepoints = set(self._normalize_points(chokepoints))
        self.blockade_bonus = blockade_bonus
        self.chokepoint_bonus = chokepoint_bonus
        self.safe_zone_bonus = safe_zone_bonus
        self.leave_safe_penalty = leave_safe_penalty
        self.progress_weight = progress_weight
        self.risk_penalty = risk_penalty

    @staticmethod
    def _normalize_points(chokepoints: Optional[Sequence[int]]) -> Iterable[int]:
        if chokepoints is not None:
            return chokepoints
        default = {1, 2, 3, 8, 9, config.PATH_LENGTH - 6}
        return default

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.forms_blockade:
            score += self.blockade_bonus
        if move.new_pos in self.chokepoints:
            score += self.chokepoint_bonus
        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty
        score += move.progress * self.progress_weight
        score -= move.risk * self.risk_penalty
        return score
