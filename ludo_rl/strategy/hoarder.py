from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar, FrozenSet, Iterable, Optional

from ludo_rl.ludo.config import config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class HoarderStrategyConfig(BaseStrategyConfig):
    chokepoint_candidates: tuple[int, ...] = (
        1,
        2,
        3,
        8,
        9,
        config.PATH_LENGTH - 8,
        config.PATH_LENGTH - 6,
        config.PATH_LENGTH - 4,
    )
    chokepoint_count_range: tuple[int, int] = (3, 6)
    blockade_bonus: tuple[float, float] = (4.5, 7.5)
    chokepoint_bonus: tuple[float, float] = (3.0, 6.0)
    safe_zone_bonus: tuple[float, float] = (1.5, 3.5)
    leave_safe_penalty: tuple[float, float] = (4.0, 6.5)
    progress_weight: tuple[float, float] = (0.1, 0.4)
    risk_penalty: tuple[float, float] = (0.6, 1.4)

    def sample(self, rng: random.Random | None = None) -> dict[str, object]:
        rng = rng or random
        pool = list(self.chokepoint_candidates)
        rng.shuffle(pool)
        min_count, max_count = self.chokepoint_count_range
        max_available = max(1, min(len(pool), max_count))
        count = rng.randint(min(min_count, max_available), max_available)
        chokepoints = sorted(pool[:count])
        return {
            "chokepoints": tuple(chokepoints),
            "blockade_bonus": rng.uniform(*self.blockade_bonus),
            "chokepoint_bonus": rng.uniform(*self.chokepoint_bonus),
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "progress_weight": rng.uniform(*self.progress_weight),
            "risk_penalty": rng.uniform(*self.risk_penalty),
        }


DEFAULT_CHOKEPOINTS: tuple[int, ...] = (
    1,
    2,
    3,
    8,
    9,
    config.PATH_LENGTH - 6,
)


@dataclass(slots=True)
class HoarderStrategy(BaseStrategy):
    """Holds chokepoints near the yard and builds blockades to stall opponents."""

    name: ClassVar[str] = "hoarder"
    config: ClassVar[HoarderStrategyConfig] = HoarderStrategyConfig()

    chokepoints: tuple[int, ...] = field(default_factory=lambda: DEFAULT_CHOKEPOINTS)
    blockade_bonus: float = 6.0
    chokepoint_bonus: float = 4.0
    safe_zone_bonus: float = 2.5
    leave_safe_penalty: float = 5.0
    progress_weight: float = 0.2
    risk_penalty: float = 1.0
    _chokepoint_set: FrozenSet[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = tuple(self._normalize_points(self.chokepoints))
        self.chokepoints = normalized
        self._chokepoint_set = frozenset(normalized)

    @staticmethod
    def _normalize_points(chokepoints: Optional[Iterable[int]]) -> Iterable[int]:
        if chokepoints is not None:
            return sorted(chokepoints)
        return DEFAULT_CHOKEPOINTS

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0
        if move.forms_blockade:
            score += self.blockade_bonus
        if move.new_pos in self._chokepoint_set:
            score += self.chokepoint_bonus
        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty
        score += move.progress * self.progress_weight
        score -= move.risk * self.risk_penalty
        return score
