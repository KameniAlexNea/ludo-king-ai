from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar, Optional

from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class BaseStrategyConfig:
    """Base configuration contract for sampling strategy parameters."""

    def sample(self, rng: random.Random | None = None) -> dict[str, object]:
        raise NotImplementedError


class BaseStrategy:
    """Base class for heuristic strategies with shared move selection."""

    name = "base"
    config: ClassVar[BaseStrategyConfig | None] = None

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

    def _score_move(
        self, ctx: StrategyContext, move: MoveOption
    ) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def create_instance(cls, rng: random.Random | None = None) -> "BaseStrategy":
        if cls.config is None:
            raise NotImplementedError(
                f"{cls.__name__} does not define a configuration for sampling."
            )
        params = cls.config.sample(rng)
        return cls(**params)
