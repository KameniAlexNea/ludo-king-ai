from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np

from .features import build_move_options
from .types import MoveOption, StrategyContext

if TYPE_CHECKING:
    from ..ludo_king.game import Game


@dataclass(slots=True)
class BaseStrategyConfig:
    """Base configuration contract for sampling strategy parameters."""

    def sample(self, rng: random.Random | None = None) -> dict[str, object]:
        raise NotImplementedError


class BaseStrategy:
    """Base class for heuristic strategies with shared move selection."""

    name = "base"
    config: ClassVar[BaseStrategyConfig | None] = None

    def decide(
        self,
        board_stack: np.ndarray,
        dice_roll: int,
        action_mask: np.ndarray,
        move_choices: list[dict | None],
    ) -> Optional[MoveOption]:
        ctx = build_move_options(board_stack, int(dice_roll), action_mask, move_choices)
        return self.select_move(ctx)

    def update_history(self, game: "Game", mover_index: int, dice_roll: int) -> None:
        """Optional hook for strategies to maintain a per-player history buffer.

        Default: no-op. RL strategies override this to accumulate a 10-step
        token-sequence history compatible with training.
        """
        return None

    def select_move(self, ctx: StrategyContext) -> Optional[MoveOption]:
        scored_moves: list[tuple[MoveOption, float]] = []

        for move in ctx.iter_legal():
            scored_moves.append((move, self._score_move(ctx, move)))

        if not scored_moves:
            return None

        weights = self._softmax([score for _, score in scored_moves])
        population = [move for move, _ in scored_moves]
        return random.choices(population, weights=weights, k=1)[0]

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        if max_score == float("-inf"):
            return [1.0] * len(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total = sum(exps)
        if total <= 0:
            return [1.0] * len(scores)
        return [value / total for value in exps]

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
