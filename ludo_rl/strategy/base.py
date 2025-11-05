from typing import Optional

from .types import MoveOption, StrategyContext


class BaseStrategy:
    """Base class for heuristic strategies with shared move selection."""

    name = "base"

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
