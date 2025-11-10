from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any
import inspect
import numpy as np

from .enums import Color
from .piece import Piece
from .types import Move


@dataclass(slots=True)
class Player:
    color: int | Color
    strategy: object | None = None
    pieces: list[Piece] = field(init=False)
    has_finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        color_val = int(self.color)
        self.pieces = [Piece(color=color_val, piece_id=i) for i in range(4)]

    def active_positions(self) -> list[int]:
        return [p.position for p in self.pieces]

    def check_won(self) -> bool:
        if self.has_finished:
            return True
        if all(p.position == 57 for p in self.pieces):
            self.has_finished = True
            return True
        return False

    def choose(
        self,
        board_stack: np.ndarray,
        dice_roll: int,
        legal_moves: Sequence[Move],
    ) -> Move | None:
        """Delegate move selection to attached strategy.

        Expected signature for strategies in this package is compatible with the
        legacy decide(board, dice, moves) contract. For flexibility, we also
        support simple strategies that accept only (moves).
        """
        if not self.strategy:
            return None

        # Prefer a 'decide' method if present (board, dice, moves)
        decide = getattr(self.strategy, "decide", None)
        if callable(decide):
            try:
                return decide(board_stack, dice_roll, legal_moves)
            except TypeError:
                pass

        # Fallback to 'select_move' with flexible arity
        sel = getattr(self.strategy, "select_move", None)
        if callable(sel):
            try:
                sig = inspect.signature(sel)
                params = len(sig.parameters)
            except Exception:
                params = 3

            if params >= 3:
                try:
                    return sel(board_stack, dice_roll, legal_moves)
                except TypeError:
                    pass
            try:
                return sel(legal_moves)
            except TypeError:
                return None

        return None
