from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .enums import Color
from .piece import Piece
from .strategy_base import Strategy, RandomStrategy
from .types import Move


@dataclass(slots=True)
class Player:
    color: int | Color
    strategy: Strategy = field(default_factory=RandomStrategy)
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

    def choose(self, legal_moves: Sequence[Move]) -> Move | None:
        return self.strategy.select_move(legal_moves)
