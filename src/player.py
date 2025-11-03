from dataclasses import dataclass, field
from typing import Optional

from src.config import config


@dataclass
class Piece:
    color: int  # 0, 1, 2, or 3
    piece_id: int  # 0, 1, 2, or 3
    position: int = 0  # 0 = Yard

    def is_safe(self, abs_pos):
        """Checks if an absolute board position is a safe 'star' square."""
        return abs_pos in config.SAFE_SQUARES_ABS

    def is_finished(self):
        return self.position == 57


@dataclass
class Player:
    color: int
    start_square: Optional[int] = None
    pieces: Optional[list[Piece]] = field(default_factory=lambda: [])
    has_finished: Optional[bool] = False

    def __post_init__(self):
        self.start_square = config.PLAYER_START_SQUARES[self.color]
        self.pieces = [Piece(self.color, i) for i in range(config.PIECES_PER_PLAYER)]

    def has_won(self):
        """Checks if all pieces are at the home position."""
        if self.has_finished:  # Don't re-check if already done
            return True

        if all(p.is_finished() for p in self.pieces):
            self.has_finished = True
            return True
        return False
