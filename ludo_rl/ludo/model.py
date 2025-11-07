from dataclasses import dataclass

from .config import config


@dataclass(slots=True)
class Piece:
    color: int  # 0, 1, 2, or 3
    piece_id: int  # 0, 1, 2, or 3
    position: int = 0  # 0 = Yard

    def is_safe(self, abs_pos):
        """Checks if an absolute board position is a safe 'star' square."""
        return abs_pos in config.SAFE_SQUARES_ABS

    def is_finished(self):
        return self.position == 57
