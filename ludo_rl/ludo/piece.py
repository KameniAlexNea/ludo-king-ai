from dataclasses import dataclass

from .config import config

MAIN_TRACK_END = 51
HOME_COLUMN_START = 52
HOME_FINISH = 57
ENTRY_POSITION = 1


@dataclass(slots=True)
class Piece:
    color: int  # 0, 1, 2, or 3
    piece_id: int  # 0, 1, 2, or 3
    position: int = 0  # 0 = Yard

    def is_safe(self, abs_pos: int) -> bool:
        """Checks if an absolute board position is a safe 'star' square."""
        return abs_pos in config.SAFE_SQUARES_ABS

    def is_finished(self) -> bool:
        return self.position == HOME_FINISH

    def in_yard(self) -> bool:
        return self.position == 0

    def in_home_column(self) -> bool:
        return HOME_COLUMN_START <= self.position <= HOME_FINISH - 1

    def destination_for_roll(self, dice_roll: int) -> int | None:
        """Return the relative destination for a given dice roll, if legal."""
        if self.in_yard():
            return ENTRY_POSITION if dice_roll == 6 else None

        if self.is_finished():
            return None

        if self.in_home_column():
            candidate = self.position + dice_roll
            return candidate if candidate <= HOME_FINISH else None

        candidate = self.position + dice_roll
        if candidate > MAIN_TRACK_END:
            overflow = candidate - MAIN_TRACK_END
            if overflow > 6:
                return None
            return HOME_COLUMN_START + overflow - 1

        return candidate

    def move_to(self, new_position: int) -> None:
        self.position = new_position

    def send_home(self) -> None:
        self.position = 0

    def to_absolute(self) -> int:
        """
        Convert the piece's relative position to an absolute board position.
        Used by interface only
        """
        if self.in_yard():
            return -1
        if self.in_home_column() or self.is_finished():
            return self.position

        start_square = config.PLAYER_START_SQUARES[self.color]
        abs_pos = ((start_square + self.position - 2 + 52) % 52) + 1
        return abs_pos
