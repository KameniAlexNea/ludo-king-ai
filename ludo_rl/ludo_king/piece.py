from dataclasses import dataclass


@dataclass(slots=True)
class Piece:
    """Lightweight piece model. Holds state only.

    Rule logic (legal destinations, finishing, captures) is handled by the
    rules/engine, not the piece. Board-relative/absolute mapping is also
    performed by Board utilities.
    """

    color: int  # enum value 0..3
    piece_id: int  # 0..3 per player
    position: int = 0  # 0 = yard; 1..51 main ring; 52..56 home col; 57 finished

    def move_to(self, new_position: int) -> None:
        self.position = new_position

    def send_home(self) -> None:
        self.position = 0
