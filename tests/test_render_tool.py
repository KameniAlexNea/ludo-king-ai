from typing import Dict, List

from PIL import Image

from ludo_interface.board_viz import BOARD_SIZE
from ludo_rl.ludo_king.render import render_from_pieces
from ludo_rl.ludo_king.types import Color


class FakePiece:
    def __init__(self, piece_id: int, position: int):
        self.piece_id = piece_id
        self.position = position


def test_render_from_pieces_returns_image():
    pieces: Dict[Color, List[FakePiece]] = {
        Color.RED: [FakePiece(0, 0), FakePiece(1, 1)],
        Color.GREEN: [FakePiece(0, 10)],
        Color.YELLOW: [FakePiece(2, 52)],  # home column range
        Color.BLUE: [FakePiece(3, 57)],  # finished
    }

    img = render_from_pieces(pieces, show_ids=False)
    assert isinstance(img, Image.Image)
    assert img.size == (BOARD_SIZE, BOARD_SIZE)
