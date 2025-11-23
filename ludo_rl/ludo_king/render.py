from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List

from PIL import Image

from ludo_interface.board_viz import draw_board
from ludo_interface.models import PlayerColor

if TYPE_CHECKING:
    from .game import Game
    from .piece import Piece
    from ..ludo_env import LudoEnv

from .types import Color

_COLOR_MAP: Dict[Color, PlayerColor] = {
    Color.RED: PlayerColor.RED,
    Color.GREEN: PlayerColor.GREEN,
    Color.YELLOW: PlayerColor.YELLOW,
    Color.BLUE: PlayerColor.BLUE,
}


def _as_viz_payload(
    pieces_by_color: Dict[Color, Iterable["Piece"]],
) -> Dict[PlayerColor, List["Piece"]]:
    payload: Dict[PlayerColor, List["Piece"]] = {
        PlayerColor.RED: [],
        PlayerColor.GREEN: [],
        PlayerColor.YELLOW: [],
        PlayerColor.BLUE: [],
    }
    for c, plist in pieces_by_color.items():
        payload[_COLOR_MAP[c]].extend(list(plist))
    return payload


def render_from_pieces(
    pieces_by_color: Dict[Color, Iterable["Piece"]], show_ids: bool = True
):
    """Render a Ludo board image with provided pieces placed.

    - pieces_by_color: mapping from engine Color -> iterable of Piece
    - returns: PIL.Image (RGB)
    """
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow not installed. Please `pip install pillow`.")
    tokens = _as_viz_payload(pieces_by_color)
    return draw_board(tokens, show_ids=show_ids)


def render_from_game(game: "Game", show_ids: bool = True):
    """Render from a Game instance."""
    pieces_by_color: Dict[Color, List["Piece"]] = {}
    for pl in game.players:
        # pl.color is Color, pl.pieces is List[Piece]
        pieces_by_color[pl.color] = list(pl.pieces)
    return render_from_pieces(pieces_by_color, show_ids=show_ids)


def render_from_env(env: "LudoEnv", show_ids: bool = True):
    """Render from a LudoEnv instance (no import cycle by type-hint)."""
    if getattr(env, "game", None) is None:
        raise ValueError("Environment has no active game to render.")
    return render_from_game(env.game, show_ids=show_ids)
