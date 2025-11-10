from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .config import config
from .piece import Piece


def _compute_relative_translations() -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    translations: List[List[Tuple[int, ...]]] = []
    for player_start in config.PLAYER_START_SQUARES:
        player_rows: List[Tuple[int, ...]] = []
        for agent_start in config.PLAYER_START_SQUARES:
            mapping = [-1] * config.PATH_LENGTH
            for rel_pos in range(1, config.MAIN_TRACK_END + 1):
                abs_pos = ((player_start + rel_pos - 2 + 52) % 52) + 1
                agent_rel = ((abs_pos - agent_start + 52) % 52) + 1
                mapping[rel_pos] = agent_rel
            player_rows.append(tuple(mapping))
        translations.append(tuple(player_rows))
    return tuple(translations)


_RELATIVE_TRANSLATIONS = _compute_relative_translations()


@dataclass(slots=True)
class Board:
    """Owns piece placement and mapping utilities (no rule logic)."""

    players: Sequence[Sequence[Piece]]  # players -> their pieces
    _tensor_buffer: np.ndarray | None = field(default=None, init=False, repr=False)

    def absolute_position(self, player_color: int, relative_pos: int) -> int:
        if not 1 <= relative_pos <= config.MAIN_TRACK_END:
            return -1
        start = config.PLAYER_START_SQUARES[player_color]
        return ((start + relative_pos - 2 + 52) % 52) + 1

    def relative_position(self, player_color: int, abs_pos: int) -> int:
        if not 1 <= abs_pos <= 52:
            return -1
        start = config.PLAYER_START_SQUARES[player_color]
        return ((abs_pos - start + 52) % 52) + 1

    def translate_relative(self, src_color: int, dst_color: int, rel_pos: int) -> int:
        mapping = _RELATIVE_TRANSLATIONS[src_color][dst_color]
        return mapping[rel_pos] if 0 <= rel_pos < len(mapping) else -1

    def pieces_at_absolute(self, abs_pos: int, *, exclude_color: int | None = None) -> list[tuple[int, Piece]]:
        out: list[tuple[int, Piece]] = []
        for color, pieces in enumerate(self.players):
            if exclude_color is not None and color == exclude_color:
                continue
            r = self.relative_position(color, abs_pos)
            if r == -1:
                continue
            for pc in pieces:
                if pc.position == r:
                    out.append((color, pc))
        return out

    def count_at_relative(self, player_color: int, rel_pos: int) -> int:
        return sum(1 for p in self.players[player_color] if p.position == rel_pos)

    def build_tensor(self, agent_color: int, out: np.ndarray | None = None) -> np.ndarray:
        if out is not None:
            board = out
            if board.shape != (10, config.PATH_LENGTH):
                raise ValueError("Expected board tensor of shape (10, PATH_LENGTH)")
        else:
            if self._tensor_buffer is None:
                self._tensor_buffer = np.zeros((10, config.PATH_LENGTH), dtype=np.float32)
            board = self._tensor_buffer

        board.fill(0.0)

        # safe channel
        safe = board[4]
        safe[config.HOME_COLUMN_START:config.HOME_FINISH] = 1.0
        for abs_pos in config.SAFE_SQUARES_ABS:
            rel = self.relative_position(agent_color, abs_pos)
            if rel != -1:
                safe[rel] = 1.0

        # my + opponents
        for color, pieces in enumerate(self.players):
            # fixed channel layout my, opp1, opp2, opp3
            rel_idx = (color - agent_color) % 4
            ch = board[rel_idx]
            for pc in pieces:
                pos = pc.position
                if pos == 0:
                    ch[0] += 1.0
                elif pos <= config.MAIN_TRACK_END:
                    if rel_idx == 0:
                        ch[pos] += 1.0
                    else:
                        translated = self.translate_relative(color, agent_color, pos)
                        if translated != -1:
                            ch[translated] += 1.0
                # home column and finished not represented on ring channels here
        return board
