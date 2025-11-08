from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np

from .config import config

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .piece import Piece
    from .player import Player


def _compute_safe_relative_positions() -> Tuple[Tuple[int, ...], ...]:
    safe_positions: List[Tuple[int, ...]] = []
    for agent_start in config.PLAYER_START_SQUARES:
        relative_indices = {
            ((abs_pos - agent_start + 52) % 52) + 1
            for abs_pos in config.SAFE_SQUARES_ABS
        }
        safe_positions.append(tuple(sorted(relative_indices)))
    return tuple(safe_positions)


def _compute_relative_translations() -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    translations: List[List[Tuple[int, ...]]] = []
    for player_start in config.PLAYER_START_SQUARES:
        player_rows: List[Tuple[int, ...]] = []
        for agent_start in config.PLAYER_START_SQUARES:
            mapping = [-1] * config.PATH_LENGTH
            for rel_pos in range(1, 52):
                abs_pos = ((player_start + rel_pos - 2 + 52) % 52) + 1
                agent_rel = ((abs_pos - agent_start + 52) % 52) + 1
                mapping[rel_pos] = agent_rel
            player_rows.append(tuple(mapping))
        translations.append(tuple(player_rows))
    return tuple(translations)


_SAFE_RELATIVE_POSITIONS = _compute_safe_relative_positions()
_RELATIVE_TRANSLATIONS = _compute_relative_translations()


@dataclass(slots=True)
class LudoBoard:
    """Container responsible for board geometry and player occupancy."""

    players: Sequence["Player"]
    _tensor_buffer: np.ndarray | None = field(default=None, init=False, repr=False)

    def is_safe_square(self, abs_pos: int) -> bool:
        return abs_pos in config.SAFE_SQUARES_ABS

    def absolute_position(self, player_index: int, relative_pos: int) -> int:
        """Map a player's relative position to the absolute track coordinate."""
        if not 1 <= relative_pos <= 51:
            return -1
        start = self.players[player_index].start_square
        return ((start + relative_pos - 2 + 52) % 52) + 1

    def relative_position(self, player_index: int, abs_pos: int) -> int:
        """Map an absolute coordinate back into the player's relative frame."""
        if not 1 <= abs_pos <= 52:
            return -1
        start = self.players[player_index].start_square
        return ((abs_pos - start + 52) % 52) + 1

    def count_player_pieces(self, player_index: int, relative_pos: int) -> int:
        return sum(
            1
            for piece in self.players[player_index].pieces
            if piece.position == relative_pos
        )

    def pieces_at_absolute(
        self, abs_pos: int, *, exclude_player: int | None = None
    ) -> List[tuple[int, "Piece"]]:
        """Return all pieces occupying a given absolute square."""
        occupants: List[tuple[int, "Piece"]] = []
        for idx, player in enumerate(self.players):
            if exclude_player is not None and idx == exclude_player:
                continue
            rel_pos = self.relative_position(idx, abs_pos)
            if rel_pos == -1:
                continue
            for piece in player.pieces:
                if piece.position == rel_pos:
                    occupants.append((idx, piece))
        return occupants

    def build_tensor(
        self, agent_index: int, out: np.ndarray | None = None
    ) -> np.ndarray:
        if out is not None:
            board = out
            if board.shape != (10, config.PATH_LENGTH):
                raise ValueError("Expected board tensor of shape (10, PATH_LENGTH)")
        else:
            if self._tensor_buffer is None:
                self._tensor_buffer = np.zeros(
                    (10, config.PATH_LENGTH), dtype=np.float32
                )
            board = self._tensor_buffer

        board.fill(0.0)

        safe_channel = board[4]
        safe_channel[52:57] = 1.0
        for rel_pos in _SAFE_RELATIVE_POSITIONS[agent_index]:
            safe_channel[rel_pos] = 1.0

        for player_index, player in enumerate(self.players):
            relative_channel_index = (
                player_index - agent_index + config.NUM_PLAYERS
            ) % config.NUM_PLAYERS
            channel = board[relative_channel_index]

            if relative_channel_index == 0:
                for piece in player.pieces:
                    channel[piece.position] += 1.0
                continue

            mapping = _RELATIVE_TRANSLATIONS[player_index][agent_index]
            for piece in player.pieces:
                pos = piece.position
                if pos == 0:
                    channel[0] += 1.0
                elif pos > 51:
                    continue
                else:
                    target = mapping[pos]
                    if target != -1:
                        channel[target] += 1.0

        return board

    def safe_relative_positions(self, agent_index: int) -> Tuple[int, ...]:
        return _SAFE_RELATIVE_POSITIONS[agent_index]

    def relative_translations(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        return _RELATIVE_TRANSLATIONS
