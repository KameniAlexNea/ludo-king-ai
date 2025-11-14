from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from loguru import logger

from .config import config
from .piece import Piece


def _compute_relative_translations() -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    translations: List[List[Tuple[int, ...]]] = []
    for player_start in config.PLAYER_START_SQUARES:
        player_rows: List[Tuple[int, ...]] = []
        for agent_start in config.PLAYER_START_SQUARES:
            mapping = [-1] * config.PATH_LENGTH
            for rel_pos in range(1, config.MAIN_TRACK_END + 1):
                # absolute 0..51 per piece.to_absolute
                abs_pos = (player_start + rel_pos - 1) % 52
                agent_rel = ((abs_pos - agent_start + 52) % 52) + 1
                mapping[rel_pos] = agent_rel
            player_rows.append(tuple(mapping))
        translations.append(tuple(player_rows))
    return tuple(translations)


_RELATIVE_TRANSLATIONS = _compute_relative_translations()


@dataclass(slots=True)
class Board:
    """Owns piece placement and mapping utilities (no rule logic)."""

    players: Sequence[Sequence[Piece]]  # players (engine order) -> their pieces
    colors: Sequence[int]  # engine order -> color ids (0..3)
    _tensor_buffer: np.ndarray | None = field(default=None, init=False, repr=False)
    # Transition summary tracking (channels 5-10)
    movement_heatmap: np.ndarray = field(default=None, init=False, repr=False)
    my_knockouts: np.ndarray = field(default=None, init=False, repr=False)
    opp_knockouts: np.ndarray = field(default=None, init=False, repr=False)
    new_blockades: np.ndarray = field(default=None, init=False, repr=False)
    blockade_hits: np.ndarray = field(default=None, init=False, repr=False)  # Opponents hit my blockades
    reward_heatmap: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize transition summary arrays."""
        self.movement_heatmap = np.zeros(config.PATH_LENGTH, dtype=np.float32)
        self.my_knockouts = np.zeros(config.PATH_LENGTH, dtype=np.float32)
        self.opp_knockouts = np.zeros(config.PATH_LENGTH, dtype=np.float32)
        self.new_blockades = np.zeros(config.PATH_LENGTH, dtype=np.float32)
        self.blockade_hits = np.zeros(config.PATH_LENGTH, dtype=np.float32)
        self.reward_heatmap = np.zeros(config.PATH_LENGTH, dtype=np.float32)

    def _resolve_index(self, player_color: int) -> int:
        """Map a color id (0..3) to engine index in self.players/colors."""
        for idx, col in enumerate(self.colors):
            if col == player_color:
                return idx
        raise IndexError("Player color not found in Board.colors")

    def absolute_position(self, player_color: int, relative_pos: int) -> int:
        """Map a player's relative position to absolute (0..51) per piece.to_absolute."""
        if not 1 <= relative_pos <= config.MAIN_TRACK_END:
            return -1
        start = config.PLAYER_START_SQUARES[player_color]
        return (start + relative_pos - 1) % 52

    def relative_position(self, player_color: int, abs_pos: int) -> int:
        """Map absolute (0..51) back to player's relative frame (1..52).

        Note: absolute index 0 is not used as a valid ring position in our path
        mapping; return -1 for this value to match expectations in tests.
        """
        if not 1 <= abs_pos <= 51:
            return -1
        start = config.PLAYER_START_SQUARES[player_color]
        rel = (abs_pos - start + 52) % 52
        return rel + 1

    def translate_relative(self, src_color: int, dst_color: int, rel_pos: int) -> int:
        mapping = _RELATIVE_TRANSLATIONS[src_color][dst_color]
        return mapping[rel_pos] if 0 <= rel_pos < len(mapping) else -1

    def pieces_at_absolute(
        self, abs_pos: int, *, exclude_color: int | None = None
    ) -> list[tuple[int, Piece]]:
        out: list[tuple[int, Piece]] = []
        for idx, pieces in enumerate(self.players):
            color_id = self.colors[idx]
            if exclude_color is not None and color_id == exclude_color:
                continue
            r = self.relative_position(color_id, abs_pos)
            if r == -1:
                continue
            for pc in pieces:
                if pc.position == r:
                    out.append((color_id, pc))
        return out

    def count_at_relative(self, player_color: int, rel_pos: int) -> int:
        idx = self._resolve_index(player_color)
        return sum(1 for p in self.players[idx] if p.position == rel_pos)

    def reset_transition_summaries(self) -> None:
        """Reset all transition summary channels to zero."""
        self.movement_heatmap.fill(0.0)
        self.my_knockouts.fill(0.0)
        self.opp_knockouts.fill(0.0)
        self.new_blockades.fill(0.0)
        self.blockade_hits.fill(0.0)
        self.reward_heatmap.fill(0.0)

    # --- Token sequence helpers (for compact observation) ---
    def token_order_for_agent(self, agent_color: int) -> list[int]:
        """Return the color order [agent, next, across, previous] relative to agent.

        Colors are integers 0..3. This defines the 16-token fixed order.
        """
        return [
            agent_color,
            (agent_color + 1) % 4,
            (agent_color + 2) % 4,
            (agent_color + 3) % 4,
        ]

    def token_colors(self, agent_color: int) -> np.ndarray:
        """Return a (16,) array of color ids for tokens in fixed order.

        Order: 4 tokens per color block in token_order_for_agent(agent_color).
        """
        order = self.token_order_for_agent(agent_color)
        cols: list[int] = []
        for c in order:
            cols.extend([c, c, c, c])
        return np.asarray(cols, dtype=np.int64)

    def token_exists_mask(self, agent_color: int) -> np.ndarray:
        """Return a (16,) bool mask for tokens that exist (color/seat present).

        For 2-player games, non-participating colors have no pieces.
        """
        order = self.token_order_for_agent(agent_color)
        mask: list[bool] = []
        for c in order:
            try:
                idx = self._resolve_index(c)
                exists = len(self.players[idx]) > 0
            except IndexError as e:
                logger.warning(
                    f"Failed to resolve color index {c} in token_exists_mask: {e}"
                )
                exists = False
            mask.extend([exists, exists, exists, exists])
        return np.asarray(mask, dtype=np.bool_)

    def all_token_positions(self, agent_color: int) -> np.ndarray:
        """Return positions (0..57) for all 16 tokens in fixed order.

        Uses each piece's native relative position (yard=0, ring 1..51,
        home 52..56, finish 57). Order is 4 tokens per color block as in
        token_order_for_agent(agent_color), sorted by piece_id within color.
        """
        order = self.token_order_for_agent(agent_color)
        positions: list[int] = []
        for c in order:
            try:
                idx = self._resolve_index(c)
            except IndexError as e:
                logger.warning(
                    f"Failed to resolve color index {c} in all_token_positions: {e}"
                )
                # Color not present
                positions.extend([0, 0, 0, 0])
                continue
            pieces = list(self.players[idx])
            pieces.sort(key=lambda p: int(getattr(p, "piece_id", 0)))
            # Ensure exactly 4 outputs per color
            for k in range(4):
                if k < len(pieces):
                    positions.append(int(pieces[k].position))
                else:
                    positions.append(0)
        return np.asarray(positions, dtype=np.int64)

    def build_tensor(
        self, agent_color: int, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Build a (10, PATH_LENGTH) tensor representing the full board state.

        Channels:
        0: My pieces
        1-3: Opponent pieces (relative to agent)
        4: Safe zones (fixed)
        5: Movement heatmap (transitions since last agent turn)
        6: My knockouts (where agent knocked out opponents)
        7: Opponent knockouts (where opponents knocked out agent's pieces)
        8: New blockades formed
        9: Reward heatmap
        """
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

        # Channels 0-3: Piece positions
        for idx, pieces in enumerate(self.players):
            # fixed channel layout my, opp1, opp2, opp3
            color_id = self.colors[idx]
            rel_idx = (color_id - agent_color) % 4
            ch = board[rel_idx]
            for pc in pieces:
                pos = pc.position
                if pos == 0:
                    # Piece in yard
                    ch[0] += 1.0
                elif 1 <= pos <= config.MAIN_TRACK_END:
                    if rel_idx == 0:
                        ch[pos] += 1.0
                    else:
                        translated = self.translate_relative(color_id, agent_color, pos)
                        if translated != -1:
                            ch[translated] += 1.0
                elif config.HOME_COLUMN_START <= pos <= config.HOME_FINISH:
                    # Represent home column (52..56) and finished (57) directly at their indices
                    # for all players. These do not require translation.
                    ch[pos] += 1.0

        # Channel 4: Safe zones (fixed)
        safe = board[4]
        safe[config.HOME_COLUMN_START : config.HOME_FINISH] = 1.0
        for abs_pos in config.SAFE_SQUARES_ABS:
            rel = self.relative_position(agent_color, abs_pos)
            if rel != -1:
                safe[rel] = 1.0

        # Channels 5-9: Transition summaries
        board[5] = self.movement_heatmap
        board[6] = self.my_knockouts
        board[7] = self.opp_knockouts
        board[8] = self.new_blockades
        board[9] = self.blockade_hits
        board[9] = self.reward_heatmap

        return board
