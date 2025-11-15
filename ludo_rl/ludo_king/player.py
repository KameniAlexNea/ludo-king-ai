from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # avoid runtime imports to prevent circular deps
    from ludo_rl.strategy.base import BaseStrategy

from .piece import Piece
from .types import Color, Move


@dataclass(slots=True)
class Player:
    color: int | Color
    strategy: Optional["BaseStrategy"] = field(default=None)
    strategy_name: str = "unknown"
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

    def choose(
        self,
        board_stack: np.ndarray,
        dice_roll: int,
        legal_moves: Sequence[Move],
    ) -> Move | None:
        """Delegate move selection to attached strategy.

        Expected signature for strategies in this package is compatible with the
        legacy decide(board, dice, moves) contract. For flexibility, we also
        support simple strategies that accept only (moves).
        """
        # Ensure a strategy instance exists; do not replace existing strategy based on name
        try:
            if not self.strategy:
                from ludo_rl.strategy.registry import create as create_strategy

                self.strategy = create_strategy(self.strategy_name)
        except KeyError as e:
            logger.warning(
                f"Unknown strategy '{self.strategy_name}', falling back to random: {e}"
            )
            # Unknown strategy: fallback to random legal move and mark as random
            self.strategy_name = "random"
            return next(iter(legal_moves), None) if legal_moves else None

        # Build move_choices and action_mask per legacy extractor expectations
        pieces_per_player = len(self.pieces)
        move_choices: list[dict | None] = [None] * pieces_per_player
        action_mask = np.zeros(pieces_per_player, dtype=bool)
        for mv in legal_moves:
            if (
                0 <= mv.piece_id < pieces_per_player
                and move_choices[mv.piece_id] is None
            ):
                action_mask[mv.piece_id] = True
                move_choices[mv.piece_id] = {
                    "piece": self.pieces[mv.piece_id],
                    "new_pos": mv.new_pos,
                    "move": mv,
                }
        decided = self.strategy.decide(
            board_stack, int(dice_roll), action_mask, move_choices
        )
        if decided is None:
            return None
        # Map back to our Move by piece_id
        return move_choices[decided.piece_id]["move"]
