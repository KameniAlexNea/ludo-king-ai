from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any
import inspect
import numpy as np

from ludo_rl.strategy import BaseStrategy, HumanStrategy, build_move_options
from ludo_rl.ludo.config import strategy_config as legacy_strategy_cfg

from .enums import Color
from .piece import Piece
from .types import Move


@dataclass(slots=True)
class Player:
    color: int | Color
    strategy: BaseStrategy = field(default_factory=HumanStrategy)
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
        if not self.strategy:
            return None

        # Build move_choices and action_mask per legacy extractor expectations
        pieces_per_player = len(self.pieces)
        move_choices: list[dict | None] = [None] * pieces_per_player
        action_mask = np.zeros(pieces_per_player, dtype=bool)
        for mv in legal_moves:
            if 0 <= mv.piece_id < pieces_per_player and move_choices[mv.piece_id] is None:
                action_mask[mv.piece_id] = True
                move_choices[mv.piece_id] = {
                    "piece": self.pieces[mv.piece_id],
                    "new_pos": mv.new_pos,
                }
        ctx = build_move_options(board_stack, int(dice_roll), action_mask, move_choices)
        decided = self.strategy.select_move(ctx)
        if decided is None:
            return None
        # Map back to our Move by piece_id
        for mv in legal_moves:
            if mv.piece_id == decided.piece_id:
                return mv
        return None
