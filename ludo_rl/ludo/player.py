import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ludo_rl.strategy import build_move_options, create

from ..strategy.base import BaseStrategy
from .config import config
from .model import Piece


@dataclass(slots=True)
class Player:
    color: int
    start_square: Optional[int] = None
    pieces: Optional[list[Piece]] = field(default_factory=lambda: [])
    has_finished: Optional[bool] = False
    strategy_name: str = "random"
    _strategy: BaseStrategy = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.start_square = config.PLAYER_START_SQUARES[self.color]
        self.pieces = [Piece(self.color, i) for i in range(config.PIECES_PER_PLAYER)]

    def has_won(self):
        """Checks if all pieces are at the home position."""
        if self.has_finished:  # Don't re-check if already done
            return True

        if all(p.is_finished() for p in self.pieces):
            self.has_finished = True
            return True
        return False

    def decide(self, board_stack: np.ndarray, dice_roll: int, valid_moves: list[dict]):
        if not valid_moves:
            return None

        if self.strategy_name == "random":
            return random.choice(valid_moves)

        if (
            self._strategy is not None
            and getattr(self._strategy, "name", None) != self.strategy_name
        ):
            self._strategy = None

        if self._strategy is None:
            try:
                self._strategy = create(self.strategy_name)
            except KeyError:
                self.strategy_name = "random"
                return random.choice(valid_moves)

        move_choices: list[dict | None] = [None] * config.PIECES_PER_PLAYER
        action_mask = np.zeros(config.PIECES_PER_PLAYER, dtype=bool)

        for move in valid_moves:
            piece_id = move["piece"].piece_id
            action_mask[piece_id] = True
            if move_choices[piece_id] is None:
                move_choices[piece_id] = move

        ctx = build_move_options(board_stack, dice_roll, action_mask, move_choices)
        decision = self._strategy.select_move(ctx)

        if decision is None:
            return random.choice(valid_moves)

        if decision.piece_id >= len(move_choices):
            return random.choice(valid_moves)
        selected = move_choices[decision.piece_id]
        if selected is None:
            return random.choice(valid_moves)
        return selected
