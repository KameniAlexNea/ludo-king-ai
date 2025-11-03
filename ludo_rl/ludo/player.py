import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ludo_rl.strategy import build_move_options, create

from .config import config
from .model import Piece


@dataclass
class Player:
    color: int
    start_square: Optional[int] = None
    pieces: Optional[list[Piece]] = field(default_factory=lambda: [])
    has_finished: Optional[bool] = False
    strategy_name: str = "random"
    _strategy: object = field(default=None, init=False, repr=False)

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

        move_map: dict[int, dict] = {}
        action_mask = np.zeros(config.PIECES_PER_PLAYER, dtype=bool)

        for move in valid_moves:
            piece_id = move["piece"].piece_id
            action_mask[piece_id] = True
            if piece_id not in move_map:
                move_copy = move.copy()
                move_copy["dice_roll"] = dice_roll
                move_map[piece_id] = move_copy

        observation = {
            "board": board_stack,
            "dice_roll": np.array([dice_roll - 1], dtype=np.int64),
        }
        ctx = build_move_options(observation, action_mask, move_map)
        decision = self._strategy.select_move(ctx)

        if decision is None:
            return random.choice(valid_moves)

        selected = move_map.get(decision.piece_id)
        if selected is None:
            return random.choice(valid_moves)
        return selected
