from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(slots=True)
class MoveOption:
    """Structured metadata about a legal move."""

    piece_id: int
    current_pos: int
    new_pos: int
    dice_roll: int
    progress: int
    distance_to_goal: int
    can_capture: bool
    capture_count: int
    enters_home: bool
    enters_safe_zone: bool
    forms_blockade: bool
    extra_turn: bool
    risk: float
    leaving_safe_zone: bool


@dataclass(slots=True)
class StrategyContext:
    """Input payload shared by heuristic strategies."""

    board: np.ndarray  # shape (channels, path_length)
    dice_roll: int
    action_mask: np.ndarray  # shape (4,)
    moves: List[MoveOption]

    def iter_legal(self) -> Iterable[MoveOption]:
        return (move for move in self.moves if self.action_mask[move.piece_id])
