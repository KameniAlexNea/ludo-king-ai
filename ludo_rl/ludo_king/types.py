from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class Color(IntEnum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    BLUE = 3


@dataclass(slots=True)
class Move:
    player_index: int
    piece_id: int
    new_pos: int
    dice_roll: int


@dataclass(slots=True)
class KnockoutEvent:
    """Event data for when a piece is knocked out (captured)."""

    player: int  # Index of the player whose piece was knocked out
    piece_id: int  # ID of the piece that was knocked out
    abs_pos: int  # Absolute board position where knockout occurred


@dataclass(slots=True)
class BlockadeEvent:
    """Event data for when a blockade is formed."""

    player: int  # Index of the player who formed the blockade
    rel_pos: int  # Relative position where blockade was formed


@dataclass(slots=True)
class MoveEvents:
    exited_home: bool = False
    finished: bool = False
    knockouts: List[KnockoutEvent] = field(default_factory=list)
    hit_blockade: bool = False
    blockades: List[BlockadeEvent] = field(default_factory=list)
    move_resolved: bool = True


@dataclass(slots=True)
class MoveResult:
    old_position: int
    new_position: int
    events: MoveEvents
    extra_turn: bool
    rewards: Optional[Dict[int, float]] = None
