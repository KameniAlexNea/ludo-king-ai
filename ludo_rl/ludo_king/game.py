from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from .board import Board
from .config import config
from .piece import Piece

if TYPE_CHECKING:  # avoid runtime import to prevent circular deps
    from .player import Player

from .types import BlockadeEvent, KnockoutEvent, Move, MoveEvents, MoveResult


@dataclass(slots=True)
class Game:
    players: List["Player"]
    board: Board = field(init=False)
    rng: random.Random = field(default_factory=random.Random, init=False)

    def __post_init__(self) -> None:
        # Board expects players indexed by Color id (0..3). Build a fixed map.
        pieces_by_color: List[List[Piece]] = [
            [] for _ in range(config.PIECES_PER_PLAYER)
        ]
        for pl in self.players:
            pieces_by_color[int(pl.color)] = pl.pieces
        colors = list(range(len(pieces_by_color)))
        self.board = Board(players=pieces_by_color, colors=colors)

    # --- Dice ---
    def roll_dice(self) -> int:
        return self.rng.randint(1, 6)

    # --- Rules: destinations and legality ---
    @staticmethod
    def _destination_for_roll(current_pos: int, dice: int) -> int | None:
        if current_pos == 0:
            return config.START_POSITION if dice == 6 else None
        if current_pos == config.HOME_FINISH:
            return None
        if config.HOME_COLUMN_START <= current_pos <= config.HOME_FINISH - 1:
            cand = current_pos + dice
            return cand if cand <= config.HOME_FINISH else None
        cand = current_pos + dice
        if cand > config.MAIN_TRACK_END:
            overflow = cand - config.MAIN_TRACK_END
            if overflow > config.HOME_COLUMN_SIZE:
                return None
            return config.HOME_COLUMN_START + overflow - 1
        return cand

    # --- Path utilities ---
    def _iter_ring_path(self, start_rel: int, end_rel: int) -> list[int]:
        """Return ring-only squares traversed from start to destination.

        - From yard (0) to ring: include only destination if on ring.
        - From ring to ring: include all ring squares (start+1 .. end) inclusive.
        - From ring to home column: include ring squares (start+1 .. 51) inclusive.
        - From home column/finished: no ring traversal.
        """
        path: list[int] = []
        # Yard -> ring
        if start_rel == 0:
            if 1 <= end_rel <= config.MAIN_TRACK_END:
                path.append(end_rel)
            return path
        # Already in home column or finished
        if start_rel >= config.HOME_COLUMN_START:
            return path
        # Ring -> ring
        if (
            1 <= start_rel <= config.MAIN_TRACK_END
            and 1 <= end_rel <= config.MAIN_TRACK_END
        ):
            for r in range(start_rel + 1, end_rel + 1):
                path.append(r)
            return path
        # Ring -> home column: walk to the end of ring (51)
        if (
            1 <= start_rel <= config.MAIN_TRACK_END
            and end_rel >= config.HOME_COLUMN_START
        ):
            for r in range(start_rel + 1, config.MAIN_TRACK_END + 1):
                path.append(r)
            return path
        return path

    def legal_moves(self, player_idx: int, dice: int) -> List[Move]:
        player = self.players[player_idx]
        moves: List[Move] = []

        # Get blockade positions from board (centralized, computed once)
        blockade_abs_to_color = self.board.get_blockade_positions()

        for pc in player.pieces:
            dest = self._destination_for_roll(pc.position, dice)
            if dest is None:
                continue

            # Filter out moves that would cross or land on any blockade along the ring path
            path = self._iter_ring_path(int(pc.position), int(dest))
            blocked = False
            for rel in path:
                abs_pos = self.board.absolute_position(int(player.color), rel)
                if abs_pos in blockade_abs_to_color:
                    blocked = True
                    break
            if blocked:
                continue

            moves.append(
                Move(
                    player_index=player_idx,
                    piece_id=pc.piece_id,
                    new_pos=dest,
                    dice_roll=dice,
                )
            )
        return moves

    # --- Applying a move ---
    def apply_move(self, mv: Move) -> MoveResult:
        player = self.players[mv.player_index]
        pc = player.pieces[mv.piece_id]

        events = MoveEvents()
        old = pc.position

        # Get blockade positions from board (centralized, computed once)
        blockade_abs_to_color = self.board.get_blockade_positions()

        path = self._iter_ring_path(old, mv.new_pos)
        for rel in path:
            abs_pos = self.board.absolute_position(player.color, rel)
            # Any blockade on path (any color) blocks traversal
            if abs_pos in blockade_abs_to_color:
                events.hit_blockade = True
                events.move_resolved = False
                # Return without rewards - env calculates them
                return MoveResult(
                    old_position=old,
                    new_position=old,
                    events=events,
                    extra_turn=False,
                    rewards=None,
                )

        if old == 0 and mv.new_pos == config.START_POSITION:
            events.exited_home = True

        if mv.new_pos == config.HOME_FINISH:
            events.finished = True

        # tentative move
        pc.move_to(mv.new_pos)

        # resolve interactions on board (captures/blockades) only if on ring
        if 1 <= mv.new_pos <= config.MAIN_TRACK_END:
            abs_pos = self.board.absolute_position(player.color, mv.new_pos)
            occupants = self.board.pieces_at_absolute(
                abs_pos, exclude_color=player.color
            )
            # Non-safe squares: capture single opponent; block on opponent blockade
            if abs_pos not in config.SAFE_SQUARES_ABS:
                if len(occupants) == 1:
                    opp_color, opp_piece = occupants[0]
                    opp_piece.send_home()
                    # Map victim color to current game player index
                    victim_index = 0
                    for i, pl in enumerate(self.players):
                        if int(pl.color) == opp_color:
                            victim_index = i
                            break
                    events.knockouts.append(
                        KnockoutEvent(
                            player=victim_index,
                            piece_id=opp_piece.piece_id,
                            abs_pos=abs_pos,
                        )
                    )
                elif len(occupants) >= 2:
                    # can't land on an opponent blockade on non-safe squares
                    owner = blockade_abs_to_color.get(abs_pos, None)
                    if owner is not None and owner != int(player.color):
                        pc.move_to(old)
                        events.hit_blockade = True
                        events.move_resolved = False
            else:
                # Safe squares: cannot capture; also cannot land on opponent blockade
                if len(occupants) >= 2:
                    owner = blockade_abs_to_color.get(abs_pos, None)
                    if owner is not None and owner != int(player.color):
                        pc.move_to(old)
                        events.hit_blockade = True
                        events.move_resolved = False

        extra = bool(events.knockouts) or bool(events.finished) or mv.dice_roll == 6
        # If move resolved and we are on the ring, check if we formed a blockade (two of our pieces)
        if (
            events.move_resolved
            and 1 <= pc.position <= config.MAIN_TRACK_END
            and self.board.count_at_relative(player.color, pc.position) >= 2
        ):
            events.blockades.append(
                BlockadeEvent(player=mv.player_index, rel_pos=pc.position)
            )

        # Rewards calculated by env, not game - return None
        return MoveResult(
            old_position=old,
            new_position=pc.position,
            events=events,
            extra_turn=extra,
            rewards=None,
        )
