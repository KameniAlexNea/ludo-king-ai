from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from .board import Board
from .config import config
from .piece import Piece
from .player import Player
from .types import Move, MoveEvents, MoveResult


@dataclass(slots=True)
class Game:
    players: List[Player]
    board: Board = field(init=False)
    rng: random.Random = field(default_factory=random.Random, init=False)

    def __post_init__(self) -> None:
        # Board expects players indexed by Color id (0..3). Build a fixed map.
        pieces_by_color: List[List[Piece]] = [[] for _ in range(4)]
        for pl in self.players:
            pieces_by_color[int(pl.color)] = list(pl.pieces)
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

    def legal_moves(self, player_idx: int, dice: int) -> List[Move]:
        player = self.players[player_idx]
        moves: List[Move] = []
        for pc in player.pieces:
            dest = self._destination_for_roll(pc.position, dice)
            if dest is None:
                continue
            # cannot stack beyond 2 on main track
            if (
                dest <= config.MAIN_TRACK_END
                and self.board.count_at_relative(player.color, dest) >= 2
            ):
                continue
                # cannot land on an opponent blockade (two or more opponent pieces on ring)
                if 1 <= dest <= config.MAIN_TRACK_END:
                    abs_pos = self.board.absolute_position(player.color, dest)
                    occ = self.board.pieces_at_absolute(abs_pos, exclude_color=int(player.color))
                    if len(occ) >= 2:
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
            if len(occupants) == 1:
                opp_color, opp_piece = occupants[0]
                opp_piece.send_home()
                events.knockouts.append(
                    {
                        "player": opp_color,
                        "piece_id": opp_piece.piece_id,
                        "abs_pos": abs_pos,
                    }
                )
            elif len(occupants) >= 2:
                # can't land on an opponent blockade; revert
                pc.move_to(old)
                events.hit_blockade = True
                events.move_resolved = False

        extra = bool(events.knockouts) or bool(events.finished) or mv.dice_roll == 6

        return MoveResult(
            old_position=old,
            new_position=pc.position,
            events=events,
            extra_turn=extra,
            rewards=None,
        )
