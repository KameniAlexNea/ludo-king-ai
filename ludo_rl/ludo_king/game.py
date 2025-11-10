from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from .board import Board
from .config import config
from .piece import Piece
from .player import Player
from .types import Move, MoveEvents, MoveResult
from .reward import compute_move_rewards


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
            # Note: Opponent blockade handling is enforced in apply_move (including crossing/landing)
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

        # Pre-check: cannot cross a blockade on ring squares
        def iter_ring_path(start_rel: int, end_rel: int) -> list[int]:
            path: list[int] = []
            # Only consider ring traversal (1..51)
            if start_rel == 0:
                # entering from yard: only the destination on ring matters
                if 1 <= end_rel <= config.MAIN_TRACK_END:
                    path.append(end_rel)
                return path
            cur = start_rel
            # Walk forward until we reach end_rel or leave ring
            while True:
                nxt = 1 if cur >= config.MAIN_TRACK_END else cur + 1
                if 1 <= nxt <= config.MAIN_TRACK_END:
                    path.append(nxt)
                if nxt == end_rel or not (1 <= end_rel <= config.MAIN_TRACK_END):
                    break
                cur = nxt
            return path

        path = iter_ring_path(old, mv.new_pos)
        for rel in path:
            abs_pos = self.board.absolute_position(player.color, rel)
            occ = self.board.pieces_at_absolute(abs_pos)
            # Count pieces per color to detect true blockade (same color >= 2)
            counts: dict[int, int] = {}
            for col, _ in occ:
                counts[col] = counts.get(col, 0) + 1
            if any(c >= 2 for c in counts.values()):
                events.hit_blockade = True
                events.move_resolved = False
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
            # No captures on global safe squares
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
                        {
                            "player": victim_index,
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
        # If move resolved and we are on the ring, check if we formed a blockade (two of our pieces)
        if (
            events.move_resolved
            and 1 <= pc.position <= config.MAIN_TRACK_END
            and self.board.count_at_relative(player.color, pc.position) >= 2
        ):
            events.blockades.append({"player": mv.player_index, "rel": pc.position})

        # Compute per-player rewards (optional; env may or may not use)
        rewards = compute_move_rewards(
            num_players=len(self.players),
            mover_index=mv.player_index,
            old_position=old,
            new_position=pc.position,
            events={
                "move_resolved": events.move_resolved,
                "exited_home": events.exited_home,
                "finished": events.finished,
                "knockouts": events.knockouts,
                "hit_blockade": events.hit_blockade,
                "blockades": events.blockades,
            },
        )

        return MoveResult(
            old_position=old,
            new_position=pc.position,
            events=events,
            extra_turn=extra,
            rewards=rewards,
        )
