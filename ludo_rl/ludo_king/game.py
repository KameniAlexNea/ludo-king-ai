from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from .board import Board
from .config import config, reward_config
from .piece import Piece
from .player import Player
from .reward import compute_move_rewards, compute_state_potential, shaping_delta
from .types import Move, MoveEvents, MoveResult


@dataclass(slots=True)
class Game:
    players: List[Player]
    board: Board = field(init=False)
    rng: random.Random = field(default_factory=random.Random, init=False)
    # Cache for potential-based shaping: last Φ(s) per player
    _phi_cache: list[float] = field(default_factory=list, init=False, repr=False)
    _phi_valid: list[bool] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Board expects players indexed by Color id (0..3). Build a fixed map.
        pieces_by_color: List[List[Piece]] = [
            [] for _ in range(config.PIECES_PER_PLAYER)
        ]
        for pl in self.players:
            pieces_by_color[int(pl.color)] = pl.pieces
        colors = list(range(len(pieces_by_color)))
        self.board = Board(players=pieces_by_color, colors=colors)
        # Init phi cache per player index
        n = len(self.players)
        self._phi_cache = [0.0] * n
        self._phi_valid = [False] * n

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
            """Ring-only squares traversed from start to destination.

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
            if 1 <= start_rel <= config.MAIN_TRACK_END and 1 <= end_rel <= config.MAIN_TRACK_END:
                for r in range(start_rel + 1, end_rel + 1):
                    path.append(r)
                return path
            # Ring -> home column: walk to the end of ring (51)
            if 1 <= start_rel <= config.MAIN_TRACK_END and end_rel >= config.HOME_COLUMN_START:
                for r in range(start_rel + 1, config.MAIN_TRACK_END + 1):
                    path.append(r)
                return path
            return path

        # Defer computing potential until needed (lazy) to avoid overhead on blocked moves
        phi_before_computed = False
        phi_before = 0.0
        if reward_config.shaping_use and self._phi_valid[mv.player_index]:
            phi_before = self._phi_cache[mv.player_index]
            phi_before_computed = True

        # Precompute blockade absolute positions per color (main ring only)
        blockade_abs_to_color: dict[int, int] = {}
        for pi, pl in enumerate(self.players):
            color_id = int(pl.color)
            # Count pieces per relative ring position
            counts: dict[int, int] = {}
            for piece in pl.pieces:
                r = int(piece.position)
                if 1 <= r <= config.MAIN_TRACK_END:
                    counts[r] = counts.get(r, 0) + 1
            for r, cnt in counts.items():
                if cnt >= 2:
                    abs_b = self.board.absolute_position(color_id, r)
                    if abs_b != -1:
                        blockade_abs_to_color[abs_b] = color_id

        path = iter_ring_path(old, mv.new_pos)
        for rel in path:
            abs_pos = self.board.absolute_position(player.color, rel)
            # Any blockade on path (any color) blocks traversal
            if abs_pos in blockade_abs_to_color:
                events.hit_blockade = True
                events.move_resolved = False
                # Even when a move is blocked, compute rewards centrally
                rewards = compute_move_rewards(
                    num_players=len(self.players),
                    mover_index=mv.player_index,
                    old_position=old,
                    new_position=old,
                    events=events,
                )
                if reward_config.shaping_use:
                    # No state change; shaping delta is (gamma-1)*phi(s)
                    if not phi_before_computed:
                        phi_before = compute_state_potential(
                            self, mv.player_index, depth=reward_config.ro_depth
                        )
                        phi_before_computed = True
                    sd = shaping_delta(phi_before, phi_before, gamma=reward_config.shaping_gamma)
                    rewards[mv.player_index] += reward_config.shaping_alpha * sd
                return MoveResult(
                    old_position=old,
                    new_position=old,
                    events=events,
                    extra_turn=False,
                    rewards=rewards,
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
                        {
                            "player": victim_index,
                            "piece_id": opp_piece.piece_id,
                            "abs_pos": abs_pos,
                        }
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
            events.blockades.append({"player": mv.player_index, "rel": pc.position})

        # Compute per-player rewards (optional; env may or may not use)
        rewards = compute_move_rewards(
            num_players=len(self.players),
            mover_index=mv.player_index,
            old_position=old,
            new_position=pc.position,
            events=events,
        )

        # If the mover's player just won (all pieces finished), add small opponent penalty
        if self.players[mv.player_index].check_won():
            for idx in range(len(self.players)):
                if idx != mv.player_index:
                    rewards[idx] += reward_config.opp_win_penalty

        # Add potential-based shaping
        if reward_config.shaping_use:
            if not phi_before_computed:
                phi_before = compute_state_potential(
                    self, mv.player_index, depth=reward_config.ro_depth
                )
                phi_before_computed = True
            phi_after = compute_state_potential(self, mv.player_index, depth=reward_config.ro_depth)
            sd = shaping_delta(phi_before, phi_after, gamma=reward_config.shaping_gamma)
            rewards[mv.player_index] += reward_config.shaping_alpha * sd
            # Update cache for this player's latest state
            self._phi_cache[mv.player_index] = phi_after
            self._phi_valid[mv.player_index] = True

        return MoveResult(
            old_position=old,
            new_position=pc.position,
            events=events,
            extra_turn=extra,
            rewards=rewards,
        )
