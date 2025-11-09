import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, cast

import numpy as np

from ludo_rl.strategy import build_move_options, create

from ..strategy.base import BaseStrategy
from .config import config
from .piece import Piece

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .board import LudoBoard


@dataclass(slots=True)
class MoveResolution:
    old_position: int
    new_position: int
    events: Dict[str, object]
    extra_turn: bool


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

    def reset(self) -> None:
        self.has_finished = False
        for piece in self.pieces:
            piece.send_home()

    def get_valid_moves(
        self, board: "LudoBoard", dice_roll: int
    ) -> List[Dict[str, object]]:
        moves: List[Dict[str, object]] = []

        finish_position = config.PATH_LENGTH - 1

        for piece in self.pieces:
            destination = piece.destination_for_roll(dice_roll)

            if destination is None:
                continue

            if destination == finish_position:
                moves.append(
                    {"piece": piece, "new_pos": destination, "dice_roll": dice_roll}
                )
                continue

            if board.count_player_pieces(self.color, destination) < 2:
                moves.append(
                    {"piece": piece, "new_pos": destination, "dice_roll": dice_roll}
                )

        return moves

    def move_piece(
        self, board: "LudoBoard", piece: Piece, new_position: int, dice_roll: int
    ) -> MoveResolution:
        events: Dict[str, object] = {
            "knockouts": [],
            "blockades": [],
            "hit_blockade": False,
            "exited_home": False,
            "finished": False,
            "move_resolved": True,
        }

        old_position = piece.position

        if old_position == 0 and new_position == 1:
            events["exited_home"] = True
        if new_position == 57:
            events["finished"] = True

        piece.move_to(new_position)

        abs_pos = board.absolute_position(self.color, new_position)
        if abs_pos != -1 and not board.is_safe_square(abs_pos):
            occupants = board.pieces_at_absolute(abs_pos, exclude_player=self.color)

            if len(occupants) == 1:
                opponent_index, opponent_piece = occupants[0]
                opponent_piece.send_home()
                knockouts = cast(List[Dict[str, int]], events["knockouts"])
                knockouts.append(
                    {
                        "player": opponent_index,
                        "piece_id": opponent_piece.piece_id,
                        "abs_pos": abs_pos,
                    }
                )
            elif len(occupants) >= 2:
                piece.move_to(old_position)
                events["hit_blockade"] = True
                events["move_resolved"] = False

        final_position = piece.position

        if events["move_resolved"] and final_position not in (0, 57):
            if board.count_player_pieces(self.color, final_position) == 2:
                blockades = cast(List[Dict[str, int]], events["blockades"])
                blockades.append({"player": self.color, "relative_pos": final_position})

        extra_turn = (
            bool(events["knockouts"]) or bool(events["finished"]) or dice_roll == 6
        )

        return MoveResolution(
            old_position=old_position,
            new_position=final_position,
            events=events,
            extra_turn=extra_turn,
        )

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
        moves = list(ctx.iter_legal())
        if len(moves) <= 1:
            decision = moves[0] if moves else None
        else:
            decision = self._strategy.select_move(ctx)

        if decision is None:
            return random.choice(valid_moves)

        if decision.piece_id >= len(move_choices):
            return random.choice(valid_moves)
        selected = move_choices[decision.piece_id]
        if selected is None:
            return random.choice(valid_moves)
        return selected
