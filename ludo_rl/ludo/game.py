import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .board import LudoBoard
from .config import config
from .player import MoveResolution, Piece, Player
from .reward import compute_move_rewards


@dataclass(slots=True)
class TurnOutcome:
    dice_roll: int
    move: Optional[dict]
    result: Optional[Dict[str, object]]
    extra_turn: bool
    skipped: bool


@dataclass(slots=True)
class LudoGame:
    """Main Ludo Game Engine coordinating board, players, and rewards."""

    board: LudoBoard = field(init=False)
    players: list[Player] = field(init=False)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.players = [Player(i) for i in range(config.NUM_PLAYERS)]
        self.board = LudoBoard(self.players)
        self.rng = random.Random()

    def build_board_tensor(
        self, agent_index: int, out: np.ndarray | None = None
    ) -> np.ndarray:
        return self.board.build_tensor(agent_index, out)

    def get_agent_relative_pos(
        self,
        agent_index: int,
        abs_pos: int,
    ):
        return self.board.relative_position(agent_index, abs_pos)

    def get_absolute_position(self, player_index: int, relative_pos: int):
        return self.board.absolute_position(player_index, relative_pos)

    def get_valid_moves(self, player_index: int, dice_roll: int):
        return self.players[player_index].get_valid_moves(self.board, dice_roll)

    def roll_dice(self):
        return self.rng.randint(1, 6)

    def make_move(
        self, player_index: int, piece: Piece, new_position: int, dice_roll: int
    ):
        resolution: MoveResolution = self.players[player_index].move_piece(
            self.board, piece, new_position, dice_roll
        )

        rewards = compute_move_rewards(
            len(self.players),
            player_index,
            resolution.old_position,
            resolution.new_position,
            resolution.events,
        )

        return {
            "reward": rewards[player_index],
            "rewards": rewards,
            "events": resolution.events,
            "extra_turn": resolution.extra_turn,
        }

    def take_turn(
        self,
        player_index: int,
        *,
        dice_roll: Optional[int] = None,
        move: Optional[dict] = None,
        rng: Optional[random.Random] = None,
    ) -> TurnOutcome:
        rng = rng or self.rng
        dice = dice_roll if dice_roll is not None else self.roll_dice()

        valid_moves = self.get_valid_moves(player_index, dice)

        if move is not None:
            selected_move = move
            if "dice_roll" not in selected_move or selected_move["dice_roll"] != dice:
                selected_move = dict(selected_move)
                selected_move["dice_roll"] = dice

            if not any(
                selected_move["piece"] is option["piece"]
                and selected_move["new_pos"] == option["new_pos"]
                for option in valid_moves
            ):
                return TurnOutcome(dice, None, None, False, True)
        else:
            if not valid_moves:
                return TurnOutcome(dice, None, None, False, True)

            board_stack = self.build_board_tensor(player_index)
            decision = self.players[player_index].decide(board_stack, dice, valid_moves)
            selected_move = (
                decision if decision is not None else rng.choice(valid_moves)
            )
            if "dice_roll" not in selected_move or selected_move["dice_roll"] != dice:
                selected_move = dict(selected_move)
                selected_move["dice_roll"] = dice

        result = self.make_move(
            player_index,
            selected_move["piece"],
            selected_move["new_pos"],
            dice,
        )

        copied_move = dict(selected_move)
        return TurnOutcome(
            dice_roll=dice,
            move=copied_move,
            result=result,
            extra_turn=bool(result.get("extra_turn", False)),
            skipped=False,
        )

    def get_board_state(self, agent_index):
        """Generates the (58, 5) board state tensor for the given agent."""
        board = self.build_board_tensor(agent_index)
        return {
            key: v
            for key, v in zip(
                [
                    "my_pieces",
                    "opp1_pieces",
                    "opp2_pieces",
                    "opp3_pieces",
                    "safe_zones",
                ],
                board,
            )
        }
