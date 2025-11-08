import random
from dataclasses import dataclass, field

import numpy as np

from .board import LudoBoard
from .config import config
from .player import MoveResolution, Piece, Player
from .reward import compute_move_rewards


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
        self.rng.seed(42)

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
