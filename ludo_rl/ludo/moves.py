from dataclasses import dataclass, field

from .board import LudoBoard
from .player import MoveResolution, Piece, Player
from .reward import compute_move_rewards


@dataclass(slots=True)
class MoveManagement:
    players: list[Player]
    board: LudoBoard = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.board = LudoBoard(self.players)

    def is_safe(self, position: int) -> bool:
        return self.board.is_safe_square(position)

    def get_absolute_position(self, player_index: int, relative_pos: int) -> int:
        return self.board.absolute_position(player_index, relative_pos)

    def get_agent_relative_pos(self, agent_index: int, abs_pos: int) -> int:
        return self.board.relative_position(agent_index, abs_pos)

    def get_valid_moves(self, player_index: int, dice_roll: int):
        return self.players[player_index].get_valid_moves(self.board, dice_roll)

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
