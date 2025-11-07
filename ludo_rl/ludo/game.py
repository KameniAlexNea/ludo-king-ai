import random
from dataclasses import dataclass, field

import numpy as np

from .config import config
from .moves import MoveManagement
from .player import Piece, Player


def _compute_safe_relative_positions() -> tuple[tuple[int, ...], ...]:
    safe_positions: list[tuple[int, ...]] = []
    for agent_start in config.PLAYER_START_SQUARES:
        relative_indices = {
            ((abs_pos - agent_start + 52) % 52) + 1
            for abs_pos in config.SAFE_SQUARES_ABS
        }
        safe_positions.append(tuple(sorted(relative_indices)))
    return tuple(safe_positions)


def _compute_relative_translations() -> tuple[tuple[tuple[int, ...], ...], ...]:
    translations: list[list[tuple[int, ...]]] = []
    for player_start in config.PLAYER_START_SQUARES:
        player_rows: list[tuple[int, ...]] = []
        for agent_start in config.PLAYER_START_SQUARES:
            mapping = [-1] * config.PATH_LENGTH
            for rel_pos in range(1, 52):
                abs_pos = ((player_start + rel_pos - 2 + 52) % 52) + 1
                agent_rel = ((abs_pos - agent_start + 52) % 52) + 1
                mapping[rel_pos] = agent_rel
            player_rows.append(tuple(mapping))
        translations.append(tuple(player_rows))
    return tuple(translations)


_SAFE_RELATIVE_POSITIONS = _compute_safe_relative_positions()
_RELATIVE_TRANSLATIONS = _compute_relative_translations()


@dataclass(slots=True)
class LudoGame:
    """
    Main Ludo Game Engine.
    Contains all rules and state manipulation logic.
    """

    players: list[Player] = field(init=False)
    rng: random.Random = field(init=False, repr=False)
    move_manager: MoveManagement = field(init=False)
    _board_buffer: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.players = [Player(i) for i in range(config.NUM_PLAYERS)]
        self.rng = random.Random()
        self.rng.seed(42)
        self.move_manager = MoveManagement(self.players)

    def build_board_tensor(
        self, agent_index: int, out: np.ndarray | None = None
    ) -> np.ndarray:
        if out is not None:
            board = out
            if board.shape != (10, config.PATH_LENGTH):
                raise ValueError("Expected board tensor of shape (10, PATH_LENGTH)")
        else:
            if self._board_buffer is None:
                self._board_buffer = np.zeros(
                    (10, config.PATH_LENGTH), dtype=np.float32
                )
            board = self._board_buffer

        board.fill(0.0)

        safe_channel = board[4]
        safe_channel[52:57] = 1.0
        for rel_pos in _SAFE_RELATIVE_POSITIONS[agent_index]:
            safe_channel[rel_pos] = 1.0

        for player_index, player in enumerate(self.players):
            relative_channel_index = (
                player_index - agent_index + config.NUM_PLAYERS
            ) % config.NUM_PLAYERS
            channel = board[relative_channel_index]

            if relative_channel_index == 0:
                for piece in player.pieces:
                    channel[piece.position] += 1.0
                continue

            mapping = _RELATIVE_TRANSLATIONS[player_index][agent_index]
            for piece in player.pieces:
                pos = piece.position
                if pos == 0:
                    channel[0] += 1.0
                elif pos > 51:
                    continue
                else:
                    target = mapping[pos]
                    if target != -1:
                        channel[target] += 1.0

        return board

    def get_agent_relative_pos(
        self,
        agent_index: int,
        abs_pos: int,
    ):
        return self.move_manager.get_agent_relative_pos(agent_index, abs_pos)

    def get_absolute_position(self, player_index: int, relative_pos: int):
        return self.move_manager.get_absolute_position(player_index, relative_pos)

    def get_valid_moves(self, player_index: int, dice_roll: int):
        return self.move_manager.get_valid_moves(player_index, dice_roll)

    def roll_dice(self):
        return self.rng.randint(1, 6)

    def make_move(
        self, player_index: int, piece: Piece, new_position: int, dice_roll: int
    ):
        """Executes a move and returns the reward and events."""
        return self.move_manager.make_move(player_index, piece, new_position, dice_roll)

    def get_board_state(self, agent_index):
        """Generates the (58, 5) board state tensor for the given agent."""
        board = self.build_board_tensor(agent_index)
        return {
            "my_pieces": [int(value) for value in board[0]],
            "opp1_pieces": [int(value) for value in board[1]],
            "opp2_pieces": [int(value) for value in board[2]],
            "opp3_pieces": [int(value) for value in board[3]],
            "safe_zones": [int(value) for value in board[4]],
        }
