import random

from .config import config
from .moves import MoveManagement
from .player import Piece, Player


class LudoGame:
    """
    Main Ludo Game Engine.
    Contains all rules and state manipulation logic.
    """

    def __init__(self):
        self.players = [Player(i) for i in range(config.NUM_PLAYERS)]
        self.rng = random.Random()
        self.rng.seed(42)
        self.move_manager = MoveManagement(self.players)

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
        state = {
            "my_pieces": [0] * config.PATH_LENGTH,
            "opp1_pieces": [0] * config.PATH_LENGTH,  # Next player
            "opp2_pieces": [0] * config.PATH_LENGTH,  # Player across
            "opp3_pieces": [0] * config.PATH_LENGTH,  # Prev player
            "safe_zones": [0] * config.PATH_LENGTH,
        }

        # 1. Populate Safe Zones channel
        for i in range(52, 57):
            state["safe_zones"][i] = 1  # Home column is safe
        for rel_pos in range(1, 52):
            abs_pos = self.move_manager.get_absolute_position(agent_index, rel_pos)
            if self.move_manager.is_safe(abs_pos):
                state["safe_zones"][rel_pos] = 1

        # 2. Populate Piece channels
        channels = [
            state["my_pieces"],
            state["opp1_pieces"],
            state["opp2_pieces"],
            state["opp3_pieces"],
        ]

        for i in range(config.NUM_PLAYERS):
            player = self.players[i]
            relative_player_index = (
                i - agent_index + config.NUM_PLAYERS
            ) % config.NUM_PLAYERS
            target_channel = channels[relative_player_index]

            for piece in player.pieces:
                if relative_player_index == 0:
                    # This is our agent's piece
                    target_channel[piece.position] += 1
                else:
                    # This is an opponent's piece
                    if piece.position == 0:
                        target_channel[0] += 1  # Opponent in yard
                    elif piece.position > 51:
                        # Opponent in their home column, we don't map this
                        continue
                    else:
                        # Opponent on main track, map to our relative path
                        opp_abs_pos = self.move_manager.get_absolute_position(
                            i, piece.position
                        )
                        agent_relative_pos = self.move_manager.get_agent_relative_pos(
                            agent_index, opp_abs_pos
                        )
                        if agent_relative_pos != -1:
                            target_channel[agent_relative_pos] += 1
        return state
