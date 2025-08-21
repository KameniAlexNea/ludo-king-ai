"""
State encoder for converting Ludo game states to numerical vectors
suitable for reinforcement learning training.
"""

from typing import Dict, List

import numpy as np


class LudoStateEncoder:
    """Converts Ludo game states to fixed-length numerical vectors."""

    def __init__(self, board_size=52, max_tokens=4, num_players=4):
        """
        Initialize the state encoder.

        Args:
            board_size: Number of positions on the main board
            max_tokens: Maximum tokens per player
            num_players: Maximum number of players
        """
        self.board_size = board_size
        self.max_tokens = max_tokens
        self.num_players = num_players
        self.state_dim = self._calculate_state_dim()

    def _calculate_state_dim(self):
        """Calculate the total dimensionality of the state vector."""
        # Token positions: each token can be in home (-1), board (0-51), or finished (52)
        # Use one-hot encoding for each token position (54 possible positions)
        token_positions = self.num_players * self.max_tokens * (self.board_size + 2)

        # Game context features
        game_context = 4  # dice_value, consecutive_sixes, turn_count, current_player

        # Player statistics per player
        player_stats = self.num_players * 4  # tokens_home, active, finished, won

        # Valid moves encoding (up to 4 possible moves, each with features)
        valid_moves_features = (
            self.max_tokens * 6
        )  # token_id, move_type, safety, capture, strategic_value, finishing

        return token_positions + game_context + player_stats + valid_moves_features

    def encode_state(self, game_data: Dict) -> np.ndarray:
        """
        Convert game state to numerical vector.

        Args:
            game_data: Game state data with structure from GameStateSaver

        Returns:
            np.ndarray: Fixed-length state vector
        """
        state = np.zeros(self.state_dim)
        idx = 0

        # Extract game context
        game_context = game_data["game_context"]
        current_situation = game_context["current_situation"]
        player_state = game_context["player_state"]
        opponents = game_context.get("opponents", [])
        valid_moves = game_context.get("valid_moves", [])

        # Encode token positions for all players
        current_player_color = current_situation["player_color"]
        player_id = self._get_player_id(current_player_color)

        # Encode current player's tokens first
        tokens = player_state["tokens"]
        idx = self._encode_player_tokens(state, idx, tokens, 0)

        # Encode opponent tokens (simplified - we only have limited opponent info)
        for i, opponent in enumerate(opponents[: self.num_players - 1]):
            # Create dummy token data for opponents based on available info
            dummy_tokens = self._create_opponent_dummy_tokens(opponent)
            idx = self._encode_player_tokens(state, idx, dummy_tokens, i + 1)

        # Fill remaining player slots with zeros if fewer than max players
        remaining_players = self.num_players - 1 - len(opponents)
        for i in range(remaining_players):
            idx += self.max_tokens * (self.board_size + 2)

        # Encode game context
        state[idx] = current_situation["dice_value"] / 6.0  # normalize dice value
        state[idx + 1] = (
            min(current_situation["consecutive_sixes"], 3) / 3.0
        )  # normalize
        state[idx + 2] = (
            min(current_situation["turn_count"], 200) / 200.0
        )  # normalize turn count
        state[idx + 3] = player_id / (self.num_players - 1)  # current player ID
        idx += 4

        # Encode player statistics
        # Current player stats
        state[idx] = player_state["tokens_in_home"] / self.max_tokens
        state[idx + 1] = player_state["active_tokens"] / self.max_tokens
        state[idx + 2] = player_state["finished_tokens"] / self.max_tokens
        state[idx + 3] = 1.0 if player_state["has_won"] else 0.0
        idx += 4

        # Opponent stats
        for opponent in opponents[: self.num_players - 1]:
            state[idx] = (
                self.max_tokens - opponent["tokens_active"]
            ) / self.max_tokens  # estimated home tokens
            state[idx + 1] = opponent["tokens_active"] / self.max_tokens
            state[idx + 2] = opponent["tokens_finished"] / self.max_tokens
            state[idx + 3] = 0.0  # assume not won (we'd know if they won)
            idx += 4

        # Fill remaining opponent stats with zeros
        remaining_opponents = self.num_players - 1 - len(opponents)
        for i in range(remaining_opponents):
            idx += 4

        # Encode valid moves features
        for i in range(self.max_tokens):
            if i < len(valid_moves):
                move = valid_moves[i]
                state[idx] = move["token_id"] / (
                    self.max_tokens - 1
                )  # normalize token ID
                state[idx + 1] = self._encode_move_type(move["move_type"])
                state[idx + 2] = 1.0 if move["is_safe_move"] else 0.0
                state[idx + 3] = 1.0 if move["captures_opponent"] else 0.0
                state[idx + 4] = (
                    min(move["strategic_value"], 30.0) / 30.0
                )  # normalize strategic value
                state[idx + 5] = 1.0 if move["move_type"] == "finish" else 0.0
            # else: zeros for unused move slots
            idx += 6

        return state

    def _encode_player_tokens(
        self, state: np.ndarray, start_idx: int, tokens: List[Dict], player_offset: int
    ) -> int:
        """Encode tokens for a specific player using one-hot encoding."""
        idx = start_idx

        for i in range(self.max_tokens):
            if i < len(tokens):
                token = tokens[i]
                position = token["position"]

                # Convert position to one-hot encoding
                if position == -1:  # home
                    state[idx] = 1.0
                elif token.get("is_finished", False):  # finished
                    state[idx + self.board_size + 1] = 1.0
                else:  # on board
                    # Clamp position to valid range
                    board_pos = max(0, min(position, self.board_size - 1))
                    state[idx + 1 + board_pos] = 1.0

            idx += self.board_size + 2

        return idx

    def _create_opponent_dummy_tokens(self, opponent: Dict) -> List[Dict]:
        """Create dummy token data for opponents based on available information."""
        dummy_tokens = []
        tokens_active = opponent["tokens_active"]
        tokens_finished = opponent["tokens_finished"]
        tokens_home = self.max_tokens - tokens_active - tokens_finished

        # Create dummy tokens
        for i in range(self.max_tokens):
            if i < tokens_home:
                dummy_tokens.append({"position": -1, "is_finished": False})
            elif i < tokens_home + tokens_active:
                # Place active tokens at random board positions (we don't know exact positions)
                dummy_tokens.append({"position": 25, "is_finished": False})  # mid-board
            else:
                dummy_tokens.append({"position": 52, "is_finished": True})

        return dummy_tokens

    def _get_player_id(self, color: str) -> int:
        """Convert player color to numeric ID."""
        color_map = {"red": 0, "green": 1, "yellow": 2, "blue": 3}
        return color_map.get(color.lower(), 0)

    def _encode_move_type(self, move_type: str) -> float:
        """Encode move type as normalized value."""
        type_map = {
            "exit_home": 0.0,
            "advance_main_board": 0.33,
            "enter_home_column": 0.66,
            "finish": 1.0,
        }
        return type_map.get(move_type, 0.5)

    def encode_action(self, chosen_move: int, valid_moves: List[Dict]) -> int:
        """
        Convert chosen move to action index.

        Args:
            chosen_move: Index of chosen move in valid_moves list
            valid_moves: List of valid moves

        Returns:
            int: Action index (0-3, representing token to move)
        """
        if chosen_move < len(valid_moves):
            return valid_moves[chosen_move]["token_id"]
        return 0  # default to first token
