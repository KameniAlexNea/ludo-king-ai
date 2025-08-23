from typing import Dict, List

import numpy as np


class DynamicActionEncoder:
    """Handle variable number of valid moves more effectively."""

    def __init__(self, max_moves: int = 4, action_features: int = 8):
        """
        Initialize action encoder.

        Args:
            max_moves: Maximum number of moves to encode
            action_features: Number of features per move
        """
        self.max_moves = max_moves
        self.action_features = action_features

    def encode_action_space(self, valid_moves: List[Dict]) -> np.ndarray:
        """Create action space encoding for current valid moves."""
        action_encoding = np.zeros((self.max_moves, self.action_features))

        for i, move in enumerate(valid_moves[: self.max_moves]):
            action_encoding[i] = self._encode_single_move(move)

        return action_encoding.flatten()

    def _encode_single_move(self, move: Dict) -> np.ndarray:
        """Encode features of a single move."""
        features = np.zeros(self.action_features)

        features[0] = move.get("token_id", 0) / 3.0  # Normalized token ID
        features[1] = self._encode_move_type_numeric(move.get("move_type", ""))
        features[2] = 1.0 if move.get("is_safe_move", True) else 0.0
        features[3] = 1.0 if move.get("captures_opponent", False) else 0.0
        features[4] = move.get("strategic_value", 0) / 30.0
        features[5] = (move.get("target_position", 0) + 1) / 53.0  # Normalized position
        features[6] = len(move.get("captured_tokens", []))
        features[7] = 1.0 if move.get("move_type") == "finish" else 0.0

        return features

    def _encode_move_type_numeric(self, move_type: str) -> float:
        """Convert move type to numeric value."""
        move_type_mapping = {
            "exit_home": 0.2,
            "advance_main_board": 0.4,
            "enter_home_column": 0.6,
            "advance_home_column": 0.8,
            "finish": 1.0,
        }
        return move_type_mapping.get(move_type, 0.0)
