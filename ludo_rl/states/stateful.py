from collections import deque
from typing import Dict

import numpy as np

from .base import LudoStateEncoder


class StatefulLudoEncoder(LudoStateEncoder):
    """Encoder with memory of previous states for better decision making."""

    def __init__(self, history_length: int = 3, add_noise: bool = False):
        """
        Initialize stateful encoder with history tracking.

        Args:
            history_length: Number of previous states to remember
            add_noise: Whether to add regularization noise
        """
        super().__init__(add_noise=add_noise)
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        # Expand state dimension to include history
        self.state_dim_with_history = 64 + (64 * (history_length - 1))

    def encode_state_with_history(self, game_data: Dict) -> np.ndarray:
        """Encode state including recent history."""
        current_state = super().encode_state(game_data)
        self.state_history.append(current_state)

        # Combine current state with history
        if len(self.state_history) < self.history_length:
            # Pad with zeros if not enough history
            padded_history = [
                np.zeros(64)
                for _ in range(self.history_length - len(self.state_history))
            ]
            full_history = padded_history + list(self.state_history)
        else:
            full_history = list(self.state_history)

        return np.concatenate(full_history)

    def reset_history(self):
        """Reset the state history (e.g., at the start of a new game)."""
        self.state_history.clear()
