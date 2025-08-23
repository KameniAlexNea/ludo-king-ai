"""
Improved Deep Q-Network implementation for Ludo RL training with Dueling DQN architecture.
"""

import torch
import torch.nn as nn


class LudoDQN(nn.Module):
    """Dueling DQN architecture for Ludo decision making with better feature extraction."""

    def __init__(self, state_dim: int, max_actions: int = 4, hidden_dim: int = 256):
        """
        Initialize the DQN with Dueling architecture.

        Args:
            state_dim: Dimensionality of the state space
            max_actions: Maximum number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(LudoDQN, self).__init__()
        self.state_dim = state_dim
        self.max_actions = max_actions

        # Better feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Dueling DQN: Separate value and advantage streams
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights with proper initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dueling DQN.

        Args:
            state: Input state tensor

        Returns:
            torch.Tensor: Q-values for each action
        """
        features = self.feature_extractor(state)

        # Separate value and advantage computation
        value = self.value_head(features)
        advantages = self.advantage_head(features)

        # Dueling DQN combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values
