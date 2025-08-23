"""
Improved Deep Q-Network implementation for Ludo RL training with Dueling DQN architecture.
"""

from typing import List, Tuple

import numpy as np


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for more efficient learning."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, experience: Tuple):
        """Add experience to buffer with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        # Assign maximum priority to new experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
