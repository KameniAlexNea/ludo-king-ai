"""
Deep Q-Network implementation for Ludo RL training.
"""

import random
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LudoDQN(nn.Module):
    """Deep Q-Network for Ludo decision making."""

    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 512):
        """
        Initialize the DQN.

        Args:
            state_dim: Dimensionality of the state space
            action_dim: Number of possible actions (tokens to move)
            hidden_dim: Size of hidden layers
        """
        super(LudoDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.network(state)


class LudoDQNAgent:
    """DQN Agent for Ludo with experience replay and target network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim: Dimensionality of state space
            action_dim: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
            memory_size: Size of experience replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Frequency of target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_step = 0

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = LudoDQN(state_dim, action_dim).to(self.device)
        self.target_network = LudoDQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Initialize target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, valid_moves: List[Dict]) -> int:
        """
        Choose action using epsilon-greedy policy with valid move masking.

        Args:
            state: Current state
            valid_moves: List of valid moves available

        Returns:
            int: Action index (token ID to move)
        """
        if not valid_moves:
            return 0

        # Extract valid token IDs
        valid_token_ids = [move["token_id"] for move in valid_moves]

        # Epsilon-greedy exploration
        if np.random.random() <= self.epsilon:
            return random.choice(valid_token_ids)

        # Exploitation: choose best action among valid ones
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().data.numpy()[0]

            # Mask invalid actions by setting their Q-values to very low values
            masked_q_values = np.full(self.action_dim, -float("inf"))
            for token_id in valid_token_ids:
                if token_id < self.action_dim:
                    masked_q_values[token_id] = q_values[token_id]

            # Choose action with highest Q-value among valid actions
            best_action = np.argmax(masked_q_values)

            # Ensure the action is valid
            if best_action in valid_token_ids:
                return best_action
            else:
                return valid_token_ids[0]  # fallback to first valid action

    def replay(self) -> float:
        """
        Train the agent on a batch of experiences.

        Returns:
            float: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first, then to tensors for efficiency
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.training_step = checkpoint.get("training_step", 0)

    def set_eval_mode(self):
        """Set agent to evaluation mode (no exploration)."""
        self.epsilon = 0.0
        self.q_network.eval()

    def set_train_mode(self):
        """Set agent to training mode."""
        self.q_network.train()
