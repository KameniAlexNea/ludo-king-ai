"""
Improved Deep Q-Network implementation for Ludo RL training with Dueling DQN architecture.
"""

import random
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

from .ludo_buffer import PrioritizedReplayBuffer
from .ludo_dqn import LudoDQN


class LudoDQNAgent:
    """Enhanced DQN Agent with Dueling DQN, Prioritized Replay, and Double DQN."""

    def __init__(
        self,
        state_dim: int,
        max_actions: int = 4,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_prioritized_replay: bool = True,
        use_double_dqn: bool = True,
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim: Dimensionality of state space
            max_actions: Maximum number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
            memory_size: Size of experience replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Frequency of target network updates
            use_prioritized_replay: Whether to use prioritized experience replay
            use_double_dqn: Whether to use Double DQN
        """
        self.state_dim = state_dim
        self.max_actions = max_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_prioritized_replay = use_prioritized_replay
        self.use_double_dqn = use_double_dqn
        self.training_step = 0

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Neural networks
        self.q_network = LudoDQN(state_dim, max_actions).to(self.device)
        self.target_network = LudoDQN(state_dim, max_actions).to(self.device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=lr, weight_decay=1e-5
        )

        # Experience replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
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
        """Store experience in replay buffer."""
        experience = (state, action, reward, next_state, done)

        if self.use_prioritized_replay:
            self.memory.push(experience)
        else:
            self.memory.append(experience)

    def act(self, state: np.ndarray, valid_moves: List[Dict]) -> int:
        """
        Choose action using improved epsilon-greedy policy with move evaluation.

        Args:
            state: Current state
            valid_moves: List of valid moves available

        Returns:
            int: Action index (move index in valid_moves)
        """
        if not valid_moves:
            return 0

        # Epsilon-greedy exploration
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, len(valid_moves))

        # Evaluate each valid move
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get Q-values for all possible actions
            q_values = self.q_network(state_tensor).cpu().data.numpy()[0]

            # For each valid move, calculate its value
            move_values = []
            for i, move in enumerate(valid_moves):
                # Use a simple heuristic: combine Q-value with strategic value
                token_id = move.get("token_id", 0)
                base_q_value = q_values[min(token_id, self.max_actions - 1)]

                # Add strategic bonus
                strategic_value = move.get("strategic_value", 0) / 30.0  # Normalize
                safety_bonus = 0.1 if move.get("is_safe_move", True) else -0.1
                capture_bonus = 0.2 if move.get("captures_opponent", False) else 0.0

                total_value = (
                    base_q_value + strategic_value + safety_bonus + capture_bonus
                )
                move_values.append(total_value)

            # Choose best move
            best_move_idx = np.argmax(move_values)
            return best_move_idx

    def replay(self) -> float:
        """Train the agent on a batch of experiences using improved learning."""
        if self.use_prioritized_replay:
            if len(self.memory) < self.batch_size:
                return 0.0

            # Sample from prioritized replay buffer
            experiences, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.memory) < self.batch_size:
                return 0.0

            # Sample from regular replay buffer
            experiences = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # Convert to tensors
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(
            self.device
        )
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).to(
            self.device
        )
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).to(
            self.device
        )
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(
            self.device
        )
        dones = torch.BoolTensor(np.array([e[4] for e in experiences])).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).max(1)[1]
                next_q_values = (
                    self.target_network(next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze()
                )
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute weighted loss
        td_errors = target_q_values - current_q_values.squeeze()
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            td_errors_np = td_errors.abs().cpu().data.numpy()
            self.memory.update_priorities(indices, td_errors_np)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def save_model(self, filepath: str):
        """Save the trained model with additional metadata."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "state_dim": self.state_dim,
                "max_actions": self.max_actions,
                "use_prioritized_replay": self.use_prioritized_replay,
                "use_double_dqn": self.use_double_dqn,
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
        return self

    def set_train_mode(self):
        self.q_network.train()
        return self
