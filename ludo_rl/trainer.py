"""
Training pipeline for Ludo RL agents with better reward engineering and training logic.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from .config import TRAINING_CONFIG
from .model.dqn_model import LudoDQNAgent
from .online_env import OnlineLudoEnv
from .states import LudoStateEncoder
from .trainers.data_loader import DataLoader
from .trainers.evaluator import Evaluator
from .trainers.model_manager import ModelManager
from .trainers.sequence_builder import SequenceBuilder


class LudoRLTrainer:
    """Orchestrates the training process for the Ludo RL agent."""

    def __init__(
        self,
        use_prioritized_replay: bool = True,
        use_double_dqn: bool = True,
    ):
        """
        Initialize the trainer.

        Args:
            use_prioritized_replay: Whether to use prioritized experience replay.
            use_double_dqn: Whether to use Double DQN.
        """
        self.encoder = LudoStateEncoder()
        self.agent = LudoDQNAgent(
            state_dim=self.encoder.state_dim,
            max_actions=4,
            lr=TRAINING_CONFIG.LEARNING_RATE,
            gamma=TRAINING_CONFIG.GAMMA,
            epsilon=TRAINING_CONFIG.EPSILON_START,
            epsilon_decay=TRAINING_CONFIG.EPSILON_DECAY,
            epsilon_min=TRAINING_CONFIG.EPSILON_END,
            memory_size=TRAINING_CONFIG.MEMORY_SIZE,
            batch_size=TRAINING_CONFIG.BATCH_SIZE,
            target_update_freq=TRAINING_CONFIG.TARGET_UPDATE_FREQ,
            use_prioritized_replay=use_prioritized_replay,
            use_double_dqn=use_double_dqn,
        )
        self.game_data = DataLoader().load_from_hf()
        self.sequence_builder = SequenceBuilder(self.game_data, self.encoder)
        self.model_manager = ModelManager(self.agent, self.encoder)
        self.evaluator = Evaluator(self.agent, self.game_data, self.encoder)
        self.online_env = None  # type: ignore

        print(f"Loaded {len(self.game_data)} game decision records for training")
        print(f"Using state encoding with {self.encoder.state_dim} features")

    def train(
        self,
        epochs: int = 1000,
        save_freq: int = 100,
        model_save_path: str = "ludo_dqn_model.pth",
        validation_split: float = 0.1,
        early_stopping_patience: int = 50,
        **kwargs,
    ) -> Dict:
        """
        Train the RL agent with training loop and validation.

        Args:
            epochs: Number of training epochs
            save_freq: Frequency of model saving
            model_save_path: Path to save trained model
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            Dict: Training statistics
        """
        use_online = kwargs.get("online", False)
        if use_online:
            # Initialize online environment (single-agent vs random opponents)
            if self.online_env is None:
                self.online_env = OnlineLudoEnv(agent_color="red")
            sequences = []  # Not used in online mode
        else:
            sequences = self.sequence_builder.create_training_sequences()

        if not use_online and not sequences:
            print("No training sequences available!")
            return {}

        if not use_online:
            # Split data for validation
            val_size = int(len(sequences) * validation_split)
            train_sequences = sequences[:-val_size] if val_size > 0 else sequences
            val_sequences = sequences[-val_size:] if val_size > 0 else []

            print(
                f"Training on {len(train_sequences)} sequences, validating on {len(val_sequences)}"
            )

            # Populate experience replay buffer
            for sequence in train_sequences:
                for experience in sequence:
                    self.agent.remember(*experience)

            print(
                f"Experience buffer populated with {len(self.agent.memory)} experiences"
            )
        else:
            train_sequences = []
            val_sequences = []
            print("Online training mode: generating experience through self-play.")

        # Training loop with early stopping
        best_val_reward = float("-inf")
        patience_counter = 0

        self.model_manager.training_losses = []
        self.model_manager.training_rewards = []
        self.model_manager.training_accuracy = []

        # Prepare CSV logging
        import csv
        import os

        # Separate logs for online vs offline to avoid mixed headers
        log_filename = "training_log_online.csv" if use_online else "training_log.csv"
        log_path = os.path.join(os.path.dirname(model_save_path), log_filename)
        write_header = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        csv_writer = csv.writer(log_file)
        if write_header:
            if use_online:
                csv_writer.writerow(
                    [
                        "epoch",
                        "loss",
                        "recent_reward",
                        "policy_acc",
                        "epsilon",
                        "buffer_size",
                    ]
                )
            else:
                csv_writer.writerow(
                    [
                        "epoch",
                        "loss",
                        "recent_reward",
                        "policy_acc",
                        "val_reward",
                        "val_accuracy",
                        "epsilon",
                        "buffer_size",
                    ]
                )

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Adaptive batches per epoch based on buffer size
            buffer_size = len(self.agent.memory)
            if hasattr(self.agent.memory, "__len__"):
                batches_per_epoch = max(
                    1, min(10, buffer_size // (self.agent.batch_size * 5))
                )
            else:
                batches_per_epoch = max(1, buffer_size // (self.agent.batch_size * 10))

            # Online data generation (if enabled)
            if use_online:
                episodes_per_epoch = kwargs.get("episodes_per_epoch", 5)
                max_steps = kwargs.get("max_steps_per_episode", 200)
                for ep in range(episodes_per_epoch):
                    state = self.online_env.reset()
                    done = False
                    steps = 0
                    # prepare agent turn
                    state, valid_moves = self.online_env.agent_turn_prepare()
                    while not done and steps < max_steps:
                        # choose action
                        action_idx = self.agent.act(state, valid_moves)
                        next_state, reward, done = self.online_env.step(action_idx)
                        # store
                        self.agent.remember(state, action_idx, reward, next_state, done)
                        state = next_state
                        valid_moves = (
                            self.online_env.get_current_valid_moves()
                            if not done
                            else []
                        )
                        steps += 1
            # Training batches from buffer
            for _ in range(batches_per_epoch):
                if (
                    hasattr(self.agent.memory, "__len__")
                    and len(self.agent.memory) >= self.agent.batch_size
                ):
                    loss = self.agent.replay()
                elif (
                    hasattr(self.agent.memory, "buffer")
                    and len(self.agent.memory.buffer) >= self.agent.batch_size
                ):
                    loss = self.agent.replay()
                else:
                    loss = 0.0
                if loss > 0:
                    epoch_loss += loss
                    num_batches += 1

            # Calculate metrics
            avg_loss = epoch_loss / max(num_batches, 1)
            self.model_manager.training_losses.append(avg_loss)

            # Offline dataset reward is mostly static; also compute policy accuracy vs stored actions
            recent_rewards = self._calculate_recent_performance()
            policy_acc = self._calculate_policy_accuracy(sample_size=512)
            self.model_manager.training_rewards.append(recent_rewards)

            # Validation
            val_reward = 0.0
            val_accuracy = 0.0
            if (
                not use_online and val_sequences and epoch % 10 == 0
            ):  # Validate every 10 epochs
                val_metrics = self.evaluator.evaluate_model(val_sequences)
                val_reward = val_metrics.get("avg_reward_per_sequence", 0)
                val_accuracy = val_metrics.get("accuracy", 0)
                self.model_manager.training_accuracy.append(val_accuracy)

                # Early stopping check
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    patience_counter = 0
                    # Save best model
                    self.model_manager.save_model(f"{model_save_path}.best")
                else:
                    patience_counter += 1

            # Progress reporting
            # Console progress (sparser)
            if epoch % 50 == 0:
                epsilon = self.agent.epsilon
                if use_online:
                    print(
                        f"Epoch {epoch}: loss={avg_loss:.4f} reward={recent_rewards:.2f} policy_acc={policy_acc:.3f} eps={epsilon:.3f} buffer={len(self.agent.memory)}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: loss={avg_loss:.4f} reward={recent_rewards:.2f} policy_acc={policy_acc:.3f} val_reward={val_reward:.2f} val_acc={val_accuracy:.3f} eps={epsilon:.3f}"
                    )

            # CSV log each epoch
            buffer_size = (
                len(self.agent.memory)
                if hasattr(self.agent.memory, "__len__")
                else len(self.agent.memory.buffer)
            )
            if use_online:
                csv_writer.writerow(
                    [
                        epoch,
                        f"{avg_loss:.6f}",
                        f"{recent_rewards:.4f}",
                        f"{policy_acc:.4f}",
                        f"{self.agent.epsilon:.4f}",
                        buffer_size,
                    ]
                )
            else:
                csv_writer.writerow(
                    [
                        epoch,
                        f"{avg_loss:.6f}",
                        f"{recent_rewards:.4f}",
                        f"{policy_acc:.4f}",
                        f"{val_reward:.4f}",
                        f"{val_accuracy:.4f}",
                        f"{self.agent.epsilon:.4f}",
                        buffer_size,
                    ]
                )

            # Early stopping
            if (
                not use_online
                and patience_counter >= early_stopping_patience
                and epoch > 100
            ):
                print(f"Early stopping at epoch {epoch} (patience: {patience_counter})")
                break

            # Save model periodically
            if epoch % save_freq == 0 and epoch > 0:
                self.model_manager.save_model(f"{model_save_path}.epoch_{epoch}")

        # Close log file
        log_file.close()

        # Save final model
        self.model_manager.save_model(model_save_path)

        print("Training completed!")

        return {
            "final_loss": (
                self.model_manager.training_losses[-1]
                if self.model_manager.training_losses
                else 0
            ),
            "final_reward": (
                self.model_manager.training_rewards[-1]
                if self.model_manager.training_rewards
                else 0
            ),
            "best_val_reward": best_val_reward,
            "total_experiences": (
                len(self.agent.memory)
                if hasattr(self.agent.memory, "__len__")
                else len(self.agent.memory.buffer)
            ),
            "num_sequences": len(train_sequences),
            "final_epsilon": self.agent.epsilon,
            "training_epochs": len(self.model_manager.training_losses),
            "mode": "online" if use_online else "offline",
        }

    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance from replay buffer."""
        if hasattr(self.agent.memory, "buffer"):
            # Prioritized replay buffer
            buffer = self.agent.memory.buffer
        else:
            # Regular deque
            buffer = self.agent.memory

        if len(buffer) > 0:
            recent_size = min(1000, len(buffer))
            recent_experiences = list(buffer)[-recent_size:]
            recent_rewards = [exp[2] for exp in recent_experiences]
            return np.mean(recent_rewards)
        return 0.0

    def _calculate_policy_accuracy(self, sample_size: int = 256) -> float:
        """Estimate policy accuracy vs stored actions in replay buffer."""
        if hasattr(self.agent.memory, "buffer"):
            data = self.agent.memory.buffer
        else:
            data = self.agent.memory
        if len(data) == 0:
            return 0.0
        sample_size = min(sample_size, len(data))
        sample = random.sample(list(data), sample_size)
        states = [s for (s, a, r, ns, d) in sample]
        actions = [a for (s, a, r, ns, d) in sample]
        import numpy as np
        import torch

        self.agent.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(states)).to(self.agent.device)
            q_vals = self.agent.q_network(state_tensor).cpu().numpy()
        predicted = q_vals.argmax(axis=1)
        correct = sum(1 for p, a in zip(predicted, actions) if p == a)
        self.agent.q_network.train()
        return correct / sample_size

    # Plotting removed in refactor (CSV logging instead)
    def plot_training_progress(self, save_path: str = "training_progress.png"):
        print("Plotting disabled. Use training_log.csv for analysis.")

    def save_model(self, path: str):
        """Save the trained model with metadata."""
        self.model_manager.save_model(path)

    def load_model(self, path: str):
        """Load a trained model with metadata."""
        self.model_manager.load_model(path)

    def evaluate_model(self, test_sequences: List[List[Tuple]] = None) -> Dict:
        """
        Evaluate the trained model with comprehensive metrics.

        Args:
            test_sequences: Test sequences (if None, uses 10% of training data)

        Returns:
            Dict: Evaluation metrics
        """
        return self.evaluator.evaluate_model(test_sequences)

    def cross_validate(self, k_folds: int = 5) -> List[Dict]:
        """
        Perform k-fold cross-validation.

        Args:
            k_folds: Number of folds for cross-validation

        Returns:
            List[Dict]: Results for each fold
        """
        sequences = self.sequence_builder.create_training_sequences()
        if len(sequences) < k_folds:
            print(f"Not enough sequences ({len(sequences)}) for {k_folds}-fold CV")
            return []

        fold_size = len(sequences) // k_folds
        results = []

        for fold in range(k_folds):
            print(f"Training fold {fold + 1}/{k_folds}")

            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(sequences)

            test_sequences = sequences[start_idx:end_idx]
            train_sequences = sequences[:start_idx] + sequences[end_idx:]

            # Create new trainer for this fold
            fold_trainer = LudoRLTrainer()

            # Populate experience replay buffer
            for sequence in train_sequences:
                for experience in sequence:
                    fold_trainer.agent.remember(*experience)

            # Train on fold
            fold_trainer.train(epochs=200, save_freq=1000)  # Shorter training for CV

            # Evaluate
            metrics = fold_trainer.evaluate_model(test_sequences)
            metrics["fold"] = fold
            results.append(metrics)

        return results
