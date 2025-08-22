"""
Training pipeline for Ludo RL agents with better reward engineering and training logic.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple
import datasets

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .config import REWARDS, TRAINING_CONFIG
from .model.dqn_model import LudoDQNAgent
from .state_encoder import LudoStateEncoder


class LudoRLTrainer:
    """Enhanced training pipeline for Ludo RL agents with better learning algorithms."""

    def __init__(
        self,
        game_data_file: str = None,
        state_saver_dir: str = "saved_states",
        use_prioritized_replay: bool = True,
        use_double_dqn: bool = True,
    ):
        """
        Initialize the trainer.

        Args:
            game_data_file: Path to specific game data file (optional)
            state_saver_dir: Directory containing saved game states
            use_prioritized_replay: Whether to use prioritized experience replay
            use_double_dqn: Whether to use Double DQN
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

        # Load game data
        self.game_data = self.load_from_hf(state_saver_dir)

        # Training metrics
        self.training_losses = []
        self.training_rewards = []
        self.training_accuracy = []

        print(f"Loaded {len(self.game_data)} game decision records for training")
        print(f"Using state encoding with {self.encoder.state_dim} features")
    
    def load_from_hf(self, repo_id = "alexneakameni/ludo-king-rl"):
        ds = datasets.load_dataset(repo_id, split="train")
        print("Dataset Loaded")
        return ds.to_list()

    def calculate_reward(
        self, game_data: Dict, game_index: int, next_game_data: Dict = None
    ) -> float:
        """
        Calculate reward with better engineering and game context awareness.

        Args:
            game_data: Current game state data
            game_index: Index of current game in sequence
            next_game_data: Next game state data (if available)

        Returns:
            float: Calculated reward
        """
        outcome = game_data.get("outcome", {})
        context = game_data["game_context"]
        player_state = context["player_state"]
        current_situation = context["current_situation"]
        valid_moves = context.get("valid_moves", [])
        strategic_analysis = context.get("strategic_analysis", {})

        reward = 0.0

        # Basic success/failure with higher penalties for invalid moves
        if outcome.get("success", False):
            reward += REWARDS.SUCCESS
        else:
            reward += REWARDS.FAILS  # Stronger penalty for invalid moves

        # Strategic rewards with better scaling
        captured_tokens = outcome.get("captured_tokens", [])
        if captured_tokens:
            reward += REWARDS.CAPTURE * len(captured_tokens)

        if outcome.get("token_finished", False):
            reward += REWARDS.TOKEN_FINISHED

        if outcome.get("extra_turn", False):
            reward += REWARDS.EXTRA_TURN

        # Progress-based rewards with better scaling
        old_pos = outcome.get("old_position", -1)
        new_pos = outcome.get("new_position", -1)
        if outcome.get("success", False) and old_pos != -1 and new_pos != old_pos:
            if new_pos > old_pos:
                progress = (new_pos - old_pos) / 52.0
                reward += progress * REWARDS.PROGRESS_WEIGHT
            elif old_pos > 40 and new_pos < 10:  # Wrapped around the board
                progress = (52 - old_pos + new_pos) / 52.0
                reward += progress * REWARDS.PROGRESS_WEIGHT

        # Strategic value integration
        chosen_move_idx = game_data.get("chosen_move", 0)
        if chosen_move_idx < len(valid_moves):
            chosen_move = valid_moves[chosen_move_idx]
            strategic_value = chosen_move.get("strategic_value", 0)
            reward += strategic_value * REWARDS.STRATEGIC_WEIGHT

            # Safety and risk assessment
            if chosen_move.get("is_safe_move", True):
                reward += REWARDS.SAFETY_BONUS
            else:
                reward += REWARDS.RISK_PENALTY

            # Best move bonus
            best_move = strategic_analysis.get("best_strategic_move", {})
            if best_move and chosen_move.get("token_id") == best_move.get("token_id"):
                reward += REWARDS.BEST_MOVE

        # Game phase awareness
        active_tokens = player_state.get("active_tokens", 0)
        finished_tokens = player_state.get("finished_tokens", 0)
        tokens_in_home = player_state.get("tokens_in_home", 0)

        # First token out bonus
        if old_pos == -1 and new_pos >= 0 and active_tokens == 1:
            reward += REWARDS.FIRST_TOKEN_OUT

        # Game phase bonuses
        total_progress = active_tokens + finished_tokens
        if total_progress <= 1:  # Early game
            reward += REWARDS.EARLY_GAME_BONUS
        elif finished_tokens >= 2:  # End game
            reward += REWARDS.END_GAME_BONUS

        # Relative positioning reward
        opponents = context.get("opponents", [])
        if opponents:
            # Calculate relative advantage
            max_opp_finished = max(
                (opp.get("tokens_finished", 0) for opp in opponents), default=0
            )
            relative_advantage = finished_tokens - max_opp_finished
            reward += relative_advantage * REWARDS.RELATIVE_POSITION_WEIGHT

        # Game winning bonus
        if player_state.get("has_won", False):
            reward += REWARDS.WON

        # Turn momentum bonus
        consecutive_sixes = current_situation.get("consecutive_sixes", 0)
        if consecutive_sixes > 0:
            reward += min(consecutive_sixes, 2) * REWARDS.EXTRA_TURN * 0.5

        # Penalty for stalling in home when possible to move out
        if (
            current_situation.get("dice_value", 1) == 6
            and tokens_in_home > 0
            and chosen_move_idx < len(valid_moves)
            and valid_moves[chosen_move_idx].get("move_type") != "exit_home"
        ):
            reward -= 1.0  # Penalty for not using 6 to exit home

        return reward

    def create_training_sequences(self) -> List[List[Tuple]]:
        """
        Convert game data to training sequences with precise game boundary detection using exp_id.
        
        Uses exp_id (filename) to reliably identify game boundaries, which is much more accurate
        than timestamp or turn-based heuristics since all records from the same game file 
        belong to the same game session.

        Returns:
            List[List[Tuple]]: List of game sequences, each containing (s, a, r, s', done) tuples
        """
        sequences = []
        current_sequence = []
        last_exp_id = None

        for i, game_data in enumerate(self.game_data):
            context = game_data["game_context"]
            current_player = context["current_situation"]["player_color"]
            current_exp_id = game_data.get("exp_id", "")

            # Game boundary detection using exp_id
            is_new_game = False
            if last_exp_id is not None and current_exp_id != last_exp_id:
                is_new_game = True

            if is_new_game and current_sequence:
                # Add sequence completion rewards
                self._add_sequence_completion_rewards(current_sequence)
                sequences.append(current_sequence)
                current_sequence = []

            # Encode current state
            state = self.encoder.encode_state(game_data)

            # Get action (use move index instead of token ID)
            action_idx = game_data.get("chosen_move", 0)
            valid_moves = context.get("valid_moves", [])
            if action_idx >= len(valid_moves):
                action_idx = 0

            # Calculate reward
            reward = self.calculate_reward(game_data, i)

            # Determine next state and done flag
            next_state, done = self._get_next_state_and_done(i, current_player)

            current_sequence.append((state, action_idx, reward, next_state, done))

            # Update tracking variables
            last_exp_id = current_exp_id

        # Add the final sequence
        if current_sequence:
            self._add_sequence_completion_rewards(current_sequence)
            sequences.append(current_sequence)

        print(f"Created {len(sequences)} training sequences")
        avg_length = np.mean([len(seq) for seq in sequences]) if sequences else 0
        print(f"Average sequence length: {avg_length:.1f}")

        return sequences

    def _add_sequence_completion_rewards(self, sequence: List[Tuple]):
        """Add completion rewards to the sequence based on final outcome."""
        if not sequence:
            return

        # Get final state info from the last experience
        final_state, final_action, final_reward, _, _ = sequence[-1]

        # Add modest completion bonus/penalty
        completion_bonus = 0.0
        if final_reward > 50:  # Likely won the game
            completion_bonus = 10.0
        elif final_reward < -10:  # Poor performance
            completion_bonus = -5.0

        # Apply completion bonus to last few moves
        for i in range(max(0, len(sequence) - 3), len(sequence)):
            state, action, reward, next_state, done = sequence[i]
            sequence[i] = (
                state,
                action,
                reward + completion_bonus * 0.1,
                next_state,
                done,
            )

    def _get_next_state_and_done(
        self, current_index: int, current_player: str
    ) -> Tuple[np.ndarray, bool]:
        """Get next state and done flag with logic using exp_id."""
        current_exp_id = self.game_data[current_index].get("exp_id", "")
        
        # Look ahead to find next state for same player in same game
        for j in range(current_index + 1, min(current_index + 20, len(self.game_data))):
            next_data = self.game_data[j]
            next_exp_id = next_data.get("exp_id", "")
            
            # If we've moved to a different game, episode is done
            if next_exp_id != current_exp_id:
                break
                
            next_context = next_data["game_context"]
            next_player = next_context["current_situation"]["player_color"]

            if next_player == current_player:
                # Found next state for same player in same game
                next_state = self.encoder.encode_state(next_data)
                return next_state, False

        # No next state found for same player in same game - episode done
        current_state = self.encoder.encode_state(self.game_data[current_index])
        return current_state, True

    def _extract_game_id(self, game_data: Dict, index: int) -> str:
        """Extract a game identifier from game data using exp_id."""
        # Use exp_id as the primary game identifier
        exp_id = game_data.get("exp_id", "")
        if exp_id:
            return exp_id
        
        # Fallback to index-based grouping if exp_id is missing
        return f"game_{index // 50}"

    def train(
        self,
        epochs: int = 1000,
        target_update_freq: int = None,
        save_freq: int = 100,
        model_save_path: str = "ludo_dqn_model.pth",
        validation_split: float = 0.1,
        early_stopping_patience: int = 50,
    ) -> Dict:
        """
        Train the RL agent with training loop and validation.

        Args:
            epochs: Number of training epochs
            target_update_freq: Frequency of target network updates (uses agent default if None)
            save_freq: Frequency of model saving
            model_save_path: Path to save trained model
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            Dict: Training statistics
        """
        sequences = self.create_training_sequences()

        if not sequences:
            print("No training sequences available!")
            return {}

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

        print(f"Experience buffer populated with {len(self.agent.memory)} experiences")

        # Training loop with early stopping
        best_val_reward = float("-inf")
        patience_counter = 0
        epoch_losses = []
        epoch_rewards = []
        epoch_accuracies = []

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

            # Training batches
            for _ in range(batches_per_epoch):
                if (
                    hasattr(self.agent.memory, "__len__")
                    and len(self.agent.memory) >= self.agent.batch_size
                ) or (
                    hasattr(self.agent.memory, "buffer")
                    and len(self.agent.memory.buffer) >= self.agent.batch_size
                ):
                    loss = self.agent.replay()
                    if loss > 0:  # Valid loss
                        epoch_loss += loss
                        num_batches += 1

            # Calculate metrics
            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)

            # Calculate training reward from recent experiences
            recent_rewards = self._calculate_recent_performance()
            epoch_rewards.append(recent_rewards)

            # Validation
            val_reward = 0.0
            val_accuracy = 0.0
            if val_sequences and epoch % 10 == 0:  # Validate every 10 epochs
                val_metrics = self.evaluate_model(val_sequences)
                val_reward = val_metrics.get("avg_reward_per_sequence", 0)
                val_accuracy = val_metrics.get("accuracy", 0)
                epoch_accuracies.append(val_accuracy)

                # Early stopping check
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    patience_counter = 0
                    # Save best model
                    self.save_model(f"{model_save_path}.best")
                else:
                    patience_counter += 1

            # Progress reporting
            if epoch % 20 == 0:
                epsilon = self.agent.epsilon
                print(
                    f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, "
                    f"Reward: {recent_rewards:.2f}, Val Reward: {val_reward:.2f}, "
                    f"Val Acc: {val_accuracy:.3f}, Epsilon: {epsilon:.3f}"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience and epoch > 100:
                print(f"Early stopping at epoch {epoch} (patience: {patience_counter})")
                break

            # Save model periodically
            if epoch % save_freq == 0 and epoch > 0:
                self.save_model(f"{model_save_path}.epoch_{epoch}")

        # Save final model
        self.save_model(model_save_path)

        # Store training metrics
        self.training_losses = epoch_losses
        self.training_rewards = epoch_rewards
        self.training_accuracy = epoch_accuracies

        print("Training completed!")

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else 0,
            "final_reward": epoch_rewards[-1] if epoch_rewards else 0,
            "best_val_reward": best_val_reward,
            "total_experiences": (
                len(self.agent.memory)
                if hasattr(self.agent.memory, "__len__")
                else len(self.agent.memory.buffer)
            ),
            "num_sequences": len(train_sequences),
            "final_epsilon": self.agent.epsilon,
            "training_epochs": len(epoch_losses),
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

    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot comprehensive training metrics."""
        if not self.training_losses and not self.training_rewards:
            print("No training data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot losses
        if self.training_losses:
            axes[0, 0].plot(self.training_losses)
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True)

        # Plot rewards
        if self.training_rewards:
            axes[0, 1].plot(self.training_rewards)
            axes[0, 1].set_title("Average Reward")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Reward")
            axes[0, 1].grid(True)

        # Plot accuracy if available
        if self.training_accuracy:
            axes[1, 0].plot(self.training_accuracy)
            axes[1, 0].set_title("Validation Accuracy")
            axes[1, 0].set_xlabel("Epoch (×10)")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].grid(True)

        # Plot epsilon decay
        if hasattr(self.agent, "epsilon"):
            epochs = len(self.training_losses)
            epsilon_values = [
                TRAINING_CONFIG.EPSILON_START * (TRAINING_CONFIG.EPSILON_DECAY**i)
                for i in range(epochs)
            ]
            epsilon_values = [
                max(e, TRAINING_CONFIG.EPSILON_END) for e in epsilon_values
            ]

            axes[1, 1].plot(epsilon_values)
            axes[1, 1].set_title("Epsilon Decay")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Epsilon")
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training progress saved to {save_path}")

    def save_model(self, path: str):
        """Save the trained model with metadata."""
        self.agent.save_model(path)

        # Save training metadata
        metadata = {
            "state_dim": self.encoder.state_dim,
            "training_losses": self.training_losses,
            "training_rewards": self.training_rewards,
            "training_accuracy": self.training_accuracy,
            "config": {
                "batch_size": TRAINING_CONFIG.BATCH_SIZE,
                "learning_rate": TRAINING_CONFIG.LEARNING_RATE,
                "gamma": TRAINING_CONFIG.GAMMA,
                "use_prioritized_replay": TRAINING_CONFIG.USE_PRIORITIZED_REPLAY,
                "use_double_dqn": TRAINING_CONFIG.USE_DOUBLE_DQN,
            },
        }

        metadata_path = path.replace(".pth", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model and metadata saved to {path}")

    def load_model(self, path: str):
        """Load a trained model with metadata."""
        self.agent.load_model(path)

        # Load training metadata if available
        metadata_path = path.replace(".pth", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.training_losses = metadata.get("training_losses", [])
            self.training_rewards = metadata.get("training_rewards", [])
            self.training_accuracy = metadata.get("training_accuracy", [])
            print(f"Model and metadata loaded from {path}")
        else:
            print(f"Model loaded from {path} (no metadata found)")

    def evaluate_model(self, test_sequences: List[List[Tuple]] = None) -> Dict:
        """
        Evaluate the trained model with comprehensive metrics.

        Args:
            test_sequences: Test sequences (if None, uses 10% of training data)

        Returns:
            Dict: Evaluation metrics
        """
        if test_sequences is None:
            # Use last 10% of sequences for testing
            all_sequences = self.create_training_sequences()
            test_size = max(1, len(all_sequences) // 10)
            test_sequences = all_sequences[-test_size:]

        self.agent.set_eval_mode()

        total_reward = 0.0
        total_steps = 0
        correct_actions = 0
        strategic_value_errors = []
        move_type_accuracy = {"exit_home": 0, "advance": 0, "capture": 0, "finish": 0}
        move_type_counts = {"exit_home": 0, "advance": 0, "capture": 0, "finish": 0}

        for sequence in test_sequences:
            sequence_reward = 0.0
            for state, true_action, reward, _, _ in sequence:
                # Create dummy valid moves for evaluation
                dummy_moves = [{"token_id": i, "strategic_value": 0} for i in range(4)]
                predicted_action = self.agent.act(state, dummy_moves)

                if predicted_action == true_action:
                    correct_actions += 1

                sequence_reward += reward
                total_steps += 1

            total_reward += sequence_reward

        accuracy = correct_actions / total_steps if total_steps > 0 else 0
        avg_reward_per_sequence = (
            total_reward / len(test_sequences) if test_sequences else 0
        )
        avg_reward_per_step = total_reward / total_steps if total_steps > 0 else 0

        self.agent.set_train_mode()

        return {
            "accuracy": accuracy,
            "avg_reward_per_sequence": avg_reward_per_sequence,
            "avg_reward_per_step": avg_reward_per_step,
            "total_test_steps": total_steps,
            "num_test_sequences": len(test_sequences),
            "total_reward": total_reward,
        }

    def cross_validate(self, k_folds: int = 5) -> List[Dict]:
        """
        Perform k-fold cross-validation.

        Args:
            k_folds: Number of folds for cross-validation

        Returns:
            List[Dict]: Results for each fold
        """
        sequences = self.create_training_sequences()
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
            fold_trainer.encoder = self.encoder
            fold_trainer.agent = LudoDQNAgent(
                state_dim=self.encoder.state_dim,
                max_actions=4,
                **TRAINING_CONFIG.__dict__,
            )

            # Convert sequences back to game data format (simplified)
            fold_trainer.game_data = self._sequences_to_game_data(train_sequences)

            # Train on fold
            fold_trainer.train(epochs=200, save_freq=1000)  # Shorter training for CV

            # Evaluate
            metrics = fold_trainer.evaluate_model(test_sequences)
            metrics["fold"] = fold
            results.append(metrics)

        return results

    def _sequences_to_game_data(self, sequences: List[List[Tuple]]) -> List[Dict]:
        """Convert sequences back to simplified game data format."""
        # This is a simplified conversion - in practice you'd want to preserve
        # the original game data structure
        game_data = []
        for seq_idx, sequence in enumerate(sequences):
            for step_idx, (state, action, reward, next_state, done) in enumerate(
                sequence
            ):
                # Create minimal game data entry
                game_data.append(
                    {
                        "game_context": {
                            "valid_moves": [{"token_id": i} for i in range(4)],
                            "player_state": {
                                "tokens": [{"position": -1} for _ in range(4)]
                            },
                            "current_situation": {
                                "dice_value": 1,
                                "player_color": "red",
                            },
                            "opponents": [],
                            "strategic_analysis": {},
                        },
                        "chosen_move": action,
                        "outcome": {"success": reward > 0},
                        "timestamp": f"2024-{seq_idx:02d}-{step_idx:02d}T12:00:00Z",
                    }
                )
        return game_data
