"""
Training pipeline for Ludo RL agents.
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .dqn_model import LudoDQNAgent
from .state_encoder import LudoStateEncoder


class LudoRLTrainer:
    """Training pipeline for Ludo RL agents using saved game data."""

    def __init__(
        self, game_data_file: str = None, state_saver_dir: str = "saved_states"
    ):
        """
        Initialize the trainer.

        Args:
            game_data_file: Path to specific game data file (optional)
            state_saver_dir: Directory containing saved game states
        """
        self.encoder = LudoStateEncoder()
        self.agent = LudoDQNAgent(self.encoder.state_dim)

        # Load game data
        if game_data_file:
            self.game_data = self.load_game_data_file(game_data_file)
        else:
            self.game_data = self.load_all_game_data(state_saver_dir)

        # Training metrics
        self.training_losses = []
        self.training_rewards = []

        print(f"Loaded {len(self.game_data)} game decision records for training")

    def load_game_data_file(self, file_path: str) -> List[Dict]:
        """Load game data from a specific file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading game data from {file_path}: {e}")
            return []

    def load_all_game_data(self, save_dir: str) -> List[Dict]:
        """Load all game data from saved states directory."""
        all_data = []

        if not os.path.exists(save_dir):
            print(f"Save directory {save_dir} does not exist")
            return all_data

        # Get list of JSON files
        json_files = [f for f in os.listdir(save_dir) if f.endswith(".json")]
        
        if not json_files:
            print(f"No JSON files found in {save_dir}")
            return all_data

        print(f"Loading game data from {len(json_files)} files...")
        
        # Load files with progress bar
        for filename in tqdm(json_files, desc="Loading game files"):
            file_path = os.path.join(save_dir, filename)
            data = self.load_game_data_file(file_path)
            all_data.extend(data)

        return all_data

    def calculate_reward(self, game_data: Dict, next_game_data: Dict = None) -> float:
        """
        Calculate reward based on game outcome and strategic value.

        Args:
            game_data: Current game state data
            next_game_data: Next game state data (if available)

        Returns:
            float: Calculated reward
        """
        outcome = game_data.get("outcome", {})
        context = game_data["game_context"]

        reward = 0.0

        # Basic success/failure
        if outcome.get("success", False):
            reward += 1.0
        else:
            reward -= 2.0  # penalty for invalid moves

        # Strategic rewards
        captured_tokens = outcome.get("captured_tokens", [])
        if captured_tokens:
            reward += 10.0 * len(captured_tokens)

        if outcome.get("token_finished", False):
            reward += 25.0

        if outcome.get("extra_turn", False):
            reward += 3.0

        # Use strategic value from the game analysis
        valid_moves = context.get("valid_moves", [])
        chosen_move_idx = game_data.get("chosen_move", 0)

        if chosen_move_idx < len(valid_moves):
            chosen_move = valid_moves[chosen_move_idx]
            strategic_value = chosen_move.get("strategic_value", 0)
            reward += strategic_value * 0.1

            # Bonus for choosing the best strategic move
            strategic_analysis = context.get("strategic_analysis", {})
            best_move = strategic_analysis.get("best_strategic_move", {})
            if best_move and chosen_move.get("token_id") == best_move.get("token_id"):
                reward += 5.0

        # Game winning bonus
        player_state = context.get("player_state", {})
        if player_state.get("has_won", False):
            reward += 100.0

        # Progress reward: reward for advancing tokens
        if outcome.get("success", False):
            old_pos = outcome.get("old_position", -1)
            new_pos = outcome.get("new_position", -1)
            if old_pos != -1 and new_pos > old_pos:
                reward += (new_pos - old_pos) * 0.1

        return reward

    def create_training_sequences(self) -> List[List[Tuple]]:
        """
        Convert game data to training sequences.

        Returns:
            List[List[Tuple]]: List of game sequences, each containing (s, a, r, s', done) tuples
        """
        sequences = []
        current_sequence = []
        current_game_id = None

        for i, game_data in enumerate(self.game_data):
            # Check if this is a new game (simple heuristic based on timestamp or strategy changes)
            game_id = self._extract_game_id(game_data, i)

            if current_game_id is not None and game_id != current_game_id:
                # Start new sequence
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = []

            current_game_id = game_id

            # Encode current state
            state = self.encoder.encode_state(game_data)

            # Get action
            action = self.encoder.encode_action(
                game_data["chosen_move"], game_data["game_context"]["valid_moves"]
            )

            # Calculate reward
            reward = self.calculate_reward(game_data)

            # Determine next state and done flag
            if i < len(self.game_data) - 1:
                next_game_data = self.game_data[i + 1]
                next_game_id = self._extract_game_id(next_game_data, i + 1)

                if next_game_id == current_game_id:
                    next_state = self.encoder.encode_state(next_game_data)
                    done = False
                else:
                    next_state = state  # terminal state
                    done = True
            else:
                next_state = state  # terminal state
                done = True

            current_sequence.append((state, action, reward, next_state, done))

        # Add the final sequence
        if current_sequence:
            sequences.append(current_sequence)

        print(f"Created {len(sequences)} training sequences")
        return sequences

    def _extract_game_id(self, game_data: Dict, index: int) -> str:
        """Extract a game identifier from game data."""
        # Use timestamp patterns to identify games
        timestamp = game_data.get("timestamp", "")
        strategy = game_data.get("strategy", "unknown")

        # Simple heuristic: group by hour and strategy
        if timestamp:
            game_id = f"{strategy}_{timestamp[:13]}"  # YYYY-MM-DDTHH
        else:
            game_id = f"{strategy}_{index // 50}"  # Group by chunks

        return game_id

    def train(
        self,
        epochs: int = 1000,
        target_update_freq: int = 100,
        save_freq: int = 100,
        model_save_path: str = "ludo_dqn_model.pth",
    ) -> Dict:
        """
        Train the RL agent.

        Args:
            epochs: Number of training epochs
            target_update_freq: Frequency of target network updates
            save_freq: Frequency of model saving
            model_save_path: Path to save trained model

        Returns:
            Dict: Training statistics
        """
        sequences = self.create_training_sequences()

        if not sequences:
            print("No training sequences available!")
            return {}

        print(f"Training for {epochs} epochs on {len(sequences)} game sequences...")

        # Populate experience replay buffer
        for sequence in sequences:
            for experience in sequence:
                self.agent.remember(*experience)

        print(f"Experience buffer populated with {len(self.agent.memory)} experiences")

        # Training loop
        epoch_losses = []
        epoch_rewards = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_batches = 0

            # Train on multiple batches per epoch
            batches_per_epoch = max(
                1, len(self.agent.memory) // (self.agent.batch_size * 10)
            )

            for _ in range(batches_per_epoch):
                if len(self.agent.memory) >= self.agent.batch_size:
                    loss = self.agent.replay()
                    epoch_loss += loss
                    num_batches += 1

            # Calculate average loss
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                epoch_losses.append(avg_loss)

            # Calculate average reward from recent experiences in replay buffer
            if len(self.agent.memory) > 0:
                # Sample recent experiences to calculate current performance
                recent_size = min(1000, len(self.agent.memory))
                recent_experiences = list(self.agent.memory)[-recent_size:]
                recent_rewards = [exp[2] for exp in recent_experiences]
                avg_reward = np.mean(recent_rewards)
                epoch_rewards.append(avg_reward)

            # Progress reporting
            if epoch % 100 == 0:
                epsilon = self.agent.epsilon
                avg_loss_str = f"{avg_loss:.4f}" if num_batches > 0 else "N/A"
                avg_reward_str = f"{avg_reward:.2f}" if sequences else "N/A"
                print(
                    f"Epoch {epoch}/{epochs} - Loss: {avg_loss_str}, "
                    f"Avg Reward: {avg_reward_str}, Epsilon: {epsilon:.3f}"
                )

            # Save model periodically
            if epoch % save_freq == 0 and epoch > 0:
                self.save_model(f"{model_save_path}.epoch_{epoch}")

        # Save final model
        self.save_model(model_save_path)

        # Store training metrics
        self.training_losses = epoch_losses
        self.training_rewards = epoch_rewards

        print("Training completed!")

        return {
            "final_loss": epoch_losses[-1] if epoch_losses else 0,
            "final_reward": epoch_rewards[-1] if epoch_rewards else 0,
            "total_experiences": len(self.agent.memory),
            "num_sequences": len(sequences),
            "final_epsilon": self.agent.epsilon,
        }

    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot training metrics."""
        if not self.training_losses and not self.training_rewards:
            print("No training data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot losses
        if self.training_losses:
            ax1.plot(self.training_losses)
            ax1.set_title("Training Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.grid(True)

        # Plot rewards
        if self.training_rewards:
            ax2.plot(self.training_rewards)
            ax2.set_title("Average Reward")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Reward")
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Training progress saved to {save_path}")

    def save_model(self, path: str):
        """Save the trained model."""
        self.agent.save_model(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model."""
        self.agent.load_model(path)
        print(f"Model loaded from {path}")

    def evaluate_model(self, test_sequences: List[List[Tuple]] = None) -> Dict:
        """
        Evaluate the trained model on test data.

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

        for sequence in test_sequences:
            for state, true_action, reward, _, _ in sequence:
                # Get agent's action (without exploration)
                dummy_moves = [{"token_id": i} for i in range(4)]  # dummy valid moves
                predicted_action = self.agent.act(state, dummy_moves)

                if predicted_action == true_action:
                    correct_actions += 1

                total_reward += reward
                total_steps += 1

        accuracy = correct_actions / total_steps if total_steps > 0 else 0
        avg_reward = total_reward / len(test_sequences) if test_sequences else 0

        self.agent.set_train_mode()

        return {
            "accuracy": accuracy,
            "avg_reward_per_sequence": avg_reward,
            "total_test_steps": total_steps,
            "num_test_sequences": len(test_sequences),
        }
