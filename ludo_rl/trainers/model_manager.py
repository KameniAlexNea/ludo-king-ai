"""
Handles model-related operations like saving, loading, and plotting training progress.
"""

import json
import os

import matplotlib.pyplot as plt

from ..config import TRAINING_CONFIG
from ..model.dqn_model import LudoDQNAgent
from ..model.ludo_dqn import LudoDQN


class ModelManager:
    """Manages the DQN model."""

    def __init__(self, agent: LudoDQNAgent, encoder: LudoDQN):
        self.agent = agent
        self.encoder = encoder
        self.training_losses = []
        self.training_rewards = []
        self.training_accuracy = []

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
