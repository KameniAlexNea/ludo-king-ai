"""
Reinforcement Learning module for Ludo AI training.
Simplified RL tools for training and using a DQN-based Ludo agent.
"""

from .config import REWARDS, TRAINING_CONFIG

# Import core components
from .model.dqn_model import LudoDQN, LudoDQNAgent
from .rl_player import create_rl_strategy
from .states import LudoStateEncoder
from .trainer import LudoRLTrainer

__all__ = [
    # Core components
    "LudoStateEncoder",
    # DQN models
    "LudoDQN",
    "LudoDQNAgent",
    # Trainer
    "LudoRLTrainer",
    # Strategy factory
    "create_rl_strategy",
    # Configuration
    "REWARDS",
    "TRAINING_CONFIG",
]
