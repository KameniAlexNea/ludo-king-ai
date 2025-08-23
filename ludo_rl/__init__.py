"""
Reinforcement Learning module for Ludo AI training.
Provides enhanced tools for training RL agents on Ludo game data with modern techniques.
"""

from .config import REWARDS, TRAINING_CONFIG

# Import core components
from .model.dqn_model import LudoDQN, LudoDQNAgent
from .rl_player import create_rl_strategy
from .states import LudoStateEncoder
from .trainer import LudoRLTrainer
from .validator import LudoRLValidator, ModelComparator

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
    # validators optional
    "LudoRLValidator",
    "ModelComparator",
]
