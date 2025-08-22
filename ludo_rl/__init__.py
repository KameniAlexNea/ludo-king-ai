"""
Reinforcement Learning module for Ludo AI training.
Provides enhanced tools for training RL agents on Ludo game data with modern techniques.
"""

from .config import REWARDS, TRAINING_CONFIG

# Import core components
from .dqn_model import LudoDQN, LudoDQNAgent
from .rl_player import LudoRLPlayer, create_rl_strategy
from .state_encoder import LudoStateEncoder
from .trainer import LudoRLTrainer

# Optional validator import (may not be needed in all environments)
try:
    from .validator import LudoRLValidator, ModelComparator

    _has_validator = True
except ImportError:
    _has_validator = False

__all__ = [
    # Core components
    "LudoStateEncoder",
    # DQN models
    "LudoDQN",
    "LudoDQNAgent",
    # Trainer
    "LudoRLTrainer",
    # Player
    "LudoRLPlayer",
    # Strategy factory
    "create_rl_strategy",
    # Configuration
    "REWARDS",
    "TRAINING_CONFIG",
]

# Add validator components if available
if _has_validator:
    __all__.extend(
        [
            "LudoRLValidator",
            "ModelComparator",
        ]
    )
