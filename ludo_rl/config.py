"""
Configuration for improved Ludo RL training with better reward engineering.
"""


class REWARDS:
    """Improved reward configuration with better balance and scaling."""

    # Basic action rewards (scaled down for better learning stability)
    SUCCESS = 0.1
    FAILS = -0.5  # Moderate penalty for invalid moves

    # Strategic rewards with better scaling
    CAPTURE = 1.5  # Good reward for capturing
    TOKEN_FINISHED = 3.0  # High reward for finishing tokens
    EXTRA_TURN = 0.3
    BEST_MOVE = 0.5
    WON = 10.0

    # Progress and positioning rewards
    PROGRESS_WEIGHT = 0.2  # Reward for advancing tokens
    SAFETY_BONUS = 0.2  # Bonus for safe moves
    RISK_PENALTY = -0.2  # Penalty for risky moves
    FIRST_TOKEN_OUT = 1.0  # Bonus for getting first token out

    # Strategic weights
    STRATEGIC_WEIGHT = 0.02  # Weight for strategic value
    RELATIVE_POSITION_WEIGHT = 0.1  # Weight for relative positioning

    # Game phase bonuses
    EARLY_GAME_BONUS = 0.05  # Bonus for early progress
    END_GAME_BONUS = 0.2  # Bonus for late game positioning


class TRAINING_CONFIG:
    """Training hyperparameters and configuration."""

    # Network architecture
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    # Training parameters
    BATCH_SIZE = 64
    MEMORY_SIZE = 50000
    TARGET_UPDATE_FREQ = 100  # More frequent updates for online training

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    # Faster decay for online training - reach exploration minimum sooner
    EPSILON_DECAY = 0.9995  # applied per replay update

    # Discount factor
    GAMMA = 0.99

    # Advanced features
    USE_PRIORITIZED_REPLAY = True
    USE_DOUBLE_DQN = True
    USE_DUELING_DQN = True

    # Prioritized replay parameters
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA = 0.4

    # Gradient clipping
    GRAD_CLIP_NORM = 1.0
