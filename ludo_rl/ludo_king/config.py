import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Config:
    HISTORY_LENGTH: int = int(os.getenv("HISTORY_LENGTH", 12))
    RANK_ENV: bool = bool(int(os.getenv("RANK_ENV", 0)))
    MAX_EXTRA_TURNS: int = int(os.getenv("MAX_EXTRA_TURNS", 5))
    # --- Constants ---
    PATH_LENGTH: int = 58  # 0=yard, 1-51=track, 52-56=home, 57=finished
    NUM_PLAYERS: int = int(os.getenv("NUM_PLAYERS", 4))
    PIECES_PER_PLAYER: int = 4
    MAX_TURNS: int = int(os.getenv("MAX_TURNS", 1000))

    # Absolute positions on the 52-square board
    PLAYER_START_SQUARES: list[int] = field(
        default_factory=lambda: [1, 14, 27, 40]
    )  # Red, Green, Yellow, Blue
    SAFE_SQUARES_ABS: list[int] = field(
        default_factory=lambda: [1, 9, 14, 22, 27, 35, 40, 48]
    )

    # Viz Variables
    HOME_ENTRY: list[int] = field(
        default_factory=lambda: [51, 12, 25, 38]
    )  # Red, Green, Yellow, Blue
    HOME_COLUMN_ENTRIES: int = 52  # All enter home column at position 52
    HOME_COLUMN_SIZE: int = 6
    STAR_SQUARES: list[int] = field(default_factory=lambda: [9, 22, 35, 48])
    START_POSITION: int = 1

    # Derived (populated in __post_init__ due to slots)
    MAIN_TRACK_END: int = 0
    HOME_COLUMN_START: int = 0
    HOME_FINISH: int = 0

    FIXED_OPPONENTS_STEPS: int = int(os.getenv("FIXED_OPPONENTS_STEPS", 100))

    # Curriculum configuration
    CURRICULUM_TOTAL_TIMESTEPS: int = int(
        os.getenv("CURRICULUM_TOTAL_TIMESTEPS", 50_000_000)
    )

    # Derived positions (computed in __post_init__ for convenience)
    def __post_init__(self):
        # Main ring covers 1..51
        self.MAIN_TRACK_END = self.HOME_COLUMN_ENTRIES - 1
        # Home column starts at 52 and ends at 56
        self.HOME_COLUMN_START = self.HOME_COLUMN_ENTRIES
        self.HOME_FINISH = self.PATH_LENGTH - 1

        if self.NUM_PLAYERS < 2 or self.NUM_PLAYERS > 4:
            raise ValueError("NUM_PLAYERS must be between 2 and 4")


@dataclass(slots=True)
class NetworkConfig:
    embed_dim: int = int(os.getenv("EMBED_DIM", 128))  # Output features dimension
    token_embed_dim: int = int(
        os.getenv("TOKEN_EMBED_DIM", 16)
    )  # Embedding dimension for tokens
    pooled_output_size: int = 4
    pi: list[int] = field(
        default_factory=lambda: [int(x) for x in os.getenv("PI", "64").split(",")]
    )
    vf: list[int] = field(
        default_factory=lambda: [int(x) for x in os.getenv("VF", "64").split(",")]
    )
    # Transformer hyperparameters (configurable via env)
    trans_nhead: int = int(os.getenv("TRANS_NHEAD", 4))
    trans_num_layers: int = int(os.getenv("TRANS_NUM_LAYERS", 2))
    # Feed-forward layer size multiplier relative to token/embed dim
    trans_ff_mult: int = int(os.getenv("TRANS_FF_MULT", 3))


@dataclass(slots=True)
class StrategyConfig:
    board_channel_my: int = 0
    board_channel_safe: int = 4
    board_channel_opp_start: int = 1
    board_channel_opp_end: int = 3

    main_track_end: int = 51
    home_start: int = 52
    home_finish: int = 57


# Sparse rewards only - no shaping, no COEF scaling
# All intermediate signals removed to eliminate conflicting gradients
@dataclass(slots=True)
class Reward:
    # Terminal rewards (sparse)
    win: float = 5.0
    lose: float = -5.0
    draw: float = -1.0

    # Sparse milestone rewards
    finish: float = 0.5  # Piece reaches home finish
    capture: float = 0.1  # Capture an opponent's piece
    got_captured: float = -0.15  # Agent's piece got captured

    # Minor penalties (very small to avoid reward hacking)
    invalid_action: float = -0.01
    skipped_turn: float = 0.0  # No penalty for skipped turns (dice luck)


config = Config()
net_config = NetworkConfig()
strategy_config = StrategyConfig()
reward_config = Reward()

# Arena results: higher score = stronger opponent. Used by curriculum sampling.
ARENA_SCORES = {
    "defensive": 4962,
    "cautious": 4710,
    "homebody": 4371,
    "hoarder": 4276,
    "probability": 3582,
    "killer": 3528,
    "finish_line": 3455,
    "heatseeker": 3398,
    "rusher": 3343,
    "retaliator": 2994,
    "support": 981,
}
