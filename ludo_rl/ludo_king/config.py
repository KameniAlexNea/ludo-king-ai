import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Config:
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

    FIXED_OPPONENTS_STEPS: int = int(os.getenv("FIXED_OPPONENTS_STEPS", 10_000))

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
    conv_configs: list[int] = field(default_factory=lambda: [32, 24])
    kernel_sizes: list[int] = field(default_factory=lambda: [7, 5])
    paddings: list[int] = field(default_factory=lambda: [3, 2])
    embed_dim: int = 128  # Output features dimension
    token_embed_dim: int = 16  # Embedding dimension for tokens
    pooled_output_size: int = 4
    pi: list[int] = field(default_factory=lambda: [128, 64])
    vf: list[int] = field(default_factory=lambda: [128, 64])


@dataclass(slots=True)
class StrategyConfig:
    board_channel_my: int = 0
    board_channel_safe: int = 4
    board_channel_opp_start: int = 1
    board_channel_opp_end: int = 3

    main_track_end: int = 51
    home_start: int = 52
    home_finish: int = 57


COEF = 5


@dataclass(slots=True)
class Reward:
    win: float = 50
    lose: float = -50
    finish: float = 1 * COEF
    capture: float = 0.2 * COEF
    got_capture: float = -0.5 * COEF
    blockade: float = 0.15 * COEF
    hit_blockade: float = -0.1 * COEF
    blockade_hit: float = 0.1 * COEF  # Bonus when opponent hits your blockade
    exit_home: float = 0.1 * COEF
    progress: float = 0.01
    safe_position: float = 0.05 * COEF
    draw: float = -2 * COEF
    skipped_turn: float = -0.01


config = Config()
net_config = NetworkConfig()
strategy_config = StrategyConfig()
reward_config = Reward()
