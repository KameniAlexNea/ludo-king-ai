import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Config:
    HISTORY_LENGTH: int = int(os.getenv("HISTORY_LENGTH", 4))
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
    use_scheduler: bool = True
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


COEF = 5


@dataclass(slots=True)
class Reward:
    win: float = 50
    lose: float = -50
    finish: float = 1 * COEF
    capture: float = 0.2 * COEF
    got_capture: float = -0.5 * COEF
    blockade: float = 0.05 * COEF
    hit_blockade: float = -0.1 * COEF
    blockade_hit: float = 0.1 * COEF
    exit_home: float = 0.1 * COEF
    progress: float = 0.001
    safe_position: float = 0.05 * COEF
    draw: float = -2 * COEF
    skipped_turn: float = -0.001

    # Risk/Opportunity shaping (potential-based) parameters
    shaping_use: bool = bool(int(os.getenv("SHAPING_USE", 1)))
    shaping_alpha: float = float(os.getenv("SHAPING_ALPHA", 2.0))
    shaping_gamma: float = float(os.getenv("SHAPING_GAMMA", 0.99))
    ro_depth: int = int(
        os.getenv("RO_DEPTH", 3)
    )  # lookahead depth in plies (approximate)
    # Weights for potential components
    ro_w_progress: float = float(os.getenv("RO_W_PROGRESS", 0.3))
    ro_w_cap_opp: float = float(os.getenv("RO_W_CAP_OPP", 0.4))
    ro_w_cap_risk: float = float(os.getenv("RO_W_CAP_RISK", 0.6))
    ro_w_finish_opp: float = float(os.getenv("RO_W_FINISH_OPP", 0.3))

    # Opponent progress penalties (sparse signals to encourage urgency)
    opp_exit_home_penalty: float = float(
        os.getenv("OPP_EXIT_HOME_PENALTY", -0.05 * COEF)
    )
    opp_piece_finished_penalty: float = float(
        os.getenv("OPP_PIECE_FINISHED_PENALTY", -0.3 * COEF)
    )
    opp_win_penalty: float = float(os.getenv("OPP_WIN_PENALTY", -0.2 * COEF))


config = Config()
net_config = NetworkConfig()
strategy_config = StrategyConfig()
reward_config = Reward()
