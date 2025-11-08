from dataclasses import dataclass
import os

@dataclass(slots=True)
class Config:
    # --- Constants ---
    PATH_LENGTH = 58  # 0=yard, 1-51=track, 52-56=home, 57=finished
    NUM_PLAYERS = 4
    PIECES_PER_PLAYER = 4
    MAX_TURNS = int(os.getenv("MAX_TURNS", 1000))

    # Absolute positions on the 52-square board
    PLAYER_START_SQUARES = [1, 14, 27, 40]  # Red, Green, Yellow, Blue
    SAFE_SQUARES_ABS = [1, 9, 14, 22, 27, 35, 40, 48]

    # Viz Variables
    HOME_ENTRY = [51, 12, 25, 38]  # Red, Green, Yellow, Blue
    HOME_COLUMN_ENTRIES = 52  # All enter home column at position 52
    HOME_COLUMN_SIZE = 6
    STAR_SQUARES = [9, 22,  35, 48]


@dataclass(slots=True)
class NetworkConfig:
    conv_configs = [32, 24]
    kernel_sizes = [7, 5]
    paddings = [3, 2]
    embed_dim = 128
    pooled_output_size = 4
    pi = [256, 128, 64]
    vf = [256, 128, 64]


@dataclass(slots=True)
class StrategyConfig:
    board_channel_my = 0
    board_channel_safe = 4
    board_channel_opp_start = 1
    board_channel_opp_end = 3

    main_track_end = 51
    home_start = 52
    home_finish = 57


config = Config()
net_config = NetworkConfig()
strategy_config = StrategyConfig()
