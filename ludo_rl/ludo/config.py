class Config:
    # --- Constants ---
    PATH_LENGTH = 58  # 0=yard, 1-51=track, 52-56=home, 57=finished
    NUM_PLAYERS = 4
    PIECES_PER_PLAYER = 4
    MAX_TURNS = 200

    # Absolute positions on the 52-square board
    PLAYER_START_SQUARES = [1, 14, 27, 40]  # Blue, Red, Green, Yellow
    SAFE_SQUARES_ABS = [1, 9, 14, 22, 27, 35, 40, 48]


class NetworkConfig:
    conv_configs = [16, 32]
    kernel_sizes = [7, 5]
    paddings = [3, 2]
    embed_dim = 32
    pooled_output_size = 4
    pi = [32, 16]
    vf = [32, 16]


config = Config()
net_config = NetworkConfig()
