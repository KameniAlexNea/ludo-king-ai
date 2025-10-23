import numpy as np
from gymnasium import spaces
from ludo_engine.models import ALL_COLORS, GameConstants


def get_space_config():
    tokens = GameConstants.TOKENS_PER_PLAYER
    opponents = len(ALL_COLORS) - 1
    token_total_max = float(tokens)
    opponent_total_max = float(tokens * opponents)
    return spaces.Dict(
        {
            "agent_color": spaces.Box(
                low=0.0, high=1.0, shape=(len(ALL_COLORS),), dtype=np.float32
            ),
            "agent_progress": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_distance_to_finish": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_vulnerable": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_safe": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_home": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_on_board": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_capture_available": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_finish_available": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_threat_level": spaces.Box(
                low=0.0, high=1.0, shape=(tokens,), dtype=np.float32
            ),
            "agent_tokens_at_home": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "agent_tokens_finished": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "agent_tokens_on_safe": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "agent_total_progress": spaces.Box(
                low=0.0, high=token_total_max, shape=(1,), dtype=np.float32
            ),
            "opponents_positions": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(tokens * opponents,),
                dtype=np.float32,
            ),
            "opponents_active": spaces.Box(
                low=0.0, high=1.0, shape=(opponents,), dtype=np.float32
            ),
            "opponent_total_progress": spaces.Box(
                low=0.0,
                high=opponent_total_max,
                shape=(1,),
                dtype=np.float32,
            ),
            "opponent_best_progress": spaces.Box(
                low=0.0, high=token_total_max, shape=(1,), dtype=np.float32
            ),
            "opponent_tokens_at_home": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "opponent_tokens_finished": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "opponent_tokens_on_safe": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "progress_lead": spaces.Box(
                low=-token_total_max,
                high=token_total_max,
                shape=(1,),
                dtype=np.float32,
            ),
            "agent_rank": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "dice": spaces.Box(
                low=0.0, high=1.0, shape=(GameConstants.DICE_MAX,), dtype=np.float32
            ),
            "dice_value_norm": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "dice_is_six": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "dice_is_even": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "home_exit_ready": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "capture_any": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "finish_any": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
    )
