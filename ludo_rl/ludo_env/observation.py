from typing import List, Dict, Any

import numpy as np
# NOTE: These domain classes (LudoGame, EnvConfig, PlayerColor, etc.) 
# are assumed to be defined in your ludo_engine and ludo_rl packages.
from ludo_engine.core import LudoGame 
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor 
from ludo_rl.config import EnvConfig 
from ludo_rl.utils.reward_calculator import token_progress, token_progress_pos


class ObservationBuilderBase:
    """Base class providing common utilities for Ludo RL observation."""
    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self.start_pos = BoardConstants.START_POSITIONS[self.agent_color]
        # Total path size (51 main + 6 home column positions)
        self.total_path = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        self.agent_player = next(
            p for p in self.game.players if p.color == self.agent_color
        )
        # Cache colors present in the game 
        self.present_colors = {p.color for p in self.game.players}

    def build(self, turn_counter: int, dice: int) -> Dict[str, np.ndarray]:
        """Abstract method to build the observation dictionary."""
        raise NotImplementedError()

    def is_vulnerable(self, pos: int) -> bool:
        """Check if a position is not HOME and not a safe square."""
        return not (
            pos == GameConstants.HOME_POSITION or BoardConstants.is_safe_position(pos)
        )


class ContinuousObservationBuilder(ObservationBuilderBase):
    """
    Builds a structured continuous observation (float32 dictionary) suitable 
    for Stable Baselines3's MultiInputPolicy.
    """
    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor):
        super().__init__(cfg, game, agent_color)

    def build(self, turn_counter: int, dice: int) -> Dict[str, np.ndarray]:
        """Returns observation as a dictionary of float32 NumPy arrays."""
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        # Agent color one-hot encoding
        agent_color_onehot = [0.0] * len(ALL_COLORS)
        agent_color_onehot[ALL_COLORS.index(self.agent_color)] = 1.0

        # Agent progress (normalized 0-1) - this replaces positions since they're redundant
        agent_progress = [token_progress(t.position, self.start_pos) for t in self.agent_player.tokens]
        
        # Agent vulnerable flags
        agent_vulnerable = [1.0 if self.is_vulnerable(t.position) else 0.0 for t in self.agent_player.tokens]

        # Opponents (ordered by seat position)
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1 :] + ALL_COLORS[:start_idx]
        opp_positions = []
        opp_active = []
        home_pos_normalized = 0

        for color in ordered:
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                start_pos = BoardConstants.START_POSITIONS[color]
                for t in p.tokens:
                    opp_positions.append(token_progress(t.position, start_pos))
                opp_active.append(1.0)
            else:
                for _ in range(tokens_per_player):
                    opp_positions.append(home_pos_normalized)
                opp_active.append(0.0)

        # Dice encoding
        if self.cfg.obs.include_dice_one_hot:
            dice_vec = [0.0] * 6
            if 1 <= dice <= 6:
                dice_vec[dice - 1] = 1.0
        else:
            dice_vec = [((dice - 1) / 6.0) if (1 <= dice <= 6) else 0.0]

        return {
            "agent_color": np.asarray(agent_color_onehot, dtype=np.float32),
            "agent_progress": np.asarray(agent_progress, dtype=np.float32),
            "agent_vulnerable": np.asarray(agent_vulnerable, dtype=np.float32),
            "opponents_positions": np.asarray(opp_positions, dtype=np.float32),
            "opponents_active": np.asarray(opp_active, dtype=np.float32),
            "dice": np.asarray(dice_vec, dtype=np.float32),
        }


class DiscreteObservationBuilder(ObservationBuilderBase):
    """
    Builds a structured discrete observation (int64 dictionary) suitable 
    for Stable Baselines3's MultiInputPolicy (using MultiDiscrete/Discrete spaces).
    """
    def __init__(self, cfg, game, agent_color):
        super().__init__(cfg, game, agent_color)

    def build(self, turn_counter: int, dice: int) -> Dict[str, np.ndarray]:
        """Returns observation as a dictionary of int64 NumPy arrays."""
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        # Agent color one-hot encoding (discrete version)
        agent_color_onehot = [0] * len(ALL_COLORS)
        agent_color_onehot[ALL_COLORS.index(self.agent_color)] = 1

        # Agent progress (discrete bins 0-10) - positions removed as redundant
        agent_progress = [
            token_progress_pos(t.position, self.start_pos)
            for t in self.agent_player.tokens
        ]
        
        # Agent vulnerable flags
        agent_vulnerable = [1 if self.is_vulnerable(t.position) else 0 for t in self.agent_player.tokens]

        # Opponents (ordered by seat position)
        opp_positions = []
        opp_active = []
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1:] + ALL_COLORS[:start_idx]
        home_pos_int = 0  # HOME maps to 0 in discrete

        for color in ordered:
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                start_pos = BoardConstants.START_POSITIONS[color]
                for t in p.tokens:
                    # Use discrete position mapping for opponents too
                    opp_positions.append(token_progress_pos(t.position, start_pos))
                opp_active.append(1)
            else:
                for _ in range(tokens_per_player):
                    opp_positions.append(home_pos_int)
                opp_active.append(0)

        # Dice (discrete value)
        dice_val = dice if (1 <= dice <= 6) else 0

        return {
            "agent_color": np.asarray(agent_color_onehot, dtype=np.int64),
            "agent_progress": np.asarray(agent_progress, dtype=np.int64),
            "agent_vulnerable": np.asarray(agent_vulnerable, dtype=np.int64),
            "opponents_positions": np.asarray(opp_positions, dtype=np.int64),
            "opponents_active": np.asarray(opp_active, dtype=np.int64),
            "dice": np.asarray([dice_val], dtype=np.int64),
        }


# Backward compatibility alias
ObservationBuilder = ContinuousObservationBuilder
