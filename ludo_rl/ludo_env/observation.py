from typing import List, Dict, Any

import numpy as np
# NOTE: These domain classes (LudoGame, EnvConfig, PlayerColor, etc.) 
# are assumed to be defined in your ludo_engine and ludo_rl packages.
from ludo_engine.core import LudoGame 
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor 
from ludo_rl.config import EnvConfig 
from ludo_rl.utils.reward_calculator import token_progress


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

    def normalize_pos(self, pos: int) -> float:
        """
        Normalizes a token's absolute position into a float in [-1.0, 1.0].
        HOME_POSITION maps to -1.0. Other positions are relative to the agent's start.
        """
        if pos == GameConstants.HOME_POSITION:
            return -1.0
        rank = token_progress(pos, self.start_pos)
        return rank * 2.0 - 1.0

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

        # 1. Agent-Specific Continuous Features
        agent_tokens = [self.normalize_pos(t.position) for t in self.agent_player.tokens]
        agent_progress = [token_progress(t.position, self.start_pos) for t in self.agent_player.tokens]
        agent_vulnerable = [1.0 if self.is_vulnerable(t.position) else 0.0 for t in self.agent_player.tokens]

        # 2. Opponent Continuous Features (Order matters for NN input consistency)
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1 :] + ALL_COLORS[:start_idx]
        opp_positions = []
        opp_active = []
        home_pos_normalized = self.normalize_pos(GameConstants.HOME_POSITION)

        for color in ordered:
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                for t in p.tokens:
                    opp_positions.append(self.normalize_pos(t.position))
                opp_active.append(1.0)
            else:
                for _ in range(tokens_per_player):
                    opp_positions.append(home_pos_normalized)
                opp_active.append(0.0)

        # 3. Dice Continuous/Categorical Features
        if self.cfg.obs.include_dice_one_hot:
            d = [0.0] * 6
            if 1 <= dice <= 6:
                d[dice - 1] = 1.0
            dice_vec = d
        else:
            # Normalized scalar [0.0, ~1.0]
            dice_vec = [((dice - 1) / 6.0)]

        # Final Structured Output
        return {
            "agent_tokens": np.asarray(agent_tokens, dtype=np.float32),
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
        obs: Dict[str, np.ndarray] = {}

        def pos_to_int(p: int) -> int:
            """
            Maps token position to an integer rank:
            HOME -> 0
            Path ranks -> 1..total_path
            """
            if p == GameConstants.HOME_POSITION:
                return 0
            
            if p >= BoardConstants.HOME_COLUMN_START:
                # Home column positions map to 52..57
                rank = GameConstants.MAIN_BOARD_SIZE + (p - BoardConstants.HOME_COLUMN_START)
            else:
                # Main board positions shifted relative to agent's start
                q = (
                    p - self.start_pos
                    if p >= self.start_pos
                    else GameConstants.MAIN_BOARD_SIZE - self.start_pos + p
                )
                rank = q
            
            # Since rank goes from 0 to 57, we map to 1 to 58 (total_path + 1 bins)
            return 1 + int(rank)

        # 1. Agent-Specific Discrete Features
        obs["agent_positions"] = np.asarray(
            [pos_to_int(t.position) for t in self.agent_player.tokens],
            dtype=np.int64
        )
        
        # Agent progress quantized to 0..10
        obs["agent_progress"] = np.asarray(
            [
                max(0, min(10, int(round(token_progress(t.position, self.start_pos) * 10.0)))) 
                for t in self.agent_player.tokens
            ],
            dtype=np.int64
        )
        
        # Agent vulnerable flag (0 or 1)
        obs["agent_vulnerable"] = np.asarray(
            [1 if self.is_vulnerable(t.position) else 0 for t in self.agent_player.tokens],
            dtype=np.int64
        )

        # 2. Opponent Discrete Features (Order matters)
        opp_positions = []
        opp_active = []
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1:] + ALL_COLORS[:start_idx]
        home_pos_int = pos_to_int(GameConstants.HOME_POSITION)

        for color in ordered:
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                for t in p.tokens:
                    opp_positions.append(pos_to_int(t.position))
                opp_active.append(1)
            else:
                # Pad with home positions for absent players
                for _ in range(tokens_per_player):
                    opp_positions.append(home_pos_int)
                opp_active.append(0)
        
        obs["opponents_positions"] = np.asarray(opp_positions, dtype=np.int64)
        obs["opponents_active"] = np.asarray(opp_active, dtype=np.int64)
        
        # 3. Dice Discrete Feature
        # Dice roll (0=no roll, 1..6)
        obs["dice"] = np.asarray([dice if (1 <= dice <= 6) else 0], dtype=np.int64)

        return obs


# Backward compatibility alias
ObservationBuilder = ContinuousObservationBuilder
