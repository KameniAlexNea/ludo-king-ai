"""General utilities for LudoGymEnv."""

from typing import List

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import BoardConstants, GameConstants, ValidMove

from ..models.model_base import BaseEnvConfig


class MoveUtils:
    """General utility functions for moves and game state."""

    def __init__(self, cfg: BaseEnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color

    def _roll_dice(self) -> int:
        # Use core game mechanics (seeding done via global random seed)
        return self.game.roll_dice()

    def _roll_new_agent_dice(self):
        dice_value = self._roll_dice()
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )
        valid_moves = self.game.get_valid_moves(agent_player, dice_value)
        return dice_value, valid_moves

    def _snapshot_agent_tokens(self) -> List[int]:
        player = next(p for p in self.game.players if p.color.value == self.agent_color)
        return [t.position for t in player.tokens]

    def _compute_agent_progress_sum(self) -> float:
        player = self.game.get_player_from_color(self.agent_color)
        start_pos = BoardConstants.START_POSITIONS.get(self.agent_color)
        total_path = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        total = 0.0
        for t in player.tokens:
            if t.position == GameConstants.HOME_POSITION:
                continue
            if t.position == GameConstants.FINISH_POSITION:
                total += 1.0
                continue
            if 0 <= t.position < GameConstants.MAIN_BOARD_SIZE:
                # Normalize relative to agent's start
                if t.position >= start_pos:
                    steps_done = t.position - start_pos
                else:
                    steps_done = GameConstants.MAIN_BOARD_SIZE - start_pos + t.position
                total += steps_done / total_path
            elif t.position >= BoardConstants.HOME_COLUMN_START:
                # Home column: add main board progress + home progress
                home_steps = t.position - BoardConstants.HOME_COLUMN_START
                total += (GameConstants.MAIN_BOARD_SIZE + home_steps) / total_path
        return total

    def action_masks(self, pending_valid_moves: List[ValidMove]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
        if pending_valid_moves:
            valid_ids = {m.token_id for m in pending_valid_moves}
            for i in range(GameConstants.TOKENS_PER_PLAYER):
                if i in valid_ids:
                    mask[i] = 1
        return mask
