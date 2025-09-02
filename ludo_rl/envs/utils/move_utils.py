"""General utilities for LudoGymEnv."""

from typing import Dict, List

import numpy as np

from ludo.constants import BoardConstants, GameConstants
from ludo.game import LudoGame
from ludo.player import Player

from ..model import EnvConfig


class MoveUtils:
    """General utility functions for moves and game state."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
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
        player = next(p for p in self.game.players if p.color.value == self.agent_color)
        total = 0.0
        for t in player.tokens:
            if t.position == -1:
                continue
            if 0 <= t.position < GameConstants.MAIN_BOARD_SIZE:
                # Main board progress: 0-51 normalized to 0-0.5
                total += t.position / float(
                    GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
                )
            elif t.position >= BoardConstants.HOME_COLUMN_START:
                # Home column progress: 100-105 mapped to 0.5-1.0 range
                home_progress = (t.position - BoardConstants.HOME_COLUMN_START) / float(
                    GameConstants.HOME_COLUMN_SIZE - 1
                )
                total += 0.5 + (home_progress * 0.5)  # Map to 0.5-1.0 range
        return total

    def action_masks(self, pending_valid_moves: List[Dict]) -> np.ndarray:
        mask = np.zeros(GameConstants.TOKENS_PER_PLAYER, dtype=np.int8)
        if pending_valid_moves:
            valid_ids = {m["token_id"] for m in pending_valid_moves}
            for i in range(GameConstants.TOKENS_PER_PLAYER):
                if i in valid_ids:
                    mask[i] = 1
        return mask

    def _make_strategy_context(
        self, player: Player, dice_value: int, valid_moves: List[Dict]
    ):
        # Basic context bridging existing strategies expecting a structure similar to tournaments
        board_state = self.game.board.get_board_state_for_ai(player)
        opponents = []
        for p in self.game.players:
            if p is player:
                continue
            opponents.append(p.get_game_state())
        ctx = {
            "player_state": player.get_game_state(),
            "board": board_state,
            "valid_moves": valid_moves,
            "dice_value": dice_value,
            "opponents": opponents,
        }
        return ctx
