from __future__ import annotations

from typing import List

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import BoardConstants, GameConstants, PlayerColor

from ludo_rl.config import EnvConfig


class ObservationBuilder:
    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self.start_pos = BoardConstants.START_POSITIONS[self.agent_color]
        self.size = self.compute_size()
        self.total_path = GameConstants.MAIN_BOARD_SIZE + (
            GameConstants.FINISH_POSITION - BoardConstants.HOME_COLUMN_START
        )

    def compute_size(self) -> int:
        base = 0
        base += 4  # agent token positions
        base += 12  # opponents token positions
        base += 4  # finished tokens per player
        base += 6  # dice one-hot
        if self.cfg.obs.include_turn_index:
            base += 1
        return base

    def normalize_pos(self, pos: int) -> float:
        if pos == GameConstants.HOME_POSITION:
            return -1.0
        if pos >= BoardConstants.HOME_COLUMN_START:
            # map 100..105 to 52..57
            rank = GameConstants.MAIN_BOARD_SIZE + (
                pos - BoardConstants.HOME_COLUMN_START
            )
        else:
            # shift so agent start is 0
            p = (
                pos - self.start_pos
                if pos >= self.start_pos
                else GameConstants.MAIN_BOARD_SIZE - self.start_pos + pos
            )
            rank = p
        # ranks 0..57 -> [-1, 1]
        return (
            rank / (GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE)
        ) * 2.0 - 1.0

    def token_progress(self, pos: int, start_pos: int) -> float:
        if pos == GameConstants.HOME_POSITION:
            return 0.0
        if pos >= BoardConstants.HOME_COLUMN_START:
            home_steps = (
                min(GameConstants.FINISH_POSITION, pos)
                - BoardConstants.HOME_COLUMN_START
            )
            return (GameConstants.MAIN_BOARD_SIZE + max(0, home_steps)) / float(
                self.total_path
            )
        # on main board: forward distance from start to current pos
        if pos >= start_pos:
            steps = pos - start_pos
        else:
            steps = GameConstants.MAIN_BOARD_SIZE - start_pos + pos
        return steps / float(self.total_path)

    def build(self, turn_counter: int, dice: int) -> np.ndarray:
        obs: List[float] = []

        # agent tokens
        agent = self.game.get_player_from_color(self.agent_color)
        for t in agent.tokens:
            obs.append(self.normalize_pos(t.position))

        # opponents tokens
        for p in self.game.players:
            if p.color == self.agent_color:
                continue
            for t in p.tokens:
                obs.append(self.normalize_pos(t.position))

        # average progress per player (smooth signal 0..1)
        for player in self.game.players:
            sp = BoardConstants.START_POSITIONS[player.color]
            prog_sum = 0.0
            for t in player.tokens:
                prog_sum += self.token_progress(t.position, sp)
            obs.append(prog_sum / float(GameConstants.TOKENS_PER_PLAYER))

        # dice one-hot (1..6)
        d = [0.0] * 6
        if 1 <= dice <= 6:
            d[dice - 1] = 1.0
        obs.extend(d)

        if self.cfg.obs.include_turn_index:
            obs.append(min(1.0, turn_counter / float(self.cfg.max_turns)))

        return np.asarray(obs, dtype=np.float32)
