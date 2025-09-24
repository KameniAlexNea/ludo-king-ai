from typing import List

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor

from ludo_rl.config import EnvConfig
from ludo_rl.utils.reward_calculator import token_progress


class ObservationBuilder:
    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self.start_pos = BoardConstants.START_POSITIONS[self.agent_color]
        self.size = self.compute_size()
        self.total_path = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        self.agent_player = next(p for p in self.game.players if p.color == self.agent_color)

    def compute_size(self) -> int:
        base = 0
        base += 4  # agent token positions
        base += 4  # agent color one-hot
        base += 12  # opponents token positions
        base += 4  # agent token progresses
        base += 4  # agent token safety flags
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

    def build(self, turn_counter: int, dice: int) -> np.ndarray:
        obs: List[float] = []

        # agent color one-hot
        color_onehot = [0.0] * 4
        color_onehot[ALL_COLORS.index(self.agent_color)] = 1.0
        obs.extend(color_onehot)

        # agent and opponents tokens
        for p in self.game.players:
            for t in p.tokens:
                obs.append(self.normalize_pos(t.position))

        # token progresses and safety flags (only agent)
        for t in self.agent_player.tokens:
            obs.append(
                token_progress(
                    t.position, self.start_pos
                )
            )
            obs.append(self.is_vulnerable(t.position))

        # dice one-hot (1..6)
        d = [0.0] * 6
        if 1 <= dice <= 6:
            d[dice - 1] = 1.0
        obs.extend(d)

        if self.cfg.obs.include_turn_index:
            obs.append(min(1.0, turn_counter / float(self.cfg.max_turns)))

        return np.asarray(obs, dtype=np.float32)

    def is_vulnerable(self, pos: int) -> bool:
        return not (
            pos == GameConstants.HOME_POSITION or BoardConstants.is_safe_position(pos)
        )
