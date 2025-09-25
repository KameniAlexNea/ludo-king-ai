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
        self.agent_player = next(
            p for p in self.game.players if p.color == self.agent_color
        )
        # cache which colors are present in this game to avoid unnecessary lookups
        self.present_colors = {p.color for p in self.game.players}

    def compute_size(self) -> int:
        base = 0
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER
        max_players = len(ALL_COLORS)
        # color encoding: either one-hot (max_players) or 1 scalar
        if self.cfg.obs.include_color_one_hot:
            base += max_players
        else:
            base += 1
        base += tokens_per_player  # agent token positions
        # opponents token positions: (max_players - 1) * tokens_per_player
        base += (max_players - 1) * tokens_per_player
        # binary active flag for each opponent (1/0)
        base += (max_players - 1)
        base += 4  # agent token progresses
        base += 4  # agent token safety flags
        # dice encoding: either one-hot (6) or 1 scalar
        if self.cfg.obs.include_dice_one_hot:
            base += 6
        else:
            base += 1
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
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        # agent color encoding
        if self.cfg.obs.include_color_one_hot:
            color_onehot = [0.0] * len(ALL_COLORS)
            color_onehot[ALL_COLORS.index(self.agent_color)] = 1.0
            obs.extend(color_onehot)
        else:
            # normalized scalar in [0,1]
            obs.append(ALL_COLORS.index(self.agent_color) / float(len(ALL_COLORS)))

        # agent and opponents tokens
        # Write agent tokens first (agent_player.tokens)
        for t in self.agent_player.tokens:
            obs.append(self.normalize_pos(t.position))

        # For opponents, iterate colors in seat-relative order starting after agent
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1 :] + ALL_COLORS[:start_idx]
        for color in ordered:
            # find player with this color in the current game (may be absent in 2-player)
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                for t in p.tokens:
                    obs.append(self.normalize_pos(t.position))
                # active flag 1 for present player
                obs.append(1.0)
            else:
                # pad with home positions for each token
                for _ in range(tokens_per_player):
                    obs.append(self.normalize_pos(GameConstants.HOME_POSITION))
                # active flag 0 for absent player
                obs.append(0.0)

        # token progresses and safety flags (only agent)
        for t in self.agent_player.tokens:
            obs.append(token_progress(t.position, self.start_pos))
            # explicit float for boolean
            obs.append(1.0 if self.is_vulnerable(t.position) else 0.0)

        # dice encoding
        if self.cfg.obs.include_dice_one_hot:
            d = [0.0] * 6
            if 1 <= dice <= 6:
                d[dice - 1] = 1.0
            obs.extend(d)
        else:
            # normalized scalar in [0,1], map die face index -> index/6
            # we use (dice-1)/6 to mirror color index / len normalization style
            if 1 <= dice <= 6:
                obs.append((dice - 1) / float(6))
            else:
                obs.append(0.0)

        if self.cfg.obs.include_turn_index:
            obs.append(min(1.0, turn_counter / float(self.cfg.max_turns)))

        return np.asarray(obs, dtype=np.float32)

    def is_vulnerable(self, pos: int) -> bool:
        return not (
            pos == GameConstants.HOME_POSITION or BoardConstants.is_safe_position(pos)
        )
