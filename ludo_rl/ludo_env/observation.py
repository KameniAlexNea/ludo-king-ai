from typing import List

import gymnasium as gym
import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor

from ludo_rl.config import EnvConfig
from ludo_rl.utils.reward_calculator import token_progress


class ObservationBuilderBase:
    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self.start_pos = BoardConstants.START_POSITIONS[self.agent_color]
        self.total_path = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        self.agent_player = next(
            p for p in self.game.players if p.color == self.agent_color
        )
        # cache which colors are present in this game to avoid unnecessary lookups
        self.present_colors = {p.color for p in self.game.players}

    def build(self, turn_counter: int, dice: int) -> np.ndarray:
        raise NotImplementedError()

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

    def is_vulnerable(self, pos: int) -> bool:
        return not (
            pos == GameConstants.HOME_POSITION or BoardConstants.is_safe_position(pos)
        )


class ContinuousObservationBuilder(ObservationBuilderBase):
    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor):
        super().__init__(cfg, game, agent_color)
        self.size = self.compute_size()

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
        base += max_players - 1
        base += 4  # agent token progresses
        base += 4  # agent token safety flags
        # dice encoding: either one-hot (6) or 1 scalar
        if self.cfg.obs.include_dice_one_hot:
            base += 6
        else:
            base += 1
        return base

    def build(self, turn_counter: int, dice: int):
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        # Agent tokens normalized positions
        agent_tokens = [self.normalize_pos(t.position) for t in self.agent_player.tokens]
        agent_progress = [token_progress(t.position, self.start_pos) for t in self.agent_player.tokens]
        agent_vulnerable = [1.0 if self.is_vulnerable(t.position) else 0.0 for t in self.agent_player.tokens]

        # Opponents aggregated seat-next single opponent (for simple example)
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1 :] + ALL_COLORS[:start_idx]
        # Aggregate across opponents by averaging positions into a 4-vector and a single active flag (1 if any present)
        opp_pos_accum = [0.0] * tokens_per_player
        opp_count = 0
        for color in ordered:
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                for i, t in enumerate(p.tokens):
                    opp_pos_accum[i] += self.normalize_pos(t.position)
                opp_count += 1
        if opp_count > 0:
            opp_positions = [v / float(opp_count) for v in opp_pos_accum]
            opp_active = [1.0]
        else:
            opp_positions = [self.normalize_pos(GameConstants.HOME_POSITION)] * tokens_per_player
            opp_active = [0.0]

        # Dice encoding
        if self.cfg.obs.include_dice_one_hot:
            d = [0.0] * 6
            if 1 <= dice <= 6:
                d[dice - 1] = 1.0
            dice_vec = d
        else:
            dice_vec = [((dice - 1) / 6.0) if (1 <= dice <= 6) else 0.0]

        return {
            "agent_tokens": np.asarray(agent_tokens, dtype=np.float32),
            "agent_progress": np.asarray(agent_progress, dtype=np.float32),
            "agent_vulnerable": np.asarray(agent_vulnerable, dtype=np.float32),
            "opponents": {
                "positions": np.asarray(opp_positions, dtype=np.float32),
                "active": np.asarray(opp_active, dtype=np.float32),
            },
            "dice": np.asarray(dice_vec, dtype=np.float32),
        }


class DiscreteObservationBuilder(ObservationBuilderBase):
    def __init__(self, cfg, game, agent_color):
        super().__init__(cfg, game, agent_color)
        self.size = self.compute_size()

    def compute_size(self) -> list:
        """Return the nvec list for a MultiDiscrete observation encoding."""
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER
        max_players = len(ALL_COLORS)
        dims: list = []

        # agent color scalar (0..max_players-1)
        dims.append(max_players)

        # agent token positions (0 = HOME, 1..total_path)
        pos_bins = self.total_path + 1
        for _ in range(tokens_per_player):
            dims.append(pos_bins)

        # opponents token positions and active flags
        for _ in range(max_players - 1):
            for _ in range(tokens_per_player):
                dims.append(pos_bins)
            dims.append(2)  # active flag

        # token progress (quantized) and vulnerable flag for agent tokens
        progress_bins = 11
        for _ in range(tokens_per_player):
            dims.append(progress_bins)
            dims.append(2)  # vulnerable flag

        # dice (0=no dice, 1..6)
        dims.append(7)

        return dims

    def build(self, turn_counter: int, dice: int) -> np.ndarray:
        """Build a MultiDiscrete-style integer observation vector.

        Mappings:
        - positions: HOME -> 0, main/home ranks -> 1..total_path
        - progress: quantized 0..10
        - vulnerable: 0/1
        - dice: 0..6 (0 means no dice rolled yet)
        - turn index: 0..100
        """
        obs: List[int] = []
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        # agent color scalar
        obs.append(ALL_COLORS.index(self.agent_color))

        # helper to map position -> int
        def pos_to_int(p: int) -> int:
            if p == GameConstants.HOME_POSITION:
                return 0
            if p >= BoardConstants.HOME_COLUMN_START:
                rank = GameConstants.MAIN_BOARD_SIZE + (
                    p - BoardConstants.HOME_COLUMN_START
                )
            else:
                q = (
                    p - self.start_pos
                    if p >= self.start_pos
                    else GameConstants.MAIN_BOARD_SIZE - self.start_pos + p
                )
                rank = q
            return 1 + int(rank)

        # agent tokens
        for t in self.agent_player.tokens:
            obs.append(pos_to_int(t.position))

        # opponents in seat-relative order
        start_idx = ALL_COLORS.index(self.agent_color)
        ordered = ALL_COLORS[start_idx + 1 :] + ALL_COLORS[:start_idx]
        for color in ordered:
            if color in self.present_colors:
                p = self.game.get_player_from_color(color)
                for t in p.tokens:
                    obs.append(pos_to_int(t.position))
                obs.append(1)
            else:
                for _ in range(tokens_per_player):
                    obs.append(pos_to_int(GameConstants.HOME_POSITION))
                obs.append(0)

        # token progress and vulnerable flags
        for t in self.agent_player.tokens:
            prog = int(round(token_progress(t.position, self.start_pos) * 10.0))
            prog = max(0, min(10, prog))
            obs.append(prog)
            obs.append(1 if self.is_vulnerable(t.position) else 0)

        # dice
        if 1 <= dice <= 6:
            obs.append(dice)
        else:
            obs.append(0)

        return np.asarray(obs, dtype=np.int64)


# Backward compatibility: alias ObservationBuilder to ContinuousObservationBuilder
ObservationBuilder = ContinuousObservationBuilder
