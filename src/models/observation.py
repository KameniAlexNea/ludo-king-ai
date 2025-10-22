"""Observation builders for the minimal Ludo environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor

from models.config import EnvConfig

_TOTAL_PATH = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE


def _progress_fraction(position: int, start_pos: int) -> float:
    """Return progress in [0, 1] for the token relative to its start square."""
    if position == GameConstants.HOME_POSITION:
        return 0.0
    if position == GameConstants.FINISH_POSITION:
        return 1.0
    if position >= GameConstants.HOME_COLUMN_START:
        steps = GameConstants.MAIN_BOARD_SIZE + (
            position - GameConstants.HOME_COLUMN_START + 1
        )
    else:
        steps = (position - start_pos) % GameConstants.MAIN_BOARD_SIZE
    return float(steps) / float(_TOTAL_PATH)


def _progress_bucket(position: int, start_pos: int) -> int:
    """Discrete progress for MultiDiscrete style observations."""
    return int(round(_progress_fraction(position, start_pos) * _TOTAL_PATH))


@dataclass
class ObservationBuilderBase:
    cfg: EnvConfig
    game: LudoGame
    agent_color: PlayerColor

    def __post_init__(self) -> None:
        self._agent = self.game.get_player_from_color(self.agent_color)
        self._start_pos = BoardConstants.START_POSITIONS[self.agent_color]
        self._present_colors = {p.color for p in self.game.players}

    def build(self, dice: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def _ordered_opponent_colors(agent_color: PlayerColor) -> Iterable[PlayerColor]:
        pivot = ALL_COLORS.index(agent_color)
        reorder: List[PlayerColor] = list(ALL_COLORS[pivot + 1 :] + ALL_COLORS[:pivot])
        return reorder

    @staticmethod
    def _is_vulnerable(position: int) -> bool:
        return not (
            position == GameConstants.HOME_POSITION
            or BoardConstants.is_safe_position(position)
        )


@dataclass
class ContinuousObservationBuilder(ObservationBuilderBase):
    def build(self, dice: int) -> Dict[str, np.ndarray]:
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        agent_color_onehot = np.zeros(len(ALL_COLORS), dtype=np.float32)
        agent_color_onehot[ALL_COLORS.index(self.agent_color)] = 1.0

        agent_progress = np.array(
            [
                _progress_fraction(t.position, self._start_pos)
                for t in self._agent.tokens
            ],
            dtype=np.float32,
        )
        agent_vulnerable = np.array(
            [self._is_vulnerable(t.position) for t in self._agent.tokens],
            dtype=np.float32,
        )

        opp_positions: List[float] = []
        opp_active: List[float] = []
        for color in self._ordered_opponent_colors(self.agent_color):
            if color in self._present_colors:
                player = self.game.get_player_from_color(color)
                start_pos = BoardConstants.START_POSITIONS[color]
                opp_positions.extend(
                    _progress_fraction(t.position, start_pos) for t in player.tokens
                )
                opp_active.append(1.0)
            else:
                opp_positions.extend([0.0] * tokens_per_player)
                opp_active.append(0.0)

        dice_vec = np.zeros(6, dtype=np.float32)
        if 1 <= dice <= 6:
            dice_vec[dice - 1] = 1.0

        return {
            "agent_color": agent_color_onehot,
            "agent_progress": agent_progress,
            "agent_vulnerable": agent_vulnerable,
            "opponents_positions": np.array(opp_positions, dtype=np.float32),
            "opponents_active": np.array(opp_active, dtype=np.float32),
            "dice": dice_vec,
        }


@dataclass
class DiscreteObservationBuilder(ObservationBuilderBase):
    def build(self, dice: int) -> Dict[str, np.ndarray]:
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        agent_color_onehot = np.zeros(len(ALL_COLORS), dtype=np.int64)
        agent_color_index = ALL_COLORS.index(self.agent_color)
        agent_color_onehot[agent_color_index] = 1

        agent_progress = np.array(
            [_progress_bucket(t.position, self._start_pos) for t in self._agent.tokens],
            dtype=np.int64,
        )
        agent_vulnerable = np.array(
            [int(self._is_vulnerable(t.position)) for t in self._agent.tokens],
            dtype=np.int64,
        )

        opp_positions: List[int] = []
        opp_active: List[int] = []
        for color in self._ordered_opponent_colors(self.agent_color):
            if color in self._present_colors:
                player = self.game.get_player_from_color(color)
                start_pos = BoardConstants.START_POSITIONS[color]
                opp_positions.extend(
                    _progress_bucket(t.position, start_pos) for t in player.tokens
                )
                opp_active.append(1)
            else:
                opp_positions.extend([0] * tokens_per_player)
                opp_active.append(0)

        dice_val = dice if 1 <= dice <= 6 else 0

        return {
            "agent_color": agent_color_onehot,
            "agent_progress": agent_progress,
            "agent_vulnerable": agent_vulnerable,
            "opponents_positions": np.array(opp_positions, dtype=np.int64),
            "opponents_active": np.array(opp_active, dtype=np.int64),
            "dice": np.array([dice_val], dtype=np.int64),
        }


def make_observation_builder(
    cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor
) -> ObservationBuilderBase:
    """Factory helper that picks the continuous or discrete builder."""
    if cfg.obs.discrete:
        return DiscreteObservationBuilder(cfg, game, agent_color)
    return ContinuousObservationBuilder(cfg, game, agent_color)
