"""Shared utilities and base class for Ludo observation builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor

from ...configs.config import EnvConfig

_TOTAL_PATH = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
_DICE_RANGE = range(GameConstants.DICE_MIN, GameConstants.DICE_MAX + 1)


def _progress_fraction(position: int, start_pos: int) -> float:
    """Return progress in [0, 1] relative to start square."""
    if _is_home(position):
        return 0.0
    if _is_finished(position):
        return 1.0
    if position >= GameConstants.HOME_COLUMN_START:
        steps = GameConstants.MAIN_BOARD_SIZE + (
            position - GameConstants.HOME_COLUMN_START + 1
        )
    else:
        steps = (position - start_pos) % GameConstants.MAIN_BOARD_SIZE
    return float(steps) / float(_TOTAL_PATH)


def _is_on_main_board(position: int) -> bool:
    return 0 <= position < GameConstants.MAIN_BOARD_SIZE


def _is_finished(position: int) -> bool:
    return position == GameConstants.FINISH_POSITION


def _is_home(position: int) -> bool:
    return position == GameConstants.HOME_POSITION


def _threat_steps_to_agent(
    game: LudoGame, agent_color: PlayerColor, token_position: int
) -> int | None:
    """Return minimal forward steps opponents need to capture a token."""
    if not _is_on_main_board(token_position):
        return None

    min_steps: int | None = None
    for player in game.players:
        if player.color == agent_color:
            continue
        for token in player.tokens:
            opponent_pos = token.position
            if not _is_on_main_board(opponent_pos):
                continue
            steps = (token_position - opponent_pos) % GameConstants.MAIN_BOARD_SIZE
            if steps == 0 or steps > GameConstants.DICE_MAX:
                continue
            min_steps = steps if min_steps is None else min(min_steps, steps)
    return min_steps


@dataclass
class ObservationBuilderBase:
    cfg: EnvConfig
    game: LudoGame
    agent_color: PlayerColor

    def __post_init__(self) -> None:
        self._agent = self.game.get_player_from_color(self.agent_color)
        self._start_pos = BoardConstants.START_POSITIONS[self.agent_color]
        self._present_colors = {p.color for p in self.game.players}
        self._opponent_colors = [
            color for color in self._present_colors if color != self.agent_color
        ]

    def build(self, dice: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def _ordered_opponent_colors(agent_color: PlayerColor) -> Iterable[PlayerColor]:
        pivot = ALL_COLORS.index(agent_color)
        reorder: List[PlayerColor] = list(ALL_COLORS[pivot + 1 :] + ALL_COLORS[:pivot])
        return reorder

    @staticmethod
    def _is_vulnerable(position: int) -> bool:
        return not (_is_home(position) or BoardConstants.is_safe_position(position))


__all__ = [
    "ObservationBuilderBase",
    "_DICE_RANGE",
    "_is_finished",
    "_is_home",
    "_is_on_main_board",
    "_progress_fraction",
    "_threat_steps_to_agent",
]
