"""Observation builders for the minimal Ludo environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from ludo_engine.core import LudoGame
from ludo_engine.models import ALL_COLORS, BoardConstants, GameConstants, PlayerColor

from .config import EnvConfig

_TOTAL_PATH = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
_DICE_RANGE = range(GameConstants.DICE_MIN, GameConstants.DICE_MAX + 1)


def _progress_fraction(position: int, start_pos: int) -> float:
    """Return progress in [0, 1] for the token relative to its start square."""
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
    """Return minimal forward steps opponents need to capture this token."""
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
            c for c in self._present_colors if c != self.agent_color
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


@dataclass
class ContinuousObservationBuilder(ObservationBuilderBase):
    def build(self, dice: int) -> Dict[str, np.ndarray]:
        tokens_per_player = GameConstants.TOKENS_PER_PLAYER

        agent_color_onehot = np.zeros(len(ALL_COLORS), dtype=np.float32)
        agent_color_onehot[ALL_COLORS.index(self.agent_color)] = 1.0

        agent_tokens = self._agent.tokens
        agent_positions = [token.position for token in agent_tokens]
        agent_progress = np.array(
            [_progress_fraction(pos, self._start_pos) for pos in agent_positions],
            dtype=np.float32,
        )
        agent_distance_to_finish = 1.0 - agent_progress

        agent_on_board = np.array(
            [_is_on_main_board(pos) for pos in agent_positions], dtype=np.float32
        )
        agent_home = np.array(
            [_is_home(pos) for pos in agent_positions], dtype=np.float32
        )
        agent_finished = np.array(
            [_is_finished(pos) for pos in agent_positions], dtype=np.float32
        )
        agent_safe = np.array(
            [BoardConstants.is_safe_position(pos) for pos in agent_positions],
            dtype=np.float32,
        )
        agent_vulnerable = 1.0 - agent_safe

        valid_moves = self.game.get_valid_moves(self._agent, dice)
        capture_flags = np.zeros(tokens_per_player, dtype=np.float32)
        finish_flags = np.zeros(tokens_per_player, dtype=np.float32)
        for move in valid_moves:
            index = move.token_id
            if move.captures_opponent:
                capture_flags[index] = 1.0
            if _is_finished(move.target_position):
                finish_flags[index] = 1.0

        threat_scores = np.zeros(tokens_per_player, dtype=np.float32)
        for idx, pos in enumerate(agent_positions):
            if not agent_vulnerable[idx]:
                continue
            steps = _threat_steps_to_agent(self.game, self.agent_color, pos)
            if steps is not None:
                threat_scores[idx] = (
                    GameConstants.DICE_MAX + 1 - steps
                ) / GameConstants.DICE_MAX

        agent_tokens_at_home = float(agent_home.mean())
        agent_tokens_finished = float(agent_finished.mean())
        agent_tokens_on_safe = float(agent_safe.mean())
        agent_total_progress = float(agent_progress.sum())

        opp_positions: List[float] = []
        opp_active: List[float] = []
        opponent_progress_totals: List[float] = []
        opponent_home_count = 0.0
        opponent_finished_count = 0.0
        opponent_safe_count = 0.0
        for color in self._ordered_opponent_colors(self.agent_color):
            if color in self._present_colors:
                player = self.game.get_player_from_color(color)
                start_pos = BoardConstants.START_POSITIONS[color]
                progresses = np.array(
                    [_progress_fraction(t.position, start_pos) for t in player.tokens],
                    dtype=np.float32,
                )
                opp_positions.extend(progresses.tolist())
                opp_active.append(1.0)
                opponent_progress_totals.append(float(progresses.sum()))
                opponent_home_count += sum(_is_home(t.position) for t in player.tokens)
                opponent_finished_count += sum(
                    _is_finished(t.position) for t in player.tokens
                )
                opponent_safe_count += sum(
                    self._is_vulnerable(t.position) for t in player.tokens
                )
            else:
                opp_positions.extend([0.0] * tokens_per_player)
                opp_active.append(0.0)

        opponent_total_progress = float(sum(opponent_progress_totals))
        opponent_best_progress = float(max(opponent_progress_totals, default=0.0))

        total_opponent_tokens = float(
            tokens_per_player * max(1, len(self._opponent_colors))
        )
        opponent_tokens_at_home = float(opponent_home_count / total_opponent_tokens)
        opponent_tokens_finished = float(
            opponent_finished_count / total_opponent_tokens
        )
        opponent_tokens_on_safe = float(opponent_safe_count / total_opponent_tokens)

        totals_with_agent = [agent_total_progress] + opponent_progress_totals
        sorted_totals = sorted(totals_with_agent, reverse=True)
        agent_rank = (
            sorted_totals.index(agent_total_progress) / max(1, len(sorted_totals) - 1)
            if len(sorted_totals) > 1
            else 0.0
        )
        progress_lead = agent_total_progress - max(
            opponent_progress_totals, default=0.0
        )

        max_dice = GameConstants.DICE_MAX
        dice_vec = np.zeros(max_dice, dtype=np.float32)
        if dice in _DICE_RANGE:
            dice_vec[dice - 1] = 1.0
        dice_value_norm = np.array(
            [dice / max_dice if dice in _DICE_RANGE else 0.0], dtype=np.float32
        )
        dice_is_six = np.array(
            [1.0 if dice == GameConstants.EXIT_HOME_ROLL else 0.0], dtype=np.float32
        )
        dice_is_even = np.array(
            [1.0 if dice in _DICE_RANGE and dice % 2 == 0 else 0.0], dtype=np.float32
        )
        home_exit_ready = np.array(
            [1.0 if dice == GameConstants.EXIT_HOME_ROLL and agent_home.any() else 0.0],
            dtype=np.float32,
        )

        capture_any = np.array([capture_flags.max()], dtype=np.float32)
        finish_any = np.array([finish_flags.max()], dtype=np.float32)

        return {
            "agent_color": agent_color_onehot,
            "agent_progress": agent_progress,
            "agent_distance_to_finish": agent_distance_to_finish,
            "agent_vulnerable": agent_vulnerable,
            "agent_safe": agent_safe,
            "agent_home": agent_home,
            "agent_on_board": agent_on_board,
            "agent_capture_available": capture_flags,
            "agent_finish_available": finish_flags,
            "agent_threat_level": threat_scores,
            "agent_tokens_at_home": np.array([agent_tokens_at_home], dtype=np.float32),
            "agent_tokens_finished": np.array(
                [agent_tokens_finished], dtype=np.float32
            ),
            "agent_tokens_on_safe": np.array([agent_tokens_on_safe], dtype=np.float32),
            "agent_total_progress": np.array([agent_total_progress], dtype=np.float32),
            "opponents_positions": np.array(opp_positions, dtype=np.float32),
            "opponents_active": np.array(opp_active, dtype=np.float32),
            "opponent_total_progress": np.array(
                [opponent_total_progress], dtype=np.float32
            ),
            "opponent_best_progress": np.array(
                [opponent_best_progress], dtype=np.float32
            ),
            "opponent_tokens_at_home": np.array(
                [opponent_tokens_at_home], dtype=np.float32
            ),
            "opponent_tokens_finished": np.array(
                [opponent_tokens_finished], dtype=np.float32
            ),
            "opponent_tokens_on_safe": np.array(
                [opponent_tokens_on_safe], dtype=np.float32
            ),
            "progress_lead": np.array([progress_lead], dtype=np.float32),
            "agent_rank": np.array([agent_rank], dtype=np.float32),
            "dice": dice_vec,
            "dice_value_norm": dice_value_norm,
            "dice_is_six": dice_is_six,
            "dice_is_even": dice_is_even,
            "home_exit_ready": home_exit_ready,
            "capture_any": capture_any,
            "finish_any": finish_any,
        }


class FlattenedObservationBuilder(ContinuousObservationBuilder):
    def build(self, dice: int) -> np.ndarray:
        obs_dict = super().build(dice)
        flat_obs = np.concatenate(list(obs_dict.values())).astype(np.float32)
        return flat_obs


def make_observation_builder(
    cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor
) -> ObservationBuilderBase:
    if cfg.multi_agent:
        return FlattenedObservationBuilder(cfg, game, agent_color)
    return ContinuousObservationBuilder(cfg, game, agent_color)
