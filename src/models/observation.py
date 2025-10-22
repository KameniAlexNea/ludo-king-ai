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
            opp_pos = token.position
            if not _is_on_main_board(opp_pos):
                continue
            steps = (token_position - opp_pos) % GameConstants.MAIN_BOARD_SIZE
            if 0 < steps <= 6:
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
        agent_distance_to_finish = np.clip(1.0 - agent_progress, 0.0, 1.0)

        agent_vulnerable = np.array(
            [float(self._is_vulnerable(t.position)) for t in self._agent.tokens],
            dtype=np.float32,
        )
        agent_safe = 1.0 - agent_vulnerable

        agent_home = np.array(
            [float(_is_home(t.position)) for t in self._agent.tokens],
            dtype=np.float32,
        )
        agent_on_board = np.array(
            [float(_is_on_main_board(t.position)) for t in self._agent.tokens],
            dtype=np.float32,
        )

        valid_moves = self.game.get_valid_moves(self._agent, dice)
        capture_flags = np.zeros(tokens_per_player, dtype=np.float32)
        finish_flags = np.zeros(tokens_per_player, dtype=np.float32)
        for mv in valid_moves:
            idx = mv.token_id
            if getattr(mv, "captures_opponent", False):
                capture_flags[idx] = 1.0
            if (
                getattr(mv, "finished_token", False)
                or getattr(mv, "target_position", None) == GameConstants.FINISH_POSITION
            ):
                finish_flags[idx] = 1.0

        threat_scores = np.zeros(tokens_per_player, dtype=np.float32)
        for i, token in enumerate(self._agent.tokens):
            if not agent_vulnerable[i]:
                continue
            steps = _threat_steps_to_agent(self.game, self.agent_color, token.position)
            if steps is not None:
                threat_scores[i] = (7 - steps) / 6.0

        agent_tokens_at_home = float(agent_home.sum() / max(1, tokens_per_player))
        agent_tokens_finished = float(
            sum(float(_is_finished(t.position)) for t in self._agent.tokens)
            / max(1, tokens_per_player)
        )
        agent_tokens_on_safe = float(agent_safe.sum() / max(1, tokens_per_player))
        agent_total_progress = float(agent_progress.sum())

        opp_positions: List[float] = []
        opp_active: List[float] = []
        opponent_total_progress = 0.0
        opponent_best_progress = 0.0
        opponent_home_count = 0.0
        opponent_finished_count = 0.0
        opponent_safe_count = 0.0
        for color in self._ordered_opponent_colors(self.agent_color):
            if color in self._present_colors:
                player = self.game.get_player_from_color(color)
                start_pos = BoardConstants.START_POSITIONS[color]
                progresses = [
                    _progress_fraction(t.position, start_pos) for t in player.tokens
                ]
                opp_positions.extend(progresses)
                opp_active.append(1.0)
                total_progress = float(sum(progresses))
                opponent_total_progress += total_progress
                opponent_best_progress = max(opponent_best_progress, total_progress)
                opponent_home_count += sum(_is_home(t.position) for t in player.tokens)
                opponent_finished_count += sum(
                    _is_finished(t.position) for t in player.tokens
                )
                opponent_safe_count += sum(
                    not self._is_vulnerable(t.position) for t in player.tokens
                )
            else:
                opp_positions.extend([0.0] * tokens_per_player)
                opp_active.append(0.0)

        total_opponent_tokens = max(1.0, tokens_per_player * len(self._opponent_colors))
        opponent_tokens_at_home = float(opponent_home_count / total_opponent_tokens)
        opponent_tokens_finished = float(
            opponent_finished_count / total_opponent_tokens
        )
        opponent_tokens_on_safe = float(opponent_safe_count / total_opponent_tokens)

        all_totals = [(self.agent_color, agent_total_progress)] + [
            (
                color,
                sum(
                    _progress_fraction(
                        t.position, BoardConstants.START_POSITIONS[color]
                    )
                    for t in self.game.get_player_from_color(color).tokens
                ),
            )
            for color in self._opponent_colors
        ]
        all_totals.sort(key=lambda pair: pair[1], reverse=True)
        rank_index = next(
            (idx for idx, pair in enumerate(all_totals) if pair[0] == self.agent_color),
            0,
        )
        agent_rank = (
            rank_index / max(1, len(all_totals) - 1) if len(all_totals) > 1 else 0.0
        )
        progress_lead = agent_total_progress - max(
            (val for color, val in all_totals if color != self.agent_color),
            default=0.0,
        )

        max_dice = GameConstants.DICE_MAX
        dice_vec = np.zeros(max_dice, dtype=np.float32)
        if 1 <= dice <= max_dice:
            dice_vec[dice - 1] = 1.0
        dice_value_norm = np.array(
            [dice / max_dice if 1 <= dice <= max_dice else 0.0], dtype=np.float32
        )
        dice_is_six = np.array([1.0 if dice == 6 else 0.0], dtype=np.float32)
        dice_is_even = np.array(
            [1.0 if dice > 0 and dice % 2 == 0 else 0.0], dtype=np.float32
        )
        home_exit_ready = np.array(
            [1.0 if dice == 6 and agent_tokens_at_home > 0 else 0.0],
            dtype=np.float32,
        )

        capture_any = np.array([float(capture_flags.max())], dtype=np.float32)
        finish_any = np.array([float(finish_flags.max())], dtype=np.float32)

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


def make_observation_builder(
    cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor
) -> ObservationBuilderBase:
    return ContinuousObservationBuilder(cfg, game, agent_color)
