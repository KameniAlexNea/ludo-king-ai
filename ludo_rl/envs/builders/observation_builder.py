"""Observation building utilities for LudoGymEnv."""

from typing import List

import numpy as np

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.game import LudoGame

from ..model import EnvConfig


class ObservationBuilder:
    """Handles observation construction for the Ludo environment."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color

    def _compute_observation_size(self) -> int:
        base = 0
        # agent token positions (4)
        base += 4
        # opponents token positions (12)
        base += 12
        # finished tokens per player (4)
        base += 4
        # scalar flags / stats:
        # can_finish, dice_norm, agent_finished_fraction, opp_mean_finished_fraction, agent_mean_token_progress
        base += 5
        if self.cfg.obs_cfg.include_turn_index:
            base += 1
        if self.cfg.obs_cfg.include_blocking_count:
            base += 1
        return base

    def _normalize_position(self, pos: int) -> float:
        if pos == GameConstants.HOME_POSITION:
            return -1.0
        if pos >= BoardConstants.HOME_COLUMN_START:
            depth = (
                pos - BoardConstants.HOME_COLUMN_START
            ) / GameConstants.HOME_COLUMN_DEPTH_SCALE  # 0..1
            return (
                GameConstants.POSITION_NORMALIZATION_FACTOR
                + depth * GameConstants.POSITION_NORMALIZATION_FACTOR
            )
        return (
            pos / (GameConstants.MAIN_BOARD_SIZE - 1)
        ) * GameConstants.POSITION_NORMALIZATION_FACTOR  # [0,0.5]

    def _build_observation(
        self, turns: int, pending_agent_dice: int = None
    ) -> np.ndarray:
        # Cache map for faster lookups
        players_by_color = {p.color.value: p for p in self.game.players}
        agent_player = players_by_color[self.agent_color]

        vec: List[float] = []
        # agent tokens
        for t in agent_player.tokens:
            vec.append(self._normalize_position(t.position))
        # opponents tokens in fixed global order excluding agent
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = players_by_color[color]
            for t in opp.tokens:
                vec.append(self._normalize_position(t.position))
        # finished counts
        for color in Colors.ALL_COLORS:
            pl = players_by_color[color]
            vec.append(pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER)
        # can any finish (include home column tokens)
        can_finish = 0.0
        for t in agent_player.tokens:
            if t.position >= 0:
                remaining = GameConstants.FINISH_POSITION - t.position
                if 0 < remaining <= GameConstants.DICE_MAX:
                    can_finish = 1.0
                    break
        vec.append(can_finish)
        # dice norm (pending dice for current decision)
        if pending_agent_dice is None:
            vec.append(0.0)
        else:
            dice_scale = (GameConstants.DICE_MAX - GameConstants.DICE_MIN) / 2.0
            dice_norm = (pending_agent_dice - GameConstants.DICE_NORMALIZATION_MEAN) / dice_scale
            dice_norm = float(max(-1.0, min(1.0, dice_norm)))
            vec.append(dice_norm)
        # progress stats
        agent_progress = (
            agent_player.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
        )
        opp_progresses = []
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            pl = players_by_color[color]
            opp_progresses.append(
                pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
            )
        opp_mean = (
            sum(opp_progresses) / max(1, len(opp_progresses)) if opp_progresses else 0.0
        )
        vec.append(agent_progress)
        vec.append(opp_mean)
        # agent mean token progress (path coverage)
        total_path = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
        start_pos = BoardConstants.START_POSITIONS.get(self.agent_color)
        token_progress_vals = []
        for t in agent_player.tokens:
            pos = t.position
            if pos < 0:
                token_progress_vals.append(0.0)
                continue
            if pos >= BoardConstants.HOME_COLUMN_START:
                steps = GameConstants.MAIN_BOARD_SIZE + (pos - BoardConstants.HOME_COLUMN_START)
            else:
                if start_pos is None:
                    steps = 0
                elif pos >= start_pos:
                    steps = pos - start_pos
                else:
                    steps = (GameConstants.MAIN_BOARD_SIZE - start_pos) + pos
            token_progress_vals.append(min(1.0, steps / total_path))
        mean_token_progress = sum(token_progress_vals) / max(1, len(token_progress_vals))
        vec.append(mean_token_progress)
        # turn index scaled
        if self.cfg.obs_cfg.include_turn_index:
            vec.append(
                min(GameConstants.TURN_INDEX_MAX_SCALE, turns / self.cfg.max_turns)
            )
        # blocking count
        if self.cfg.obs_cfg.include_blocking_count:
            blocking_positions = self.game.board.get_blocking_positions(
                self.agent_color
            )
            vec.append(
                min(
                    GameConstants.TURN_INDEX_MAX_SCALE,
                    len(blocking_positions)
                    / GameConstants.BLOCKING_COUNT_NORMALIZATION,
                )
            )  # normalize roughly
        obs = np.asarray(vec, dtype=np.float32)
        expected = self._compute_observation_size()
        if obs.shape[0] != expected:
            raise ValueError(
                f"Observation length mismatch: got {obs.shape[0]} expected {expected}"
            )
        return obs
