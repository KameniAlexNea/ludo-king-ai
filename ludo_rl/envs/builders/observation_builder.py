"""Observation building utilities for LudoGymEnv."""

from typing import List

import numpy as np

from ludo.constants import BoardConstants, Colors, GameConstants
from ludo.game import LudoGame
from ludo.player import Player

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
        # flags: can_finish, dice_value, progress stats, turn index
        base += 4  # can_finish, dice_norm, agent_progress, opp_mean_progress
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

    def _build_observation(self, turns: int, pending_agent_dice: int = None) -> np.ndarray:
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )
        vec: List[float] = []
        # agent tokens
        for t in agent_player.tokens:
            vec.append(self._normalize_position(t.position))
        # opponents tokens in fixed global order excluding agent
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = next(p for p in self.game.players if p.color.value == color)
            for t in opp.tokens:
                vec.append(self._normalize_position(t.position))
        # finished counts
        for color in Colors.ALL_COLORS:
            pl = next(p for p in self.game.players if p.color.value == color)
            vec.append(pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER)
        # can any finish
        can_finish = 0.0
        for t in agent_player.tokens:
            if 0 <= t.position < BoardConstants.HOME_COLUMN_START:
                remaining = GameConstants.FINISH_POSITION - t.position
                if remaining <= GameConstants.DICE_MAX:
                    can_finish = 1.0
                    break
        vec.append(can_finish)
        # dice norm (pending dice for current decision)
        if pending_agent_dice is None:
            vec.append(0.0)
        else:
            vec.append(
                (pending_agent_dice - GameConstants.DICE_NORMALIZATION_MEAN)
                / GameConstants.DICE_NORMALIZATION_MEAN
            )
        # progress stats
        agent_progress = (
            agent_player.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
        )
        opp_progresses = []
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            pl = next(p for p in self.game.players if p.color.value == color)
            opp_progresses.append(
                pl.get_finished_tokens_count() / GameConstants.TOKENS_PER_PLAYER
            )
        opp_mean = (
            sum(opp_progresses) / max(1, len(opp_progresses)) if opp_progresses else 0.0
        )
        vec.append(agent_progress)
        vec.append(opp_mean)
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
        return np.asarray(vec, dtype=np.float32)
