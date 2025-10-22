"""Minimal reward calculator used by the simplified Ludo environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ludo_engine.core import LudoGame
from ludo_engine.models import BoardConstants, GameConstants, MoveResult, PlayerColor

from models.config import EnvConfig

_TOTAL_PATH = GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE


def _progress_fraction(position: int, start_pos: int) -> float:
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


@dataclass
class RewardBreakdown:
    progress: float = 0.0
    capture: float = 0.0
    finish: float = 0.0
    got_captured: float = 0.0
    illegal: float = 0.0
    time_penalty: float = 0.0
    terminal: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


class AdvancedRewardCalculator:
    """Small helper that returns shaped rewards compatible with PPO training."""

    def reset_for_new_episode(self) -> None:  # Kept for API parity
        return None

    def compute(
        self,
        game: LudoGame,
        agent_color: PlayerColor,
        move_result: Optional[MoveResult],
        cfg: EnvConfig,
        *,
        is_illegal: bool = False,
        opponent_captures: int = 0,
        terminated: bool = False,
    ) -> tuple[float, Dict[str, float]]:
        reward = 0.0
        breakdown = RewardBreakdown()

        breakdown.time_penalty = cfg.reward.time_penalty
        reward += breakdown.time_penalty

        if move_result is not None and move_result.success:
            start_pos = BoardConstants.START_POSITIONS[agent_color]
            old_progress = _progress_fraction(move_result.old_position, start_pos)
            new_progress = _progress_fraction(move_result.new_position, start_pos)
            delta = max(0.0, new_progress - old_progress)
            if delta > 0.0:
                breakdown.progress = cfg.reward.progress_scale * delta
                reward += breakdown.progress

            if move_result.captured_tokens:
                captures = len(move_result.captured_tokens)
                breakdown.capture = captures * cfg.reward.capture
                reward += breakdown.capture

            if (
                move_result.finished_token
                or move_result.new_position == GameConstants.FINISH_POSITION
            ):
                breakdown.finish = cfg.reward.finish_token
                reward += breakdown.finish

        if is_illegal:
            breakdown.illegal = cfg.reward.illegal_action
            reward += breakdown.illegal

        if opponent_captures > 0:
            breakdown.got_captured = cfg.reward.got_captured * opponent_captures
            reward += breakdown.got_captured

        if terminated:
            if game.winner is not None:
                if game.winner.color == agent_color:
                    breakdown.terminal = cfg.reward.win
                else:
                    breakdown.terminal = cfg.reward.lose
            else:
                breakdown.terminal = cfg.reward.draw
            reward += breakdown.terminal

        return reward, breakdown.to_dict()
