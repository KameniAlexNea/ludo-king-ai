from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .features import nearest_opponent_distance
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class DefensiveStrategyConfig(BaseStrategyConfig):
    enter_home_bonus: tuple[float, float] = (5.0, 8.0)
    safe_zone_bonus: tuple[float, float] = (4.0, 7.0)
    blockade_bonus: tuple[float, float] = (2.5, 4.5)
    extra_turn_bonus: tuple[float, float] = (1.0, 3.5)
    risk_penalty: tuple[float, float] = (1.5, 3.0)
    leave_safe_penalty: tuple[float, float] = (5.0, 8.0)
    progress_weight: tuple[float, float] = (0.2, 0.6)
    # Added defensive refinements
    home_depth_weight: tuple[float, float] = (0.3, 0.8)
    break_block_penalty: tuple[float, float] = (2.0, 4.0)
    safe_capture_weight: tuple[float, float] = (1.5, 3.5)
    prey_progress_weight: tuple[float, float] = (0.4, 1.0)
    allowed_capture_risk: tuple[float, float] = (0.0, 0.6)
    distance_weight: tuple[float, float] = (0.3, 0.8)
    risk_band_threshold: tuple[float, float] = (1.5, 3.0)
    high_risk_penalty: tuple[float, float] = (0.5, 1.2)
    min_active_tokens: tuple[int, int] = (1, 2)
    allowed_exit_risk: tuple[float, float] = (0.0, 0.6)
    exit_home_penalty: tuple[float, float] = (2.0, 5.0)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        return {
            "enter_home_bonus": rng.uniform(*self.enter_home_bonus),
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "blockade_bonus": rng.uniform(*self.blockade_bonus),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "progress_weight": rng.uniform(*self.progress_weight),
            "home_depth_weight": rng.uniform(*self.home_depth_weight),
            "break_block_penalty": rng.uniform(*self.break_block_penalty),
            "safe_capture_weight": rng.uniform(*self.safe_capture_weight),
            "prey_progress_weight": rng.uniform(*self.prey_progress_weight),
            "allowed_capture_risk": rng.uniform(*self.allowed_capture_risk),
            "distance_weight": rng.uniform(*self.distance_weight),
            "risk_band_threshold": rng.uniform(*self.risk_band_threshold),
            "high_risk_penalty": rng.uniform(*self.high_risk_penalty),
            "min_active_tokens": rng.randint(*self.min_active_tokens),
            "allowed_exit_risk": rng.uniform(*self.allowed_exit_risk),
            "exit_home_penalty": rng.uniform(*self.exit_home_penalty),
        }


@dataclass(slots=True)
class DefensiveStrategy(BaseStrategy):
    """Keeps pieces sheltered and avoids exposing them to captures."""

    name: ClassVar[str] = "defensive"
    config: ClassVar[DefensiveStrategyConfig] = DefensiveStrategyConfig()

    enter_home_bonus: float = 6.0
    safe_zone_bonus: float = 5.0
    blockade_bonus: float = 3.5
    extra_turn_bonus: float = 2.0
    risk_penalty: float = 2.0
    leave_safe_penalty: float = 6.0
    progress_weight: float = 0.3
    home_depth_weight: float = 0.5
    break_block_penalty: float = 3.0
    safe_capture_weight: float = 2.5
    prey_progress_weight: float = 0.7
    allowed_capture_risk: float = 0.4
    distance_weight: float = 0.5
    risk_band_threshold: float = 2.0
    high_risk_penalty: float = 0.8
    min_active_tokens: int = 2
    allowed_exit_risk: float = 0.4
    exit_home_penalty: float = 3.5

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0

        # 1) Finish immediately preferred
        if move.enters_home:
            score += self.enter_home_bonus

        # 2) Safe zones and home column depth
        if move.enters_safe_zone:
            score += self.safe_zone_bonus
        if move.new_pos >= strategy_config.home_start:
            depth = move.new_pos - strategy_config.home_start
            score += depth * self.home_depth_weight

        # 3) Blocks: forming is good, breaking without compensation is bad
        if move.forms_blockade:
            score += self.blockade_bonus
        # penalise leaving an existing block unless we form another or go safe/home
        if 1 <= move.current_pos <= strategy_config.main_track_end:
            if (
                ctx.safe_channel[move.current_pos] == 0
                and ctx.my_distribution[move.current_pos] >= 2
            ):
                breaks_block = (
                    not move.forms_blockade
                    and not move.enters_safe_zone
                    and move.new_pos < strategy_config.home_start
                )
                if breaks_block:
                    score -= self.break_block_penalty

        # 4) Safe captures with positional benefit
        if move.can_capture and (
            move.enters_safe_zone or move.risk <= self.allowed_capture_risk
        ):
            score += move.capture_count * self.safe_capture_weight
            if 1 <= move.new_pos <= strategy_config.main_track_end:
                prey_progress_frac = move.new_pos / float(
                    strategy_config.main_track_end
                )
                score += (
                    prey_progress_frac * self.prey_progress_weight * move.capture_count
                )

        # 5) Tempo helps when safe
        if move.extra_turn:
            score += self.extra_turn_bonus

        # 6) Risk banding: heavy penalty overall; extra if above threshold
        score -= move.risk * self.risk_penalty
        if move.risk > self.risk_band_threshold:
            score -= (move.risk - self.risk_band_threshold) * self.high_risk_penalty

        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        # 7) Exit home conservatively based on presence
        # approximate active tokens on board/home column
        active_tokens = int(min(4, max(0, round(float(ctx.my_distribution[1:].sum())))))
        if move.current_pos == 0 and not move.enters_safe_zone:
            if (
                active_tokens < self.min_active_tokens
                and move.risk <= self.allowed_exit_risk
            ):
                # slight encouragement to maintain presence
                score += 0.5
            else:
                score -= self.exit_home_penalty

        # 8) Prefer landing further from opponents
        nearest = nearest_opponent_distance(ctx.opponent_distribution, move.new_pos)
        score += min(nearest, 6) * self.distance_weight

        # 9) Modest progress to avoid stalling
        score += move.progress * self.progress_weight

        return score
