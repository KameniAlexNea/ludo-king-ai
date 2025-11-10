from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo_king.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .features import nearest_opponent_distance
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class CautiousStrategyConfig(BaseStrategyConfig):
    safe_bonus: tuple[float, float] = (3.0, 6.0)
    blockade_bonus: tuple[float, float] = (2.5, 4.5)
    risk_penalty: tuple[float, float] = (5.0, 7.5)
    leave_safe_penalty: tuple[float, float] = (4.0, 6.5)
    progress_weight: tuple[float, float] = (0.6, 1.2)
    finish_bonus: tuple[float, float] = (12.0, 18.0)
    home_depth_weight: tuple[float, float] = (0.3, 0.8)
    safe_capture_weight: tuple[float, float] = (2.0, 5.0)
    allowed_risk_for_capture: tuple[float, float] = (0.0, 0.5)
    allowed_risk_for_exit: tuple[float, float] = (0.0, 0.5)
    distance_weight: tuple[float, float] = (0.3, 0.8)
    exit_home_penalty: tuple[float, float] = (3.0, 7.0)
    extra_turn_bonus: tuple[float, float] = (0.5, 1.5)
    near_opponent_penalty: tuple[float, float] = (0.08, 0.14)
    far_opponent_penalty: tuple[float, float] = (0.0, 0.06)

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        rng = rng or random
        return {
            "safe_bonus": rng.uniform(*self.safe_bonus),
            "blockade_bonus": rng.uniform(*self.blockade_bonus),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "progress_weight": rng.uniform(*self.progress_weight),
            "finish_bonus": rng.uniform(*self.finish_bonus),
            "home_depth_weight": rng.uniform(*self.home_depth_weight),
            "safe_capture_weight": rng.uniform(*self.safe_capture_weight),
            "allowed_risk_for_capture": rng.uniform(*self.allowed_risk_for_capture),
            "allowed_risk_for_exit": rng.uniform(*self.allowed_risk_for_exit),
            "distance_weight": rng.uniform(*self.distance_weight),
            "exit_home_penalty": rng.uniform(*self.exit_home_penalty),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "near_opponent_penalty": rng.uniform(*self.near_opponent_penalty),
            "far_opponent_penalty": rng.uniform(*self.far_opponent_penalty),
        }


@dataclass(slots=True)
class CautiousStrategy(BaseStrategy):
    """Prioritises safety, safe zones, and avoiding exposure."""

    name: ClassVar[str] = "cautious"
    config: ClassVar[CautiousStrategyConfig] = CautiousStrategyConfig()

    safe_bonus: float = 4.0
    blockade_bonus: float = 3.0
    risk_penalty: float = 6.0
    leave_safe_penalty: float = 5.0
    progress_weight: float = 0.8
    finish_bonus: float = 15.0
    home_depth_weight: float = 0.5
    safe_capture_weight: float = 3.0
    allowed_risk_for_capture: float = 0.2
    allowed_risk_for_exit: float = 0.2
    distance_weight: float = 0.5
    exit_home_penalty: float = 5.0
    extra_turn_bonus: float = 1.0
    near_opponent_penalty: float = 0.1
    far_opponent_penalty: float = 0.03

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0

        # 1) Absolute safety priorities
        if move.enters_home:
            score += self.finish_bonus

        if move.enters_safe_zone:
            score += self.safe_bonus
            # Prefer deeper progress inside home column when applicable
            if move.new_pos >= strategy_config.home_start:
                depth = move.new_pos - strategy_config.home_start
                score += depth * self.home_depth_weight

        # 2) Safe captures first: only when landing is safe or effectively threat-free
        if move.can_capture and (
            move.risk <= self.allowed_risk_for_capture or move.enters_safe_zone
        ):
            score += move.capture_count * self.safe_capture_weight

        # 3) Protective structures
        if move.forms_blockade:
            score += self.blockade_bonus

        # 4) Tempo is useful but not at the expense of safety
        if move.extra_turn:
            score += self.extra_turn_bonus

        # 5) Positional risk handling and conservative advancement
        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        # Avoid leaving home unless landing is acceptably safe
        if (
            move.current_pos == 0
            and not move.enters_safe_zone
            and move.risk > self.allowed_risk_for_exit
        ):
            score -= self.exit_home_penalty

        # Prefer being far from opponents after the move
        nearest = nearest_opponent_distance(ctx.opponent_distribution, move.new_pos)
        score += min(nearest, 6) * self.distance_weight

        # Conservative progress: keep small incentive, modulated by nearby threats
        threat_scaled_penalty = (
            self.near_opponent_penalty if nearest <= 3 else self.far_opponent_penalty
        )
        score += move.progress * self.progress_weight
        score -= min(move.distance_to_goal, 20) * threat_scaled_penalty

        return score
