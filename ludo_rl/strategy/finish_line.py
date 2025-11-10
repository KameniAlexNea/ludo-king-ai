from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class FinishLineStrategyConfig(BaseStrategyConfig):
    finish_bonus: tuple[float, float] = (12.0, 18.0)
    distance_weight: tuple[float, float] = (0.4, 0.8)
    progress_weight: tuple[float, float] = (1.0, 1.5)
    safe_zone_bonus: tuple[float, float] = (3.0, 5.5)
    capture_weight: tuple[float, float] = (1.5, 3.5)
    risk_penalty: tuple[float, float] = (0.5, 1.2)
    leave_safe_penalty: tuple[float, float] = (1.0, 3.0)
    safe_threshold: tuple[int, int] = (
        strategy_config.home_start - 4,
        strategy_config.home_start + 4,
    )
    # Winner-aligned refinements
    home_depth_weight: tuple[float, float] = (0.4, 1.0)
    safe_capture_weight: tuple[float, float] = (2.0, 5.0)
    capture_progress_weight: tuple[float, float] = (0.6, 1.6)
    allowed_capture_risk: tuple[float, float] = (0.0, 0.6)
    exit_min_active: tuple[int, int] = (1, 2)
    allowed_exit_risk: tuple[float, float] = (0.0, 0.6)
    exit_home_bonus: tuple[float, float] = (0.4, 1.2)
    exit_home_penalty: tuple[float, float] = (0.8, 2.0)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        low, high = self.safe_threshold
        low = max(low, strategy_config.home_start - 6)
        high = min(high, strategy_config.home_finish)
        if high < low:
            high = low
        return {
            "finish_bonus": rng.uniform(*self.finish_bonus),
            "distance_weight": rng.uniform(*self.distance_weight),
            "progress_weight": rng.uniform(*self.progress_weight),
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "capture_weight": rng.uniform(*self.capture_weight),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "safe_threshold": rng.randint(low, high),
            "home_depth_weight": rng.uniform(*self.home_depth_weight),
            "safe_capture_weight": rng.uniform(*self.safe_capture_weight),
            "capture_progress_weight": rng.uniform(*self.capture_progress_weight),
            "allowed_capture_risk": rng.uniform(*self.allowed_capture_risk),
            "exit_min_active": rng.randint(*self.exit_min_active),
            "allowed_exit_risk": rng.uniform(*self.allowed_exit_risk),
            "exit_home_bonus": rng.uniform(*self.exit_home_bonus),
            "exit_home_penalty": rng.uniform(*self.exit_home_penalty),
        }


@dataclass(slots=True)
class FinishLineStrategy(BaseStrategy):
    """Pushes the leading runner across the finish line as quickly as possible."""

    name: ClassVar[str] = "finish_line"
    config: ClassVar[FinishLineStrategyConfig] = FinishLineStrategyConfig()

    finish_bonus: float = 15.0
    distance_weight: float = 0.6
    progress_weight: float = 1.2
    safe_zone_bonus: float = 4.0
    capture_weight: float = 2.0
    risk_penalty: float = 0.8
    leave_safe_penalty: float = 2.0
    safe_threshold: int = strategy_config.home_start
    home_depth_weight: float = 0.7
    safe_capture_weight: float = 3.2
    capture_progress_weight: float = 1.0
    allowed_capture_risk: float = 0.4
    exit_min_active: int = 2
    allowed_exit_risk: float = 0.4
    exit_home_bonus: float = 0.8
    exit_home_penalty: float = 1.2

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0

        # 1) Finish outweighs everything
        if move.enters_home:
            score += self.finish_bonus

        # 2) Home column depth: push the runner deeper when inside
        if move.new_pos >= strategy_config.home_start:
            depth = move.new_pos - strategy_config.home_start
            score += depth * self.home_depth_weight

        # 3) Convert progress efficiently while getting closer to goal
        score -= move.distance_to_goal * self.distance_weight
        score += move.progress * self.progress_weight

        # 4) Safe zone preference near the endgame threshold
        if move.enters_safe_zone and move.new_pos >= self.safe_threshold:
            score += self.safe_zone_bonus

        # 5) Captures: prefer safe/low-risk and meaningful progress removal
        if move.can_capture:
            # Baseline capture value
            score += move.capture_count * self.capture_weight

            # Only boost strongly if landing is safe or risk is very low
            if move.enters_safe_zone or move.risk <= self.allowed_capture_risk:
                score += move.capture_count * self.safe_capture_weight
                if 1 <= move.new_pos <= strategy_config.main_track_end:
                    prey_progress_frac = move.new_pos / float(
                        strategy_config.main_track_end
                    )
                    score += (
                        prey_progress_frac
                        * self.capture_progress_weight
                        * move.capture_count
                    )

        # 6) Safety costs
        score -= move.risk * self.risk_penalty
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        # 7) Exit home only when necessary (maintain board presence)
        active_tokens = int(min(4, max(0, round(float(ctx.my_distribution[1:].sum())))))
        if move.current_pos == 0 and not move.enters_safe_zone:
            if (
                active_tokens < self.exit_min_active
                and move.risk <= self.allowed_exit_risk
            ):
                score += self.exit_home_bonus
            else:
                score -= self.exit_home_penalty

        return score
