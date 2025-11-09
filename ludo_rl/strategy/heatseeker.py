from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .features import nearest_opponent_distance, opponent_density_within
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class HeatSeekerStrategyConfig(BaseStrategyConfig):
    capture_weight: tuple[float, float] = (9.0, 14.0)
    proximity_radius: tuple[int, int] = (4, 8)
    proximity_weight: tuple[float, float] = (1.5, 3.0)
    density_radius: tuple[int, int] = (3, 6)
    density_weight: tuple[float, float] = (0.5, 1.0)
    progress_weight: tuple[float, float] = (0.3, 0.8)
    risk_penalty: tuple[float, float] = (0.4, 0.9)
    # Safety-aware engagement
    safe_bonus: tuple[float, float] = (1.0, 2.0)
    leave_safe_penalty: tuple[float, float] = (0.8, 1.8)
    rear_threat_weight: tuple[float, float] = (0.2, 0.6)
    engagement_risk_alpha: tuple[float, float] = (0.2, 0.6)
    # Tempo and prediction
    extra_turn_bonus: tuple[float, float] = (1.0, 2.5)
    predictive_weight: tuple[float, float] = (0.2, 0.6)
    # Endgame bias and misc
    endgame_threshold: tuple[int, int] = (
        max(1, strategy_config.home_start - 8),
        strategy_config.home_start,
    )
    endgame_engage_scale: tuple[float, float] = (0.5, 0.8)
    endgame_progress_boost: tuple[float, float] = (1.1, 1.6)
    stack_penalty: tuple[float, float] = (0.3, 0.9)
    distance_weight: tuple[float, float] = (0.02, 0.06)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        low, high = self.endgame_threshold
        low = max(1, min(low, strategy_config.home_start))
        high = max(low, min(high, strategy_config.home_start))
        return {
            "capture_weight": rng.uniform(*self.capture_weight),
            "proximity_radius": rng.randint(*self.proximity_radius),
            "proximity_weight": rng.uniform(*self.proximity_weight),
            "density_radius": rng.randint(*self.density_radius),
            "density_weight": rng.uniform(*self.density_weight),
            "progress_weight": rng.uniform(*self.progress_weight),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "safe_bonus": rng.uniform(*self.safe_bonus),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "rear_threat_weight": rng.uniform(*self.rear_threat_weight),
            "engagement_risk_alpha": rng.uniform(*self.engagement_risk_alpha),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "predictive_weight": rng.uniform(*self.predictive_weight),
            "endgame_threshold": rng.randint(low, high),
            "endgame_engage_scale": rng.uniform(*self.endgame_engage_scale),
            "endgame_progress_boost": rng.uniform(*self.endgame_progress_boost),
            "stack_penalty": rng.uniform(*self.stack_penalty),
            "distance_weight": rng.uniform(*self.distance_weight),
        }


@dataclass(slots=True)
class HeatSeekerStrategy(BaseStrategy):
    """Moves toward concentrations of opponents to force engagements."""

    name: ClassVar[str] = "heatseeker"
    config: ClassVar[HeatSeekerStrategyConfig] = HeatSeekerStrategyConfig()

    capture_weight: float = 12.0
    proximity_radius: int = 6
    proximity_weight: float = 2.0
    density_radius: int = 4
    density_weight: float = 0.7
    progress_weight: float = 0.5
    risk_penalty: float = 0.6
    safe_bonus: float = 1.5
    leave_safe_penalty: float = 1.2
    rear_threat_weight: float = 0.4
    engagement_risk_alpha: float = 0.4
    extra_turn_bonus: float = 1.5
    predictive_weight: float = 0.4
    endgame_threshold: int = strategy_config.home_start - 6
    endgame_engage_scale: float = 0.7
    endgame_progress_boost: float = 1.3
    stack_penalty: float = 0.5
    distance_weight: float = 0.04

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        # Captures still prioritized, but include tempo and safety
        if move.can_capture:
            score = move.capture_count * self.capture_weight
            if move.extra_turn:
                score += self.extra_turn_bonus
            if move.enters_safe_zone:
                score += self.safe_bonus
            score -= move.risk * self.risk_penalty
            return score

        # Engagement terms
        distance = nearest_opponent_distance(ctx.opponent_distribution, move.new_pos)
        density_sum = opponent_density_within(
            ctx.opponent_distribution, move.new_pos, radius=self.density_radius
        )
        # Normalize density by window length to avoid over-weighting big clusters
        window = 2 * self.density_radius + 1
        density_norm = density_sum / float(max(1, window))

        proximity_term = (
            max(0, self.proximity_radius - distance) * self.proximity_weight
        )
        density_term = density_norm * self.density_weight

        # Rear threats within 1..6 behind landing (skip safe squares)
        rear_threats = 0.0
        if 1 <= move.new_pos <= strategy_config.main_track_end:
            for step in range(1, 7):
                idx = move.new_pos - step
                if idx <= 0:
                    idx += strategy_config.main_track_end
                if ctx.safe_channel[idx]:
                    continue
                rear_threats += ctx.opponent_distribution[idx]

        # Risk-damped engagement
        risk_damp = 1.0 / (1.0 + self.engagement_risk_alpha * max(0.0, move.risk))
        engagement = (
            proximity_term + density_term
        ) * risk_damp - rear_threats * self.rear_threat_weight

        # Predictive engagements: opponents within 1..6 ahead (skip safe squares)
        predictive = 0.0
        if (
            1 <= move.new_pos <= strategy_config.main_track_end
            and not move.enters_safe_zone
        ):
            for step in range(1, 7):
                idx = move.new_pos + step
                if idx > strategy_config.main_track_end:
                    idx -= strategy_config.main_track_end
                if ctx.safe_channel[idx]:
                    continue
                predictive += ctx.opponent_distribution[idx]
        predictive *= self.predictive_weight

        # Base score assembly
        score = 0.0
        score += engagement + predictive
        score += move.progress * self.progress_weight
        score -= move.risk * self.risk_penalty
        score -= move.distance_to_goal * self.distance_weight

        if move.enters_safe_zone:
            score += self.safe_bonus
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty
        if move.extra_turn:
            score += self.extra_turn_bonus

        # Avoid stacking unless explicitly desired
        if move.forms_blockade and not move.enters_safe_zone:
            score -= self.stack_penalty

        # Endgame bias: reduce engagement, boost progress near home
        if move.new_pos >= self.endgame_threshold:
            score += (
                move.progress
                * (self.endgame_progress_boost - 1.0)
                * self.progress_weight
            )
            score += (self.endgame_engage_scale - 1.0) * (proximity_term + density_term)

        return score
