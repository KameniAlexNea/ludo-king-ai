from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class KillerStrategyConfig(BaseStrategyConfig):
    # Core aggression knobs
    capture_weight: tuple[float, float] = (9.0, 14.0)
    progress_weight: tuple[float, float] = (0.6, 1.2)
    risk_discount: tuple[float, float] = (0.6, 1.2)
    extra_turn_bonus: tuple[float, float] = (3.0, 6.0)
    safe_bonus: tuple[float, float] = (0.5, 1.5)

    # Added to better align priorities with improved Killer spec
    finish_bonus: tuple[float, float] = (10.0, 16.0)
    capture_chain_bonus: tuple[float, float] = (1.5, 3.0)
    blockade_bonus: tuple[float, float] = (0.8, 2.0)
    prey_progress_weight: tuple[float, float] = (0.8, 2.0)
    predictive_radius: tuple[int, int] = (4, 6)
    predictive_weight: tuple[float, float] = (0.3, 0.8)
    leave_safe_penalty: tuple[float, float] = (0.5, 1.5)
    weak_prey_penalty: tuple[float, float] = (0.8, 2.0)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        return {
            "capture_weight": rng.uniform(*self.capture_weight),
            "progress_weight": rng.uniform(*self.progress_weight),
            "risk_discount": rng.uniform(*self.risk_discount),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "safe_bonus": rng.uniform(*self.safe_bonus),
            "finish_bonus": rng.uniform(*self.finish_bonus),
            "capture_chain_bonus": rng.uniform(*self.capture_chain_bonus),
            "blockade_bonus": rng.uniform(*self.blockade_bonus),
            "prey_progress_weight": rng.uniform(*self.prey_progress_weight),
            "predictive_radius": rng.randint(*self.predictive_radius),
            "predictive_weight": rng.uniform(*self.predictive_weight),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "weak_prey_penalty": rng.uniform(*self.weak_prey_penalty),
        }


@dataclass(slots=True)
class KillerStrategy(BaseStrategy):
    """Aggressively hunts opponent pieces even at higher risk."""

    name: ClassVar[str] = "killer"
    config: ClassVar[KillerStrategyConfig] = KillerStrategyConfig()

    capture_weight: float = 10.0
    progress_weight: float = 1.0
    risk_discount: float = 0.9
    extra_turn_bonus: float = 4.5
    safe_bonus: float = 1.0
    finish_bonus: float = 13.0
    capture_chain_bonus: float = 2.0
    blockade_bonus: float = 1.2
    prey_progress_weight: float = 1.2
    predictive_radius: int = 5
    predictive_weight: float = 0.5
    leave_safe_penalty: float = 1.0
    weak_prey_penalty: float = 1.2

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        score = 0.0

        # 1) Finishing should beat routine captures
        if move.enters_home:
            score += self.finish_bonus

        # 2) Captures with integrated tactical factors
        if move.can_capture:
            # Base capture value (multi-capture supported)
            score += move.capture_count * self.capture_weight

            # Prefer removing prey that are further progressed along the ring
            if 1 <= move.new_pos <= strategy_config.main_track_end:
                prey_progress_frac = move.new_pos / float(
                    strategy_config.main_track_end
                )
                score += (
                    prey_progress_frac * self.prey_progress_weight * move.capture_count
                )

            # Extra chain potential beyond the generic extra_turn bonus
            if move.extra_turn:
                score += self.capture_chain_bonus

        # 3) Predictive aggression (future capture setup) as a secondary cue
        #    Count opponents ahead within dice range if not already safe/home.
        if not move.can_capture and not move.enters_home and not move.enters_safe_zone:
            ahead_density = 0.0
            for step in range(1, 7):
                idx = move.new_pos + step
                if idx > strategy_config.main_track_end:
                    idx -= strategy_config.main_track_end
                # ignore safe landing squares for predictive aggression
                if ctx.safe_channel[idx]:
                    continue
                ahead_density += ctx.opponent_distribution[idx]
            # weight only within a bounded radius emphasis
            radius_scale = min(self.predictive_radius, 6) / 6.0
            score += ahead_density * self.predictive_weight * radius_scale

        # 4) Positional bonuses and penalties
        if move.enters_safe_zone:
            score += self.safe_bonus
        if move.forms_blockade and not move.enters_safe_zone:
            score += self.blockade_bonus
        if move.extra_turn:
            score += self.extra_turn_bonus

        # 5) Progress helps keep pressure
        score += move.progress * self.progress_weight

        # 6) Risk handling: killer tolerates more risk (smaller discount)
        score -= move.risk * self.risk_discount
        if move.leaving_safe_zone and not move.can_capture and not move.enters_home:
            score -= self.leave_safe_penalty

        # 7) Avoid chasing weak, risky prey
        if move.can_capture and 1 <= move.new_pos <= strategy_config.main_track_end:
            prey_progress_frac = move.new_pos / float(strategy_config.main_track_end)
            if prey_progress_frac < 0.2 and move.risk > 0:
                score -= self.weak_prey_penalty

        # 8) Slight nudge to keep pieces advancing toward goal
        score -= move.distance_to_goal * 0.015

        return score
