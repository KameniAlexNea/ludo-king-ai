from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class SupportStrategyConfig(BaseStrategyConfig):
    yard_bonus: tuple[float, float] = (3.0, 5.5)
    lag_weight: tuple[float, float] = (0.2, 0.5)
    lead_penalty: tuple[float, float] = (0.15, 0.35)
    safe_zone_bonus: tuple[float, float] = (1.0, 2.5)
    risk_penalty: tuple[float, float] = (0.6, 1.2)
    # Balancing refinements
    variance_weight: tuple[float, float] = (0.2, 0.6)
    min_progress_weight: tuple[float, float] = (0.3, 0.9)
    stack_penalty: tuple[float, float] = (0.4, 1.0)
    laggard_safe_bonus: tuple[float, float] = (0.4, 1.0)
    laggard_risk_scale: tuple[float, float] = (0.2, 0.6)
    # Presence & exits
    target_active: tuple[int, int] = (2, 3)
    allowed_exit_risk: tuple[float, float] = (0.0, 0.6)
    exit_home_bonus: tuple[float, float] = (0.8, 1.8)
    exit_home_penalty: tuple[float, float] = (0.8, 1.6)
    # Finish guard and race tie-breakers
    finish_guard_threshold: tuple[int, int] = (
        max(1, strategy_config.home_start - 6),
        strategy_config.home_start,
    )
    finish_guard_min_ready: tuple[int, int] = (1, 2)
    finish_bonus: tuple[float, float] = (0.5, 1.5)
    extra_turn_bonus: tuple[float, float] = (0.6, 1.4)
    distance_weight: tuple[float, float] = (0.02, 0.06)

    def sample(self, rng: random.Random | None = None) -> dict[str, float]:
        rng = rng or random
        low, high = self.finish_guard_threshold
        low = max(1, min(low, strategy_config.home_start))
        high = max(low, min(high, strategy_config.home_start))
        return {
            "yard_bonus": rng.uniform(*self.yard_bonus),
            "lag_weight": rng.uniform(*self.lag_weight),
            "lead_penalty": rng.uniform(*self.lead_penalty),
            "safe_zone_bonus": rng.uniform(*self.safe_zone_bonus),
            "risk_penalty": rng.uniform(*self.risk_penalty),
            "variance_weight": rng.uniform(*self.variance_weight),
            "min_progress_weight": rng.uniform(*self.min_progress_weight),
            "stack_penalty": rng.uniform(*self.stack_penalty),
            "laggard_safe_bonus": rng.uniform(*self.laggard_safe_bonus),
            "laggard_risk_scale": rng.uniform(*self.laggard_risk_scale),
            "target_active": rng.randint(*self.target_active),
            "allowed_exit_risk": rng.uniform(*self.allowed_exit_risk),
            "exit_home_bonus": rng.uniform(*self.exit_home_bonus),
            "exit_home_penalty": rng.uniform(*self.exit_home_penalty),
            "finish_guard_threshold": rng.randint(low, high),
            "finish_guard_min_ready": rng.randint(*self.finish_guard_min_ready),
            "finish_bonus": rng.uniform(*self.finish_bonus),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "distance_weight": rng.uniform(*self.distance_weight),
        }


@dataclass(slots=True)
class SupportStrategy(BaseStrategy):
    """Keeps the team's pieces evenly developed instead of racing a single runner."""

    name: ClassVar[str] = "support"
    config: ClassVar[SupportStrategyConfig] = SupportStrategyConfig()

    yard_bonus: float = 4.0
    lag_weight: float = 0.3
    lead_penalty: float = 0.2
    safe_zone_bonus: float = 1.5
    risk_penalty: float = 0.8
    variance_weight: float = 0.4
    min_progress_weight: float = 0.6
    stack_penalty: float = 0.6
    laggard_safe_bonus: float = 0.8
    laggard_risk_scale: float = 0.4
    target_active: int = 2
    allowed_exit_risk: float = 0.3
    exit_home_bonus: float = 1.2
    exit_home_penalty: float = 1.0
    finish_guard_threshold: int = strategy_config.home_start - 4
    finish_guard_min_ready: int = 1
    finish_bonus: float = 1.0
    extra_turn_bonus: float = 1.0
    distance_weight: float = 0.04

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        counts = ctx.my_distribution
        positions = np.arange(counts.shape[0])
        total_pieces = float(counts.sum())
        mean_progress = (
            float(np.dot(positions, counts) / total_pieces) if total_pieces > 0 else 0.0
        )

        # Weighted median for robust center
        def weighted_median(vals: np.ndarray, w: np.ndarray) -> float:
            cumsum = np.cumsum(w)
            if cumsum[-1] <= 0:
                return 0.0
            cutoff = 0.5 * cumsum[-1]
            idx = int(np.searchsorted(cumsum, cutoff))
            return float(vals[min(idx, len(vals) - 1)])

        median_progress = weighted_median(positions, counts)
        min_pos_indices = np.nonzero(counts > 0)[0]
        min_progress = int(min_pos_indices[0]) if len(min_pos_indices) > 0 else 0

        is_laggard = move.current_pos <= median_progress

        score = 0.0

        # Safe board presence: exiting yard
        active_tokens = int(max(0, counts.sum() - counts[0]))
        if move.current_pos == 0 and ctx.dice_roll == 6:
            if move.enters_safe_zone or move.risk <= self.allowed_exit_risk:
                score += self.yard_bonus + (
                    self.exit_home_bonus
                    * (1.2 if active_tokens < self.target_active else 1.0)
                )
            else:
                score -= self.exit_home_penalty

        # Laggard-first principle
        lag_bonus = max(0.0, mean_progress - move.current_pos)
        score += lag_bonus * self.lag_weight
        if move.current_pos == min_progress:
            score += (move.new_pos - move.current_pos) * self.min_progress_weight

        # Leader dampening
        lead_distance = max(0.0, move.new_pos - mean_progress)
        score -= lead_distance * self.lead_penalty

        # Safety and risk — scaled for laggards
        if move.enters_safe_zone:
            score += self.safe_zone_bonus
            if is_laggard:
                score += self.laggard_safe_bonus
        risk_scale = 1.0 + (self.laggard_risk_scale if is_laggard else 0.0)
        score -= move.risk * self.risk_penalty * risk_scale

        # Anti-bunching: avoid non-safe stacks
        if move.forms_blockade and not move.enters_safe_zone:
            score -= self.stack_penalty

        # Gentle race pressure and tempo
        if move.extra_turn:
            score += self.extra_turn_bonus * (1.15 if is_laggard else 1.0)
        score -= move.distance_to_goal * self.distance_weight

        # Variance reduction (approximate, ignoring mean shift): prefer equalization
        if total_pieces > 0:
            delta_var = (
                (move.new_pos - mean_progress) ** 2
                - (move.current_pos - mean_progress) ** 2
            ) / total_pieces
            if delta_var < 0:
                score += (-delta_var) * self.variance_weight

        # Finish guard: small bonus only if team is not too imbalanced
        if move.enters_home:
            ready_count = int(
                (counts[self.finish_guard_threshold :].sum())
                if self.finish_guard_threshold < counts.shape[0]
                else 0
            )
            if ready_count >= self.finish_guard_min_ready:
                score += self.finish_bonus

        return score
