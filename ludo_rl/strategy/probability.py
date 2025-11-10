from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import ClassVar

from ludo_rl.ludo.config import strategy_config

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class ProbabilityStrategyConfig(BaseStrategyConfig):
    # Core terms
    progress_weight: tuple[float, float] = (1.0, 1.8)
    capture_weight: tuple[float, float] = (4.0, 7.0)
    risk_weight: tuple[float, float] = (2.0, 4.0)
    safe_bonus: tuple[float, float] = (1.5, 3.0)
    extra_turn_bonus: tuple[float, float] = (2.0, 4.0)

    # Hybrid probabilistic refinements
    risk_power: tuple[float, float] = (1.0, 1.4)
    proximity_ref: tuple[float, float] = (3.0, 5.0)
    proximity_cap: tuple[float, float] = (2.0, 3.0)
    cluster_increment: tuple[float, float] = (0.12, 0.25)
    impact_base: tuple[float, float] = (0.9, 1.2)
    impact_progress_power: tuple[float, float] = (1.2, 1.8)
    capture_progress_weight: tuple[float, float] = (0.6, 1.4)
    home_depth_weight: tuple[float, float] = (0.3, 0.7)
    future_safety_bonus: tuple[float, float] = (0.2, 0.5)
    spread_active_target: tuple[int, int] = (2, 3)
    spread_bonus: tuple[float, float] = (0.6, 1.2)
    distance_weight: tuple[float, float] = (0.02, 0.06)
    leave_safe_penalty: tuple[float, float] = (0.6, 1.4)
    allowed_exit_risk: tuple[float, float] = (0.0, 0.6)
    exit_home_penalty: tuple[float, float] = (0.8, 2.0)
    extra_turn_ev_coeff: tuple[float, float] = (0.6, 1.2)

    def sample(self, rng: random.Random | None = None) -> dict[str, float | int]:
        rng = rng or random
        return {
            "progress_weight": rng.uniform(*self.progress_weight),
            "capture_weight": rng.uniform(*self.capture_weight),
            "risk_weight": rng.uniform(*self.risk_weight),
            "safe_bonus": rng.uniform(*self.safe_bonus),
            "extra_turn_bonus": rng.uniform(*self.extra_turn_bonus),
            "risk_power": rng.uniform(*self.risk_power),
            "proximity_ref": rng.uniform(*self.proximity_ref),
            "proximity_cap": rng.uniform(*self.proximity_cap),
            "cluster_increment": rng.uniform(*self.cluster_increment),
            "impact_base": rng.uniform(*self.impact_base),
            "impact_progress_power": rng.uniform(*self.impact_progress_power),
            "capture_progress_weight": rng.uniform(*self.capture_progress_weight),
            "home_depth_weight": rng.uniform(*self.home_depth_weight),
            "future_safety_bonus": rng.uniform(*self.future_safety_bonus),
            "spread_active_target": rng.randint(*self.spread_active_target),
            "spread_bonus": rng.uniform(*self.spread_bonus),
            "distance_weight": rng.uniform(*self.distance_weight),
            "leave_safe_penalty": rng.uniform(*self.leave_safe_penalty),
            "allowed_exit_risk": rng.uniform(*self.allowed_exit_risk),
            "exit_home_penalty": rng.uniform(*self.exit_home_penalty),
            "extra_turn_ev_coeff": rng.uniform(*self.extra_turn_ev_coeff),
        }


@dataclass(slots=True)
class ProbabilityStrategy(BaseStrategy):
    """Balances progress with estimated knockout risk."""

    name: ClassVar[str] = "probability"
    config: ClassVar[ProbabilityStrategyConfig] = ProbabilityStrategyConfig()

    progress_weight: float = 1.4
    capture_weight: float = 5.5
    risk_weight: float = 3.2
    safe_bonus: float = 2.2
    extra_turn_bonus: float = 3.0

    # Hybrid refinements (defaults tuned conservatively)
    risk_power: float = 1.2
    proximity_ref: float = 4.0
    proximity_cap: float = 2.5
    cluster_increment: float = 0.18
    impact_base: float = 1.0
    impact_progress_power: float = 1.5
    capture_progress_weight: float = 1.0
    home_depth_weight: float = 0.5
    future_safety_bonus: float = 0.3
    spread_active_target: int = 2
    spread_bonus: float = 0.8
    distance_weight: float = 0.04
    leave_safe_penalty: float = 1.0
    allowed_exit_risk: float = 0.3
    exit_home_penalty: float = 1.2
    extra_turn_ev_coeff: float = 0.9

    def _score_move(self, ctx: StrategyContext, move: MoveOption) -> float:
        # Opportunity terms
        opp_score = 0.0

        # Non-linear progress along main track; small home depth incentive
        opp_score += move.progress * self.progress_weight
        if move.new_pos >= strategy_config.home_start:
            depth = move.new_pos - strategy_config.home_start
            opp_score += depth * self.home_depth_weight

        # Safe landing bonus
        if move.enters_safe_zone:
            opp_score += self.safe_bonus

        # Capture value scaled by foe progress proxy (approx via landing position)
        if move.can_capture:
            opp_score += move.capture_count * self.capture_weight
            if 1 <= move.new_pos <= strategy_config.main_track_end:
                prey_progress_frac = move.new_pos / float(
                    strategy_config.main_track_end
                )
                opp_score += (
                    prey_progress_frac
                    * self.capture_progress_weight
                    * move.capture_count
                )

        # Tempo: extra turn (capture or rolling six) expected value
        if move.extra_turn:
            opp_score += self.extra_turn_bonus
        # Expected value of an extra future action: capture gives ~1, rolling a six ~1/6
        ev_extra = (1.0 if move.can_capture else 0.0) + (1.0 / 6.0)
        opp_score += ev_extra * self.extra_turn_ev_coeff

        # Future safety potential: reachable safe squares next turn
        if 1 <= move.new_pos <= strategy_config.main_track_end:
            potential = 0.0
            for d in range(1, 7):
                nxt = move.new_pos + d
                if nxt > strategy_config.main_track_end:
                    nxt -= strategy_config.main_track_end
                if ctx.safe_channel[nxt]:
                    potential += 1.0
            if potential > 0:
                opp_score += potential * self.future_safety_bonus

        # Spread activation when presence is low and exiting home
        if move.current_pos == 0:
            active_tokens = int(
                min(4, max(0, round(float(ctx.my_distribution[1:].sum()))))
            )
            if active_tokens < self.spread_active_target:
                opp_score += self.spread_bonus
            elif move.risk > self.allowed_exit_risk and not move.enters_safe_zone:
                opp_score -= self.exit_home_penalty

        # Gentle pull toward finish for tie-breaking
        opp_score -= move.distance_to_goal * self.distance_weight

        # Risk terms: blended and modulated by proximity, clustering, and impact
        risk_score = self._blended_risk(ctx, move)

        # Composite hybrid: opportunity minus weighted risk with a mild nonlinearity
        score = opp_score - self.risk_weight * (risk_score**self.risk_power)

        # Penalties for abandoning safety
        if move.leaving_safe_zone:
            score -= self.leave_safe_penalty

        return score

    # --- Risk modelling helpers ---
    def _blended_risk(self, ctx: StrategyContext, move: MoveOption) -> float:
        pos = move.new_pos
        home_start = strategy_config.home_start
        main_end = strategy_config.main_track_end
        if pos <= 0 or pos > main_end or pos >= home_start:
            return 0.0
        if ctx.safe_channel[pos]:
            return 0.0

        # Immediate (1 turn) probability approximation
        immediate_threats = 0.0
        for d in range(1, 7):
            idx = pos - d
            if idx <= 0:
                idx += main_end
            if ctx.safe_channel[idx]:
                continue
            immediate_threats += ctx.opponent_distribution[idx]
        p_immediate = (
            0.0 if immediate_threats <= 0 else 1 - (5 / 6) ** immediate_threats
        )

        # Horizon risk (up to 3 dice throws away), aggregated crudely by bands
        band_counts = {1: 0.0, 2: 0.0, 3: 0.0}
        for d in range(1, 19):
            idx = pos - d
            if idx <= 0:
                idx += main_end
            if ctx.safe_channel[idx]:
                continue
            count = ctx.opponent_distribution[idx]
            if 1 <= d <= 6:
                band_counts[1] += count
            elif 7 <= d <= 12:
                band_counts[2] += count
            else:
                band_counts[3] += count

        def p_single(distance_band: int) -> float:
            if distance_band == 1:
                return 1.0 / 6.0
            if distance_band == 2:
                return 1.0 / 36.0
            return 1.0 / 216.0

        p_no = 1.0
        for band, cnt in band_counts.items():
            p = p_single(band)
            if cnt > 0:
                p_no *= (1 - p) ** cnt
        p_horizon = 1.0 - p_no

        # Proximity multiplier and clustering
        proximity = self._proximity_factor(ctx, pos)
        cluster = self._cluster_factor(ctx, pos)

        # Impact weighting: more advanced token => heavier risk
        impact = self._impact_weight(move)

        # Blend: emphasize immediate slightly via geometric mean style
        blended = (0.6 * p_immediate + 0.4 * p_horizon) * proximity * cluster * impact
        return float(max(0.0, min(1.0, blended)))

    def _proximity_factor(self, ctx: StrategyContext, pos: int) -> float:
        main_end = strategy_config.main_track_end
        min_d = None
        for d in range(1, 7):
            idx = pos - d
            if idx <= 0:
                idx += main_end
            if ctx.safe_channel[idx]:
                continue
            if ctx.opponent_distribution[idx] > 0:
                min_d = d
                break
        if min_d is None:
            return 1.0
        val = math.exp(max(0.0, (self.proximity_ref - float(min_d))) / 3.0)
        return float(min(self.proximity_cap, max(1.0, val)))

    def _cluster_factor(self, ctx: StrategyContext, pos: int) -> float:
        main_end = strategy_config.main_track_end
        close = 0.0
        for d in range(1, 7):
            idx = pos - d
            if idx <= 0:
                idx += main_end
            if ctx.safe_channel[idx]:
                continue
            close += ctx.opponent_distribution[idx]
        if close <= 1.0:
            return 1.0
        return 1.0 + self.cluster_increment * (close - 1.0)

    def _impact_weight(self, move: MoveOption) -> float:
        cur = move.current_pos
        if cur <= 0:
            return self.impact_base
        if cur >= strategy_config.home_start:
            return self.impact_base
        norm = cur / float(strategy_config.main_track_end)
        return self.impact_base + (norm**self.impact_progress_power) * 1.0
