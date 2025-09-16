"""HybridProbStrategy

Combines strengths of probabilistic strategies (v1/v2) plus selected modular
components (subset of planned v3) with a focus on:
  - Immediate + horizon blended risk
  - Adaptive risk weight based on variance-aware lead factor
  - Proximity & cluster amplification
  - Impact weighting (losing advanced token costlier)
  - Capture value scaled by captured token progress
  - Non-linear progress & home depth scoring
  - Extra-turn expected value (capture or rolling a six)
  - Future safety potential (reachable safe squares next turn)
  - Threat suppression (removing nearby threats) & spread activation bonus

Structured for extensibility while keeping computation modest.
Attaches rich diagnostics to each move for analysis / RL shaping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ludo_engine.constants import BoardConstants, GameConstants, StrategyConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.utils import get_opponent_main_positions_with_fallback

MoveDict = Dict[str, object]


@dataclass
class HybridConfig:
    horizon_turns: int = 3
    use_proximity: bool = True
    use_cluster: bool = True
    use_future_safety: bool = True
    use_threat_suppression: bool = True
    use_extra_turn_ev: bool = True
    use_spread_bonus: bool = True
    use_capture_progress_scaling: bool = True
    normalize: bool = True
    pareto_prune: bool = True


class HybridProbStrategy(Strategy):
    def __init__(self, config: Optional[HybridConfig] = None):
        super().__init__(
            "HybridProb",
            "Hybrid probabilistic strategy blending risk horizon, progress and safety",
        )
        self.cfg = config or HybridConfig()

    # ---- Public API ----
    def decide(self, game_context: Dict) -> int:  # type: ignore[override]
        moves: List[MoveDict] = self._get_valid_moves(game_context)
        if not moves:
            return 0

        player_state = game_context.get("player_state", {})
        current_color = player_state.get("color")
        opponents = game_context.get("opponents", [])

        finished = float(player_state.get("finished_tokens", 0))
        my_progress = finished / float(GameConstants.TOKENS_PER_PLAYER)
        opp_progresses = [
            o.get("tokens_finished", 0) / float(GameConstants.TOKENS_PER_PLAYER)
            for o in opponents
        ]
        opp_mean = (
            sum(opp_progresses) / max(1, len(opp_progresses)) if opp_progresses else 0.0
        )
        opp_var = (
            sum((p - opp_mean) ** 2 for p in opp_progresses)
            / max(1, len(opp_progresses))
            if opp_progresses
            else 0.0
        )
        opp_std = math.sqrt(opp_var)
        lead_factor = (my_progress - opp_mean) / max(0.05, opp_std)

        # Adaptive risk weight & horizon
        if lead_factor > StrategyConstants.HYBRID_LEAD_FACTOR_STRONG:
            risk_weight = 1.3
            horizon = max(2, self.cfg.horizon_turns - 1)
        elif lead_factor < StrategyConstants.HYBRID_BEHIND_FACTOR_STRONG:
            risk_weight = 0.75
            horizon = self.cfg.horizon_turns + 1
        else:
            risk_weight = 1.0
            horizon = self.cfg.horizon_turns

        opponent_positions = self._collect_opponent_positions(
            game_context, current_color
        )
        opp_token_progress_map = self._collect_opponent_token_progress(game_context)
        baseline_active = player_state.get("active_tokens", 0)

        scored: List[MoveDict] = []

        for mv in moves:
            # Finish priority
            if mv.get("move_type") == "finish":
                return mv["token_id"]

            immediate_risk = self._immediate_risk(mv, opponent_positions)
            horizon_risk = self._horizon_risk(mv, opponent_positions, horizon)
            blended_risk = (
                StrategyConstants.HYBRID_IMMEDIATE_RISK_WEIGHT * immediate_risk
                + (1 - StrategyConstants.HYBRID_IMMEDIATE_RISK_WEIGHT) * horizon_risk
            )

            proximity_factor = (
                self._proximity_factor(mv, opponent_positions)
                if self.cfg.use_proximity
                else 1.0
            )
            cluster_factor = (
                self._cluster_factor(mv, opponent_positions)
                if self.cfg.use_cluster
                else 1.0
            )
            impact_weight = self._impact_weight(mv)
            risk_score = (
                blended_risk * proximity_factor * cluster_factor * impact_weight
            )

            # Opportunity components
            opp_score = 0.0
            opp_score += self._capture_value(mv, opp_token_progress_map)
            opp_score += self._progress_value(mv)
            opp_score += self._home_column_value(mv)
            if mv.get("is_safe_move"):
                opp_score += StrategyConstants.HYBRID_SAFE_LANDING_BONUS
            if self.cfg.use_extra_turn_ev:
                opp_score += self._extra_turn_ev(mv)
            if self.cfg.use_threat_suppression:
                opp_score += self._risk_suppression_bonus(mv, opponent_positions)
            if self.cfg.use_spread_bonus:
                opp_score += self._spread_bonus(mv, baseline_active)
            if self.cfg.use_future_safety:
                opp_score += self._future_safety_potential(mv)

            # Phase scaling (late game) using mean progress
            avg_progress = (my_progress + opp_mean) / 2.0
            if avg_progress < 0.25:
                opp_score *= StrategyConstants.HYBRID_EARLY_GAME_PROGRESS_MULT
            elif avg_progress > 0.65:
                opp_score *= StrategyConstants.HYBRID_LATE_GAME_PROGRESS_MULT

            composite_raw = opp_score - risk_weight * (
                risk_score**StrategyConstants.HYBRID_COMPOSITE_RISK_POWER
            )

            mv["hy_risk_immediate"] = immediate_risk
            mv["hy_risk_horizon"] = horizon_risk
            mv["hy_risk_blended"] = blended_risk
            mv["hy_risk_proximity_factor"] = proximity_factor
            mv["hy_risk_cluster_factor"] = cluster_factor
            mv["hy_risk_impact_weight"] = impact_weight
            mv["hy_risk_score"] = risk_score
            mv["hy_opportunity"] = opp_score
            mv["hy_composite_raw"] = composite_raw
            mv["hy_lead_factor"] = lead_factor

            scored.append(mv)

        # Optional Pareto pruning
        candidates = self._pareto_filter(scored) if self.cfg.pareto_prune else scored

        # Normalization
        if self.cfg.normalize and len(candidates) > 1:
            self._normalize_composite(candidates)
        else:
            for m in candidates:
                m["hy_composite"] = m.get("hy_composite_raw", 0.0)

        # Select best; tie-breaker: lower risk_score then higher opportunity
        best = max(
            candidates,
            key=lambda m: (
                m["hy_composite"],
                -m["hy_risk_score"],
                m["hy_opportunity"],
            ),
        )
        return best["token_id"]

    # ---- Risk helpers ----
    def _immediate_risk(self, move: MoveDict, opponent_positions: List[int]) -> float:
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 0.0
        if (
            move.get("is_safe_move")
            or move.get("move_type") in {"finish", "advance_home_column"}
            or tgt >= BoardConstants.HOME_COLUMN_START
        ):
            return 0.0
        threats = 0
        for opp in opponent_positions:
            dist = self._backward_distance(tgt, opp)
            if dist is not None and 1 <= dist <= 6:
                threats += 1
        if threats == 0:
            return 0.0
        return 1 - (5 / 6) ** threats

    def _horizon_risk(
        self, move: MoveDict, opponent_positions: List[int], turns: int
    ) -> float:
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 0.0
        if (
            move.get("is_safe_move")
            or move.get("move_type") in {"finish", "advance_home_column"}
            or tgt >= BoardConstants.HOME_COLUMN_START
        ):
            return 0.0
        if not opponent_positions:
            return 0.0
        p_no_capture = 1.0
        for opp in opponent_positions:
            dist = self._backward_distance(tgt, opp)
            if dist is None:
                continue
            p_turn = self._single_turn_capture_probability(dist)
            p_no_capture *= (1 - p_turn) ** max(1, turns)
        return 1.0 - p_no_capture

    def _proximity_factor(self, move: MoveDict, opponent_positions: List[int]) -> float:
        if not opponent_positions:
            return 1.0
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 1.0
        dists = [self._backward_distance(tgt, opp) for opp in opponent_positions]
        dists = [d for d in dists if d is not None]
        if not dists:
            return 1.0
        min_d = min(dists)
        val = math.exp(max(0.0, (StrategyConstants.HYBRID_PROXIMITY_REF - min_d)) / 3.0)
        return min(StrategyConstants.HYBRID_PROXIMITY_PENALTY_CAP, max(1.0, val))

    def _cluster_factor(self, move: MoveDict, opponent_positions: List[int]) -> float:
        if not opponent_positions:
            return 1.0
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 1.0
        close = 0
        for opp in opponent_positions:
            dist = self._backward_distance(tgt, opp)
            if dist is not None and dist <= 6:
                close += 1
        if close <= 1:
            return 1.0
        return 1.0 + StrategyConstants.HYBRID_CLUSTER_INCREMENT * (close - 1)

    def _impact_weight(self, move: MoveDict) -> float:
        cur = move.get("current_position")
        if not isinstance(cur, int):
            return 1.0
        if cur < 0:
            return 1.0
        if cur >= BoardConstants.HOME_COLUMN_START:
            return 1.0
        norm = cur / float(GameConstants.MAIN_BOARD_SIZE)
        return (
            StrategyConstants.HYBRID_IMPACT_BASE
            + (norm**StrategyConstants.HYBRID_IMPACT_PROGRESS_POWER) * 1.3
        )

    # ---- Opportunity helpers ----
    def _capture_value(
        self, move: MoveDict, opp_token_progress_map: Dict[str, float]
    ) -> float:
        if not move.get("captures_opponent"):
            return 0.0
        captured = move.get("captured_tokens", [])
        total_scale = 0.0
        for c in captured:
            tid = c.get("token_id")
            prog = opp_token_progress_map.get(tid, 0.5)
            total_scale += 1.0 + prog
        return StrategyConstants.HYBRID_CAPTURE_BASE * max(1.0, total_scale)

    def _progress_value(self, move: MoveDict) -> float:
        cur = move.get("current_position")
        tgt = move.get("target_position")
        if not isinstance(cur, int) or not isinstance(tgt, int):
            return 0.0
        delta = 0.0
        if (
            0 <= cur < GameConstants.MAIN_BOARD_SIZE
            and 0 <= tgt < GameConstants.MAIN_BOARD_SIZE
        ):
            raw = (
                (tgt - cur)
                if tgt >= cur
                else (GameConstants.MAIN_BOARD_SIZE - cur + tgt)
            )
            delta = raw / float(GameConstants.MAIN_BOARD_SIZE)
        elif tgt >= BoardConstants.HOME_COLUMN_START:
            delta = 0.25
        if delta <= 0:
            return 0.0
        return (
            delta**StrategyConstants.HYBRID_PROGRESS_POWER
        ) * StrategyConstants.HYBRID_PROGRESS_SCALE

    def _home_column_value(self, move: MoveDict) -> float:
        mt = move.get("move_type")
        if mt == "finish":
            return StrategyConstants.HYBRID_FINISH_BONUS
        if mt == "advance_home_column":
            pos = move.get("target_position")
            if isinstance(pos, int):
                depth = pos - GameConstants.HOME_COLUMN_START
                return (
                    StrategyConstants.HYBRID_ADVANCE_HOME_BONUS
                    + depth * StrategyConstants.HYBRID_HOME_DEPTH_FACTOR * 0.1
                )
            return StrategyConstants.HYBRID_ADVANCE_HOME_BONUS
        if mt == "exit_home":
            return StrategyConstants.HYBRID_EXIT_HOME_BONUS
        return 0.0

    def _extra_turn_ev(self, move: MoveDict) -> float:
        capture_turn = 1.0 if move.get("captures_opponent") else 0.0
        roll_six_prob = 1.0 / 6.0
        expected_additional = capture_turn + roll_six_prob
        return (
            expected_additional
            * StrategyConstants.HYBRID_EXTRA_TURN_PROGRESS_NORM
            * StrategyConstants.HYBRID_EXTRA_TURN_COEFF
        )

    def _risk_suppression_bonus(
        self, move: MoveDict, opponent_positions: List[int]
    ) -> float:
        if not move.get("captures_opponent"):
            return 0.0
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 0.0
        removed = 0
        for opp in opponent_positions:
            dist = self._backward_distance(tgt, opp)
            if dist is not None and 1 <= dist <= 6:
                removed += 1
        if removed == 0:
            return 0.0
        return removed * StrategyConstants.HYBRID_RISK_SUPPRESSION_COEFF

    def _spread_bonus(self, move: MoveDict, baseline_active: int) -> float:
        if (
            move.get("move_type") == "exit_home"
            and baseline_active < StrategyConstants.HYBRID_SPREAD_ACTIVE_TARGET
        ):
            return StrategyConstants.HYBRID_SPREAD_BONUS
        return 0.0

    def _future_safety_potential(self, move: MoveDict) -> float:
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or tgt >= BoardConstants.HOME_COLUMN_START:
            return 0.0
        safe_set = BoardConstants.get_all_safe_squares()
        potential = 0.0
        for d in range(1, 7):
            np = (tgt + d) % GameConstants.MAIN_BOARD_SIZE
            if np in safe_set:
                potential += StrategyConstants.HYBRID_FUTURE_SAFETY_BONUS
        return potential

    # ---- Selection helpers ----
    def _pareto_filter(self, moves: Sequence[MoveDict]) -> List[MoveDict]:
        result: List[MoveDict] = []
        for m in moves:
            dominated = False
            for o in moves:
                if o is m:
                    continue
                if (
                    o.get("hy_opportunity", 0.0) >= m.get("hy_opportunity", 0.0)
                    and o.get("hy_risk_score", 1.0) <= m.get("hy_risk_score", 1.0)
                    and (
                        o.get("hy_opportunity", 0.0) > m.get("hy_opportunity", 0.0)
                        or o.get("hy_risk_score", 1.0) < m.get("hy_risk_score", 1.0)
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                result.append(m)
        return result if result else list(moves)

    def _normalize_composite(self, moves: Sequence[MoveDict]):
        scores = [m.get("hy_composite_raw", 0.0) for m in moves]
        if not scores:
            return
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores))
        std = math.sqrt(var) or 1.0
        for m in moves:
            m["hy_composite"] = (m.get("hy_composite_raw", 0.0) - mean) / std

    # ---- Generic helpers ----
    def _collect_opponent_positions(
        self, game_context: Dict, current_color: str
    ) -> List[int]:
        res = get_opponent_main_positions_with_fallback(game_context, current_color)
        return res

    def _collect_opponent_token_progress(self, game_context: Dict) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for opp in game_context.get("opponents", []):
            for t in opp.get("tokens", []):
                tid = t.get("token_id")
                finished = float(t.get("finished_steps", 0))
                result[tid] = min(
                    1.0, max(0.0, finished / float(GameConstants.FINISH_POSITION))
                )
        return result

    def _backward_distance(self, from_pos: int, opp_pos: int) -> Optional[int]:
        if not (isinstance(from_pos, int) and isinstance(opp_pos, int)):
            return None
        if not (
            0 <= from_pos < GameConstants.MAIN_BOARD_SIZE
            and 0 <= opp_pos < GameConstants.MAIN_BOARD_SIZE
        ):
            return None
        if opp_pos <= from_pos:
            return from_pos - opp_pos
        return from_pos + (GameConstants.MAIN_BOARD_SIZE - opp_pos)

    def _single_turn_capture_probability(self, distance: Optional[int]) -> float:
        if distance is None:
            return 0.0
        if 1 <= distance <= 6:
            return 1.0 / 6.0
        if 7 <= distance <= 12:
            return 1.0 / 36.0
        if 12 <= distance <= 18:
            return 1.0 / 216.0
        return 0.0


__all__ = ["HybridProbStrategy", "HybridConfig"]
