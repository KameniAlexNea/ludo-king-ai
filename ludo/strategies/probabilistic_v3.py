"""ProbabilisticV3 Strategy

Incorporates advanced risk / opportunity modeling with modular components
that can be toggled on/off for experimentation and RL feature shaping.

Features (all configurable):
  Risk refinements:
    - Immediate vs horizon risk blending
    - Impact weighting (late-progress tokens costlier to lose)
    - Threat cluster amplification
    - Discounted multi-turn capture probability
    - Proximity non-linear penalty
    - Chase deterrence (lower risk if opponents would endanger themselves by chasing)

  Opportunity refinements:
    - Capture value scaled by captured token progress
    - Extra-turn expected value (capture or rolling a six)
    - Risk suppression bonus for removing threats
    - Spread bonus for increasing number of active tokens
    - Future safety potential (nearby safe squares reachable next turn)
    - Home column non-linear depth scaling
    - Non-linear progress delta

  Selection refinements:
    - Pareto pruning of dominated moves
    - Optional normalization (per-turn z-score) for stable composite scale
    - Optional soft top-k stochastic selection (exploration)

  RL integration:
    - Rich per-move diagnostics attached directly to move dicts
    - Configurable exploration temperature

All heuristics are intentionally encapsulated for easy removal or
replacement. Each sub-score returns interpretable values.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ludo_engine.constants import BoardConstants, GameConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.utils import (
    get_my_main_positions_with_fallback,
    get_opponent_main_positions_with_fallback,
)

MoveDict = Dict[str, object]


@dataclass
class V3Config:
    # Risk toggles
    use_impact_weight: bool = True
    use_dual_horizon_risk: bool = True
    use_cluster_factor: bool = True
    use_discounted_horizon: bool = True
    use_proximity_penalty: bool = True
    use_chase_deterrence: bool = True

    # Opportunity toggles
    use_capture_progress_scaling: bool = True
    use_extra_turn_ev: bool = True
    use_risk_suppression_capture: bool = True
    use_spread_bonus: bool = True
    use_future_safety: bool = True
    use_home_column_nonlinear: bool = True
    use_progress_nonlinear: bool = True

    # Selection & RL
    use_pareto_prune: bool = True
    use_normalization: bool = True
    use_soft_topk: bool = False
    soft_topk_k: int = 3
    softmax_temperature: float = 0.9
    exploration_epsilon: float = 0.0  # optional epsilon-random among valid

    log_diagnostics: bool = True

    # Weights / parameters
    horizon_turns: int = 3
    alpha_immediate: float = 0.6  # blend immediate vs horizon
    cluster_increment: float = 0.15
    discount_lambda: float = 0.85
    chase_deterrence_unit: float = 0.05
    extra_turn_progress_norm: float = 3.5 / GameConstants.MAIN_BOARD_SIZE
    extra_turn_coeff: float = 2.2
    risk_suppression_coeff: float = 0.7
    spread_bonus_value: float = 0.5
    future_safety_bonus: float = 0.2
    home_column_depth_factor: float = 2.5
    progress_nonlinear_power: float = 1.4
    progress_nonlinear_scale: float = 3.0
    finish_bonus: float = 4.5
    advance_home_bonus: float = 2.0
    exit_home_bonus: float = 1.2
    safe_landing_bonus: float = 1.0
    base_capture_value: float = 2.0
    composite_risk_power: float = 1.05
    proximity_penalty_cap: float = 6.0
    proximity_ref: int = 7


class ProbabilisticV3Strategy(Strategy):
    def __init__(self, config: Optional[V3Config] = None):
        super().__init__(
            "ProbabilisticV3",
            "Full-featured probabilistic strategy with modular risk & opportunity components",
        )
        self.cfg = config or V3Config()

    # ---- public API ----
    def decide(self, game_context: Dict) -> int:  # type: ignore[override]
        moves: List[MoveDict] = self._get_valid_moves(game_context)
        if not moves:
            return 0

        player_state = game_context.get("player_state", {})
        current_color = player_state.get("color")
        opponents = game_context.get("opponents", [])

        my_progress = player_state.get("finished_tokens", 0) / float(
            GameConstants.TOKENS_PER_PLAYER
        )
        opp_progresses = [
            opp.get("tokens_finished", 0) / float(GameConstants.TOKENS_PER_PLAYER)
            for opp in opponents
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

        # Dynamic risk weight & horizon
        if lead_factor > 0.8:
            risk_weight = 1.3
            horizon_turns = max(2, self.cfg.horizon_turns - 1)
        elif lead_factor < -0.8:
            risk_weight = 0.75
            horizon_turns = self.cfg.horizon_turns + 1
        else:
            risk_weight = 1.0
            horizon_turns = self.cfg.horizon_turns

        opponent_positions = self._collect_opponent_positions(
            game_context, current_color
        )
        own_positions = self._collect_own_positions(game_context, current_color)
        opp_token_progress_map = self._collect_opponent_token_progress(game_context)

        baseline_active_tokens = player_state.get("active_tokens", 0)

        scored_moves: List[MoveDict] = []

        for mv in moves:
            if mv.get("move_type") == "finish":
                # Immediate finish trump
                return mv["token_id"]

            # RISK COMPONENTS ------------------------------------
            immediate_risk = self._single_step_risk(mv, opponent_positions)
            horizon_risk = self._horizon_risk(mv, opponent_positions, horizon_turns)
            if self.cfg.use_dual_horizon_risk:
                combined_prob = (
                    self.cfg.alpha_immediate * immediate_risk
                    + (1 - self.cfg.alpha_immediate) * horizon_risk
                )
            else:
                combined_prob = horizon_risk

            proximity_factor = (
                self._proximity_factor(mv, opponent_positions)
                if self.cfg.use_proximity_penalty
                else 1.0
            )
            cluster_factor = (
                self._cluster_factor(mv, opponent_positions)
                if self.cfg.use_cluster_factor
                else 1.0
            )
            risk_prob = combined_prob * proximity_factor * cluster_factor

            impact_weight = (
                self._impact_weight(mv) if self.cfg.use_impact_weight else 1.0
            )
            risk_score = risk_prob * impact_weight

            # Chase deterrence lowers effective risk
            if self.cfg.use_chase_deterrence:
                deterrence = self._chase_deterrence(
                    mv, opponent_positions, own_positions
                )
                risk_score *= max(0.0, 1.0 - deterrence)

            # OPPORTUNITY COMPONENTS -----------------------------
            opportunity = 0.0
            opportunity += self._capture_value(mv, opp_token_progress_map)
            opportunity += self._progress_value(mv)
            opportunity += self._home_column_value(mv)
            if mv.get("is_safe_move"):
                opportunity += self.cfg.safe_landing_bonus
            if self.cfg.use_extra_turn_ev:
                opportunity += self._extra_turn_ev(mv)
            if self.cfg.use_risk_suppression_capture:
                opportunity += self._risk_suppression_bonus(mv, opponent_positions)
            if self.cfg.use_spread_bonus:
                opportunity += self._spread_bonus(mv, baseline_active_tokens)
            if self.cfg.use_future_safety:
                opportunity += self._future_safety_potential(mv)

            # Phase scaling (late game)
            opportunity *= self._phase_multiplier(my_progress, opp_mean)

            # COMPOSITE ------------------------------------------
            composite = opportunity - risk_weight * (
                risk_score**self.cfg.composite_risk_power
            )

            mv["v3_risk_prob"] = risk_prob
            mv["v3_risk_score"] = risk_score
            mv["v3_opportunity"] = opportunity
            mv["v3_composite_raw"] = composite
            mv["v3_components"] = {
                "immediate_risk": immediate_risk,
                "horizon_risk": horizon_risk,
                "proximity_factor": proximity_factor,
                "cluster_factor": cluster_factor,
                "impact_weight": impact_weight,
            }
            scored_moves.append(mv)

        # Selection refinements
        candidates = scored_moves
        if self.cfg.use_pareto_prune:
            candidates = self._pareto_filter(candidates)

        if self.cfg.use_normalization and len(candidates) > 1:
            self._normalize_scores(candidates)
        else:
            for mv in candidates:
                mv["v3_composite"] = mv.get("v3_composite_raw")

        # Epsilon exploration
        if (
            self.cfg.exploration_epsilon > 0
            and random.random() < self.cfg.exploration_epsilon
        ):
            return random.choice(candidates)["token_id"]

        # Soft top-k exploration
        if self.cfg.use_soft_topk and len(candidates) > 1:
            chosen = self._soft_topk_choice(candidates)
            return chosen["token_id"]

        # Deterministic best
        best = max(candidates, key=lambda m: m["v3_composite"])
        return best["token_id"]

    # ---- RISK helpers ----
    def _single_step_risk(self, move: MoveDict, opponent_positions: List[int]) -> float:
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 0.0
        if (
            move.get("move_type") == "finish"
            or (isinstance(tgt, int) and tgt >= BoardConstants.HOME_COLUMN_START)
            or move.get("is_safe_move")
        ):
            return 0.0
        threats = 0
        for opp in opponent_positions:
            dist = self._backward_distance(tgt, opp)
            if 1 <= dist <= 6:
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
            move.get("move_type") == "finish"
            or move.get("is_safe_move")
            or (isinstance(tgt, int) and tgt >= BoardConstants.HOME_COLUMN_START)
        ):
            return 0.0
        if not opponent_positions:
            return 0.0
        if self.cfg.use_discounted_horizon:
            # discounted independent approximation
            p_no_capture = 1.0
            for opp in opponent_positions:
                d = self._backward_distance(tgt, opp)
                p_turn = (
                    self._single_turn_capture_probability(d) if d is not None else 0.0
                )
                # geometric discount across turns
                effective_fail = 1.0
                for t in range(turns):
                    weight = self.cfg.discount_lambda**t
                    effective_fail *= 1.0 - weight * p_turn
                p_no_capture *= effective_fail
            return 1.0 - p_no_capture
        else:
            # simple multi-turn as earlier versions
            p_no = 1.0
            for opp in opponent_positions:
                d = self._backward_distance(tgt, opp)
                p_turn = (
                    self._single_turn_capture_probability(d) if d is not None else 0.0
                )
                p_fail = (1 - p_turn) ** max(1, turns)
                p_no *= p_fail
            return 1.0 - p_no

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
        val = math.exp(max(0.0, (self.cfg.proximity_ref - min_d)) / 3.0)
        return min(self.cfg.proximity_penalty_cap, max(1.0, val))

    def _cluster_factor(self, move: MoveDict, opponent_positions: List[int]) -> float:
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 1.0
        close = 0
        for opp in opponent_positions:
            d = self._backward_distance(tgt, opp)
            if d is not None and 1 <= d <= 8:
                close += 1
        if close <= 1:
            return 1.0
        return 1.0 + self.cfg.cluster_increment * (close - 1)

    def _impact_weight(self, move: MoveDict) -> float:
        cur = move.get("current_position")
        if not isinstance(cur, int):
            return 1.0
        if cur < 0:
            return 0.7  # still at home
        if cur >= BoardConstants.HOME_COLUMN_START:
            return 0.3  # already in home column -> nearly safe
        # normalized progress on loop
        norm = cur / float(GameConstants.MAIN_BOARD_SIZE)
        return 0.5 + (norm**1.2) * 1.3

    def _chase_deterrence(
        self, move: MoveDict, opponent_positions: List[int], own_positions: List[int]
    ) -> float:
        """Estimate reduction in risk because opponents would expose themselves if they chase.
        Heuristic: count opponents within 1..6 behind whose own backward distance to one
        of our OTHER tokens (not the moved one) is <=6, implying potential counter-capture.
        deterrence = unit * count (clamped 0..0.5)
        """
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or not own_positions:
            return 0.0
        count = 0
        for opp in opponent_positions:
            d = self._backward_distance(tgt, opp)
            if d is None or d < 1 or d > 6:
                continue
            # would they land near one of ours if they advanced toward us?
            for own in own_positions:
                if own == tgt:
                    continue
                # if we are within 6 behind their potential landing spot, we threaten them
                back = self._backward_distance(opp, own)
                if back is not None and 1 <= back <= 6:
                    count += 1
                    break
        return min(0.5, count * self.cfg.chase_deterrence_unit)

    # ---- OPPORTUNITY helpers ----
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
            total_scale += 1.0 + prog if self.cfg.use_capture_progress_scaling else 1.0
        return self.cfg.base_capture_value * max(1.0, total_scale)

    def _progress_value(self, move: MoveDict) -> float:
        cur = move.get("current_position")
        tgt = move.get("target_position")
        if not isinstance(cur, int) or not isinstance(tgt, int):
            return 0.0
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
        else:
            delta = 0.0
        if self.cfg.use_progress_nonlinear:
            return (
                delta**self.cfg.progress_nonlinear_power
            ) * self.cfg.progress_nonlinear_scale
        return delta

    def _home_column_value(self, move: MoveDict) -> float:
        mt = move.get("move_type")
        if mt == "finish":
            return self.cfg.finish_bonus
        if mt == "advance_home_column":
            if self.cfg.use_home_column_nonlinear:
                tgt = move.get("target_position")
                if isinstance(tgt, int) and tgt >= BoardConstants.HOME_COLUMN_START:
                    depth = (tgt - BoardConstants.HOME_COLUMN_START) / 5.0  # 0..1
                    return (
                        self.cfg.advance_home_bonus
                        + depth * self.cfg.home_column_depth_factor
                    )
            return self.cfg.advance_home_bonus
        if mt == "exit_home":
            return self.cfg.exit_home_bonus
        return 0.0

    def _extra_turn_ev(self, move: MoveDict) -> float:
        capture_bonus_turn = 1.0 if move.get("captures_opponent") else 0.0
        roll_six_prob = 1.0 / 6.0
        expected_additional = capture_bonus_turn + roll_six_prob
        # value of an extra turn approximated by avg forward progress
        return (
            expected_additional
            * self.cfg.extra_turn_progress_norm
            * self.cfg.extra_turn_coeff
        )

    def _risk_suppression_bonus(
        self, move: MoveDict, opponent_positions: List[int]
    ) -> float:
        if not move.get("captures_opponent"):
            return 0.0
        tgt = move.get("target_position")
        if not isinstance(tgt, int):
            return 0.0
        removed_threats = 0
        for opp in opponent_positions:
            if opp == tgt:  # captured those exactly on square
                removed_threats += 1
        if removed_threats == 0:
            return 0.0
        return removed_threats * self.cfg.risk_suppression_coeff

    def _spread_bonus(self, move: MoveDict, baseline_active: int) -> float:
        # If exiting home increases number of active tokens
        if move.get("move_type") == "exit_home" and baseline_active == 0:
            return self.cfg.spread_bonus_value
        return 0.0

    def _future_safety_potential(self, move: MoveDict) -> float:
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or tgt >= BoardConstants.HOME_COLUMN_START:
            return 0.0
        for d in range(1, 7):
            nxt = (tgt + d) % GameConstants.MAIN_BOARD_SIZE
            if BoardConstants.is_safe_position(nxt):
                return self.cfg.future_safety_bonus
        return 0.0

    # ---- Selection helpers ----
    def _pareto_filter(self, moves: Sequence[MoveDict]) -> List[MoveDict]:
        result: List[MoveDict] = []
        for m in moves:
            dominated = False
            for n in moves:
                if n is m:
                    continue
                if (
                    n.get("v3_risk_score", 0) <= m.get("v3_risk_score", 0)
                    and n.get("v3_opportunity", 0) >= m.get("v3_opportunity", 0)
                    and (
                        n.get("v3_risk_score", 0) < m.get("v3_risk_score", 0)
                        or n.get("v3_opportunity", 0) > m.get("v3_opportunity", 0)
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                result.append(m)
        return result if result else list(moves)

    def _normalize_scores(self, moves: Sequence[MoveDict]):
        comps = [m.get("v3_composite_raw", 0.0) for m in moves]
        if not comps:
            return
        mean = sum(comps) / len(comps)
        var = sum((c - mean) ** 2 for c in comps) / max(1, len(comps))
        std = math.sqrt(var) or 1.0
        for m in moves:
            m["v3_composite"] = (m.get("v3_composite_raw", 0.0) - mean) / std

    def _soft_topk_choice(self, moves: Sequence[MoveDict]) -> MoveDict:
        k = min(self.cfg.soft_topk_k, len(moves))
        sorted_moves = sorted(moves, key=lambda m: m["v3_composite"], reverse=True)
        top = sorted_moves[:k]
        temp = max(1e-6, self.cfg.softmax_temperature)
        logits = [m["v3_composite"] / temp for m in top]
        max_logit = max(logits)
        exps = [math.exp(log - max_logit) for log in logits]
        total = sum(exps) or 1.0
        r = random.random() * total
        acc = 0.0
        for m, w in zip(top, exps):
            acc += w
            if r <= acc:
                return m
        return top[-1]

    # ---- Generic helpers ----
    def _collect_opponent_positions(
        self, game_context: Dict, current_color: str
    ) -> List[int]:
        return get_opponent_main_positions_with_fallback(game_context, current_color)

    def _collect_own_positions(
        self, game_context: Dict, current_color: str
    ) -> List[int]:
        return get_my_main_positions_with_fallback(game_context, current_color)

    def _collect_opponent_token_progress(self, game_context: Dict) -> Dict[str, float]:
        # Placeholder: opponents list may not include per-token progress; fallback mid.
        result: Dict[str, float] = {}
        # Game context opponents may contain token arrays
        for opp in game_context.get("opponents", []):
            for t in opp.get("tokens", []):
                tid = t.get("token_id")
                prog = t.get("progress_norm") or 0.5
                if tid is not None:
                    result[tid] = max(0.0, min(1.0, float(prog)))
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

    def _phase_multiplier(self, my_progress: float, opp_mean: float) -> float:
        avg = (my_progress + opp_mean) / 2.0
        if avg < 0.25:
            return 0.9
        if avg < 0.65:
            return 1.0
        return 1.15

    def __repr__(self):  # pragma: no cover
        return f"ProbabilisticV3Strategy(cfg={self.cfg})"


__all__ = ["ProbabilisticV3Strategy", "V3Config"]
