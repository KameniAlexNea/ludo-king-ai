from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..constants import BoardConstants, GameConstants
from .base import Strategy
from .utils import get_opponent_main_positions_with_fallback

# Utility type hint for a move dict
MoveDict = Dict[str, object]


class ProbabilisticV2Strategy(Strategy):
    """
    Probabilistic strategy v2.

    Decision logic:
    - Compute for each valid move a probabilistic multi turn risk estimate
      based on opponent positions and an opportunity score that is non linear
      in progress and reward for valuable captures.
    - Adapt risk tolerance depending on lead factor computed from opponent
      progress mean and variance and current progress.
    - Choose the move maximizing composite score = opportunity - risk_weight * risk_score.
    """

    def __init__(self):
        super().__init__(
            "ProbabilisticV2",
            "Adaptive prob strategy with multi turn risk and non linear scoring",
        )

    # ---- public API ----
    def decide(self, game_context: Dict) -> int:  # type: ignore[override]
        valid_moves: List[MoveDict] = self._get_valid_moves(game_context)
        if not valid_moves:
            return 0

        player_state = game_context.get("player_state", {})
        opponents = game_context.get("opponents", [])
        # board = game_context.get("board", {})
        current_color = player_state.get("color")

        # phase and progress metrics
        my_finished = float(player_state.get("finished_tokens", 0))
        my_progress = my_finished / float(GameConstants.TOKENS_PER_PLAYER)
        opp_progresses = [
            float(opp.get("tokens_finished", 0))
            / float(GameConstants.TOKENS_PER_PLAYER)
            for opp in opponents
        ]
        opp_mean = sum(opp_progresses) / max(1.0, len(opp_progresses))
        opp_var = sum((p - opp_mean) ** 2 for p in opp_progresses) / max(
            1.0, len(opp_progresses)
        )
        opp_std = math.sqrt(opp_var)

        # lead factor normalized by variance to get dynamic threshold
        lead_factor = (my_progress - opp_mean) / max(0.05, opp_std)

        # adaptive risk weight: more than 1 when ahead, less than 1 when behind
        if lead_factor > 0.8:
            risk_weight = 1.3
            lookahead_turns = 2  # safer, shorter horizon
        elif lead_factor < -0.8:
            risk_weight = 0.75
            lookahead_turns = 4  # be bolder, consider longer horizon
        else:
            # moderate stance based on how close game is
            risk_weight = 1.0
            lookahead_turns = 3

        # collect opponent positions on main path
        opponent_positions = self._collect_opponent_positions(
            game_context, current_color
        )

        best_move: Optional[MoveDict] = None
        best_score = float("-inf")

        # pre compute opponent token progress map if available
        opp_token_progress_map = self._collect_opponent_token_progress(game_context)

        for move in valid_moves:
            # immediate finish takes precedence
            if move.get("move_type") == "finish":
                return move["token_id"]

            # compute probabilistic multi turn risk
            risk_prob = self._multi_turn_capture_probability(
                move.get("target_position"),
                opponent_positions,
                lookahead_turns,
            )

            # convert to a non linear risk score, penalize close threats more harshly
            min_dist = self._min_backward_distance_to_any_opponent(
                move.get("target_position"), opponent_positions
            )
            proximity_penalty = (
                math.exp(max(0.0, (7 - min_dist)) / 3.0)
                if min_dist is not None
                else 1.0
            )
            # clamp proximity factor
            proximity_penalty = min(6.0, max(1.0, proximity_penalty))

            # final risk score non linear
            risk_score = risk_prob * (proximity_penalty**1.1)

            # opportunity evaluation
            opportunity_score = self._opportunity_v2(
                move, current_color, opp_token_progress_map
            )

            # adjust opportunity based on phase of game
            phase_multiplier = self._phase_multiplier(my_progress, opp_mean)
            opportunity_score *= phase_multiplier

            # composite objective
            composite = opportunity_score - risk_weight * (risk_score**1.05)

            # attach diagnostics for logging and RL use
            move["v2_prob_risk"] = risk_prob
            move["v2_risk_score"] = risk_score
            move["v2_opportunity"] = opportunity_score
            move["v2_composite"] = composite

            if composite > best_score:
                best_score = composite
                best_move = move

        # fallback
        if best_move:
            return best_move["token_id"]
        return valid_moves[0]["token_id"]

    # ---- helpers ----
    def _collect_opponent_positions(
        self, game_context: Dict, current_color: str
    ) -> List[int]:
        """Return list of opponent token positions on main loop 0..51."""
        return get_opponent_main_positions_with_fallback(game_context, current_color)

    def _collect_opponent_token_progress(self, game_context: Dict) -> Dict[str, float]:
        """Map token id to its normalized progress from 0 to 1 if info exists."""
        result = {}
        opponents = game_context.get("opponents", [])
        for opp in opponents:
            # store per token if available
            for t in opp.get("tokens", []):
                tid = t.get("token_id")
                finished = float(t.get("finished_steps", 0))  # optional
                result[tid] = min(
                    1.0,
                    max(
                        0.0,
                        finished
                        / float(
                            GameConstants.FINISH_POSITION
                            if GameConstants.FINISH_POSITION
                            else 1
                        ),
                    ),
                )
        return result

    def _circular_backward_distance(self, from_pos: int, opp_pos: int) -> Optional[int]:
        """Distance moving backward from from_pos to opp_pos along 0..51 loop."""
        if not isinstance(from_pos, int) or not isinstance(opp_pos, int):
            return None
        if not (
            0 <= from_pos < GameConstants.MAIN_BOARD_SIZE
            and 0 <= opp_pos < GameConstants.MAIN_BOARD_SIZE
        ):
            return None
        if opp_pos <= from_pos:
            return from_pos - opp_pos
        return from_pos + (GameConstants.MAIN_BOARD_SIZE - opp_pos)

    def _min_backward_distance_to_any_opponent(
        self, target: object, opponent_positions: List[int]
    ) -> Optional[int]:
        if not isinstance(target, int):
            return None
        distances = []
        for opp in opponent_positions:
            d = self._circular_backward_distance(target, opp)
            if d is not None:
                distances.append(d)
        return min(distances) if distances else None

    def _single_turn_capture_probability(self, distance: int) -> float:
        """
        Approximate probability an opponent captures exact distance in one of their turns.
        Model:
        1..6 -> 1/6
        7..12 -> approx 1/36 (needs a 6 then remainder)
        >12 -> negligible in one turn, return 0
        """
        if distance is None:
            return 0.0
        if 1 <= distance <= 6:
            return 1.0 / 6.0
        if 7 <= distance <= 12:
            return 1.0 / 36.0
        if 12 <= distance <= 18:
            return 1.0 / 216.0
        return 0.0

    def _multi_turn_capture_probability(
        self, target: object, opponent_positions: List[int], turns: int
    ) -> float:
        """
        Approximate probability at least one opponent captures within next `turns` opponent moves.
        We approximate each opponent token as independent and each opponent move as identical.
        This is an approximation but it balances cost and usefulness.
        """
        if not isinstance(target, int):
            return 0.0
        # for each opponent token compute its per turn capture prob
        p_no_capture = 1.0
        for opp in opponent_positions:
            d = self._circular_backward_distance(target, opp)
            p_turn = self._single_turn_capture_probability(d) if d is not None else 0.0
            # probability this token fails to capture in all upcoming turns
            p_fail_all_turns = (1.0 - p_turn) ** max(1, turns)
            # combine independent tokens
            p_no_capture *= p_fail_all_turns
        return 1.0 - p_no_capture

    def _opportunity_v2(
        self,
        move: MoveDict,
        player_color: str,
        opp_token_progress_map: Dict[str, float],
    ) -> float:
        """
        Opportunity components:
        - immediate capture reward scaled with captured token progress
        - finishing bonus
        - entering home column bonus
        - safety landing bonus
        - non linear progress delta reward
        """
        opportunity = 0.0

        # capture bonus
        if move.get("captures_opponent"):
            captured = move.get("captured_tokens", [])
            # scale reward by how advanced captured tokens were
            total_scale = 0.0
            for c in captured:
                tid = c.get("token_id")
                prog = opp_token_progress_map.get(tid, 0.5)  # fallback mid value
                total_scale += 1.0 + prog  # prefer removing advanced tokens
            opportunity += 2.0 * max(1.0, total_scale)

        # finishing and home column
        mt = move.get("move_type")
        if mt == "finish":
            opportunity += 4.0
        elif mt == "advance_home_column":
            opportunity += 2.0
        elif mt == "exit_home":
            opportunity += 1.2

        # safety
        if move.get("is_safe_move"):
            opportunity += 1.0

        # progress delta non linear
        cur = move.get("current_position")
        tgt = move.get("target_position")
        progress_delta = 0.0
        if isinstance(cur, int) and isinstance(tgt, int):
            if (
                0 <= cur < GameConstants.MAIN_BOARD_SIZE
                and 0 <= tgt < GameConstants.MAIN_BOARD_SIZE
            ):
                raw = (
                    (tgt - cur)
                    if tgt >= cur
                    else (GameConstants.MAIN_BOARD_SIZE - cur + tgt)
                )
                progress_delta = raw / float(GameConstants.MAIN_BOARD_SIZE)
            elif tgt >= BoardConstants.HOME_COLUMN_START:
                progress_delta = 0.25
        # non linear boost to favor larger advances
        opportunity += (progress_delta**1.4) * 3.0

        return opportunity

    def _phase_multiplier(self, my_progress: float, opp_mean: float) -> float:
        """
        Slightly favor progress in late game, favor safety in lead.
        """
        # phase rough cut
        avg_progress = (my_progress + opp_mean) / 2.0
        if avg_progress < 0.25:
            return 0.9  # early game prefer spreading and safety
        if avg_progress < 0.65:
            return 1.0  # mid game neutral
        # late game increase reward for progress and finishing
        return 1.15

    def __repr__(self) -> str:  # pragma: no cover
        return "ProbabilisticV2Strategy(v2 multi turn risk, adaptive)"
