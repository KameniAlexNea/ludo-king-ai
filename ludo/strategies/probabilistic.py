"""
Probabilistic Balanced Strategy - Uses risk & opportunity estimates.

This strategy extends the idea of the existing Balanced strategy by
explicitly computing:

* Risk: probability the moved token is captured before its next turn.
  Approximated as 1 - (5/6)^k where k = number of opponent tokens that
  would be 1..6 squares behind the landing square (circular on main path).
  Risk is zero on safe squares, in home column, or when finishing.

* Opportunity: composite of (capture value, progress, safety bonuses,
  entering home column, finishing).

It then adapts its risk tolerance depending on relative progress:
  - If ahead: weight risk higher (play safer)
  - If behind: weight risk lower (play more boldly)
  - Otherwise: neutral weighting

Returned move is the one with highest (opportunity - risk_weight * risk).

The extra per-move metrics (risk, opportunity, composite_score) are added
back onto the move dict so they can later feed into RL reward shaping or
logging without changing the core engine.
"""

from __future__ import annotations

from typing import Dict, List

from ludo_engine.constants import BoardConstants, GameConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.utils import get_opponent_main_positions_with_fallback


class ProbabilisticStrategy(Strategy):
    """Adaptive strategy using explicit probability-based evaluation."""

    def __init__(self):
        super().__init__(
            "Probabilistic",
            "Adaptive strategy using probability of capture vs. opportunity gain",
        )

    # ---- Public API ----
    def decide(self, game_context: Dict) -> int:  # type: ignore[override]
        valid_moves = self._get_valid_moves(game_context)
        if not valid_moves:
            return 0

        player_state = game_context.get("player_state", {})
        opponents = game_context.get("opponents", [])
        current_color = player_state.get("color")

        # Relative progress assessment
        my_progress = player_state.get("finished_tokens", 0) / 4.0
        opponent_max_progress = (
            max([opp.get("tokens_finished", 0) for opp in opponents], default=0) / 4.0
        )
        behind = my_progress < opponent_max_progress - 0.25
        ahead = my_progress > opponent_max_progress + 0.25

        if ahead:
            risk_weight = 1.2
        elif behind:
            risk_weight = 0.8
        else:
            risk_weight = 1.0

        opponent_positions = self._collect_opponent_positions(
            game_context, current_color
        )

        # Compute scores for each move
        best_move = None
        best_score = float("-inf")
        for move in valid_moves:
            risk = self._estimate_risk(move, opponent_positions, current_color)
            opportunity = self._estimate_opportunity(move, current_color)
            composite = opportunity - risk_weight * risk

            # Attach diagnostics (non-breaking; strategies can ignore)
            move["prob_risk"] = risk
            move["prob_opportunity"] = opportunity
            move["prob_composite_score"] = composite

            # Hard priority: always finish immediately
            if move.get("move_type") == "finish":
                return move["token_id"]

            if composite > best_score:
                best_score = composite
                best_move = move

        return best_move["token_id"] if best_move else valid_moves[0]["token_id"]

    # ---- Internal helpers ----
    def _collect_opponent_positions(
        self, game_context: Dict, current_color: str
    ) -> List[int]:
        """Extract opponent positions on main path with utils; fallback to board map."""
        return get_opponent_main_positions_with_fallback(game_context, current_color)

    def _circular_distance_backward(self, from_pos: int, opp_pos: int) -> int:
        """Distance moving backward along circular 52 path from from_pos to opp_pos.

        If opponent is behind within 1..6 inclusive, it threatens with a single dice.
        """
        if not (
            0 <= from_pos < GameConstants.MAIN_BOARD_SIZE
            and 0 <= opp_pos < GameConstants.MAIN_BOARD_SIZE
        ):
            return 999  # Not on main path / irrelevant
        if opp_pos < from_pos:
            return from_pos - opp_pos
        # wrap-around
        return GameConstants.MAIN_BOARD_SIZE - (opp_pos - from_pos)

    def _estimate_risk(
        self, move: Dict, opponent_positions: List[int], player_color: str
    ) -> float:
        target = move.get("target_position")
        move_type = move.get("move_type")
        # Safe contexts eliminate immediate capture risk
        if (
            move.get("is_safe_move")
            or move_type == "finish"
            or (isinstance(target, int) and target >= BoardConstants.HOME_COLUMN_START)
        ):
            return 0.0

        # Count distinct opponent tokens that would be 1..6 behind after move
        threatening = 0
        for opp_pos in opponent_positions:
            dist = self._circular_distance_backward(target, opp_pos)
            if 1 <= dist <= 6:
                threatening += 1

        if threatening == 0:
            return 0.0
        # Probability at least one gets exact roll (independent approx)
        return 1 - (5 / 6) ** threatening

    def _estimate_opportunity(self, move: Dict, player_color: str) -> float:
        target = move.get("target_position")
        move_type = move.get("move_type")
        opportunity = 0.0

        # Capture bonus (already flagged by engine)
        if move.get("captures_opponent"):
            captured = move.get("captured_tokens", [])
            opportunity += 2.0 * max(1, len(captured))

        # Finishing / home column progression
        if move_type == "finish":
            opportunity += 3.0
        elif move_type == "advance_home_column":
            opportunity += 1.5
        elif move_type == "exit_home":
            opportunity += 1.2

        # Safe landing
        if move.get("is_safe_move"):
            opportunity += 1.0

        # Progress component (normalized); crude but monotonic
        current_pos = move.get("current_position")
        if isinstance(target, int) and isinstance(current_pos, int):
            progress_delta = 0.0
            if (
                0 <= current_pos < GameConstants.MAIN_BOARD_SIZE
                and 0 <= target < GameConstants.MAIN_BOARD_SIZE
            ):
                # forward distance along path ignoring wrap risk; simple delta
                raw_delta = (
                    (target - current_pos)
                    if target >= current_pos
                    else ((GameConstants.MAIN_BOARD_SIZE - current_pos) + target)
                )
                progress_delta = raw_delta / GameConstants.MAIN_BOARD_SIZE
            elif (
                target >= BoardConstants.HOME_COLUMN_START
            ):  # entering / inside home column
                progress_delta = 0.2  # modest constant
            opportunity += progress_delta

        return opportunity

    def __repr__(self):  # pragma: no cover - debug helper
        return "ProbabilisticStrategy(risk/opportunity adaptive)"


__all__ = ["ProbabilisticStrategy"]
