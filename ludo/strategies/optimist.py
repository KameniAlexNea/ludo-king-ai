"""Optimist Strategy.

Bold, opportunistic: prefers high-value risky advancements, proactive exits,
and future capture potential while still finishing when convenient.
"""

from typing import Dict, List, Tuple

from ludo_engine.constants import BoardConstants, GameConstants, StrategyConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.utils import (
    forward_distance,
    get_opponent_main_positions,
    is_safe_or_home,
)


class OptimistStrategy(Strategy):
    """Optimistic, high-upside decision policy."""

    def __init__(self):
        super().__init__(
            "Optimist",
            "Optimistic strategy taking calculated risks, prioritizing upside and momentum",
        )

    def decide(self, game_context: Dict) -> int:
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        # Extract state for aggressive heuristics
        player_state = game_context.get("player_state", {})
        active_tokens = player_state.get("active_tokens", 0)
        # finished_tokens = player_state.get("finished_tokens", 0)

        # 1. High-value risky moves (above threshold)
        risky_moves = self._get_risky_moves(moves)
        high_value_risky = [
            m
            for m in risky_moves
            if m["strategic_value"] >= StrategyConstants.OPTIMIST_HIGH_RISK_THRESHOLD
        ]
        if high_value_risky:
            # Prefer moves with future capture potential
            best = self._choose_future_capture(
                high_value_risky, game_context, fallback=True
            )
            if best:
                return best["token_id"]

        # 2. Capture moves (prioritize progressed prey & forming stacks)
        capture_moves = self._get_capture_moves(moves)
        if capture_moves:
            best_capture = self._score_captures(capture_moves, game_context)
            if best_capture:
                return best_capture["token_id"]

        # 3. Aggressive exit to increase board presence until target count reached
        if active_tokens < StrategyConstants.OPTIMIST_EXIT_EARLY_ACTIVE_TARGET:
            exit_move = self._get_move_by_type(moves, "exit_home")
            if exit_move:
                return exit_move["token_id"]

        # 4. Finish tokens (still beneficial but not first priority)
        finish_move = self._get_move_by_type(moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # 5. Secondary risky moves (any remaining risky)
        if risky_moves:
            best_secondary = self._choose_future_capture(
                risky_moves, game_context, fallback=True
            )
            if best_secondary:
                return best_secondary["token_id"]

        # 6. High-upside future capture positioning among safe moves
        safe_moves = self._get_safe_moves(moves)
        if safe_moves:
            future_pos = self._choose_future_capture(
                safe_moves, game_context, fallback=False
            )
            if future_pos:
                return future_pos["token_id"]

        # 7. Default: highest value momentum move
        best_move = self._get_highest_value_move(moves)
        return best_move["token_id"] if best_move else 0

    # --- Capture scoring ---
    def _score_captures(self, captures: List[Dict], ctx: Dict) -> Dict | None:
        entries = BoardConstants.HOME_COLUMN_ENTRIES
        scored: List[Tuple[float, Dict]] = []
        for mv in captures:
            base = StrategyConstants.CAPTURE_BONUS
            progress_bonus = 0.0
            for ct in mv.get("captured_tokens", []):
                remaining = self._distance_to_finish_proxy(
                    mv["target_position"], entries[ct["player_color"]]
                )
                progress_bonus += (
                    (60 - remaining)
                    * StrategyConstants.OPTIMIST_CAPTURE_PROGRESS_WEIGHT
                    * 0.01
                )
            stack_bonus = (
                StrategyConstants.OPTIMIST_STACK_BONUS
                if (not mv.get("is_safe_move") and mv["strategic_value"] > 10)
                else 0.0
            )
            total = base + progress_bonus + stack_bonus
            scored.append((total, mv))
        if not scored:
            return None
        return max(scored, key=lambda x: x[0])[1]

    # --- Future capture positioning ---
    def _choose_future_capture(
        self, moves: List[Dict], ctx: Dict, fallback: bool
    ) -> Dict | None:
        scored: List[Tuple[float, Dict]] = []
        for mv in moves:
            landing = mv["target_position"]
            if BoardConstants.is_home_column_position(landing):
                continue
            potential = self._count_targets_in_range(landing, ctx)
            if potential == 0 and not fallback:
                continue
            risk_reward = (
                StrategyConstants.OPTIMIST_RISK_REWARD_BONUS
                if not mv.get("is_safe_move")
                else 0.0
            )
            score = (
                potential * StrategyConstants.OPTIMIST_FUTURE_CAPTURE_WEIGHT
                + mv.get("strategic_value", 0)
                + risk_reward
            )
            scored.append((score, mv))
        if not scored:
            return None
        return max(scored, key=lambda x: x[0])[1]

    # --- Utilities ---
    def _count_targets_in_range(self, landing: int, ctx: Dict) -> int:
        """Count opponents ahead within 1..6 squares from landing (main path only).

        Uses forward distance wrapping around 52. Ignores opponents in home columns
        and tokens off-board (<0). Landing in home column is handled by caller.
        """
        # Use shared utils for accurate opponent scanning on main board.
        opponent_positions = get_opponent_main_positions(ctx)
        count = 0
        for opp in opponent_positions:
            # Skip opponents that are safe or in home columns (cannot be captured)
            if is_safe_or_home(opp):
                continue
            # distance forward from landing to opponent using shared helper
            dist = forward_distance(landing, opp)
            if 1 <= dist <= 6:
                count += 1
        return count

    @staticmethod
    def _distance_to_finish_proxy(position: int, entry: int) -> int:
        if BoardConstants.is_home_column_position(position):
            return GameConstants.FINISH_POSITION - position
        if position <= entry:
            to_entry = entry - position
        else:
            to_entry = (GameConstants.MAIN_BOARD_SIZE - position) + entry
        return to_entry + GameConstants.HOME_COLUMN_SIZE
