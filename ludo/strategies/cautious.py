"""Cautious Strategy.

Highly risk-averse: prioritizes guaranteed safety, minimizes exposure, delays
exiting home unless required, and slightly relaxes rules only when falling
behind late game.
"""

from typing import Dict, List, Tuple

from .base import Strategy
from ..constants import StrategyConstants, GameConstants, BoardConstants


class CautiousStrategy(Strategy):
    """Extremely conservative decision policy."""

    def __init__(self):
        super().__init__(
            "Cautious",
            "Conservative strategy favoring safe squares, home column advancement, and minimal exposure",
        )

    # --- Public API ---
    def decide(self, game_context: Dict) -> int:
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        player_state = game_context.get("player_state", {})
        finished = player_state.get("finished_tokens", 0)
        active_tokens = player_state.get("active_tokens", 0)

        opponents = game_context.get("opponents", [])
        leading_opponent_finished = max((o.get("tokens_finished", 0) for o in opponents), default=0)

        late_game = leading_opponent_finished >= 3 and finished <= 1

        threat_info = self._annotate_threats(moves, game_context)

        # 1. Finish
        finish_move = self._get_move_by_type(moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # 2. Advance in home column (depth preference)
        home_moves = self._get_moves_by_type(moves, "advance_home_column")
        if home_moves:
            # deeper (target_position larger) is safer
            best_home = max(home_moves, key=lambda m: m["target_position"])
            return best_home["token_id"]

        # 3. Fully safe main-board moves (no incoming threat allowed)
        allowed_threat = (
            StrategyConstants.CAUTIOUS_LATE_GAME_ALLOWED_THREAT if late_game else StrategyConstants.CAUTIOUS_MAX_ALLOWED_THREAT
        )
        safe_moves = self._get_safe_moves(moves)
        zero_or_allowed_threat: List[Dict] = [
            m for m in safe_moves if threat_info[m["token_id"]][0] <= allowed_threat
        ]
        if zero_or_allowed_threat:
            # prefer lowest threat then deeper strategic safety ranking
            zero_or_allowed_threat.sort(
                key=lambda m: (
                    threat_info[m["token_id"]][0],  # threat count
                    threat_info[m["token_id"]][1],  # min distance
                    -m["strategic_value"],  # larger strategic value last for caution
                )
            )
            return zero_or_allowed_threat[0]["token_id"]

        # 4. Exit home (only if board presence low or late game pressure)
        if active_tokens < StrategyConstants.CAUTIOUS_MIN_ACTIVE_TOKENS or late_game:
            exit_move = self._get_move_by_type(moves, "exit_home")
            if exit_move:
                # Ensure exit square not threatened unless forced
                tid = exit_move["token_id"]
                if threat_info.get(tid, (99,))[0] <= allowed_threat:
                    return tid

        # 5. Choose least threatened remaining safe move (even if above threshold)
        if safe_moves:
            safe_moves.sort(
                key=lambda m: (
                    threat_info[m["token_id"]][0],
                    threat_info[m["token_id"]][1],
                    -m["strategic_value"],
                )
            )
            return safe_moves[0]["token_id"]

        # 6. Last resort: any move with minimal exposure
        moves.sort(
            key=lambda m: (
                threat_info.get(m["token_id"], (99, 99))[0],
                threat_info.get(m["token_id"], (99, 99))[1],
                -m["strategic_value"],
            )
        )
        return moves[0]["token_id"]

    # --- Threat Analysis ---
    def _annotate_threats(self, moves: List[Dict], ctx: Dict) -> Dict[int, Tuple[int, int]]:
        """Return mapping token_id -> (threat_count, min_forward_distance).

        threat_count: number of opponent tokens that could reach landing square in 1..6.
        min_forward_distance: smallest such distance (large if none found).
        """
        current_color = ctx["current_situation"]["player_color"]
        opponent_positions = [
            t["position"]
            for p in ctx.get("players", [])
            if p["color"] != current_color
            for t in p["tokens"]
            if t["position"] >= 0 and not BoardConstants.is_home_column_position(t["position"])
        ]
        result: Dict[int, Tuple[int, int]] = {}
        for mv in moves:
            landing = mv["target_position"]
            if BoardConstants.is_home_column_position(landing):
                result[mv["token_id"]] = (0, 99)
                continue
            threat_count = 0
            min_dist = 99
            for opp in opponent_positions:
                if landing <= opp:
                    dist = opp - landing
                else:
                    dist = (GameConstants.MAIN_BOARD_SIZE - landing) + opp
                if 1 <= dist <= 6:
                    threat_count += 1
                    if dist < min_dist:
                        min_dist = dist
            result[mv["token_id"]] = (threat_count, min_dist)
        return result
