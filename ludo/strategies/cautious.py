"""Cautious Strategy.

Highly risk-averse: prioritizes guaranteed safety, minimizes exposure, delays
exiting home unless required, and slightly relaxes rules only when falling
behind late game.
"""

from typing import Dict, List, Set

from ludo_engine.constants import BoardConstants, StrategyConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.utils import (
    LARGE_THREAT_COUNT,
    NO_THREAT_DISTANCE,
    compute_threats_for_moves,
    get_my_main_positions,
)


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
        leading_opponent_finished = max(
            (o.get("tokens_finished", 0) for o in opponents), default=0
        )

        # Backward compatibility: retain original late_game flag
        late_game = leading_opponent_finished >= 3 and finished <= 1
        # Refined urgency detection (normal, behind, desperate, late_game)
        urgency = self._get_urgency_level(game_context)
        threat_info = compute_threats_for_moves(moves, game_context)

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

        # 3. Safe captures before generic safe moves (conservative but not blind)
        # Determine allowed threat based on urgency
        allowed_threat = (
            StrategyConstants.CAUTIOUS_LATE_GAME_ALLOWED_THREAT
            if urgency in ("behind", "desperate", "late_game")
            else StrategyConstants.CAUTIOUS_MAX_ALLOWED_THREAT
        )
        safe_moves = self._get_safe_moves(moves)
        my_main_positions = get_my_main_positions(game_context)

        capture_moves = self._get_capture_moves(moves)
        safe_captures: List[Dict] = [
            m
            for m in capture_moves
            if threat_info.get(m["token_id"], (LARGE_THREAT_COUNT,))[0] == 0
            or BoardConstants.is_safe_position(m["target_position"])  # star/start
        ]
        if safe_captures:
            # Prefer zero threat, larger min distance to nearest attacker, then value
            safe_captures.sort(
                key=lambda m: (
                    threat_info[m["token_id"]][0],
                    threat_info[m["token_id"]][1],
                    -m["strategic_value"],
                )
            )
            return safe_captures[0]["token_id"]

        # 4. Fully safe main-board moves (no/limited incoming threat allowed)
        zero_or_allowed_threat: List[Dict] = [
            m for m in safe_moves if threat_info[m["token_id"]][0] <= allowed_threat
        ]
        if zero_or_allowed_threat:
            # prefer lowest threat then deeper strategic safety ranking
            zero_or_allowed_threat.sort(
                key=lambda m: (
                    threat_info[m["token_id"]][0],  # threat count
                    threat_info[m["token_id"]][1],  # min distance
                    -int(self._creates_block(m, my_main_positions)),  # prefer blocks
                    -m["strategic_value"],  # then value
                )
            )
            return zero_or_allowed_threat[0]["token_id"]

        # 5. Exit home (only if board presence low or late game pressure)
        if active_tokens < StrategyConstants.CAUTIOUS_MIN_ACTIVE_TOKENS or late_game:
            exit_move = self._get_move_by_type(moves, "exit_home")
            if exit_move:
                # Ensure exit square not threatened unless forced
                tid = exit_move["token_id"]
                if threat_info.get(tid, (LARGE_THREAT_COUNT,))[0] <= allowed_threat:
                    return tid

        # 6. Choose least threatened remaining safe move (even if above threshold)
        if safe_moves:
            safe_moves.sort(
                key=lambda m: (
                    threat_info[m["token_id"]][0],
                    threat_info[m["token_id"]][1],
                    -int(self._creates_block(m, my_main_positions)),
                    -m["strategic_value"],
                )
            )
            return safe_moves[0]["token_id"]

        # 7. Last resort: any move with minimal exposure
        moves.sort(
            key=lambda m: (
                threat_info.get(
                    m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE)
                )[0],
                threat_info.get(
                    m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE)
                )[1],
                -int(self._creates_block(m, my_main_positions)),
                -m["strategic_value"],
            )
        )
        return moves[0]["token_id"]

    # --- Helpers ---

    def _creates_block(self, move: Dict, my_positions: Set[int]) -> bool:
        """Check if move lands on own token to form a protective block on main board."""
        landing = move.get("target_position", -2)
        return landing in my_positions and not BoardConstants.is_home_column_position(
            landing
        )

    def _get_urgency_level(self, ctx: Dict) -> str:
        """Classify urgency based on finished-token deficit and phase.

        Returns one of: "normal", "behind", "desperate", "late_game".
        """
        player_state = ctx.get("player_state", {})
        my_finished = player_state.get("finished_tokens", 0)
        opponents = ctx.get("opponents", [])
        max_opp_finished = max(
            (o.get("tokens_finished", 0) for o in opponents), default=0
        )
        deficit = max_opp_finished - my_finished

        if max_opp_finished >= 3 and my_finished <= 1:
            return "late_game"
        if deficit >= 2:
            return "desperate"
        if deficit >= 1:
            return "behind"
        return "normal"
