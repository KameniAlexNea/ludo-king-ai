"""Cautious Strategy.

Highly risk-averse: prioritizes guaranteed safety, minimizes exposure, delays
exiting home unless required, and slightly relaxes rules only when falling
behind late game.
"""

from typing import Dict, List, Tuple, Set

from ..constants import BoardConstants, GameConstants, StrategyConstants

# Derived sentinels (avoid magic numbers):
# - No-threat distance uses one less than home column start (100-1=99 by default)
# - Large threat count upper-bounds any realistic attacker count
NO_THREAT_DISTANCE = GameConstants.HOME_COLUMN_START - 1
LARGE_THREAT_COUNT = GameConstants.MAX_PLAYERS * GameConstants.TOKENS_PER_PLAYER + 1
from .base import Strategy


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

        # 3. Safe captures before generic safe moves (conservative but not blind)
        # Determine allowed threat based on urgency
        allowed_threat = (
            StrategyConstants.CAUTIOUS_LATE_GAME_ALLOWED_THREAT
            if urgency in ("behind", "desperate", "late_game")
            else StrategyConstants.CAUTIOUS_MAX_ALLOWED_THREAT
        )
        safe_moves = self._get_safe_moves(moves)
        my_main_positions = self._get_my_mainboard_positions(game_context)

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
                threat_info.get(m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))[0],
                threat_info.get(m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))[1],
                -int(self._creates_block(m, my_main_positions)),
                -m["strategic_value"],
            )
        )
        return moves[0]["token_id"]

    # --- Threat Analysis ---
    def _annotate_threats(
        self, moves: List[Dict], ctx: Dict
    ) -> Dict[int, Tuple[int, int]]:
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
            if t["position"] >= 0
            and not BoardConstants.is_home_column_position(t["position"])
        ]
        result: Dict[int, Tuple[int, int]] = {}
        for mv in moves:
            landing = mv["target_position"]
            # Treat home column and safe squares (stars, starts) as immune
            if BoardConstants.is_home_column_position(landing) or BoardConstants.is_safe_position(landing):
                result[mv["token_id"]] = (0, NO_THREAT_DISTANCE)
                continue
            # Treat forming a block (two of our tokens) as immune
            my_positions = self._get_my_mainboard_positions(ctx)
            if landing in my_positions:
                result[mv["token_id"]] = (0, NO_THREAT_DISTANCE)
                continue
            threat_count = 0
            min_dist = NO_THREAT_DISTANCE
            for opp in opponent_positions:
                # forward distance from opponent to landing with wrap-around
                if opp <= landing:
                    dist = landing - opp
                else:
                    dist = (GameConstants.MAIN_BOARD_SIZE - opp) + landing
                if 1 <= dist <= 6:
                    threat_count += 1
                    if dist < min_dist:
                        min_dist = dist
            result[mv["token_id"]] = (threat_count, min_dist)
        return result

    # --- Helpers ---
    def _get_my_mainboard_positions(self, ctx: Dict) -> Set[int]:
        """Return set of this player's main-board positions (non-home, >=0)."""
        current_color = ctx.get("current_situation", {}).get("player_color")
        positions: Set[int] = set()
        for p in ctx.get("players", []):
            if p.get("color") != current_color:
                continue
            for t in p.get("tokens", []):
                pos = t.get("position", -1)
                if pos >= 0 and not BoardConstants.is_home_column_position(pos):
                    positions.add(pos)
        return positions

    def _creates_block(self, move: Dict, my_positions: Set[int]) -> bool:
        """Check if move lands on own token to form a protective block on main board."""
        landing = move.get("target_position", -2)
        return landing in my_positions and not BoardConstants.is_home_column_position(landing)

    def _get_urgency_level(self, ctx: Dict) -> str:
        """Classify urgency based on finished-token deficit and phase.

        Returns one of: "normal", "behind", "desperate", "late_game".
        """
        player_state = ctx.get("player_state", {})
        my_finished = player_state.get("finished_tokens", 0)
        opponents = ctx.get("opponents", [])
        max_opp_finished = max((o.get("tokens_finished", 0) for o in opponents), default=0)
        deficit = max_opp_finished - my_finished

        if max_opp_finished >= 3 and my_finished <= 1:
            return "late_game"
        if deficit >= 2:
            return "desperate"
        if deficit >= 1:
            return "behind"
        return "normal"
