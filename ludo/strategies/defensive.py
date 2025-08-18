"""
Defensive Strategy - Prioritizes safety and protection.
"""

from typing import Dict
from .base import Strategy


class DefensiveStrategy(Strategy):
    """
    Defensive strategy that prioritizes safety and protection.
    Avoids risks and plays conservatively.
    """

    def __init__(self):
        super().__init__(
            "Defensive",
            "Conservative strategy that prioritizes safety and avoids unnecessary risks",
        )

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        # Priority 1: Finish tokens (completely safe)
        finish_move = self._get_move_by_type(valid_moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Priority 2: Safe moves only
        safe_moves = self._get_safe_moves(valid_moves)
        if safe_moves:
            # Among safe moves, prefer advancing in home column
            home_column_safe = [
                m for m in safe_moves if m["move_type"] == "advance_home_column"
            ]
            if home_column_safe:
                return self._get_highest_value_move(home_column_safe)["token_id"]

            # Otherwise, best safe move
            best_safe = self._get_highest_value_move(safe_moves)
            return best_safe["token_id"]

        # Priority 3: Capture only if opponent is very threatening
        opponents = game_context.get("opponents", [])
        high_threat_opponents = [
            opp for opp in opponents if opp.get("threat_level", 0) > 0.7
        ]

        if high_threat_opponents:
            capture_moves = self._get_capture_moves(valid_moves)
            if capture_moves:
                return capture_moves[0]["token_id"]

        # Priority 4: Exit home only if no other choice
        exit_move = self._get_move_by_type(valid_moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # Fallback: Lowest risk move
        best_move = self._get_lowest_value_move(valid_moves)
        return best_move["token_id"]
