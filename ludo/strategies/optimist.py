"""
Optimist Strategy - Takes calculated risks and plays boldly.
"""

from typing import Dict
from .base import Strategy


class OptimistStrategy(Strategy):
    """
    Optimistic strategy that takes calculated risks.
    Believes in favorable outcomes and plays boldly.
    """

    def __init__(self):
        super().__init__(
            "Optimist",
            "Optimistic strategy that takes calculated risks and plays boldly for big gains",
        )

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        # Priority 1: High-value risky moves (optimistic about outcomes)
        risky_moves = self._get_risky_moves(valid_moves)
        high_value_risky = [m for m in risky_moves if m["strategic_value"] >= 10.0]
        if high_value_risky:
            best_risky = self._get_highest_value_move(high_value_risky)
            return best_risky["token_id"]

        # Priority 2: Capture moves (confident about not being captured back)
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            best_capture = self._get_highest_value_move(capture_moves)
            return best_capture["token_id"]

        # Priority 3: Exit home aggressively
        exit_move = self._get_move_by_type(valid_moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # Priority 4: Finish tokens when available
        finish_move = self._get_move_by_type(valid_moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Priority 5: Highest value move overall
        best_move = self._get_highest_value_move(valid_moves)
        return best_move["token_id"]
