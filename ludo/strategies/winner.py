"""
Winner Strategy - Victory-focused strategy that prioritizes finishing tokens.
"""

from typing import Dict
from .base import Strategy


class WinnerStrategy(Strategy):
    """
    Victory-focused strategy that prioritizes finishing tokens.
    Plays conservatively but efficiently toward winning.
    """

    def __init__(self):
        super().__init__(
            "Winner",
            "Victory-focused strategy that prioritizes finishing tokens and safe progression",
        )

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        # Priority 1: Finish tokens (main goal)
        finish_move = self._get_move_by_type(valid_moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Priority 2: Advance in home column (close to finishing)
        home_column_moves = self._get_moves_by_type(valid_moves, "advance_home_column")
        if home_column_moves:
            # Choose the one closest to finishing
            best_home = max(home_column_moves, key=lambda m: m["strategic_value"])
            return best_home["token_id"]

        # Priority 3: Capture only if very safe or strategically important
        capture_moves = self._get_capture_moves(valid_moves)
        safe_captures = [m for m in capture_moves if m["is_safe_move"]]
        if safe_captures:
            return safe_captures[0]["token_id"]

        # Priority 4: Safe moves toward home
        safe_moves = self._get_safe_moves(valid_moves)
        if safe_moves:
            best_safe = self._get_highest_value_move(safe_moves)
            return best_safe["token_id"]

        # Priority 5: Exit home only if needed
        exit_move = self._get_move_by_type(valid_moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # Fallback: Most conservative move
        best_move = self._get_highest_value_move(valid_moves)
        return best_move["token_id"]
