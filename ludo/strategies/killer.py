"""
Killer Strategy - Aggressive strategy focused on capturing opponents.
"""

from typing import Dict
from .base import Strategy


class KillerStrategy(Strategy):
    """
    Aggressive strategy focused on capturing opponents.
    Prioritizes offensive moves and disrupting opponents.
    """

    def __init__(self):
        super().__init__(
            "Killer",
            "Aggressive strategy that prioritizes capturing opponents and blocking their progress",
        )

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        # Priority 1: Capture opponents (highest priority)
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            # Choose capture that affects the most threatening opponent
            best_capture = max(
                capture_moves, key=lambda m: len(m.get("captured_tokens", []))
            )
            return best_capture["token_id"]

        # Priority 2: Finish tokens (secure points)
        finish_move = self._get_move_by_type(valid_moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Priority 3: Exit home with 6 to get more pieces in play
        exit_move = self._get_move_by_type(valid_moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # Priority 4: Risky moves for aggressive positioning
        risky_moves = self._get_risky_moves(valid_moves)
        if risky_moves:
            # Choose the most aggressive risky move
            best_risky = self._get_highest_value_move(risky_moves)
            return best_risky["token_id"]

        # Fallback: Highest value move
        best_move = self._get_highest_value_move(valid_moves)
        return best_move["token_id"]
