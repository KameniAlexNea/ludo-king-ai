"""
Cautious Strategy - Very cautious strategy that avoids all risks.
"""

from typing import Dict

from .base import Strategy


class CautiousStrategy(Strategy):
    """
    Very cautious strategy that avoids all risks.
    Only makes moves that are guaranteed safe.
    """

    def __init__(self):
        super().__init__(
            "Cautious",
            "Extremely conservative strategy that only makes guaranteed safe moves",
        )

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        # Priority 1: Finish tokens
        finish_move = self._get_move_by_type(valid_moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Priority 2: Home column moves (always safe)
        home_moves = self._get_moves_by_type(valid_moves, "advance_home_column")
        if home_moves:
            return self._get_highest_value_move(home_moves)["token_id"]

        # Priority 3: Only very safe moves
        safe_moves = self._get_safe_moves(valid_moves)
        very_safe = [
            m for m in safe_moves if m["strategic_value"] <= 10.0
        ]  # Conservative values
        if very_safe:
            return self._get_highest_value_move(very_safe)["token_id"]

        # Priority 4: Exit home only if absolutely necessary
        player_state: dict = game_context.get("player_state", {})
        active_tokens = player_state.get("active_tokens", 0)

        if active_tokens == 0:  # Must exit home
            exit_move = self._get_move_by_type(valid_moves, "exit_home")
            if exit_move:
                return exit_move["token_id"]

        # Fallback: Least risky move
        if safe_moves:
            return self._get_lowest_value_move(safe_moves)["token_id"]

        return self._get_lowest_value_move(valid_moves)["token_id"]
