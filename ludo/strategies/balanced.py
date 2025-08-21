"""
Balanced Strategy - Adapts based on game situation.
"""

from typing import Dict, List

from .base import Strategy


class BalancedStrategy(Strategy):
    """
    Balanced strategy that adapts based on game situation.
    Switches between offensive and defensive play as needed.
    """

    def __init__(self):
        super().__init__(
            "Balanced",
            "Adaptive strategy that balances offense and defense based on game situation",
        )

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        player_state = game_context.get("player_state", {})
        opponents = game_context.get("opponents", [])

        # Analyze game situation
        my_progress = player_state.get("finished_tokens", 0) / 4.0
        opponent_max_progress = (
            max([opp.get("tokens_finished", 0) for opp in opponents], default=0) / 4.0
        )
        behind = my_progress < opponent_max_progress - 0.25  # Significantly behind
        ahead = my_progress > opponent_max_progress + 0.25  # Significantly ahead

        # Priority 1: Always finish when possible
        finish_move = self._get_move_by_type(valid_moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Adaptive strategy based on position
        if ahead:
            # Play defensively when ahead
            return self._defensive_choice(valid_moves)
        elif behind:
            # Play aggressively when behind
            return self._aggressive_choice(valid_moves)
        else:
            # Balanced play when even
            return self._balanced_choice(valid_moves, game_context)

    def _defensive_choice(self, valid_moves: List[Dict]) -> int:
        """Make defensive choice when ahead."""
        safe_moves = self._get_safe_moves(valid_moves)
        if safe_moves:
            return self._get_highest_value_move(safe_moves)["token_id"]
        return self._get_highest_value_move(valid_moves)["token_id"]

    def _aggressive_choice(self, valid_moves: List[Dict]) -> int:
        """Make aggressive choice when behind."""
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            return self._get_highest_value_move(capture_moves)["token_id"]

        risky_moves = self._get_risky_moves(valid_moves)
        if risky_moves:
            return self._get_highest_value_move(risky_moves)["token_id"]

        return self._get_highest_value_move(valid_moves)["token_id"]

    def _balanced_choice(self, valid_moves: List[Dict], game_context: Dict) -> int:
        """Make balanced choice when even."""
        # Capture if opportunity is good
        capture_moves = self._get_capture_moves(valid_moves)
        if capture_moves:
            return capture_moves[0]["token_id"]

        # Exit home if needed
        exit_move = self._get_move_by_type(valid_moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # Otherwise, best strategic move
        return self._get_highest_value_move(valid_moves)["token_id"]
