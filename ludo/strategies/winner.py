"""Winner Strategy.

Goal-centric: aggressively converts progress into finished tokens while
maintaining safety. Prefers finishing > deep home advancement > safe captures
> safe progression > exits (only when necessary) > fallback.
"""

from typing import Dict, List, Tuple

from ..constants import BoardConstants, GameConstants, StrategyConstants
from .base import Strategy


class WinnerStrategy(Strategy):
    """Victory-focused, safety-aware finishing strategy."""

    def __init__(self):
        super().__init__(
            "Winner",
            "Prioritizes finishing tokens, deep home advancement and safe progression",
        )

    def decide(self, game_context: Dict) -> int:
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        player_state = game_context.get("player_state", {})
        active_tokens = player_state.get("active_tokens", 0)

        # 1. Finish immediately if possible
        finish_move = self._get_move_by_type(moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # 2. Home column depth advancement (closest to finish first)
        home_moves = self._get_moves_by_type(moves, "advance_home_column")
        if home_moves:
            best_home = max(
                home_moves,
                key=lambda m: (
                    m["target_position"],  # deeper is closer
                    m.get("strategic_value", 0),
                ),
            )
            return best_home["token_id"]

        # 3. Safe capture of meaningful progress (avoid jeopardizing tokens)
        capture = self._choose_safe_capture(moves)
        if capture is not None:
            return capture

        # 4. Safe forward progression
        safe_moves = self._get_safe_moves(moves)
        if safe_moves:
            # Prefer moves improving proximity to finish (higher strategic value already encodes)
            best_safe = self._get_highest_value_move(safe_moves)
            if best_safe:
                return best_safe["token_id"]

        # 5. Exit home (only to maintain board presence)
        if active_tokens < StrategyConstants.WINNER_EXIT_MIN_ACTIVE:
            exit_move = self._get_move_by_type(moves, "exit_home")
            if exit_move:
                return exit_move["token_id"]

        # 6. Fallback: highest strategic value overall
        best_move = self._get_highest_value_move(moves)
        return best_move["token_id"] if best_move else 0

    # --- Helpers ---
    def _choose_safe_capture(self, moves: List[Dict]) -> int | None:
        capture_moves = self._get_capture_moves(moves)
        if not capture_moves:
            return None
        # Only consider safe captures
        safe_caps = [m for m in capture_moves if m.get("is_safe_move")]
        if not safe_caps:
            return None
        entries = BoardConstants.HOME_COLUMN_ENTRIES
        scored: List[Tuple[float, Dict]] = []
        for mv in safe_caps:
            progress_value = 0.0
            for ct in mv.get("captured_tokens", []):
                remaining = self._distance_to_finish_proxy(
                    mv["target_position"], entries[ct["player_color"]]
                )
                progress_value += (
                    (60 - remaining)
                    * StrategyConstants.WINNER_SAFE_CAPTURE_PROGRESS_WEIGHT
                    * 0.01
                )
            scored.append((progress_value, mv))
        if not scored:
            return None
        best = max(scored, key=lambda x: x[0])[1]
        return best["token_id"]

    @staticmethod
    def _distance_to_finish_proxy(position: int, entry: int) -> int:
        if BoardConstants.is_home_column_position(position):
            return GameConstants.FINISH_POSITION - position
        if position <= entry:
            to_entry = entry - position
        else:
            to_entry = (GameConstants.MAIN_BOARD_SIZE - position) + entry
        return to_entry + GameConstants.HOME_COLUMN_SIZE
