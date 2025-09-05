"""Killer Strategy.

Improved aggressive strategy that now:
    * Prioritizes finishing before ordinary captures
    * Integrates base positional (``strategic_value``) into capture scoring
    * Uses a unified distance helper for opponent progress estimation
    * Applies graded (not binary) recapture risk based on number of nearby threats
    * Retains predictive positioning as a secondary heuristic

This is an incremental refactor (not a full rewrite) keeping the external
API and existing detail fields while correcting previous priority inversions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..constants import BoardConstants, GameConstants, StrategyConstants
from .base import Strategy
from .utils import forward_distance, get_opponent_main_positions, is_safe_or_home


def _steps_to_finish(position: int, entry: int) -> int:
    """Unified steps-to-finish estimator.

    Args:
        position: Absolute board/home-column position (>= -1)
        entry: Player-specific last main-board square before home column

    Returns:
        Approx number of steps remaining (main ring + home column).
    """
    if position < 0:
        # Still at home: rough full path length (ring + home column)
        return GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
    if BoardConstants.is_home_column_position(position):
        return GameConstants.FINISH_POSITION - position
    forward = (entry - position) % GameConstants.MAIN_BOARD_SIZE
    return forward + GameConstants.HOME_COLUMN_SIZE


def _count_recap_threats(landing: int, opponent_tokens: List[int]) -> int:
    """Count opponent tokens that could recapture within 1..6 forward steps.

    Ignores opponents already in any home column. Returns the number of
    potential attackers, enabling graded risk penalties instead of a
    coarse boolean.
    """
    threats = 0
    # Landing on home column or safe squares is immune
    if is_safe_or_home(landing):
        return 0
    for pos in opponent_tokens:
        if BoardConstants.is_home_column_position(pos) or pos < 0:
            continue
        forward = forward_distance(pos, landing)
        if 1 <= forward <= GameConstants.DICE_MAX:
            threats += 1
    return threats


@dataclass
class _CaptureScore:
    move: Dict
    score: float
    details: Dict[str, float]


class KillerStrategy(Strategy):
    """Capture-first aggressive strategy with predictive follow-up logic."""

    def __init__(self):
        super().__init__(
            "Killer",
            "Aggressive strategy that prioritizes capturing opponents and blocking their progress",
        )

    # --- Public API ---
    def decide(self, game_context: Dict) -> int:
        """Choose a token to move.

        Revised priority:
          1. Finish moves
          2. High-value captures (scored with positional baseline)
          3. Predictive aggression (future capture setup)
          4. Exit home (board presence)
          5. Risky advancement (if better than safest alternatives)
          6. Highest remaining strategic value
        """
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        # 1. Immediate finish (always take finishing over routine captures)
        finish_moves = [m for m in moves if m["move_type"] == "finish"]
        if finish_moves:
            best_finish = max(finish_moves, key=lambda m: m["strategic_value"])
            return best_finish["token_id"]

        # 2. Capture (scored with integrated positional + tactical factors)
        capture_choice = self._choose_capture(moves, game_context)
        if capture_choice is not None:
            return capture_choice

        # 3. Predictive aggression (set up future captures)
        predictive_choice = self._choose_predictive(moves, game_context)
        if predictive_choice is not None:
            return predictive_choice

        # 4. Exit home to increase board presence
        exit_move = self._get_move_by_type(moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # 5. Risky aggressive advancement (prefer the best risky move if it beats best safe by margin)
        risky_moves = self._get_risky_moves(moves)
        safe_moves = self._get_safe_moves(moves)
        best_risky = self._get_highest_value_move(risky_moves) if risky_moves else None
        best_safe = self._get_highest_value_move(safe_moves) if safe_moves else None
        if best_risky and (
            not best_safe
            or best_risky["strategic_value"] > best_safe["strategic_value"] + 5
        ):
            return best_risky["token_id"]

        # 6. Fallback highest overall value
        best_move = self._get_highest_value_move(moves)
        return best_move["token_id"] if best_move else 0

    # --- Capture scoring ---
    def _choose_capture(self, moves: List[Dict], ctx: Dict) -> int | None:
        capture_moves = self._get_capture_moves(moves)
        if not capture_moves:
            return None

        current_color = ctx["current_situation"]["player_color"]
        opponent_positions = get_opponent_main_positions(ctx)
        finished_map, max_finished = self._opponent_finished_map(ctx, current_color)
        entries = BoardConstants.HOME_COLUMN_ENTRIES

        scored: List[_CaptureScore] = []
        for mv in capture_moves:
            score, details = self._score_capture_move(
                mv, opponent_positions, finished_map, max_finished, entries
            )
            mv["killer_score"] = score
            mv["killer_details"] = details
            scored.append(_CaptureScore(mv, score, details))

        best = max(scored, key=lambda cs: cs.score)
        return best.move["token_id"]

    def _score_capture_move(
        self,
        mv: Dict,
        opponent_positions: List[int],
        finished_map: Dict[str, int],
        max_finished: int,
        entries: Dict[str, int],
    ) -> Tuple[float, Dict[str, float]]:
        # Start from the underlying positional value rather than discarding it.
        base_positional = mv.get("strategic_value", 0.0)
        base_capture = StrategyConstants.CAPTURE_BONUS
        captured = mv.get("captured_tokens", [])
        capture_count = len(captured)
        multi_bonus = 2 * capture_count
        score = base_positional + base_capture + multi_bonus
        details = {
            "positional": base_positional,
            "base_capture": base_capture,
            "multi": multi_bonus,
            "progress": 0.0,
            "threat": 0.0,
            "chain": 0.0,
            "safe": 0.0,
            "block": 0.0,
            "risk_penalty": 0.0,
            "weak_prey_penalty": 0.0,
        }

        # Prey progress component
        progress_component = 0.0
        for ct in captured:
            opp_color = ct["player_color"]
            remaining = _steps_to_finish(mv["target_position"], entries[opp_color])
            # Progress fraction (0..1 roughly) over ring+home length baseline
            baseline_total = (
                GameConstants.MAIN_BOARD_SIZE + GameConstants.HOME_COLUMN_SIZE
            )
            progress_frac = max(0.0, 1.0 - (remaining / baseline_total))
            progress_component += (
                progress_frac * StrategyConstants.KILLER_PROGRESS_WEIGHT
            )
        details["progress"] = progress_component
        score += progress_component

        # Threat emphasis (leading opponent)
        for ct in captured:
            opp_color = ct["player_color"]
            if finished_map.get(opp_color, 0) == max_finished and max_finished > 0:
                bonus = StrategyConstants.KILLER_THREAT_WEIGHT
                details["threat"] += bonus
                score += bonus

        # Extra turn chain potential (always for capture)
        chain_bonus = StrategyConstants.KILLER_CHAIN_BONUS
        details["chain"] = chain_bonus
        score += chain_bonus

        # Safety landing
        if mv.get("is_safe_move"):
            safe_bonus = StrategyConstants.KILLER_SAFE_LAND_BONUS
            details["safe"] = safe_bonus
            score += safe_bonus

        # Block formation heuristic
        if not mv.get("is_safe_move") and mv["strategic_value"] > 10:
            block_bonus = StrategyConstants.KILLER_BLOCK_BONUS * 0.5
            details["block"] = block_bonus
            score += block_bonus

        # Recapture risk
        threat_count = _count_recap_threats(mv["target_position"], opponent_positions)
        if threat_count:
            # Scale penalty by number of threats, soft-capped.
            scaled = min(threat_count, 3) / 3.0  # 0..1
            penalty = StrategyConstants.KILLER_RECAPTURE_PENALTY * scaled
            details["risk_penalty"] = -penalty
            score -= penalty

        # Weak prey penalty
        if progress_component < 0.2 and details["risk_penalty"] < 0:
            penalty2 = StrategyConstants.KILLER_WEAK_PREY_PENALTY
            details["weak_prey_penalty"] = -penalty2
            score -= penalty2

        return score, details

    # --- Predictive positioning ---
    def _choose_predictive(self, moves: List[Dict], ctx: Dict) -> int | None:
        # current_color = ctx["current_situation"]["player_color"]
        opponent_positions = get_opponent_main_positions(ctx)

        scored: List[Tuple[float, Dict]] = []
        for mv in moves:
            if mv["move_type"] == "finish":
                continue  # finishing handled later
            landing = mv["target_position"]
            if is_safe_or_home(landing):
                continue
            count = 0
            for opp_pos in opponent_positions:
                # distance forward from our landing to opponent (opponent ahead)
                if landing <= opp_pos:
                    dist = opp_pos - landing
                else:
                    dist = (GameConstants.MAIN_BOARD_SIZE - landing) + opp_pos
                if 1 <= dist <= 6:
                    count += 1
            stack_bonus = (
                0.5
                if (mv.get("strategic_value", 0) > 10 and not mv.get("is_safe_move"))
                else 0.0
            )
            score = count * StrategyConstants.KILLER_FUTURE_CAPTURE_WEIGHT + stack_bonus
            if score > 0:
                scored.append((score, mv))

        if not scored:
            return None
        best = max(scored, key=lambda x: x[0])[1]
        return best["token_id"]

    # --- Utility ---
    # Removed _collect_opponent_positions in favor of utils.get_opponent_main_positions

    @staticmethod
    def _opponent_finished_map(
        ctx: Dict, exclude_color: str
    ) -> Tuple[Dict[str, int], int]:
        finished_map: Dict[str, int] = {}
        max_finished = 0
        for p in ctx.get("players", []):
            if p["color"] == exclude_color:
                continue
            finished_map[p["color"]] = p["finished_tokens"]
            if p["finished_tokens"] > max_finished:
                max_finished = p["finished_tokens"]
        return finished_map, max_finished
