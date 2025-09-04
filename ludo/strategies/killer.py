"""Killer Strategy.

Aggressive capture-focused strategy with predictive positioning when no
immediate capture is available.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import Strategy
from ..constants import StrategyConstants, GameConstants, BoardConstants


def _distance_to_finish(position: int, entry: int) -> int:
    """Approx rough distance to finish from a main-board or home-column position."""
    if BoardConstants.is_home_column_position(position):
        return GameConstants.FINISH_POSITION - position
    # distance to entry then home column size
    if position <= entry:
        to_entry = entry - position
    else:
        to_entry = (GameConstants.MAIN_BOARD_SIZE - position) + entry
    return to_entry + GameConstants.HOME_COLUMN_SIZE

def _estimate_recap_risk(landing: int, opponent_tokens: List[int]) -> bool:
    """Return True if any opponent token is within 6 steps behind landing (simple heuristic)."""
    for pos in opponent_tokens:
        # Compute forward distance from opponent to landing through wrap
        if BoardConstants.is_home_column_position(pos):
            continue  # ignore tokens already safe
        if pos <= landing:
            forward = landing - pos
        else:
            forward = (GameConstants.MAIN_BOARD_SIZE - pos) + landing
        if 1 <= forward <= 6:
            return True
    return False


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
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        # 1. Immediate captures
        capture_choice = self._choose_capture(moves, game_context)
        if capture_choice is not None:
            return capture_choice

        # 2. Predictive aggression (position for future capture)
        predictive_choice = self._choose_predictive(moves, game_context)
        if predictive_choice is not None:
            return predictive_choice

        # 3. Finish if possible (still valuable)
        finish_move = self._get_move_by_type(moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # 4. Exit home to increase board presence
        exit_move = self._get_move_by_type(moves, "exit_home")
        if exit_move:
            return exit_move["token_id"]

        # 5. Risky aggressive advancement
        risky_moves = self._get_risky_moves(moves)
        if risky_moves:
            best_risky = self._get_highest_value_move(risky_moves)
            if best_risky:
                return best_risky["token_id"]

        # 6. Fallback: highest strategic value overall
        best_move = self._get_highest_value_move(moves)
        return best_move["token_id"] if best_move else 0

    # --- Capture scoring ---
    def _choose_capture(self, moves: List[Dict], ctx: Dict) -> int | None:
        capture_moves = self._get_capture_moves(moves)
        if not capture_moves:
            return None

        current_color = ctx["current_situation"]["player_color"]
        opponent_positions = self._collect_opponent_positions(ctx, exclude_color=current_color)
        finished_map, max_finished = self._opponent_finished_map(ctx, current_color)
        entries = BoardConstants.HOME_COLUMN_ENTRIES

        scored: List[_CaptureScore] = []
        for mv in capture_moves:
            score, details = self._score_capture_move(mv, opponent_positions, finished_map, max_finished, entries)
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
        base = StrategyConstants.CAPTURE_BONUS
        captured = mv.get("captured_tokens", [])
        capture_count = len(captured)
        score = base + 2 * capture_count  # slight multi-capture bump
        details = {
            "base": base,
            "multi": 2 * capture_count,
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
            remaining = _distance_to_finish(mv["target_position"], entries[opp_color])
            progress_component += (60 - remaining) * StrategyConstants.KILLER_PROGRESS_WEIGHT * 0.01
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
        if _estimate_recap_risk(mv["target_position"], opponent_positions):
            penalty = StrategyConstants.KILLER_RECAPTURE_PENALTY
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
        current_color = ctx["current_situation"]["player_color"]
        opponent_positions = [
            t["position"]
            for p in ctx.get("players", [])
            if p["color"] != current_color
            for t in p["tokens"]
            if t["position"] >= 0 and not BoardConstants.is_home_column_position(t["position"])
        ]

        scored: List[Tuple[float, Dict]] = []
        for mv in moves:
            if mv["move_type"] == "finish":
                continue  # finishing handled later
            landing = mv["target_position"]
            if BoardConstants.is_home_column_position(landing):
                continue
            count = 0
            for opp_pos in opponent_positions:
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
    @staticmethod
    def _collect_opponent_positions(ctx: Dict, exclude_color: str) -> List[int]:
        positions = []
        for p in ctx.get("players", []):
            if p["color"] == exclude_color:
                continue
            for t in p["tokens"]:
                positions.append(t["position"])
        return positions

    @staticmethod
    def _opponent_finished_map(ctx: Dict, exclude_color: str) -> Tuple[Dict[str, int], int]:
        finished_map: Dict[str, int] = {}
        max_finished = 0
        for p in ctx.get("players", []):
            if p["color"] == exclude_color:
                continue
            finished_map[p["color"]] = p["finished_tokens"]
            if p["finished_tokens"] > max_finished:
                max_finished = p["finished_tokens"]
        return finished_map, max_finished
