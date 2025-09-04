"""Balanced Strategy.

Adaptive hybrid that blends priorities from aggressive (capture/progression),
defensive (threat management, block maintenance), winner (finish/home depth),
and cautious (avoid needless risk when ahead). Dynamic weights shift based on
relative progress and late-game pressure.
"""

from typing import Dict, List, Tuple

from .base import Strategy
from ..constants import StrategyConstants, GameConstants, BoardConstants


class BalancedStrategy(Strategy):
    """Adaptive multi-factor decision policy."""

    def __init__(self):
        super().__init__(
            "Balanced",
            "Adaptive blend of offensive, defensive, and finishing heuristics",
        )

    # --- Public API ---
    def decide(self, game_context: Dict) -> int:
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        player_state = game_context.get("player_state", {})
        finished = player_state.get("finished_tokens", 0)
        active = player_state.get("active_tokens", 0)
        opponents = game_context.get("opponents", [])
        leading_finished = max((o.get("tokens_finished", 0) for o in opponents), default=0)

        my_progress_ratio = finished / GameConstants.TOKENS_PER_PLAYER
        opponent_max_ratio = leading_finished / GameConstants.TOKENS_PER_PLAYER if GameConstants.TOKENS_PER_PLAYER else 1

        behind = my_progress_ratio + StrategyConstants.BALANCED_RISK_TOLERANCE_MARGIN < opponent_max_ratio
        ahead = my_progress_ratio > opponent_max_ratio + StrategyConstants.BALANCED_RISK_TOLERANCE_MARGIN
        late_game_pressure = leading_finished >= StrategyConstants.BALANCED_LATE_GAME_FINISH_PUSH

        threat_map = self._compute_threats(moves, game_context)
        block_positions = self._own_block_positions(game_context)

        # Priority 1: Immediate finish
        finish_move = self._get_move_by_type(moves, "finish")
        if finish_move:
            return finish_move["token_id"]

        # Priority 2: Deep home column (weighted more under late-game pressure)
        home_moves = self._get_moves_by_type(moves, "advance_home_column")
        if home_moves:
            home_weight = StrategyConstants.BALANCED_HOME_PRIORITY * (1.2 if late_game_pressure else 1.0)
            best_home = max(home_moves, key=lambda m: (m["target_position"] * home_weight, m.get("strategic_value", 0)))
            return best_home["token_id"]

        # Priority 3: High-quality safe capture (progress + safety) esp. when behind
        capture_choice = self._choose_capture(moves, threat_map, aggressive=behind)
        if capture_choice is not None:
            return capture_choice

        # Priority 4: Maintain/create protective blocks while progressing
        block_moves = self._block_positive_moves(moves, block_positions)
        if block_moves:
            pick = self._select_weighted(block_moves, threat_map, ahead)
            if pick is not None:
                return pick

        # Priority 5: Safe forward progression (moderate threat tolerance when behind)
        safe_moves = self._get_safe_moves(moves)
        if safe_moves:
            pick = self._select_weighted(safe_moves, threat_map, ahead, behind)
            if pick is not None:
                return pick

        # Priority 6: Exit home to maintain presence if needed
        if active < StrategyConstants.BALANCED_MIN_ACTIVE_TARGET or behind:
            exit_move = self._get_move_by_type(moves, "exit_home")
            if exit_move:
                return exit_move["token_id"]

        # Priority 7: Future capture positioning (when neither ahead nor severely threatened)
        future_pos = self._future_capture_positioning(moves, threat_map)
        if future_pos is not None:
            return future_pos

        # Fallback: Highest strategic value overall
        best = self._get_highest_value_move(moves)
        return best["token_id"] if best else 0

    # --- Threat Analysis (reuse concept from cautious/defensive) ---
    def _compute_threats(self, moves: List[Dict], ctx: Dict) -> Dict[int, Tuple[int, int]]:
        color = ctx["current_situation"]["player_color"]
        opponent_positions = [
            t["position"]
            for p in ctx.get("players", []) if p["color"] != color
            for t in p["tokens"]
            if t["position"] >= 0 and not BoardConstants.is_home_column_position(t["position"])
        ]
        res: Dict[int, Tuple[int, int]] = {}
        for mv in moves:
            land = mv["target_position"]
            if BoardConstants.is_home_column_position(land):
                res[mv["token_id"]] = (0, 99)
                continue
            cnt = 0
            mind = 99
            for opp in opponent_positions:
                if land <= opp:
                    dist = opp - land
                else:
                    dist = (GameConstants.MAIN_BOARD_SIZE - land) + opp
                if 1 <= dist <= 6:
                    cnt += 1
                    if dist < mind:
                        mind = dist
            res[mv["token_id"]] = (cnt, mind)
        return res

    # --- Blocks ---
    def _own_block_positions(self, ctx: Dict) -> List[int]:
        color = ctx["current_situation"]["player_color"]
        occ: Dict[int, int] = {}
        for p in ctx.get("players", []):
            if p["color"] == color:
                for t in p["tokens"]:
                    if t["position"] >= 0 and not BoardConstants.is_home_column_position(t["position"]):
                        occ[t["position"]] = occ.get(t["position"], 0) + 1
        return [pos for pos, c in occ.items() if c >= 2]

    def _block_positive_moves(self, moves: List[Dict], blocks: List[int]) -> List[Dict]:
        out: List[Dict] = []
        for mv in moves:
            dst = mv["target_position"]
            if dst in blocks:
                out.append(mv)
        return out

    # --- Capture Evaluation ---
    def _choose_capture(self, moves: List[Dict], threat_map: Dict[int, Tuple[int, int]], aggressive: bool) -> int | None:
        captures = self._get_capture_moves(moves)
        if not captures:
            return None
        scored: List[Tuple[float, Dict]] = []
        for mv in captures:
            tid = mv["token_id"]
            threat = threat_map.get(tid, (99, 99))
            # when aggressive allow up to BALANCED_THREAT_SOFT_CAP else stricter
            max_threat_allowed = StrategyConstants.BALANCED_THREAT_SOFT_CAP if aggressive else StrategyConstants.BALANCED_AHEAD_THREAT_CAP
            if threat[0] > max_threat_allowed:
                continue
            if threat[1] <= 2 and not aggressive:
                continue  # too close to danger when not pushing
            progress_value = 0.0
            for ct in mv.get("captured_tokens", []):
                entry = BoardConstants.HOME_COLUMN_ENTRIES[ct["player_color"]]
                remaining = self._distance_to_finish_proxy(mv["target_position"], entry)
                progress_value += (60 - remaining) * 0.01
            score = (
                StrategyConstants.BALANCED_SAFE_CAPTURE_WEIGHT * (1.25 if aggressive else 1.0)
                + progress_value * StrategyConstants.BALANCED_SAFE_CAPTURE_WEIGHT
            )
            scored.append((score, mv))
        if not scored:
            return None
        best = max(scored, key=lambda x: x[0])[1]
        return best["token_id"]

    # --- Future Capture Positioning ---
    def _future_capture_positioning(self, moves: List[Dict], threat_map: Dict[int, Tuple[int, int]]) -> int | None:
        candidates = [m for m in moves if m.get("is_safe_move") and not m.get("captures_opponent")]
        if not candidates:
            return None
        scored: List[Tuple[float, Dict]] = []
        scan_range = StrategyConstants.BALANCED_FUTURE_CAPTURE_PROXIMITY
        for mv in candidates:
            tid = mv["token_id"]
            threat = threat_map.get(tid, (99, 99))
            if threat[0] > StrategyConstants.BALANCED_THREAT_SOFT_CAP:
                continue
            potential = self._estimate_future_capture_potential(mv["target_position"], scan_range)
            if potential <= 0:
                continue
            scored.append((potential * StrategyConstants.BALANCED_FUTURE_CAPTURE_WEIGHT, mv))
        if not scored:
            return None
        best = max(scored, key=lambda x: x[0])[1]
        return best["token_id"]

    def _estimate_future_capture_potential(self, position: int, rng: int) -> float:
        # Placeholder heuristic: closer to entry squares for any color modestly increases potential
        # (Could be extended with real opponent token proximity lookahead)
        entries = list(BoardConstants.HOME_COLUMN_ENTRIES.values())
        best = 0
        for e in entries:
            if position <= e:
                dist = e - position
            else:
                dist = (GameConstants.MAIN_BOARD_SIZE - position) + e
            if 0 < dist <= rng:
                best = max(best, (rng - dist + 1) / rng)
        return best

    # --- Weighted Selection for progression/safety ---
    def _select_weighted(self, moves: List[Dict], threat_map: Dict[int, Tuple[int, int]], ahead: bool, behind: bool = False) -> int | None:
        if not moves:
            return None
        scored: List[Tuple[float, Dict]] = []
        for mv in moves:
            tid = mv["token_id"]
            threat = threat_map.get(tid, (0, 99))
            threat_penalty = threat[0] * (2.0 if ahead else 1.0)  # stricter when ahead
            depth_bonus = 0.0
            if BoardConstants.is_home_column_position(mv["target_position"]):
                depth_bonus = (mv["target_position"] - GameConstants.HOME_COLUMN_START) * StrategyConstants.BALANCED_HOME_PRIORITY
            progress_component = mv.get("strategic_value", 0) * StrategyConstants.BALANCED_PROGRESS_WEIGHT
            aggressiveness = 1.2 if behind else 1.0
            composite = (progress_component + depth_bonus) * aggressiveness - threat_penalty
            scored.append((composite, mv))
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
