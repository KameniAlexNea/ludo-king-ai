"""Defensive Strategy.

Distinct from Cautious: maintains controlled board presence, forms protective
blocks (stacked tokens are immune), advances home depth steadily, and only
captures when safe and positionally beneficial. Avoids breaking its own blocks
without compensation and minimizes exposure within a limited acceptable threat
band.
"""

from typing import Dict, List, Tuple

from ludo_engine.constants import BoardConstants, GameConstants, StrategyConstants
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.utils import (
    LARGE_THREAT_COUNT,
    NO_THREAT_DISTANCE,
    compute_threats_for_moves,
    get_my_main_positions,
)


class DefensiveStrategy(Strategy):
    """Safety-first yet positionally persistent strategy."""

    def __init__(self):
        super().__init__(
            "Defensive",
            "Safety-first strategy that preserves blocks, advances home depth, and allows limited safe captures",
        )

    # --- Public Decision ---
    def decide(self, game_context: Dict) -> int:
        moves = self._get_valid_moves(game_context)
        if not moves:
            return 0

        player_state = game_context.get("player_state", {})
        active = player_state.get("active_tokens", 0)
        opponents = game_context.get("opponents", [])

        leading_finished = max(
            (o.get("tokens_finished", 0) for o in opponents), default=0
        )
        pressure = (
            leading_finished >= StrategyConstants.DEFENSIVE_EXIT_PRESSURE_THRESHOLD
        )

        threat_map = compute_threats_for_moves(moves, game_context)
        block_positions = self._own_block_positions(game_context)
        my_positions = get_my_main_positions(game_context)

        # 1. Finish immediately
        fin = self._get_move_by_type(moves, "finish")
        if fin:
            return fin["token_id"]

        # 2. Deep home column advancement (prefer depth)
        home_moves = self._get_moves_by_type(moves, "advance_home_column")
        if home_moves:
            best_home = max(
                home_moves,
                key=lambda m: (m["target_position"], m.get("strategic_value", 0)),
            )
            return best_home["token_id"]

        # 3. Maintain or form blocks via safe moves
        block_preserving = self._filter_block_friendly(
            moves, block_positions, my_positions
        )
        if block_preserving:
            choice = self._select_safest(block_preserving, threat_map)
            if choice is not None:
                return choice

        # 4. Safe capture (beneficial only) with progress weighting
        cap_choice = self._choose_safe_capture(moves, threat_map)
        if cap_choice is not None:
            return cap_choice

        # 5. Safe progression (limit threat exposure)
        safe_moves = self._get_safe_moves(moves)
        if safe_moves:
            filtered = [
                m
                for m in safe_moves
                if self._is_within_defensive_threat(threat_map[m["token_id"]])
            ]
            if filtered:
                choice = self._select_safest(filtered, threat_map)
                if choice is not None:
                    return choice
            # if none meet threat filter still pick safest safe move
            choice = self._select_safest(safe_moves, threat_map)
            if choice is not None:
                return choice

        # 6. Exit home to maintain presence (only if below target active or under pressure)
        if active < StrategyConstants.DEFENSIVE_MIN_ACTIVE_TOKENS or pressure:
            exit_move = self._get_move_by_type(moves, "exit_home")
            if exit_move and self._is_within_defensive_threat(
                threat_map.get(exit_move["token_id"], (0, NO_THREAT_DISTANCE))
            ):
                return exit_move["token_id"]

        # 7. Reposition away from higher threat squares
        reposition_candidates = [
            m for m in moves if self._reposition_improves(m, threat_map, game_context)
        ]
        if reposition_candidates:
            choice = self._select_safest(reposition_candidates, threat_map)
            if choice is not None:
                return choice

        # 8. Fallback: minimal threat then highest strategic value
        moves.sort(
            key=lambda m: (
                threat_map.get(m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))[
                    0
                ],
                threat_map.get(m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))[
                    1
                ],
                -m.get("strategic_value", 0),
            )
        )
        return moves[0]["token_id"]

    # --- Threat & Safety Helpers ---
    # Threat computation now handled by utils.compute_threats_for_moves

    @staticmethod
    def _is_within_defensive_threat(threat_tuple: Tuple[int, int]) -> bool:
        count, min_dist = threat_tuple
        if count > StrategyConstants.DEFENSIVE_MAX_THREAT_COUNT:
            return False
        if 1 <= min_dist <= StrategyConstants.DEFENSIVE_ALLOW_THREAT_DISTANCE:
            return False
        return True

    # --- Block Logic ---
    def _own_block_positions(self, ctx: Dict) -> List[int]:
        current_color = ctx["current_situation"]["player_color"]
        positions: Dict[int, int] = {}
        for p in ctx.get("players", []):
            if p["color"] == current_color:
                for t in p["tokens"]:
                    if t[
                        "position"
                    ] >= 0 and not BoardConstants.is_home_column_position(
                        t["position"]
                    ):
                        positions[t["position"]] = positions.get(t["position"], 0) + 1
        return [pos for pos, cnt in positions.items() if cnt >= 2]

    def _filter_block_friendly(
        self, moves: List[Dict], blocks: List[int], my_positions: List[int]
    ) -> List[Dict]:
        out: List[Dict] = []
        for mv in moves:
            src = mv.get("current_position")
            dst = mv["target_position"]
            from_block = (src in blocks) if src is not None else False
            creates_stack = (
                dst in my_positions and not BoardConstants.is_home_column_position(dst)
            )
            lands_on_block = dst in blocks
            # Avoid breaking a block unless we're re-forming/keeping one
            if from_block and not (creates_stack or lands_on_block):
                continue
            # Prefer moves that create or land on blocks
            if creates_stack or lands_on_block:
                out.append(mv)
        return out

    # --- Capture Logic ---
    def _choose_safe_capture(
        self, moves: List[Dict], threat_map: Dict[int, Tuple[int, int]]
    ) -> int | None:
        captures = self._get_capture_moves(moves)
        safe_caps = [m for m in captures if m.get("is_safe_move")]
        if not safe_caps:
            return None
        scored: List[Tuple[float, Dict]] = []
        for mv in safe_caps:
            tid = mv["token_id"]
            threat_ok = self._is_within_defensive_threat(
                threat_map.get(tid, (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))
            )
            if not threat_ok:
                continue
            progress_value = 0.0
            for ct in mv.get("captured_tokens", []):
                # approximate remaining distance
                entry = BoardConstants.HOME_COLUMN_ENTRIES[ct["player_color"]]
                remaining = self._distance_to_finish_proxy(mv["target_position"], entry)
                progress_value += (60 - remaining) * 0.01
            total_score = (
                StrategyConstants.DEFENSIVE_SAFE_CAPTURE_BONUS
                + progress_value * StrategyConstants.DEFENSIVE_SAFE_CAPTURE_BONUS
            )
            scored.append((total_score, mv))
        if not scored:
            return None
        best = max(scored, key=lambda x: x[0])[1]
        return best["token_id"]

    # --- Repositioning ---
    def _reposition_improves(
        self, mv: Dict, threat_map: Dict[int, Tuple[int, int]], ctx: Dict
    ) -> bool:
        tid = mv["token_id"]
        current_threat = threat_map.get(tid, (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))
        # Simpler: treat any move to home column as improvement
        if BoardConstants.is_home_column_position(mv["target_position"]):
            return True
        # If landing reduces threat count or increases min distance, consider improvement
        landing_threat = threat_map.get(tid, current_threat)
        return (
            landing_threat[0] < current_threat[0]
            or landing_threat[1] > current_threat[1]
        )

    # --- Selection Helpers ---
    def _select_safest(
        self, moves: List[Dict], threat_map: Dict[int, Tuple[int, int]]
    ) -> int | None:
        if not moves:
            return None
        # sort by (threat_count, -is_home_column, -strategic_value)
        ordered = sorted(
            moves,
            key=lambda m: (
                threat_map.get(m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))[
                    0
                ],
                threat_map.get(m["token_id"], (LARGE_THREAT_COUNT, NO_THREAT_DISTANCE))[
                    1
                ],
                -m.get("strategic_value", 0),
            ),
        )
        return ordered[0]["token_id"] if ordered else None

    @staticmethod
    def _distance_to_finish_proxy(position: int, entry: int) -> int:
        if BoardConstants.is_home_column_position(position):
            return GameConstants.FINISH_POSITION - position
        if position <= entry:
            to_entry = entry - position
        else:
            to_entry = (GameConstants.MAIN_BOARD_SIZE - position) + entry
        return to_entry + GameConstants.HOME_COLUMN_SIZE

    # --- Shared helpers & constants ---
    # _my_main_positions removed in favor of utils.get_my_main_positions
