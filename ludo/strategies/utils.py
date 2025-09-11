"""Shared helpers for strategy implementations.

Provides common constants, distance utilities, safe-position checks, and
threat computation to eliminate duplication across strategies.
"""

from typing import Dict, List, Set, Tuple

from ..constants import BoardConstants, GameConstants

# Sentinel constants derived from board geometry (avoid magic numbers)
NO_THREAT_DISTANCE: int = GameConstants.HOME_COLUMN_START - 1
LARGE_THREAT_COUNT: int = (
    GameConstants.MAX_PLAYERS * GameConstants.TOKENS_PER_PLAYER + 1
)


def is_safe_or_home(pos: int) -> bool:
    """True if position is a star/start safe square or within any home column."""
    return BoardConstants.is_home_column_position(
        pos
    ) or BoardConstants.is_safe_position(pos)


def forward_distance(start: int, end: int) -> int:
    """Distance moving forward around the main loop from start -> end (0..51).

    Requires both positions be on main board (0..51). Caller should prefilter.
    """
    if start <= end:
        return end - start
    return (GameConstants.MAIN_BOARD_SIZE - start) + end


def backward_distance(start: int, end: int) -> int:
    """Distance moving backward from start to end along the main loop.

    Equivalent to forward_distance(end, start).
    """
    return forward_distance(end, start)


def get_my_main_positions(ctx: Dict) -> Set[int]:
    """Own token positions on main board (exclude home column and off-board)."""
    color = ctx.get("current_situation", {}).get("player_color")
    positions: Set[int] = set()
    for p in ctx.get("players", []):
        if p.get("color") != color:
            continue
        for t in p.get("tokens", []):
            pos = t.get("position", -1)
            if pos >= 0 and not BoardConstants.is_home_column_position(pos):
                positions.add(pos)
    return positions


def get_opponent_main_positions(ctx: Dict) -> List[int]:
    """Opponent token positions on main board (0..51)."""
    color = ctx.get("current_situation", {}).get("player_color")
    out: List[int] = []
    for p in ctx.get("players", []):
        if p.get("color") == color:
            continue
        for t in p.get("tokens", []):
            pos = t.get("position", -1)
            if pos >= 0 and not BoardConstants.is_home_column_position(pos):
                out.append(pos)
    return out


def compute_threats_for_moves(
    moves: List[Dict], ctx: Dict, my_positions: Set[int] | None = None
) -> Dict[int, Tuple[int, int]]:
    """Compute incoming threat for each move's landing square.

    Returns mapping token_id -> (threat_count, min_forward_distance), where
    threat_count is the number of opponent tokens that could reach the landing
    in 1..6 forward steps, and min_forward_distance is the smallest such distance
    (or NO_THREAT_DISTANCE if none).

    Immunities: home column squares, safe squares (stars/start), and landing on
    own occupied main-board square (stacking to form/keep a block).
    """
    opp_positions = get_opponent_main_positions(ctx)
    if my_positions is None:
        my_positions = get_my_main_positions(ctx)
    res: Dict[int, Tuple[int, int]] = {}
    for mv in moves:
        landing = mv.get("target_position")
        if not isinstance(landing, int):
            # Treat non-integer or invalid as immune (e.g., None)
            res[mv["token_id"]] = (0, NO_THREAT_DISTANCE)
            continue
        if is_safe_or_home(landing):
            res[mv["token_id"]] = (0, NO_THREAT_DISTANCE)
            continue
        if landing in my_positions:
            res[mv["token_id"]] = (0, NO_THREAT_DISTANCE)
            continue
        count = 0
        mind = NO_THREAT_DISTANCE
        for opp in opp_positions:
            dist = forward_distance(opp, landing)
            if 1 <= dist <= GameConstants.DICE_MAX:
                count += 1
                if dist < mind:
                    mind = dist
        res[mv["token_id"]] = (count, mind)
    return res


def get_opponent_main_positions_with_fallback(
    game_context: Dict, current_color: str
) -> List[int]:
    """Get opponent positions with fallback to board_state if utils fails."""
    res = get_opponent_main_positions(game_context)
    if res:
        return res
    board_state = game_context.get("board", {})
    bp = board_state.get("board_positions", {})
    fallback: List[int] = []
    for k, tokens in bp.items():
        try:
            pos = int(k)
        except Exception:
            continue
        if pos < 0 or pos >= GameConstants.MAIN_BOARD_SIZE:
            continue
        for t in tokens:
            if t.get("player_color") != current_color:
                fallback.append(pos)
    return fallback


def get_my_main_positions_with_fallback(
    game_context: Dict, current_color: str
) -> List[int]:
    """Get own positions with fallback to board_state if utils fails."""
    mine = list(get_my_main_positions(game_context))
    if mine:
        return mine
    board_state = game_context.get("board", {})
    bp = board_state.get("board_positions", {})
    fallback: List[int] = []
    for k, tokens in bp.items():
        try:
            pos = int(k)
        except Exception:
            continue
        if pos < 0 or pos >= GameConstants.MAIN_BOARD_SIZE:
            continue
        for t in tokens:
            if t.get("player_color") == current_color:
                fallback.append(pos)
    return fallback
