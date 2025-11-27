import math
from typing import TYPE_CHECKING, Any, Dict, Union

from .config import config as king_config
from .config import reward_config
from .types import MoveEvents

if TYPE_CHECKING:
    from .game import Game
    from .board import Board


def _get(events: Union[MoveEvents, Dict[str, Any]], name: str, default=0):
    """
    Unified accessor to support both dataclass MoveEvents and dicts.
    """

    if isinstance(events, MoveEvents):
        return getattr(events, name, default)
    return (events or {}).get(name, default)


def _is_position_safe(position: int, mover_color: int, board: 'Board' = None) -> bool:
    """
    Check if a position is safe from capture.
    
    Safe positions:
    - Yard (0)
    - Finished (57)
    - Home stretch (52-56)
    - Safe squares on main track (star squares)
    """
    if position == 0 or position == king_config.HOME_FINISH:  # yard or finished
        return True
    if king_config.HOME_COLUMN_START <= position <= king_config.HOME_FINISH - 1:  # home stretch
        return True
    if 1 <= position <= king_config.MAIN_TRACK_END:  # main track
        if board is not None:
            abs_pos = board.absolute_position(mover_color, position)
            return abs_pos in king_config.SAFE_SQUARES_ABS
    return False


def _count_threats_at_position(
    board: 'Board',
    mover_color: int,
    position: int,
    opponent_positions: list[tuple[int, list[int]]],
) -> int:
    """
    Count how many dice rolls (1-6) could result in an opponent capturing
    the piece at this position.
    
    Returns a count 0-6 representing how "exposed" the position is.
    Higher = more dangerous.
    """
    if position == 0 or position >= king_config.HOME_COLUMN_START:
        # Yard or home stretch - cannot be captured
        return 0
    
    if not (1 <= position <= king_config.MAIN_TRACK_END):
        return 0
    
    abs_target = board.absolute_position(mover_color, position)
    
    # Check if on safe square - still count threats but will be used differently
    # (Note: you CAN be captured on safe squares, but it's less common)
    
    threat_dice: set[int] = set()
    
    for opp_color, opp_rels in opponent_positions:
        for opp_rel in opp_rels:
            if 1 <= opp_rel <= king_config.MAIN_TRACK_END:
                abs_opp = board.absolute_position(opp_color, opp_rel)
                # Forward distance from opponent to target on the ring
                # Ring is 52 squares (1-52, with 52 wrapping to 1)
                distance = (abs_target - abs_opp) % 52
                if 1 <= distance <= 6:
                    threat_dice.add(distance)
    
    return len(threat_dice)


def compute_exposure_delta(
    board: 'Board',
    mover_color: int,
    old_position: int,
    new_position: int,
    opponent_positions: list[tuple[int, list[int]]],
) -> float:
    """
    Compute the CHANGE in exposure from moving from old_position to new_position.
    
    Returns:
        Positive value = became MORE exposed (bad)
        Negative value = became LESS exposed (good)
        Zero = no change in exposure
    
    This measures the COST of the move in terms of safety, not the absolute exposure.
    A player already in danger who captures doesn't pay extra - they were already at risk.
    A player who leaves safety to capture pays the exposure cost.
    """
    # Exposure before the move
    threats_before = _count_threats_at_position(
        board, mover_color, old_position, opponent_positions
    )
    
    # Exposure after the move
    threats_after = _count_threats_at_position(
        board, mover_color, new_position, opponent_positions
    )
    
    # Normalize to 0-1 range (6 possible dice values)
    exposure_before = threats_before / 6.0
    exposure_after = threats_after / 6.0
    
    # Return the delta: positive = more exposed, negative = safer
    return exposure_after - exposure_before


def compute_move_rewards(
    num_players: int,
    mover_index: int,
    old_position: int,
    new_position: int,
    events: Union[MoveEvents, Dict[str, Any]],
    board: 'Board' = None,
    mover_color: int = None,
    opponent_positions: list[tuple[int, list[int]]] = None,
) -> Dict[int, float]:
    """
    Calculate per-player rewards for a completed move.

    Parameters
    ----------
    num_players:
        Total number of players participating in the match.
    mover_index:
        Index of the player who executed the move.
    old_position / new_position:
        Piece positions before and after the move (relative coordinates).
    events:
        Structured event metadata collected during the move resolution.
    board:
        Optional board object for exposure calculations.
    mover_color:
        Optional color of the mover (needed for position conversions).
    opponent_positions:
        Optional list of (color, [positions]) for each opponent.

    Returns
    -------
    Dict[int, float]
        Incremental reward for each player index.
    """

    rewards: Dict[int, float] = {idx: 0.0 for idx in range(num_players)}
    mover_reward = 0.0
    if _get(events, "move_resolved", True) and old_position != new_position:
        mover_reward += reward_config.progress
    if _get(events, "exited_home"):
        mover_reward += reward_config.exit_home
        # Penalize opponents slightly when mover exits home (urgency signal)
        for idx in range(num_players):
            if idx != mover_index:
                rewards[idx] += reward_config.opp_exit_home_penalty
    if _get(events, "finished"):
        mover_reward += reward_config.finish
        # Penalize opponents when mover finishes a piece
        for idx in range(num_players):
            if idx != mover_index:
                rewards[idx] += reward_config.opp_piece_finished_penalty
    
    knockouts = _get(events, "knockouts", []) or []
    if knockouts:
        base_capture_reward = reward_config.capture * len(knockouts)
        mover_reward += base_capture_reward
        
        for knockout in knockouts:
            # Support both KnockoutEvent dataclass and legacy dict
            victim_index = (
                knockout.player if hasattr(knockout, "player") else knockout["player"]
            )
            rewards[victim_index] += reward_config.got_capture
    
    if _get(events, "hit_blockade"):
        mover_reward += reward_config.hit_blockade
    if _get(events, "blockades"):
        mover_reward += reward_config.blockade
    
    # === EXPOSURE-BASED REWARD ADJUSTMENT ===
    # Apply to ALL moves: penalize moves that INCREASE exposure, reward moves that DECREASE it
    # This is the key insight: it's not about where you end up, but whether you made yourself
    # MORE vulnerable than before. A player already in danger doesn't pay extra for staying there.
    if board is not None and mover_color is not None and opponent_positions is not None:
        exposure_delta = compute_exposure_delta(
            board, mover_color, old_position, new_position, opponent_positions
        )
        # exposure_delta > 0 means we became MORE exposed (penalty)
        # exposure_delta < 0 means we became LESS exposed (bonus)
        # exposure_delta = 0 means no change
        # Scale by the penalty factor (negative delta becomes positive reward)
        mover_reward -= exposure_delta * reward_config.capture_exposure_penalty
    
    # Small bonus for landing on safe position (encourages safe play)
    if board is not None and mover_color is not None:
        if _is_position_safe(new_position, mover_color, board):
            # Don't double-reward finishing (already has finish bonus)
            if new_position != king_config.HOME_FINISH:
                mover_reward += reward_config.safe_landing_bonus
    
    rewards[mover_index] += mover_reward
    return rewards


# --- Risk/Opportunity shaping (potential-based) ---
def _rel_positions_for_agent(game: "Game", agent_index: int):
    agent_color = int(game.players[agent_index].color)
    my_rels = [int(p.position) for p in game.players[agent_index].pieces]
    opps = [
        (int(pl.color), [int(p.position) for p in pl.pieces])
        for i, pl in enumerate(game.players)
        if i != agent_index
    ]
    return agent_color, my_rels, opps


def _cap_opp_probability(
    game: "Game", agent_color: int, my_rels: list[int], opp_occ_abs: set[int]
) -> float:
    # Probability to capture on next move averaged across my tokens
    total = 0.0
    count = 0
    safe_abs = set(king_config.SAFE_SQUARES_ABS)
    for r in my_rels:
        if 1 <= r <= king_config.MAIN_TRACK_END:
            ks = 0
            for k in range(1, 7):
                t = r + k
                if 1 <= t <= king_config.MAIN_TRACK_END:
                    abs_pos = game.board.absolute_position(agent_color, t)
                    if abs_pos in safe_abs:
                        continue
                    if abs_pos in opp_occ_abs:
                        ks += 1
            total += ks / 6.0
            count += 1
    return (total / max(1, count)) if count else 0.0


def _cap_risk_probability_depth(
    game: "Game",
    agent_color: int,
    my_rels: list[int],
    opps: list[tuple[int, list[int]]],
    depth: int,
) -> float:
    # Approx probability my token gets captured within depth plies, fast distance math
    total = 0.0
    count = 0
    safe_abs = set(king_config.SAFE_SQUARES_ABS)
    for r in my_rels:
        if 1 <= r <= king_config.MAIN_TRACK_END:
            abs_my = game.board.absolute_position(agent_color, r)
            if abs_my in safe_abs:
                total += 0.0
                count += 1
                continue
            ks_union: set[int] = set()
            for oc, opp_rels in opps:
                for rop in opp_rels:
                    if 1 <= rop <= king_config.MAIN_TRACK_END:
                        abs_opp = game.board.absolute_position(oc, rop)
                        # forward distance on ring (1..51). Both abs are in 1..51
                        d = (abs_my - abs_opp) % 52
                        if 1 <= d <= 6:
                            ks_union.add(d)
            p1 = len(ks_union) / 6.0
            p_depth = 1.0 - math.pow(1.0 - p1, max(1, depth))
            total += p_depth
            count += 1
    return (total / max(1, count)) if count else 0.0


def _finish_opportunity_probability(my_rels: list[int]) -> float:
    # Probability to finish a piece on next move (home column only)
    total = 0.0
    for r in my_rels:
        if king_config.HOME_COLUMN_START - 1 <= r <= king_config.HOME_FINISH - 1:
            total += 1.0 / 6.0 if 1 <= king_config.HOME_FINISH - r <= 6 else 0.0
    return (total / max(1, len(my_rels))) if my_rels else 0.0


def _progress_normalized(my_rels: list[int]) -> float:
    total = 0.0
    for r in my_rels:
        total += max(0, min(r, king_config.HOME_FINISH)) / king_config.HOME_FINISH
    return (total / max(1, len(my_rels))) if my_rels else 0.0


def compute_state_potential(game: "Game", agent_index: int, depth: int) -> float:
    """
    Compute a dense potential Φ(s) from risk/opportunity signals.
    """

    agent_color, my_rels, opps = _rel_positions_for_agent(game, agent_index)
    # Build opponent occupied absolute set once
    opp_occ_abs: set[int] = set()
    for oc, opp_rels in opps:
        for r in opp_rels:
            if 1 <= r <= king_config.MAIN_TRACK_END:
                opp_occ_abs.add(game.board.absolute_position(oc, r))
    p_cap_opp = _cap_opp_probability(game, agent_color, my_rels, opp_occ_abs)
    p_cap_risk = _cap_risk_probability_depth(game, agent_color, my_rels, opps, depth)
    p_finish_opp = _finish_opportunity_probability(my_rels)
    prog = _progress_normalized(my_rels)

    phi = (
        reward_config.ro_w_progress * prog
        + reward_config.ro_w_cap_opp * p_cap_opp
        - reward_config.ro_w_cap_risk * p_cap_risk
        + reward_config.ro_w_finish_opp * p_finish_opp
    )
    return phi


def shaping_delta(phi_before: float, phi_after: float, gamma: float) -> float:
    return gamma * phi_after - phi_before


# --- Supplemental reward helpers to centralize all reward math ---
def compute_invalid_action_penalty() -> float:
    """Penalty applied when the agent selects an invalid action.

    Centralized here to keep env/game logic free of reward constants.
    """
    return float(reward_config.skipped_turn)


def compute_blockade_hits_bonus(count: float) -> float:
    """Reward bonus proportional to the number of opponent hits on agent blockades."""
    return float(reward_config.blockade_hit) * float(count)


def compute_terminal_reward(num_players: int, rank: int) -> float:
    """Return terminal reward for the agent based on final rank.

    rank == 1 => win reward
    otherwise => scaled lose reward (higher rank -> smaller penalty)
    """

    if rank == 1:
        return reward_config.win
    # Scale the (negative) lose reward linearly by placement severity:
    # 2nd -> small fraction, ... -> last -> full penalty
    # Example (4 players): rank 2 => 1/3, rank 3 => 2/3, rank 4 => 1
    den = max(1, num_players - 1)
    scale = float(max(1, rank) - 1) / float(den)
    return float(reward_config.lose) * scale


def compute_draw_reward() -> float:
    """Reward for truncated (draw) episodes."""
    return float(reward_config.draw)


def compute_skipped_turn_penalty() -> float:
    """Small negative reward when the agent has to skip a turn (no legal moves)."""
    return float(reward_config.skipped_turn)
