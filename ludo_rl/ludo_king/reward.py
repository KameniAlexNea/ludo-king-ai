import math
from typing import TYPE_CHECKING, Any, Dict, Union

from .config import config as king_config
from .config import reward_config
from .types import MoveEvents

if TYPE_CHECKING:
    from .game import Game


def _get(events: Union[MoveEvents, Dict[str, Any]], name: str, default=0):
    """
    Unified accessor to support both dataclass MoveEvents and dicts.
    """

    if isinstance(events, MoveEvents):
        return getattr(events, name, default)
    return (events or {}).get(name, default)


def compute_move_rewards(
    num_players: int,
    mover_index: int,
    old_position: int,
    new_position: int,
    events: Union[MoveEvents, Dict[str, Any]],
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
        mover_reward += reward_config.capture * len(knockouts)
        for knockout in knockouts:
            victim_index = knockout["player"]
            rewards[victim_index] += reward_config.got_capture
    if _get(events, "hit_blockade"):
        mover_reward += reward_config.hit_blockade
    if _get(events, "blockades"):
        mover_reward += reward_config.blockade
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
        if king_config.HOME_COLUMN_START <= r <= king_config.HOME_FINISH - 1:
            total += 1.0 / 6.0 if 1 <= king_config.HOME_FINISH - r <= 6 else 0.0
    return (total / max(1, len(my_rels))) if my_rels else 0.0


def _progress_normalized(my_rels: list[int]) -> float:
    total = 0.0
    for r in my_rels:
        total += max(0, min(r, king_config.HOME_FINISH)) / king_config.HOME_FINISH
    return (total / max(1, len(my_rels))) if my_rels else 0.0


def compute_state_potential(game, agent_index: int, depth: int) -> float:
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
        return float(reward_config.win)
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
