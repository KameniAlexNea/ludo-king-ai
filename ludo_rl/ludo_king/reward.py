from typing import TYPE_CHECKING, Any, Dict, Union

import numpy as np

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
    opps = []
    for i, pl in enumerate(game.players):
        if i == agent_index:
            continue
        opps.append((int(pl.color), [int(p.position) for p in pl.pieces]))
    return agent_color, my_rels, opps


def _cap_opp_probability(game: "Game", agent_color: int, my_rels: list[int]) -> float:
    """One-ply capture opportunity using ring convolution (no nested loops).

    For each of our active ring tokens at absolute index a, consider the next 6 ring
    squares (a+1..a+6) modulo ring length, exclude global safe squares, and count if any
    opponent occupies those squares. Probability per token is count/6. Return sum/4.
    """
    ring_len = int(king_config.MAIN_TRACK_END)
    if ring_len <= 0:
        return 0.0

    # Opponent occupancy on ring as 0/1 vector length ring_len (0-based index for abs 1..ring_len)
    opp_occ = np.zeros(ring_len, dtype=np.int32)
    for pl in game.players:
        if int(pl.color) == agent_color:
            continue
        for p in pl.pieces:
            rp = int(p.position)
            if 1 <= rp <= ring_len:
                abs_pos = int(game.board.absolute_position(int(pl.color), rp))
                idx = (abs_pos - 1) % ring_len
                opp_occ[idx] = 1

    if not opp_occ.any():
        return 0.0

    # Safe squares mask (0-based indexing)
    safe_mask = np.zeros(ring_len, dtype=np.int32)
    for s in king_config.SAFE_SQUARES_ABS:
        if 1 <= s <= ring_len:
            safe_mask[(s - 1) % ring_len] = 1

    # Collect our active tokens' absolute indices
    my_abs_idx = []
    for r in my_rels:
        if 1 <= r <= ring_len:
            a = int(game.board.absolute_position(agent_color, int(r)))
            my_abs_idx.append((a - 1) % ring_len)

    if not my_abs_idx:
        return 0.0

    a_idx = np.asarray(my_abs_idx, dtype=np.int32)  # shape (M,)
    ks = np.arange(1, 7, dtype=np.int32)  # (6,)
    # next positions for each token and k: shape (M,6)
    next_idx = (a_idx[:, None] + ks[None, :]) % ring_len
    # mask out safe squares
    not_safe = safe_mask[next_idx] == 0
    # occupation at those next squares
    occ_next = opp_occ[next_idx] * not_safe
    # probability per token = (#occupied among 6)/6
    prob_per_token = occ_next.sum(axis=1, dtype=np.float32) / 6.0
    # Sum over tokens, then scale by active_ratio= (#active/4)
    return float(prob_per_token.sum() / 4.0)


def _cap_risk_probability_depth(
    game: "Game",
    agent_color: int,
    my_rels: list[int],
    opps: list[tuple[int, list[int]]],
    depth: int,
) -> float:
    """Capture risk within depth plies via ring convolution (no nested loops).

    For absolute square a, one-ply capture probability p1(a) = m(a)/6 where
    m(a) is the number of k in {1..6} such that an opponent occupies (a-k) on the ring.
    Then depth aggregation: 1 - (1 - p1)^depth. Sum across our active tokens and /4.
    """
    ring_len = int(king_config.MAIN_TRACK_END)
    if ring_len <= 0:
        return 0.0

    # Opponent occupancy vector on ring
    opp_occ = np.zeros(ring_len, dtype=np.int32)
    for oc, opp_rels in opps:
        for rp in opp_rels:
            r = int(rp)
            if 1 <= r <= ring_len:
                abs_pos = int(game.board.absolute_position(int(oc), r))
                opp_occ[(abs_pos - 1) % ring_len] = 1

    if not opp_occ.any():
        return 0.0

    # Precompute sums over the 6 preceding ring squares for each absolute index via rolls
    # For any absolute index a, the predecessors are (a-1 .. a-6) modulo ring_len
    m_per_abs = np.zeros(ring_len, dtype=np.int32)
    for k in range(1, 7):
        m_per_abs += np.roll(opp_occ, k)

    safe_mask = np.zeros(ring_len, dtype=np.int32)
    for s in king_config.SAFE_SQUARES_ABS:
        if 1 <= s <= ring_len:
            safe_mask[(s - 1) % ring_len] = 1

    # Our tokens' absolute indices
    a_list = []
    for r in my_rels:
        if 1 <= r <= ring_len:
            a = int(game.board.absolute_position(agent_color, int(r)))
            a_list.append((a - 1) % ring_len)

    if not a_list:
        return 0.0

    a_idx = np.asarray(a_list, dtype=np.int32)
    # Zero risk on global safes
    m_tokens = m_per_abs[a_idx] * (1 - safe_mask[a_idx])
    p1 = m_tokens.astype(np.float32) / 6.0
    p_depth = 1.0 - np.power(1.0 - p1, max(1, int(depth)))
    return float(p_depth.sum() / 4.0)


def _finish_opportunity_probability(my_rels: list[int]) -> float:
    """Probability to finish on next move (home column), scaled by active_ratio.

    Returns sum(prob_per_piece)/4.
    """
    sum_probs = 0.0
    active = 0
    for r in my_rels:
        if king_config.HOME_COLUMN_START <= r <= king_config.HOME_FINISH - 1:
            need = king_config.HOME_FINISH - r
            sum_probs += (1.0 / 6.0) if 1 <= need <= 6 else 0.0
            active += 1
    return (sum_probs / 4.0) if active > 0 else 0.0


def _progress_normalized(my_rels: list[int]) -> float:
    """Normalized progress averaged over active tokens, scaled by active_ratio.

    Active tokens are those on the board (r > 0). Returns sum(norm)/4.
    """
    sum_norm = 0.0
    active = 0
    for r in my_rels:
        if r > 0:
            sum_norm += (
                max(0, min(r, king_config.HOME_FINISH)) / king_config.HOME_FINISH
            )
            active += 1
    return (sum_norm / 4.0) if active > 0 else 0.0


def compute_state_potential(game, agent_index: int, depth: int) -> float:
    """Compute a dense potential Φ(s) from risk/opportunity signals.

    Components are normalized to [0,1] and combined with weights from reward_config.
    """
    agent_color, my_rels, opps = _rel_positions_for_agent(game, agent_index)
    p_cap_opp = _cap_opp_probability(game, agent_color, my_rels)
    p_cap_risk = _cap_risk_probability_depth(game, agent_color, my_rels, opps, depth)
    p_finish_opp = _finish_opportunity_probability(my_rels)
    prog = _progress_normalized(my_rels)

    phi = (
        reward_config.ro_w_progress * prog
        + reward_config.ro_w_cap_opp * p_cap_opp
        - reward_config.ro_w_cap_risk * p_cap_risk
        + reward_config.ro_w_finish_opp * p_finish_opp
    )
    # Clip potential for stability
    return max(-1.0, min(1.0, float(phi)))


def shaping_delta(phi_before: float, phi_after: float, gamma: float) -> float:
    return gamma * phi_after - phi_before


# --- Supplemental reward helpers to centralize all reward math ---


def compute_invalid_action_penalty() -> float:
    """Penalty applied when the agent selects an invalid action.

    Centralized here to keep env/game logic free of reward constants.
    """
    return reward_config.skipped_turn


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
