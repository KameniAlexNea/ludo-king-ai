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

    if _get(events, "finished"):
        mover_reward += reward_config.finish

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
    """Masked-average by active (ring) tokens, then scale by active ratio.

    Effectively returns sum_per_piece_prob / 4 to retain global context.
    """
    sum_probs = 0.0
    active = 0
    for r in my_rels:
        if 1 <= r <= king_config.MAIN_TRACK_END:
            ks = 0
            for k in range(1, 7):
                t = r + k
                if 1 <= t <= king_config.MAIN_TRACK_END:
                    abs_pos = game.board.absolute_position(agent_color, t)
                    if abs_pos in king_config.SAFE_SQUARES_ABS:
                        continue
                    occ = game.board.pieces_at_absolute(
                        abs_pos, exclude_color=agent_color
                    )
                    if len(occ) >= 1:
                        ks += 1
            sum_probs += ks / 6.0
            active += 1
    # masked-average-by-active * active_ratio == (sum_probs/active) * (active/4) == sum_probs/4
    return (sum_probs / 4.0) if active > 0 else 0.0


def _cap_risk_probability_depth(
    game: "Game",
    agent_color: int,
    my_rels: list[int],
    opps: list[tuple[int, list[int]]],
    depth: int,
) -> float:
    """Approx probability of being captured within depth plies.

    Masked-average by active ring tokens, then multiply by active ratio -> sum/4.
    """
    sum_probs = 0.0
    active = 0
    for r in my_rels:
        if 1 <= r <= king_config.MAIN_TRACK_END:
            abs_my = game.board.absolute_position(agent_color, r)
            if abs_my in king_config.SAFE_SQUARES_ABS:
                # Active, but risk=0 on global safe
                active += 1
                continue
            ks_union = set()
            for oc, opp_rels in opps:
                for rop in opp_rels:
                    if 1 <= rop <= king_config.MAIN_TRACK_END:
                        for k in range(1, 7):
                            t = rop + k
                            if 1 <= t <= king_config.MAIN_TRACK_END:
                                if game.board.absolute_position(oc, t) == abs_my:
                                    ks_union.add(k)
            p1 = len(ks_union) / 6.0
            p_depth = 1.0 - math.pow(1.0 - p1, max(1, depth))
            sum_probs += p_depth
            active += 1
    return (sum_probs / 4.0) if active > 0 else 0.0


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
