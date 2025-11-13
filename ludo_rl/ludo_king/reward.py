from typing import Any, Dict, Union

from .config import reward_config
from .types import MoveEvents


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
