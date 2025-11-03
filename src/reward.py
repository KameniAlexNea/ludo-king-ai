from dataclasses import dataclass
from typing import Dict


@dataclass
class Reward:
    win: float = 50
    lose: float = -50
    finish: float = 10
    capture: float = 2
    got_capture: float = -5
    blockade: float = 3
    hit_blockade: float = -4
    exit_home: float = 1
    progress: float = 0.1
    safe_position: float = 0.5
    skipped_turn = -0.01

reward_config = Reward()


def compute_move_rewards(
    num_players: int,
    mover_index: int,
    old_position: int,
    new_position: int,
    events: Dict,
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

    if events.get("move_resolved", True) and old_position != new_position:
        mover_reward += reward_config.progress

    if events.get("exited_home"):
        mover_reward += reward_config.exit_home

    if events.get("finished"):
        mover_reward += reward_config.finish

    knockouts = events.get("knockouts", []) or []
    if knockouts:
        mover_reward += reward_config.capture * len(knockouts)
        for knockout in knockouts:
            victim_index = knockout["player"]
            rewards[victim_index] += reward_config.got_capture

    if events.get("hit_blockade"):
        mover_reward += reward_config.hit_blockade

    if events.get("blockades"):
        mover_reward += reward_config.blockade

    rewards[mover_index] += mover_reward

    return rewards