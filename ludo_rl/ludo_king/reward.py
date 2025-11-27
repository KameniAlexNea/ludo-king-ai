"""
Sparse reward computation for Ludo RL environment.

This module provides SPARSE rewards only - no shaping, no exposure calculations.
All reward signals are directly tied to game outcomes:
- Win/lose/draw (terminal)
- Piece finished (milestone)
- Capture/got captured (milestone)

The goal is to minimize conflicting gradients during PPO training.
"""

from typing import Dict

from .config import reward_config
from .types import MoveEvents


def compute_sparse_rewards(
    num_players: int,
    mover_index: int,
    events: MoveEvents,
) -> Dict[int, float]:
    """
    Calculate per-player rewards for a completed move using SPARSE signals only.

    Parameters
    ----------
    num_players:
        Total number of players in the game.
    mover_index:
        Index of the player who executed the move.
    events:
        Structured event metadata collected during the move resolution.

    Returns
    -------
    Dict[int, float]
        Incremental reward for each player index. Most entries will be 0.0.
    """
    rewards: Dict[int, float] = {idx: 0.0 for idx in range(num_players)}

    # Only reward actual outcomes - no progress, no exposure, no shaping
    mover_reward = 0.0

    # Piece reached finish
    if events.finished:
        mover_reward += reward_config.finish

    # Captured opponent piece(s)
    if events.knockouts:
        mover_reward += reward_config.capture * len(events.knockouts)
        # Penalty to victim(s)
        for knockout in events.knockouts:
            rewards[knockout.player] += reward_config.got_captured

    rewards[mover_index] += mover_reward
    return rewards


def compute_invalid_action_penalty() -> float:
    """Penalty applied when the agent selects an invalid action."""
    return float(reward_config.invalid_action)


def compute_terminal_reward(num_players: int, rank: int) -> float:
    """Return terminal reward for the agent based on final rank.

    rank == 1 => win reward
    otherwise => scaled lose reward (higher rank -> smaller penalty)
    """
    if rank == 1:
        return reward_config.win
    # Scale the (negative) lose reward linearly by placement severity:
    # 2nd -> small fraction, ..., last -> full penalty
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
