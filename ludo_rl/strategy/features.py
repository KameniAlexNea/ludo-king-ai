from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ludo_rl.ludo_king.config import strategy_config

if TYPE_CHECKING:  # avoid runtime import to prevent circular deps
    from ludo_rl.ludo_king.piece import Piece

from .types import MoveOption, StrategyContext


def _create_move_options(
    move: dict,
    dice_roll: int,
    my_channel: np.ndarray,
    safe_channel: np.ndarray,
    opponent_counts: np.ndarray,
) -> MoveOption:
    piece: "Piece" = move["piece"]
    new_pos = move["new_pos"]
    current_pos = piece.position
    progress = _compute_progress(current_pos, new_pos)
    distance_to_goal = max(strategy_config.home_finish - new_pos, 0)

    can_capture, capture_count = _check_capture(opponent_counts, safe_channel, new_pos)
    enters_home = new_pos == strategy_config.home_finish
    enters_safe_zone = _is_safe_destination(safe_channel, new_pos)
    forms_blockade = _forms_blockade(my_channel, safe_channel, new_pos)
    extra_turn = dice_roll == 6 or enters_home or can_capture
    risk = _estimate_risk(opponent_counts, safe_channel, new_pos)
    leaving_safe_zone = (
        _is_safe_destination(safe_channel, current_pos) and not enters_safe_zone
    )
    return MoveOption(
        piece_id=piece.piece_id,
        current_pos=current_pos,
        new_pos=new_pos,
        dice_roll=dice_roll,
        progress=progress,
        distance_to_goal=distance_to_goal,
        can_capture=can_capture,
        capture_count=capture_count,
        enters_home=enters_home,
        enters_safe_zone=enters_safe_zone,
        forms_blockade=forms_blockade,
        extra_turn=extra_turn,
        risk=risk,
        leaving_safe_zone=leaving_safe_zone,
    )


def build_move_options(
    board: np.ndarray,
    dice_roll: int,
    action_mask: np.ndarray,
    move_choices: Sequence[dict | None],
) -> StrategyContext:
    """Convert board state and legal moves into a strategy context."""
    dice = int(dice_roll)
    my_channel = board[strategy_config.board_channel_my]
    safe_channel = board[strategy_config.board_channel_safe]
    opponents = board[
        strategy_config.board_channel_opp_start : strategy_config.board_channel_opp_end
        + 1
    ]
    opponent_counts = opponents.sum(axis=0)

    moves = [
        _create_move_options(move, dice, my_channel, safe_channel, opponent_counts)
        for move in move_choices
        if move is not None
    ]

    return StrategyContext(
        board=board,
        dice_roll=dice,
        action_mask=action_mask,
        moves=moves,
        my_distribution=my_channel,
        opponent_distribution=opponent_counts,
        safe_channel=safe_channel,
    )


def _compute_progress(current_pos: int, new_pos: int) -> int:
    if current_pos == 0:
        return new_pos  # entering the board
    return max(new_pos - current_pos, 0)


def _check_capture(
    opponent_counts: np.ndarray, safe_channel: np.ndarray, new_pos: int
) -> tuple[bool, int]:
    if new_pos <= 0 or new_pos > strategy_config.main_track_end:
        return False, 0
    if safe_channel[new_pos] > 0:
        return False, 0
    captured = int(opponent_counts[new_pos])
    return captured > 0, captured


def _is_safe_destination(safe_channel: np.ndarray, new_pos: int) -> bool:
    if new_pos >= strategy_config.home_start:
        return True
    if new_pos <= 0:
        return False
    return bool(safe_channel[new_pos])


def _forms_blockade(
    my_channel: np.ndarray, safe_channel: np.ndarray, new_pos: int
) -> bool:
    if new_pos <= 0 or new_pos > strategy_config.main_track_end:
        return False
    if safe_channel[new_pos]:
        return False
    current_count = my_channel[new_pos]
    return current_count >= 1


def _estimate_risk(
    opponent_counts: np.ndarray, safe_channel: np.ndarray, new_pos: int
) -> float:
    if new_pos <= 0 or new_pos > strategy_config.main_track_end:
        return 0.0

    risk = 0.0

    for step in range(1, 7):
        idx = new_pos - step
        if idx <= 0:
            idx += strategy_config.main_track_end
        if safe_channel[idx]:
            continue
        threat_level = opponent_counts[idx]
        if threat_level == 0:
            continue
        weight = 1.0 - (step - 1) / 6.0
        risk += threat_level * weight

    return risk


def opponent_density_within(
    distribution: Sequence[float], center: int, radius: int = 6
) -> float:
    total = 0.0
    for offset in range(-radius, radius + 1):
        idx = center + offset
        if idx <= 0:
            idx += strategy_config.main_track_end
        elif idx > strategy_config.main_track_end:
            idx -= strategy_config.main_track_end
        total += distribution[idx]
    return float(total)


def nearest_opponent_distance(distribution: Sequence[float], position: int) -> int:
    for distance in range(1, strategy_config.main_track_end + 1):
        forward = position + distance
        backward = position - distance
        if forward > strategy_config.main_track_end:
            forward -= strategy_config.main_track_end
        if backward <= 0:
            backward += strategy_config.main_track_end
        if distribution[forward] > 0 or distribution[backward] > 0:
            return distance
    return strategy_config.main_track_end
