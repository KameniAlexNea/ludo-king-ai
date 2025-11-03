from __future__ import annotations

from typing import Dict, List

import numpy as np

from ludo_rl.ludo.model import Piece

from .types import MoveOption, StrategyContext

BOARD_CHANNEL_MY = 0
BOARD_CHANNEL_SAFE = 4
BOARD_CHANNEL_OPP_START = 1
BOARD_CHANNEL_OPP_END = 3

MAIN_TRACK_END = 51
HOME_START = 52
HOME_FINISH = 57


def build_move_options(
    observation: Dict[str, np.ndarray],
    action_mask: np.ndarray,
    move_map: Dict[int, Dict],
) -> StrategyContext:
    """Convert the environment observation + move map into a strategy context.

    Parameters
    ----------
    observation:
        Observation dictionary emitted by :class:`LudoEnv`.
    action_mask:
        Boolean mask of shape (4,) indicating which pieces can move.
    move_map:
        Mapping from piece id to move metadata, as maintained by :class:`LudoEnv`.
    """

    board = observation["board"]
    dice_roll = int(observation["dice_roll"][0]) + 1  # convert back to 1-6
    moves: List[MoveOption] = []

    for piece_id, move in move_map.items():
        piece: Piece = move["piece"]
        new_pos = move["new_pos"]
        current_pos = piece.position
        progress = _compute_progress(current_pos, new_pos)
        distance_to_goal = max(HOME_FINISH - new_pos, 0)

        can_capture, capture_count = _check_capture(board, new_pos)
        enters_home = new_pos >= HOME_START
        enters_safe_zone = _is_safe_destination(board, new_pos)
        forms_blockade = _forms_blockade(board, new_pos, move)
        extra_turn = dice_roll == 6 or enters_home or can_capture
        risk = _estimate_risk(board, new_pos)
        leaving_safe_zone = (
            _is_safe_destination(board, current_pos) and not enters_safe_zone
        )

        moves.append(
            MoveOption(
                piece_id=piece_id,
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
        )

    return StrategyContext(
        board=board, dice_roll=dice_roll, action_mask=action_mask, moves=moves
    )


def _compute_progress(current_pos: int, new_pos: int) -> int:
    if current_pos == 0:
        return new_pos  # entering the board
    return max(new_pos - current_pos, 0)


def _check_capture(board: np.ndarray, new_pos: int) -> tuple[bool, int]:
    if new_pos <= 0 or new_pos > MAIN_TRACK_END:
        return False, 0
    safe_channel = board[BOARD_CHANNEL_SAFE]
    if safe_channel[new_pos] > 0:
        return False, 0
    opponents = board[BOARD_CHANNEL_OPP_START : BOARD_CHANNEL_OPP_END + 1, new_pos]
    captured = int(opponents.sum())
    return captured > 0, captured


def _is_safe_destination(board: np.ndarray, new_pos: int) -> bool:
    if new_pos >= HOME_START:
        return True
    if new_pos <= 0:
        return False
    safe_channel = board[BOARD_CHANNEL_SAFE]
    return bool(safe_channel[new_pos])


def _forms_blockade(board: np.ndarray, new_pos: int, move: Dict) -> bool:
    if new_pos <= 0 or new_pos > MAIN_TRACK_END:
        return False
    safe_channel = board[BOARD_CHANNEL_SAFE]
    if safe_channel[new_pos]:
        return False
    current_count = board[BOARD_CHANNEL_MY, new_pos]
    return current_count >= 1


def _estimate_risk(board: np.ndarray, new_pos: int) -> float:
    if new_pos <= 0 or new_pos > MAIN_TRACK_END:
        return 0.0

    opponents = board[BOARD_CHANNEL_OPP_START : BOARD_CHANNEL_OPP_END + 1]
    safe_channel = board[BOARD_CHANNEL_SAFE]
    risk = 0.0

    for step in range(1, 7):
        idx = new_pos - step
        if idx <= 0:
            idx += MAIN_TRACK_END
        if safe_channel[idx]:
            continue
        threat_level = opponents[:, idx].sum()
        if threat_level == 0:
            continue
        weight = 1.0 - (step - 1) / 6.0
        risk += threat_level * weight

    return risk
