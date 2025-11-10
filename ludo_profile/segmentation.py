"""Game segmentation into behavioral phases."""

from typing import List, Tuple

from .models import MoveRecord


def segment_by_step_count(
    moves: List[MoveRecord],
    window_size: int = 20,
) -> List[Tuple[int, int, List[MoveRecord]]]:
    """
    Segment moves into fixed-size windows by step count.
    
    Parameters
    ----------
    moves : List[MoveRecord]
        All moves by a single player.
    window_size : int
        Number of moves per segment (default: 20).
        
    Returns
    -------
    List[Tuple[int, int, List[MoveRecord]]]
        List of (start_step, end_step, moves_in_segment).
    """
    if not moves:
        return []
    
    segments = []
    for i in range(0, len(moves), window_size):
        segment_moves = moves[i : i + window_size]
        if segment_moves:
            start_step = segment_moves[0].step
            end_step = segment_moves[-1].step
            segments.append((start_step, end_step, segment_moves))
    
    return segments


def segment_by_game_phase(
    moves: List[MoveRecord],
    all_moves: List[MoveRecord],
) -> List[Tuple[int, int, List[MoveRecord], str]]:
    """
    Segment moves into game phases: opening, midgame, endgame.
    
    Parameters
    ----------
    moves : List[MoveRecord]
        All moves by a single player.
    all_moves : List[MoveRecord]
        All moves in the game.
        
    Returns
    -------
    List[Tuple[int, int, List[MoveRecord], str]]
        List of (start_step, end_step, moves_in_segment, phase_name).
    """
    if not moves:
        return []
    
    segments = []
    
    # Opening: until all pieces are out of home (or first 1/3 of moves)
    opening_end_idx = 0
    for i, move in enumerate(moves):
        if move.board_state and move.board_state.pieces_in_yard == 0:
            opening_end_idx = i
            break
        if i >= len(moves) // 3:
            opening_end_idx = i
            break
    
    if opening_end_idx > 0:
        opening_moves = moves[: opening_end_idx + 1]
        segments.append((
            opening_moves[0].step,
            opening_moves[-1].step,
            opening_moves,
            "opening",
        ))
    
    # Endgame: from first piece finishing to end (or last 1/3 of moves)
    endgame_start_idx = len(moves) - 1
    for i, move in enumerate(moves):
        # move.events is a MoveEvents dataclass; use attribute access
        if getattr(move.events, "finished", False):
            endgame_start_idx = i
            break
    
    # If no finish detected, use last third
    if endgame_start_idx == len(moves) - 1:
        endgame_start_idx = max(opening_end_idx + 1, (2 * len(moves)) // 3)
    
    # Midgame: everything between opening and endgame
    if endgame_start_idx > opening_end_idx + 1:
        midgame_moves = moves[opening_end_idx + 1 : endgame_start_idx]
        if midgame_moves:
            segments.append((
                midgame_moves[0].step,
                midgame_moves[-1].step,
                midgame_moves,
                "midgame",
            ))
    
    # Endgame
    endgame_moves = moves[endgame_start_idx:]
    if endgame_moves:
        segments.append((
            endgame_moves[0].step,
            endgame_moves[-1].step,
            endgame_moves,
            "endgame",
        ))
    
    return segments


def segment_adaptive(
    moves: List[MoveRecord],
    min_window: int = 10,
    max_window: int = 30,
) -> List[Tuple[int, int, List[MoveRecord]]]:
    """
    Segment adaptively based on behavior change detection.
    
    This is a placeholder for more sophisticated segmentation
    that detects behavioral shifts (e.g., sudden change in aggression).
    
    Parameters
    ----------
    moves : List[MoveRecord]
        All moves by a single player.
    min_window : int
        Minimum moves per segment.
    max_window : int
        Maximum moves per segment.
        
    Returns
    -------
    List[Tuple[int, int, List[MoveRecord]]]
        List of (start_step, end_step, moves_in_segment).
    """
    # For now, use fixed windows
    # TODO: Implement change-point detection
    return segment_by_step_count(moves, window_size=20)
