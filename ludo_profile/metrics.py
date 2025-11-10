"""Metric calculators for behavior analysis."""

from typing import List

from ludo_rl.ludo_king.config import config

from .models import BehaviorCharacteristics, MoveRecord

# Safe squares on the main track (relative positions)
SAFE_POSITIONS = {1, 9, 14, 22, 27, 35, 40, 48}  # Start + safe squares


def compute_characteristics(moves: List[MoveRecord]) -> BehaviorCharacteristics:
    """
    Compute behavior characteristics from a sequence of moves.
    
    Parameters
    ----------
    moves : List[MoveRecord]
        Sequence of moves by a single player.
        
    Returns
    -------
    BehaviorCharacteristics
        Computed behavioral metrics.
    """
    if not moves:
        return BehaviorCharacteristics(
            aggression=0.0,
            risk_taking=0.0,
            exploration=0.0,
            finishing=0.0,
            blockade_usage=0.0,
            defensiveness=0.0,
        )
    
    # Aggression: captures made / capture opportunities
    captures_made = sum(1 for m in moves if m.events.knockouts)
    capture_opportunities = sum(1 for m in moves if m.was_capture_opportunity)
    aggression = captures_made / max(capture_opportunities, 1)
    
    # Risk-taking: risky moves taken / risky situations
    risky_moves = sum(1 for m in moves if m.was_risky_move)
    risky_situations = sum(1 for m in moves if m.left_safe_square)
    risk_taking = risky_moves / max(risky_situations, 1) if risky_situations > 0 else 0.0
    
    # Exploration: unique pieces moved / 4 (total pieces)
    unique_pieces = len(set(m.piece_id for m in moves))
    exploration = unique_pieces / 4.0
    
    # Finishing focus: moves advancing lead piece / total advancing moves
    if moves[0].board_state:
        lead_piece_moves = sum(1 for m in moves if m.advanced_lead_piece)
        advancing_moves = sum(
            1 for m in moves 
            if m.new_position > m.old_position and m.old_position > 0
        )
        finishing = lead_piece_moves / max(advancing_moves, 1)
    else:
        finishing = 0.0
    
    # Blockade usage: blockades formed / blockade opportunities
    blockades_formed = sum(1 for m in moves if m.events.blockades)
    blockade_opportunities = sum(1 for m in moves if m.was_blockade_opportunity)
    blockade_usage = blockades_formed / max(blockade_opportunities, 1)
    
    # Defensiveness: moves to safe squares / moves from exposed positions
    moves_to_safety = sum(1 for m in moves if m.moved_to_safe_square)
    exposed_situations = sum(
        1 for m in moves 
        if m.old_position not in SAFE_POSITIONS and 1 <= m.old_position <= config.MAIN_TRACK_END
    )
    defensiveness = moves_to_safety / max(exposed_situations, 1) if exposed_situations > 0 else 0.0
    
    return BehaviorCharacteristics(
        aggression=min(aggression, 1.0),
        risk_taking=min(risk_taking, 1.0),
        exploration=min(exploration, 1.0),
        finishing=min(finishing, 1.0),
        blockade_usage=min(blockade_usage, 1.0),
        defensiveness=min(defensiveness, 1.0),
    )


def annotate_move_opportunities(
    moves: List[MoveRecord],
    all_moves: List[MoveRecord],
    player_index: int,
) -> None:
    """
    Annotate moves with opportunity flags (in-place).
    
    This analyzes the context of each move to identify what opportunities
    were available and what choices were made.
    
    Parameters
    ----------
    moves : List[MoveRecord]
        Moves by the player being analyzed (will be modified in place).
    all_moves : List[MoveRecord]
        All moves in the game (for context).
    player_index : int
        Index of the player being analyzed.
    """
    # Build a position tracker for all players at each step
    # Track positions: player_positions[step][player_idx] = {piece_id: position}
    player_positions = {}
    num_players = max(m.player_index for m in all_moves) + 1
    
    # Initialize positions (all pieces start at 0)
    current_positions = {pid: {i: 0 for i in range(4)} for pid in range(num_players)}
    player_positions[0] = {pid: dict(positions) for pid, positions in current_positions.items()}
    
    # Simulate through all moves to track positions
    for move in sorted(all_moves, key=lambda m: m.step):
        # Update position
        current_positions[move.player_index][move.piece_id] = move.new_position
        player_positions[move.step] = {pid: dict(positions) for pid, positions in current_positions.items()}
    
    # Now annotate each move with full context
    for move in moves:
        # Get board state at this step (before the move)
        prev_step = move.step - 1
        if prev_step in player_positions:
            positions_before = player_positions[prev_step]
            own_positions = positions_before[player_index]
            
            # Check if this was actually a capture opportunity
            # (any opponent piece at new_position before move)
            move.was_capture_opportunity = False
            for pid in range(num_players):
                if pid != player_index:
                    for piece_pos in positions_before[pid].values():
                        # Need to check if new_position would land on opponent
                        # This is complex due to relative vs absolute positions
                        # For now, flag if capture actually happened
                        if len(move.events.knockouts) > 0:
                            move.was_capture_opportunity = True
                            break
            
            # Was there actually a capture opportunity among legal moves?
            # This requires knowing what other legal moves were available
            # For now: if captured, it was an opportunity taken
            if not move.was_capture_opportunity and len(move.events.knockouts) == 0:
                # Could have been opportunity if not taken - mark conservatively
                move.was_capture_opportunity = False
        
        # Risky move: leaving a safe square to exposed position
        old_safe = move.old_position in SAFE_POSITIONS or move.old_position == 0
        new_safe = move.new_position in SAFE_POSITIONS or move.new_position >= config.HOME_COLUMN_START
        move.was_risky_move = old_safe and not new_safe and 1 <= move.new_position <= config.MAIN_TRACK_END
        move.left_safe_square = move.old_position in SAFE_POSITIONS
        move.moved_to_safe_square = move.new_position in SAFE_POSITIONS
        
        # Blockade opportunity: formed blockade or could have
        move.was_blockade_opportunity = len(move.events.blockades) > 0
        if not move.was_blockade_opportunity and move.board_state:
            # Check if landing position has another own piece
            # This is approximate without full board state
            pass
        
        # Advanced lead piece: moved the piece furthest ahead
        if move.board_state:
            move.advanced_lead_piece = (
                move.old_position == move.board_state.max_position
                and move.new_position > move.old_position
            )


def generate_behavior_descriptions(
    moves: List[MoveRecord],
    characteristics: BehaviorCharacteristics,
) -> List[str]:
    """
    Generate human-readable behavior descriptions.
    
    Parameters
    ----------
    moves : List[MoveRecord]
        Moves in this segment.
    characteristics : BehaviorCharacteristics
        Computed characteristics.
        
    Returns
    -------
    List[str]
        Human-readable behavior descriptions.
    """
    descriptions = []
    
    # Exits from home
    exits = sum(1 for m in moves if m.events.exited_home)
    if exits > 0:
        descriptions.append(f"Got {exits} piece(s) out of home")
    
    # Captures
    captures = sum(len(m.events.knockouts) for m in moves)
    if captures > 0:
        if characteristics.aggression > 0.7:
            descriptions.append(f"Aggressively captured {captures} opponent piece(s)")
        else:
            descriptions.append(f"Captured {captures} opponent piece(s)")
    
    # Blockades
    blockades = sum(len(m.events.blockades) for m in moves)
    if blockades > 0:
        descriptions.append(f"Formed {blockades} blockade(s)")
    
    # Risk behavior
    if characteristics.risk_taking > 0.6:
        risky_count = sum(1 for m in moves if m.was_risky_move)
        descriptions.append(f"Took risky positions {risky_count} time(s)")
    elif characteristics.defensiveness > 0.6:
        safe_moves = sum(1 for m in moves if m.moved_to_safe_square)
        descriptions.append(f"Prioritized safety, moved to safe squares {safe_moves} time(s)")
    
    # Exploration
    unique_pieces = len(set(m.piece_id for m in moves))
    if characteristics.exploration > 0.75:
        descriptions.append(f"Spread pieces evenly (moved {unique_pieces}/4 pieces)")
    elif characteristics.exploration < 0.5 and unique_pieces < 4:
        descriptions.append(f"Focused on few pieces (only moved {unique_pieces}/4)")
    
    # Finishing
    finished = sum(1 for m in moves if m.events.finished)
    if finished > 0:
        descriptions.append(f"Finished {finished} piece(s)")
    elif characteristics.finishing > 0.7:
        descriptions.append("Focused on advancing lead pieces toward home")
    
    # Got captured
    # This requires looking at all_moves to find captures against this player
    # For now, we'll skip this or implement later with full context
    
    return descriptions if descriptions else ["No significant actions"]
