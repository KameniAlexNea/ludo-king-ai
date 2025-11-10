"""Utilities for building GameTrace from various sources."""

from typing import Dict, List, Union

from .models import (
    BlockadeEvent,
    BoardStateSummary,
    GameTrace,
    KnockoutEvent,
    MoveEvents,
    MoveRecord,
    PlayerInfo,
)


def _convert_events(events: Union[Dict, MoveEvents]) -> MoveEvents:
    """
    Convert events from dict or MoveEvents to MoveEvents dataclass.
    
    Parameters
    ----------
    events : Union[Dict, MoveEvents]
        Events as dict or dataclass.
        
    Returns
    -------
    MoveEvents
        Events as dataclass.
    """
    if isinstance(events, MoveEvents):
        return events
    
    # Convert knockouts
    knockouts = []
    for ko in events.get("knockouts", []):
        if isinstance(ko, KnockoutEvent):
            knockouts.append(ko)
        else:
            knockouts.append(KnockoutEvent(
                player=ko["player"],
                piece_id=ko["piece_id"],
                abs_pos=ko.get("abs_pos", 0),
            ))
    
    # Convert blockades
    blockades = []
    for blk in events.get("blockades", []):
        if isinstance(blk, BlockadeEvent):
            blockades.append(blk)
        else:
            blockades.append(BlockadeEvent(
                player=blk["player"],
                rel=blk.get("rel", 0),
            ))
    
    return MoveEvents(
        exited_home=events.get("exited_home", False),
        finished=events.get("finished", False),
        knockouts=knockouts,
        hit_blockade=events.get("hit_blockade", False),
        blockades=blockades,
        move_resolved=events.get("move_resolved", True),
    )


def build_trace_from_dict(data: Dict) -> GameTrace:
    """
    Build GameTrace from dictionary (e.g., loaded from JSON).
    
    Parameters
    ----------
    data : Dict
        Dictionary with game trace data.
        
    Returns
    -------
    GameTrace
        Constructed game trace.
    """
    players = [
        PlayerInfo(
            index=p["index"],
            color=p["color"],
            strategy=p["strategy"],
        )
        for p in data["players"]
    ]
    
    moves = []
    for m in data["moves"]:
        board_state = None
        if "board_state_summary" in m:
            bs = m["board_state_summary"]
            board_state = BoardStateSummary(
                pieces_in_yard=bs["pieces_in_yard"],
                pieces_on_track=bs["pieces_on_track"],
                pieces_in_home=bs["pieces_in_home"],
                pieces_finished=bs["pieces_finished"],
                avg_position=bs["avg_position"],
                max_position=bs.get("max_position", 0),
                min_position_on_track=bs.get("min_position_on_track", 0),
            )
        
        # Convert events dict to MoveEvents dataclass
        events_data = m["events"]
        events = _convert_events(events_data)
        
        move = MoveRecord(
            step=m["step"],
            player_index=m["player_index"],
            dice_roll=m["dice_roll"],
            piece_id=m["piece_id"],
            old_position=m["old_position"],
            new_position=m["new_position"],
            events=events,
            extra_turn=m["extra_turn"],
            board_state=board_state,
        )
        moves.append(move)
    
    return GameTrace(
        game_id=data["game_id"],
        num_players=data["num_players"],
        players=players,
        moves=moves,
        winner=data.get("winner"),
        total_turns=data.get("total_turns", len(moves)),
    )


def build_trace_from_game(game, move_history: List[Dict]) -> GameTrace:
    """
    Build GameTrace from a ludo_rl.ludo_king.Game instance.
    
    Parameters
    ----------
    game : ludo_rl.ludo_king.Game
        Game instance.
    move_history : List[Dict]
        History of moves with format:
        [{"player": int, "dice": int, "piece": int, "old": int, "new": int, "events": dict, "extra": bool}, ...]
        
    Returns
    -------
    GameTrace
        Constructed game trace.
    """
    from ludo_rl.ludo_king import Color
    
    # Build player info
    players = [
        PlayerInfo(
            index=i,
            color=Color(player.color).name,
            strategy=getattr(player, "strategy_name", "unknown"),
        )
        for i, player in enumerate(game.players)
    ]
    
    # Build moves with board state snapshots
    moves = []
    for step, move_data in enumerate(move_history, start=1):
        player_idx = move_data["player"]
        player = game.players[player_idx]
        
        # Compute board state summary
        pieces_in_yard = sum(1 for p in player.pieces if p.position == 0)
        pieces_finished = sum(1 for p in player.pieces if p.position == 57)
        pieces_on_track = sum(1 for p in player.pieces if 1 <= p.position <= 51)
        pieces_in_home = sum(1 for p in player.pieces if 52 <= p.position <= 56)
        
        positions = [p.position for p in player.pieces if p.position > 0]
        avg_position = sum(positions) / len(positions) if positions else 0
        max_position = max(positions) if positions else 0
        min_on_track = min([p for p in positions if 1 <= p <= 56], default=0)
        
        board_state = BoardStateSummary(
            pieces_in_yard=pieces_in_yard,
            pieces_on_track=pieces_on_track,
            pieces_in_home=pieces_in_home,
            pieces_finished=pieces_finished,
            avg_position=avg_position,
            max_position=max_position,
            min_position_on_track=min_on_track,
        )
        
        # Convert events dict to MoveEvents dataclass
        events = _convert_events(move_data["events"])
        
        move = MoveRecord(
            step=step,
            player_index=player_idx,
            dice_roll=move_data["dice"],
            piece_id=move_data["piece"],
            old_position=move_data["old"],
            new_position=move_data["new"],
            events=events,
            extra_turn=move_data["extra"],
            board_state=board_state,
        )
        moves.append(move)
    
    # Determine winner
    winner = None
    for i, player in enumerate(game.players):
        if player.check_won():
            winner = i
            break
    
    return GameTrace(
        game_id=f"game_{id(game)}",
        num_players=len(game.players),
        players=players,
        moves=moves,
        winner=winner,
        total_turns=len(moves),
    )
