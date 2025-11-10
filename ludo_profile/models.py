"""Data models for agent profiling."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlayerInfo:
    """Information about a player in the game."""

    index: int
    color: str
    strategy: str


@dataclass
class BoardStateSummary:
    """Summary of board state at a particular move."""

    pieces_in_yard: int
    pieces_on_track: int
    pieces_in_home: int
    pieces_finished: int
    avg_position: float
    max_position: int
    min_position_on_track: int


@dataclass
class KnockoutEvent:
    """Record of a knockout event."""

    player: int  # Player index who got captured
    piece_id: int
    abs_pos: int


@dataclass
class BlockadeEvent:
    """Record of a blockade formation event."""

    player: int  # Player index who formed the blockade
    rel: int  # Relative position of blockade


@dataclass
class MoveEvents:
    """Events that occurred during a move."""

    exited_home: bool = False
    finished: bool = False
    knockouts: List[KnockoutEvent] = field(default_factory=list)
    hit_blockade: bool = False
    blockades: List[BlockadeEvent] = field(default_factory=list)
    move_resolved: bool = True


@dataclass
class MoveRecord:
    """Record of a single move in the game."""

    step: int
    player_index: int
    dice_roll: int
    piece_id: int
    old_position: int
    new_position: int
    events: MoveEvents
    extra_turn: bool
    board_state: Optional[BoardStateSummary] = None
    
    # Analysis fields (computed later)
    was_capture_opportunity: bool = False
    was_risky_move: bool = False
    left_safe_square: bool = False
    moved_to_safe_square: bool = False
    was_blockade_opportunity: bool = False
    advanced_lead_piece: bool = False


@dataclass
class GameTrace:
    """Complete trace of a game."""

    game_id: str
    num_players: int
    players: List[PlayerInfo]
    moves: List[MoveRecord]
    winner: Optional[int] = None
    total_turns: int = 0


@dataclass
class BehaviorCharacteristics:
    """Quantitative characteristics of behavior in a segment."""

    aggression: float  # 0-1: capture attempts / opportunities
    risk_taking: float  # 0-1: risky moves / risky situations
    exploration: float  # 0-1: unique pieces moved / total pieces
    finishing: float  # 0-1: focus on advancing lead pieces
    blockade_usage: float  # 0-1: blockades formed / opportunities
    defensiveness: float  # 0-1: moves to safety / exposed situations


@dataclass
class ProfileSegment:
    """Profile for a segment of the game."""

    step_range: tuple[int, int]
    style: str  # e.g., "defensive", "aggressive", "finisher", "explorer"
    confidence: float
    characteristics: BehaviorCharacteristics
    behaviors: List[str]  # Human-readable behavior descriptions
    
    # Statistics
    moves_count: int = 0
    captures_made: int = 0
    got_captured: int = 0
    blockades_formed: int = 0
    pieces_finished: int = 0


@dataclass
class StyleTransition:
    """Record of a style transition during the game."""

    step: int
    from_style: str
    to_style: str
    trigger: str


@dataclass
class OverallSummary:
    """Overall summary of player behavior across the game."""

    dominant_style: str
    style_distribution: Dict[str, float]
    key_transitions: List[StyleTransition]
    total_captures: int = 0
    total_got_captured: int = 0
    total_blockades: int = 0
    total_finished: int = 0
    win_achieved: bool = False


@dataclass
class GameProfile:
    """Complete behavioral profile for a player in a game."""

    game_id: str
    player_index: int
    player_color: str
    player_strategy: str
    profile_segments: List[ProfileSegment]
    overall_summary: OverallSummary
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "player_index": self.player_index,
            "player_color": self.player_color,
            "player_strategy": self.player_strategy,
            "profile_segments": [
                {
                    "step_range": list(seg.step_range),
                    "style": seg.style,
                    "confidence": seg.confidence,
                    "characteristics": {
                        "aggression": seg.characteristics.aggression,
                        "risk_taking": seg.characteristics.risk_taking,
                        "exploration": seg.characteristics.exploration,
                        "finishing": seg.characteristics.finishing,
                        "blockade_usage": seg.characteristics.blockade_usage,
                        "defensiveness": seg.characteristics.defensiveness,
                    },
                    "behaviors": seg.behaviors,
                    "moves_count": seg.moves_count,
                    "captures_made": seg.captures_made,
                    "got_captured": seg.got_captured,
                    "blockades_formed": seg.blockades_formed,
                    "pieces_finished": seg.pieces_finished,
                }
                for seg in self.profile_segments
            ],
            "overall_summary": {
                "dominant_style": self.overall_summary.dominant_style,
                "style_distribution": self.overall_summary.style_distribution,
                "key_transitions": [
                    {
                        "step": t.step,
                        "from": t.from_style,
                        "to": t.to_style,
                        "trigger": t.trigger,
                    }
                    for t in self.overall_summary.key_transitions
                ],
                "total_captures": self.overall_summary.total_captures,
                "total_got_captured": self.overall_summary.total_got_captured,
                "total_blockades": self.overall_summary.total_blockades,
                "total_finished": self.overall_summary.total_finished,
                "win_achieved": self.overall_summary.win_achieved,
            },
        }
