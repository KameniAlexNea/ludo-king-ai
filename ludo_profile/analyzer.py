"""Main game analyzer for generating behavioral profiles."""

from typing import Dict, List, Optional

from .classifier import BehaviorClassifier
from .metrics import (
    annotate_move_opportunities,
    compute_characteristics,
    generate_behavior_descriptions,
)
from .models import (
    GameProfile,
    GameTrace,
    OverallSummary,
    ProfileSegment,
    StyleTransition,
)
from .segmentation import segment_by_game_phase, segment_by_step_count


class GameAnalyzer:
    """Analyzes game traces to generate behavioral profiles."""
    
    def __init__(self, classifier: Optional[BehaviorClassifier] = None):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        classifier : BehaviorClassifier, optional
            Behavior classifier to use. If None, creates default.
        """
        self.classifier = classifier or BehaviorClassifier()
    
    def analyze_game(
        self,
        trace: GameTrace,
        player_index: int,
        segmentation: str = "fixed",
        window_size: int = 20,
    ) -> GameProfile:
        """
        Generate behavioral profile for a player.
        
        Parameters
        ----------
        trace : GameTrace
            Complete game trace.
        player_index : int
            Index of player to profile.
        segmentation : str
            Segmentation strategy: "fixed", "phase", or "adaptive".
        window_size : int
            Window size for fixed segmentation (default: 20 moves).
            
        Returns
        -------
        GameProfile
            Complete behavioral profile.
        """
        # Extract player's moves
        player_moves = [m for m in trace.moves if m.player_index == player_index]
        
        if not player_moves:
            # No moves by this player
            return self._empty_profile(trace, player_index)
        
        # Annotate moves with opportunity flags
        annotate_move_opportunities(player_moves, trace.moves, player_index)
        
        # Segment the game
        if segmentation == "phase":
            segments = segment_by_game_phase(player_moves, trace.moves)
        elif segmentation == "adaptive":
            from .segmentation import segment_adaptive
            segments = [(s, e, m, "adaptive") for s, e, m in segment_adaptive(player_moves)]
        else:  # "fixed"
            segments = [
                (s, e, m, f"segment_{i}")
                for i, (s, e, m) in enumerate(segment_by_step_count(player_moves, window_size))
            ]
        
        # Analyze each segment
        profile_segments = []
        for start_step, end_step, segment_moves, *extra in segments:
            phase_name = extra[0] if extra else "segment"
            
            # Compute characteristics
            characteristics = compute_characteristics(segment_moves)
            
            # Classify behavior
            style, confidence = self.classifier.classify_with_context(
                characteristics, phase=phase_name if phase_name in ["opening", "midgame", "endgame"] else "midgame"
            )
            
            # Generate descriptions
            behaviors = generate_behavior_descriptions(segment_moves, characteristics)
            
            # Collect statistics
            captures_made = sum(len(m.events.knockouts) for m in segment_moves)
            blockades_formed = sum(len(m.events.blockades) for m in segment_moves)
            pieces_finished = sum(1 for m in segment_moves if m.events.finished)
            
            # Count times this player got captured
            got_captured = sum(
                1 for m in trace.moves
                if m.player_index != player_index
                and any(ko.player == player_index for ko in m.events.knockouts)
                and start_step <= m.step <= end_step
            )
            
            segment_profile = ProfileSegment(
                step_range=(start_step, end_step),
                style=style,
                confidence=confidence,
                characteristics=characteristics,
                behaviors=behaviors,
                moves_count=len(segment_moves),
                captures_made=captures_made,
                got_captured=got_captured,
                blockades_formed=blockades_formed,
                pieces_finished=pieces_finished,
            )
            profile_segments.append(segment_profile)
        
        # Compute overall summary
        overall = self._compute_overall_summary(
            profile_segments, trace, player_index
        )
        
        # Get player info
        player_info = next(p for p in trace.players if p.index == player_index)
        
        return GameProfile(
            game_id=trace.game_id,
            player_index=player_index,
            player_color=player_info.color,
            player_strategy=player_info.strategy,
            profile_segments=profile_segments,
            overall_summary=overall,
        )
    
    def _empty_profile(self, trace: GameTrace, player_index: int) -> GameProfile:
        """Create an empty profile for a player with no moves."""
        player_info = next(p for p in trace.players if p.index == player_index)
        return GameProfile(
            game_id=trace.game_id,
            player_index=player_index,
            player_color=player_info.color,
            player_strategy=player_info.strategy,
            profile_segments=[],
            overall_summary=OverallSummary(
                dominant_style="inactive",
                style_distribution={},
                key_transitions=[],
                win_achieved=False,
            ),
        )
    
    def _compute_overall_summary(
        self,
        segments: List[ProfileSegment],
        trace: GameTrace,
        player_index: int,
    ) -> OverallSummary:
        """Compute overall summary across all segments."""
        if not segments:
            return OverallSummary(
                dominant_style="unknown",
                style_distribution={},
                key_transitions=[],
                win_achieved=False,
            )
        
        # Style distribution
        style_counts: Dict[str, int] = {}
        for seg in segments:
            style_counts[seg.style] = style_counts.get(seg.style, 0) + seg.moves_count
        
        total_moves = sum(style_counts.values())
        style_distribution = {
            style: count / total_moves
            for style, count in style_counts.items()
        }
        
        # Dominant style
        dominant_style = max(style_distribution, key=style_distribution.get)
        
        # Key transitions (where style changes)
        transitions = []
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            if prev.style != curr.style:
                # Detect trigger
                trigger = "unknown"
                if "finished" in " ".join(curr.behaviors).lower():
                    trigger = "piece_finished"
                elif curr.characteristics.aggression > prev.characteristics.aggression + 0.3:
                    trigger = "became_aggressive"
                elif curr.characteristics.defensiveness > prev.characteristics.defensiveness + 0.3:
                    trigger = "became_defensive"
                elif "out of home" in " ".join(curr.behaviors).lower():
                    trigger = "all_pieces_active"
                
                transitions.append(StyleTransition(
                    step=curr.step_range[0],
                    from_style=prev.style,
                    to_style=curr.style,
                    trigger=trigger,
                ))
        
        # Total statistics
        total_captures = sum(seg.captures_made for seg in segments)
        total_got_captured = sum(seg.got_captured for seg in segments)
        total_blockades = sum(seg.blockades_formed for seg in segments)
        total_finished = sum(seg.pieces_finished for seg in segments)
        win_achieved = trace.winner == player_index
        
        return OverallSummary(
            dominant_style=dominant_style,
            style_distribution=style_distribution,
            key_transitions=transitions,
            total_captures=total_captures,
            total_got_captured=total_got_captured,
            total_blockades=total_blockades,
            total_finished=total_finished,
            win_achieved=win_achieved,
        )
    
    def analyze_all_players(
        self,
        trace: GameTrace,
        **kwargs,
    ) -> List[GameProfile]:
        """
        Generate profiles for all players in the game.
        
        Parameters
        ----------
        trace : GameTrace
            Complete game trace.
        **kwargs
            Additional arguments passed to analyze_game().
            
        Returns
        -------
        List[GameProfile]
            Profiles for all players.
        """
        return [
            self.analyze_game(trace, player.index, **kwargs)
            for player in trace.players
        ]
