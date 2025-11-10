"""Tests for agent profiling system."""

import pytest

from ludo_profile import GameAnalyzer
from ludo_profile.models import BoardStateSummary, GameTrace, MoveRecord, PlayerInfo


@pytest.fixture
def sample_trace():
    """Create a sample game trace for testing."""
    players = [
        PlayerInfo(index=0, color="RED", strategy="test_agent"),
        PlayerInfo(index=1, color="BLUE", strategy="opponent"),
    ]

    moves = [
        # Opening - get pieces out
        MoveRecord(
            step=1,
            player_index=0,
            dice_roll=6,
            piece_id=0,
            old_position=0,
            new_position=1,
            events={
                "exited_home": True,
                "knockouts": [],
                "blockades": [],
                "finished": False,
                "hit_blockade": False,
                "move_resolved": True,
            },
            extra_turn=True,
            board_state=BoardStateSummary(
                pieces_in_yard=3,
                pieces_on_track=1,
                pieces_in_home=0,
                pieces_finished=0,
                avg_position=1,
                max_position=1,
                min_position_on_track=1,
            ),
        ),
        MoveRecord(
            step=2,
            player_index=0,
            dice_roll=6,
            piece_id=1,
            old_position=0,
            new_position=1,
            events={
                "exited_home": True,
                "knockouts": [],
                "blockades": [{"player": 0, "rel": 1}],
                "finished": False,
                "hit_blockade": False,
                "move_resolved": True,
            },
            extra_turn=True,
            board_state=BoardStateSummary(
                pieces_in_yard=2,
                pieces_on_track=2,
                pieces_in_home=0,
                pieces_finished=0,
                avg_position=1,
                max_position=1,
                min_position_on_track=1,
            ),
        ),
        # Aggressive move - capture
        MoveRecord(
            step=10,
            player_index=0,
            dice_roll=4,
            piece_id=0,
            old_position=10,
            new_position=14,
            events={
                "exited_home": False,
                "knockouts": [{"player": 1, "piece_id": 0, "abs_pos": 20}],
                "blockades": [],
                "finished": False,
                "hit_blockade": False,
                "move_resolved": True,
            },
            extra_turn=True,
            board_state=BoardStateSummary(
                pieces_in_yard=1,
                pieces_on_track=3,
                pieces_in_home=0,
                pieces_finished=0,
                avg_position=12,
                max_position=14,
                min_position_on_track=10,
            ),
        ),
        # Finishing move
        MoveRecord(
            step=50,
            player_index=0,
            dice_roll=5,
            piece_id=0,
            old_position=52,
            new_position=57,
            events={
                "exited_home": False,
                "knockouts": [],
                "blockades": [],
                "finished": True,
                "hit_blockade": False,
                "move_resolved": True,
            },
            extra_turn=True,
            board_state=BoardStateSummary(
                pieces_in_yard=0,
                pieces_on_track=1,
                pieces_in_home=2,
                pieces_finished=1,
                avg_position=53,
                max_position=57,
                min_position_on_track=45,
            ),
        ),
    ]

    return GameTrace(
        game_id="test_game",
        num_players=2,
        players=players,
        moves=moves,
        winner=0,
        total_turns=50,
    )


def test_game_analyzer_basic(sample_trace):
    """Test basic profiling functionality."""
    analyzer = GameAnalyzer()
    profile = analyzer.analyze_game(sample_trace, player_index=0)

    assert profile.game_id == "test_game"
    assert profile.player_index == 0
    assert profile.player_color == "RED"
    assert profile.player_strategy == "test_agent"
    assert len(profile.profile_segments) > 0
    assert profile.overall_summary.win_achieved is True


def test_profile_to_dict(sample_trace):
    """Test profile serialization."""
    analyzer = GameAnalyzer()
    profile = analyzer.analyze_game(sample_trace, player_index=0)

    profile_dict = profile.to_dict()

    assert isinstance(profile_dict, dict)
    assert "game_id" in profile_dict
    assert "profile_segments" in profile_dict
    assert "overall_summary" in profile_dict
    assert len(profile_dict["profile_segments"]) > 0


def test_segmentation_strategies(sample_trace):
    """Test different segmentation strategies."""
    analyzer = GameAnalyzer()

    # Fixed window
    profile_fixed = analyzer.analyze_game(
        sample_trace, player_index=0, segmentation="fixed", window_size=2
    )
    assert len(profile_fixed.profile_segments) > 0

    # Phase-based
    profile_phase = analyzer.analyze_game(
        sample_trace, player_index=0, segmentation="phase"
    )
    assert len(profile_phase.profile_segments) > 0


def test_analyze_all_players(sample_trace):
    """Test analyzing all players."""
    analyzer = GameAnalyzer()
    profiles = analyzer.analyze_all_players(sample_trace)

    assert len(profiles) == 2
    assert profiles[0].player_index == 0
    assert profiles[1].player_index == 1


def test_empty_moves():
    """Test handling player with no moves."""
    players = [
        PlayerInfo(index=0, color="RED", strategy="test"),
        PlayerInfo(index=1, color="BLUE", strategy="test2"),
    ]
    trace = GameTrace(
        game_id="empty_test",
        num_players=2,
        players=players,
        moves=[],
        winner=None,
        total_turns=0,
    )

    analyzer = GameAnalyzer()
    profile = analyzer.analyze_game(trace, player_index=0)

    assert profile.player_index == 0
    assert len(profile.profile_segments) == 0
    assert profile.overall_summary.dominant_style == "inactive"
