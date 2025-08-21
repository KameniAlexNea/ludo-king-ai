"""
Simple tests for the RL training functionality.
Tests basic functionality without requiring ML dependencies.
"""

import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo import LudoGame, PlayerColor


def test_basic_functionality():
    """Test basic functionality without ML dependencies."""
    print("Testing basic game functionality...")
    
    # Test game creation and context generation
    game = LudoGame([PlayerColor.RED, PlayerColor.GREEN])
    current_player = game.get_current_player()
    
    # Test context generation
    dice_value = 6
    context = game.get_ai_decision_context(dice_value)
    
    # Verify context structure
    required_keys = ['current_situation', 'player_state', 'opponents', 'valid_moves', 'strategic_analysis']
    
    for key in required_keys:
        assert key in context, f"Missing key: {key}"
    
    # Verify current situation structure
    situation = context['current_situation']
    situation_keys = ['player_color', 'dice_value', 'consecutive_sixes', 'turn_count']
    
    for key in situation_keys:
        assert key in situation, f"Missing situation key: {key}"
    
    print("✅ Basic functionality test passed!")
    
    return context


def test_game_data_format():
    """Test the format that would be saved by GameStateSaver."""
    print("Testing game data format...")
    
    # Create sample game data in the format expected by RL training
    context = test_basic_functionality()
    
    game_data = {
        "timestamp": "2025-01-21T10:00:00.000000",
        "strategy": "test_strategy",
        "game_context": context,
        "chosen_move": 0,
        "outcome": {
            "success": True,
            "player_color": "red",
            "token_id": 0,
            "dice_value": 6,
            "old_position": -1,
            "new_position": 1,
            "captured_tokens": [],
            "token_finished": False,
            "extra_turn": False
        }
    }
    
    # Verify the structure matches what the RL system expects
    required_keys = ['timestamp', 'strategy', 'game_context', 'chosen_move', 'outcome']
    
    for key in required_keys:
        assert key in game_data, f"Missing key: {key}"
    
    # Test JSON serialization (what GameStateSaver does)
    json_str = json.dumps(game_data, indent=2)
    loaded_data = json.loads(json_str)
    
    assert loaded_data == game_data, "JSON serialization/deserialization failed"
    
    print("✅ Game data format test passed!")
    
    return game_data


def test_state_encoder_structure():
    """Test state encoder structure without ML dependencies."""
    print("Testing state encoder structure...")
    
    # Test the calculation logic without numpy
    board_size = 52
    max_tokens = 4
    num_players = 4
    
    # Calculate expected dimensions (same logic as in LudoStateEncoder)
    token_positions = num_players * max_tokens * (board_size + 2)
    game_context = 4
    player_stats = num_players * 4
    valid_moves_features = max_tokens * 6
    
    expected_dim = token_positions + game_context + player_stats + valid_moves_features
    
    print(f"Expected state dimension: {expected_dim}")
    print(f"  Token positions: {token_positions}")
    print(f"  Game context: {game_context}")
    print(f"  Player stats: {player_stats}")
    print(f"  Valid moves features: {valid_moves_features}")
    
    # Verify it's reasonable
    assert expected_dim > 0, "State dimension must be positive"
    assert expected_dim < 10000, "State dimension seems too large"
    
    print("✅ State encoder structure test passed!")
    
    return expected_dim


def test_integration_points():
    """Test integration points with existing code."""
    print("Testing integration points...")
    
    # Test that we can import the existing modules we depend on
    from ludo_stats.game_state_saver import GameStateSaver
    
    # Test GameStateSaver functionality
    saver = GameStateSaver("test_states")
    
    # Create a dummy context and save it
    game = LudoGame([PlayerColor.RED, PlayerColor.GREEN])
    context = game.get_ai_decision_context(6)
    
    saver.save_decision(
        strategy_name="test",
        game_context=context,
        chosen_move=0,
        outcome={"success": True}
    )
    
    # Check that we have the decision
    assert len(saver.current_game_states) == 1, "Decision not saved"
    
    saved_state = saver.current_game_states[0]
    assert 'game_context' in saved_state, "Game context not saved"
    assert 'chosen_move' in saved_state, "Chosen move not saved"
    
    print("✅ Integration points test passed!")


def main():
    """Run all tests."""
    print("🧪 Running RL Training Tests")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_game_data_format()
        test_state_encoder_structure()
        test_integration_points()
        
        print("\n🎉 All tests passed!")
        print("\n💡 To test the full RL functionality:")
        print("   1. Install dependencies: pip install torch numpy matplotlib")
        print("   2. Generate training data: run a tournament")
        print("   3. Run: python examples/train_rl_agent.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()