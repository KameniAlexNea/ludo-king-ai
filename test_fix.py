#!/usr/bin/env python3
"""Test the fixed strategy framework"""

from tests.test_models import TestDataFactory
from ludo.strategy import StrategyFactory

print("Testing strategy framework with corrected context...")

# Get test context
context_data = TestDataFactory.create_game_start_scenario()

# Convert to proper game format with correct field names
game_context = {
    'current_situation': {
        'player_color': context_data.game_state.current_player,
        'dice_value': context_data.game_state.dice_value,
        'turn_count': context_data.game_state.turn_count,
        'consecutive_sixes': context_data.game_state.consecutive_sixes
    },
    'player_state': {
        'tokens_home': len([t for t in context_data.game_state.players[0].tokens if t.position == -1]),
        'tokens_active': len([t for t in context_data.game_state.players[0].tokens if 0 <= t.position <= 55]),
        'tokens_finished': len([t for t in context_data.game_state.players[0].tokens if t.position >= 56])
    },
    'opponents': [],
    'valid_moves': [
        {
            'token_id': move.token_id,
            'current_position': move.from_position,
            'current_state': 'home' if move.from_position == -1 else 'active',
            'target_position': move.to_position,
            'move_type': move.move_type,
            'is_safe_move': move.reaches_safe_spot,
            'captures_opponent': move.captures_opponent,
            'strategic_value': 10.0 if move.captures_opponent else 5.0,
            'captured_tokens': []
        }
        for move in context_data.valid_moves
    ],
    'strategic_analysis': context_data.strategic_analysis
}

print("Game context valid moves:")
for move in game_context['valid_moves']:
    print(f"  Token {move['token_id']}: {move['move_type']}")

# Test all strategies
strategies = ["killer", "winner", "optimist", "defensive", "balanced", "cautious", "random"]

print("\nTesting strategies:")
for strategy_name in strategies:
    try:
        strategy = StrategyFactory.create_strategy(strategy_name)
        decision = strategy.decide(game_context)
        print(f"  {strategy_name.upper()}: Token {decision} ✅")
    except Exception as e:
        print(f"  {strategy_name.upper()}: ERROR - {e} ❌")

print("\nTest framework should now work!")
