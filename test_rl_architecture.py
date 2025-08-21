#!/usr/bin/env python3
"""
Comprehensive test demonstrating the RL training pipeline architecture.
This shows how all components work together without requiring ML dependencies.
"""

import json
import os
import sys
import tempfile
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_stats.game_state_saver import GameStateSaver


def create_sample_training_data(num_games: int = 3) -> str:
    """Create sample training data that mimics real tournament output."""
    print(f"🎲 Creating {num_games} sample games for RL training demo...")
    
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="ludo_rl_test_")
    saver = GameStateSaver(temp_dir)
    
    # Get available strategies (fallback to basic ones if none available)
    try:
        available_strategies = StrategyFactory.get_available_strategies()
        strategies = available_strategies[:2] if len(available_strategies) >= 2 else ["random", "killer"]
    except:
        strategies = ["random", "killer"]  # fallback
    
    total_decisions = 0
    
    for game_num in range(num_games):
        print(f"  Game {game_num + 1}/{num_games}")
        
        # Create game
        game = LudoGame([PlayerColor.RED, PlayerColor.GREEN])
        
        # Simulate some turns to generate decisions
        turn_count = 0
        max_turns = 20  # Keep test short
        
        while not game.game_over and turn_count < max_turns:
            current_player = game.get_current_player()
            dice_value = game.roll_dice()
            
            # Get game context (this is what the RL system trains on)
            game_context = game.get_ai_decision_context(dice_value)
            valid_moves = game_context['valid_moves']
            
            if valid_moves:
                # Choose a move (simulate strategy decision)
                chosen_move_idx = 0  # Always choose first valid move for test
                chosen_token = valid_moves[chosen_move_idx]['token_id']
                
                # Execute move
                turn_result = game.play_turn(chosen_token)
                
                # Save decision (this is what RL trains on)
                outcome = turn_result.get('move_result', {})
                strategy_name = strategies[current_player.player_id % len(strategies)]
                
                saver.save_decision(
                    strategy_name=strategy_name,
                    game_context=game_context,
                    chosen_move=chosen_move_idx,
                    outcome=outcome
                )
                
                total_decisions += 1
            else:
                # No valid moves, skip turn
                game.next_turn()
            
            turn_count += 1
    
    # Save the game data
    saver.save_game(f"rl_training_demo_{num_games}_games")
    
    print(f"✅ Generated {total_decisions} training decisions in {temp_dir}")
    return temp_dir


def analyze_training_data(data_dir: str):
    """Analyze the structure of training data."""
    print(f"\n📊 Analyzing training data in {data_dir}...")
    
    # Load all JSON files
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
    
    if not all_data:
        print("❌ No training data found!")
        return
    
    print(f"📈 Training Data Analysis:")
    print(f"   Total decisions: {len(all_data)}")
    
    # Analyze strategies
    strategies = {}
    for record in all_data:
        strategy = record.get('strategy', 'unknown')
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    print(f"   Strategies: {list(strategies.keys())}")
    for strategy, count in strategies.items():
        print(f"     {strategy}: {count} decisions")
    
    # Analyze outcomes
    successful_moves = sum(1 for r in all_data if r.get('outcome', {}).get('success', False))
    print(f"   Successful moves: {successful_moves}/{len(all_data)} ({successful_moves/len(all_data)*100:.1f}%)")
    
    # Analyze game context structure
    sample_record = all_data[0]
    game_context = sample_record['game_context']
    
    print(f"\n📋 Game Context Structure:")
    print(f"   Keys: {list(game_context.keys())}")
    print(f"   Valid moves in sample: {len(game_context.get('valid_moves', []))}")
    
    if game_context.get('valid_moves'):
        sample_move = game_context['valid_moves'][0]
        print(f"   Move features: {list(sample_move.keys())}")
    
    # Verify data is RL-ready
    required_keys = ['timestamp', 'strategy', 'game_context', 'chosen_move', 'outcome']
    missing_keys = []
    
    for record in all_data[:5]:  # Check first 5 records
        for key in required_keys:
            if key not in record:
                missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️  Missing keys found: {set(missing_keys)}")
    else:
        print("✅ All records have required RL training keys")
    
    return all_data


def simulate_rl_training_pipeline(training_data):
    """Simulate the RL training pipeline logic without ML dependencies."""
    print(f"\n🧠 Simulating RL Training Pipeline...")
    
    # Simulate state encoding
    print("1. State Encoding Simulation:")
    sample_record = training_data[0]
    game_context = sample_record['game_context']
    
    # Calculate state dimensions (same logic as LudoStateEncoder)
    board_size = 52
    max_tokens = 4
    num_players = 4
    
    token_positions = num_players * max_tokens * (board_size + 2)
    game_context_dims = 4
    player_stats = num_players * 4
    valid_moves_features = max_tokens * 6
    
    total_state_dim = token_positions + game_context_dims + player_stats + valid_moves_features
    
    print(f"   State vector dimension: {total_state_dim}")
    print(f"   Token positions: {token_positions} dims")
    print(f"   Game context: {game_context_dims} dims")
    print(f"   Player stats: {player_stats} dims")
    print(f"   Valid moves: {valid_moves_features} dims")
    
    # Simulate reward calculation
    print("\n2. Reward Calculation Simulation:")
    rewards = []
    for record in training_data[:5]:  # Sample first 5
        outcome = record.get('outcome', {})
        context = record.get('game_context', {})
        
        # Simulate reward calculation (same logic as LudoRLTrainer)
        reward = 0.0
        
        if outcome.get('success', False):
            reward += 1.0
        
        if outcome.get('captured_tokens', []):
            reward += 10.0 * len(outcome['captured_tokens'])
        
        if outcome.get('token_finished', False):
            reward += 25.0
        
        if outcome.get('extra_turn', False):
            reward += 3.0
        
        # Strategic value
        valid_moves = context.get('valid_moves', [])
        chosen_move = record.get('chosen_move', 0)
        if chosen_move < len(valid_moves):
            strategic_value = valid_moves[chosen_move].get('strategic_value', 0)
            reward += strategic_value * 0.1
        
        rewards.append(reward)
    
    print(f"   Sample rewards: {[f'{r:.1f}' for r in rewards]}")
    print(f"   Average reward: {sum(rewards)/len(rewards):.2f}")
    
    # Simulate training sequence creation
    print("\n3. Training Sequence Simulation:")
    sequences = []
    current_sequence = []
    
    for i, record in enumerate(training_data):
        # Simulate (state, action, reward, next_state, done) tuple
        state_placeholder = [0.0] * total_state_dim  # Would be encoded state
        action = record.get('chosen_move', 0)
        reward = rewards[i] if i < len(rewards) else 1.0
        
        # Determine if episode is done (simplified)
        done = (i == len(training_data) - 1) or (i > 0 and 
                training_data[i-1].get('strategy') != record.get('strategy'))
        
        next_state_placeholder = [0.0] * total_state_dim
        
        current_sequence.append((state_placeholder, action, reward, next_state_placeholder, done))
        
        if done:
            sequences.append(current_sequence)
            current_sequence = []
    
    if current_sequence:
        sequences.append(current_sequence)
    
    print(f"   Training sequences: {len(sequences)}")
    print(f"   Total experiences: {sum(len(seq) for seq in sequences)}")
    print(f"   Avg sequence length: {sum(len(seq) for seq in sequences) / len(sequences):.1f}")
    
    print("\n✅ RL Pipeline simulation completed successfully!")
    
    return {
        'state_dim': total_state_dim,
        'num_sequences': len(sequences),
        'total_experiences': sum(len(seq) for seq in sequences),
        'avg_reward': sum(rewards) / len(rewards) if rewards else 0
    }


def cleanup_test_data(data_dir: str):
    """Clean up test data."""
    print(f"\n🧹 Cleaning up test data from {data_dir}...")
    import shutil
    shutil.rmtree(data_dir)
    print("✅ Cleanup completed")


def main():
    """Run comprehensive RL architecture demonstration."""
    print("🎯 Ludo RL Training Architecture Demo")
    print("=" * 50)
    
    try:
        # Step 1: Create sample training data
        data_dir = create_sample_training_data(num_games=3)
        
        # Step 2: Analyze the data structure
        training_data = analyze_training_data(data_dir)
        
        if not training_data:
            print("❌ No training data available for pipeline simulation")
            return
        
        # Step 3: Simulate the RL training pipeline
        pipeline_stats = simulate_rl_training_pipeline(training_data)
        
        # Step 4: Summary
        print(f"\n🎉 Architecture Demo Complete!")
        print(f"📊 Summary:")
        print(f"   Training decisions processed: {len(training_data)}")
        print(f"   State vector dimension: {pipeline_stats['state_dim']}")
        print(f"   Training sequences: {pipeline_stats['num_sequences']}")
        print(f"   Total experiences: {pipeline_stats['total_experiences']}")
        print(f"   Average reward: {pipeline_stats['avg_reward']:.2f}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Install ML dependencies: pip install torch numpy matplotlib")
        print(f"   2. Generate real training data: python four_player_tournament.py")
        print(f"   3. Train RL agent: python examples/train_rl_agent.py")
        print(f"   4. Use trained agent: python examples/play_with_rl_agent.py")
        
        # Cleanup
        cleanup_test_data(data_dir)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()