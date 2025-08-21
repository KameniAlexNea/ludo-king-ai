#!/usr/bin/env python3
"""
Example script demonstrating how to train an RL agent on Ludo game data.

This script shows the complete pipeline:
1. Load saved game data from the GameStateSaver
2. Train a DQN agent on the data
3. Evaluate the trained agent
4. Save the model for later use
"""

import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo_rl import LudoRLTrainer, LudoStateEncoder, LudoDQNAgent


def main():
    """Main training example."""
    print("🎲 Ludo RL Training Example")
    print("=" * 50)
    
    # Configuration
    save_dir = "saved_states"  # Directory where GameStateSaver stores game data
    model_path = "models/ludo_dqn_model.pth"
    epochs = 1000
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Initialize trainer
    print("📊 Initializing RL trainer...")
    try:
        trainer = LudoRLTrainer(state_saver_dir=save_dir)
        
        if len(trainer.game_data) == 0:
            print("❌ No game data found!")
            print(f"   Make sure you have saved games in '{save_dir}' directory")
            print("   Run a tournament first to generate training data:")
            print("   python four_player_tournament.py")
            return
        
        print(f"✅ Loaded {len(trainer.game_data)} game records")
        
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        return
    
    # Display data statistics
    print("\n📈 Training Data Statistics:")
    strategies = {}
    outcomes = {"success": 0, "failed": 0}
    
    for record in trainer.game_data:
        strategy = record.get('strategy', 'unknown')
        if isinstance(strategy, dict):
            strategy = strategy.get('name', 'unknown')
        
        strategies[strategy] = strategies.get(strategy, 0) + 1
        
        if record.get('outcome', {}).get('success', False):
            outcomes["success"] += 1
        else:
            outcomes["failed"] += 1
    
    print(f"   Strategies represented: {list(strategies.keys())}")
    for strategy, count in strategies.items():
        print(f"   - {strategy}: {count} decisions")
    print(f"   Successful moves: {outcomes['success']}")
    print(f"   Failed moves: {outcomes['failed']}")
    
    # Train the agent
    print(f"\n🧠 Training DQN agent for {epochs} epochs...")
    try:
        training_stats = trainer.train(
            epochs=epochs,
            target_update_freq=100,
            save_freq=200,
            model_save_path=model_path
        )
        
        print("✅ Training completed!")
        print(f"   Final loss: {training_stats.get('final_loss', 'N/A'):.4f}")
        print(f"   Final reward: {training_stats.get('final_reward', 'N/A'):.2f}")
        print(f"   Total experiences: {training_stats.get('total_experiences', 0)}")
        print(f"   Training sequences: {training_stats.get('num_sequences', 0)}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Plot training progress
    print("\n📊 Generating training plots...")
    try:
        trainer.plot_training_progress("models/training_progress.png")
        print("✅ Training plots saved to models/training_progress.png")
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    # Evaluate the model
    print("\n🎯 Evaluating trained model...")
    try:
        eval_stats = trainer.evaluate_model()
        print("✅ Evaluation completed!")
        print(f"   Accuracy: {eval_stats.get('accuracy', 0):.2%}")
        print(f"   Avg reward per sequence: {eval_stats.get('avg_reward_per_sequence', 0):.2f}")
        print(f"   Test sequences: {eval_stats.get('num_test_sequences', 0)}")
        
    except Exception as e:
        print(f"⚠️  Error during evaluation: {e}")
    
    print(f"\n🎉 Training complete! Model saved to: {model_path}")
    print("\n💡 Next steps:")
    print("   1. Use the trained model in games with:")
    print("      from ludo_rl import create_rl_strategy")
    print(f"      rl_strategy = create_rl_strategy('{model_path}')")
    print("   2. Run more tournaments to gather more training data")
    print("   3. Experiment with hyperparameters and network architecture")


if __name__ == "__main__":
    main()