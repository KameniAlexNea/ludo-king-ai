#!/usr/bin/env python3
"""
Example script demonstrating how to train an RL agent on Ludo game data.

This script shows the training pipeline:
1. Load saved game data from the GameStateSaver
2. Train a DQN agent on the data
3. Evaluate the trained agent
4. Save the model for later use
"""

import argparse
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo_rl import LudoRLTrainer


def main():
    """Main training example."""
    parser = argparse.ArgumentParser(
        description="Train RL agent on Ludo game data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with existing data
  python train_rl_agent.py --epochs 1000 --output my_model.pth
  
  # Experiment with hyperparameters
  python train_rl_agent.py --epochs 2000 --target-update-freq 50
        """,
    )

    parser.add_argument(
        "--save-dir",
        default="saved_states/games",
        help="Directory where game data is stored (default: saved_states)",
    )

    parser.add_argument(
        "--output",
        "--model-path",
        default="models/ludo_dqn_model.pth",
        help="Path to save the trained model (default: models/ludo_dqn_model.pth)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )

    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=100,
        help="Frequency of target network updates (default: 100)",
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=200,
        help="Frequency of model checkpoints during training (default: 200)",
    )

    args = parser.parse_args()

    print("🎲 Ludo RL Training Example")
    print("=" * 50)

    # Create models directory if output path contains a directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Configuration from args
    save_dir = args.save_dir
    model_path = args.output
    epochs = args.epochs

    # Initialize trainer
    print("📊 Initializing RL trainer...")
    try:
        trainer = LudoRLTrainer(state_saver_dir=save_dir)

        if len(trainer.game_data) == 0:
            print("❌ No game data found!")
            print(f"   Make sure you have saved games in '{save_dir}' directory")
            print("   Run tournaments first to generate training data")
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
        strategy = record.get("strategy", "unknown")
        if isinstance(strategy, dict):
            strategy = strategy.get("name", "unknown")

        strategies[strategy] = strategies.get(strategy, 0) + 1

        if record.get("outcome", {}).get("success", False):
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
            target_update_freq=args.target_update_freq,
            save_freq=args.save_freq,
            model_save_path=model_path,
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
        plot_dir = os.path.dirname(model_path)
        plot_path = os.path.join(plot_dir, "training_progress.png")
        trainer.plot_training_progress(plot_path)
        print(f"✅ Training plots saved to {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")

    # Evaluate the model
    print("\n🎯 Evaluating trained model...")
    try:
        eval_stats = trainer.evaluate_model()
        print("✅ Evaluation completed!")
        print(f"   Accuracy: {eval_stats.get('accuracy', 0):.2%}")
        print(
            f"   Avg reward per sequence: {eval_stats.get('avg_reward_per_sequence', 0):.2f}"
        )
        print(f"   Test sequences: {eval_stats.get('num_test_sequences', 0)}")

    except Exception as e:
        print(f"⚠️  Error during evaluation: {e}")

    print(f"\n🎉 Training complete! Model saved to: {model_path}")
    print("\n💡 Next steps:")
    print("   1. Use the trained model in games with:")
    print("      from ludo_rl import LudoRLPlayer")
    print("      # Load and use the model")
    print("   2. Run tournaments separately to gather more training data")
    print("   3. Experiment with hyperparameters:")
    print(f"      python {sys.argv[0]} --epochs 2000 --target-update-freq 50")


if __name__ == "__main__":
    main()
