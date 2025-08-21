#!/usr/bin/env python3
"""
Example script demonstrating how to train an RL agent on Ludo game data.

This script shows the complete pipeline:
1. Optionally generate training data by running tournaments
2. Load saved game data from the GameStateSaver
3. Train a DQN agent on the data
4. Evaluate the trained agent
5. Save the model for later use
"""

import argparse
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo_rl import LudoRLTrainer


def generate_training_data(num_games, save_dir):
    """Generate training data by running tournaments."""
    print(f"🎮 Generating training data with {num_games} games...")

    # Import the tournament system
    from four_player_tournament import FourPlayerTournament

    # Create tournament instance
    tournament = FourPlayerTournament()

    # Set the games per matchup to generate the requested number of games
    original_games = tournament.games_per_matchup
    tournament.games_per_matchup = max(
        1, num_games // len(tournament.strategy_combinations)
    )

    print(f"   Running {tournament.games_per_matchup} games per matchup")
    print(f"   Total combinations: {len(tournament.strategy_combinations)}")

    # Run the tournament to generate data
    try:
        tournament.run_tournament()
        print("✅ Training data generation completed!")

        # Check how many games were actually saved
        if os.path.exists(save_dir):
            saved_files = [f for f in os.listdir(save_dir) if f.endswith(".json")]
            print(f"   Generated {len(saved_files)} game data files")

    except Exception as e:
        print(f"❌ Error generating training data: {e}")
        raise
    finally:
        # Restore original setting
        tournament.games_per_matchup = original_games


def main():
    """Main training example."""
    parser = argparse.ArgumentParser(
        description="Train RL agent on Ludo game data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data only
  python train_rl_agent.py --generate-data 100 --skip-training
  
  # Train with existing data
  python train_rl_agent.py --epochs 1000 --output my_model.pth
  
  # Generate data and train
  python train_rl_agent.py --generate-data 50 --epochs 500
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
        "--generate-data",
        type=int,
        metavar="NUM_GAMES",
        help="Generate training data by running NUM_GAMES tournaments before training",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training phase (useful with --generate-data)",
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

    # Generate training data if requested
    if args.generate_data:
        generate_training_data(args.generate_data, args.save_dir)

        if args.skip_training:
            print("✅ Data generation completed. Skipping training as requested.")
            return

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
            print("   Generate training data with:")
            print(f"   python {sys.argv[0]} --generate-data 100")
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
    print("   2. Run more tournaments to gather more training data:")
    print(f"      python {sys.argv[0]} --generate-data 200")
    print("   3. Experiment with hyperparameters:")
    print(f"      python {sys.argv[0]} --epochs 2000 --target-update-freq 50")


if __name__ == "__main__":
    main()
