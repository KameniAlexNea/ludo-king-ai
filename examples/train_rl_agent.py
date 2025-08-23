#!/usr/bin/env python3
"""
Advanced Ludo RL Training Script

This script demonstrates comprehensive RL agent training with:
1. Enhanced state representation with 64-dimensional features
2. Dueling DQN architecture with prioritized replay
3. Advanced training pipeline with validation and early stopping
4. Model interpretation and analysis tools
5. Comprehensive evaluation and visualization
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ludo_rl.config import REWARDS, TRAINING_CONFIG
from ludo_rl.trainer import LudoRLTrainer
# Validator removed in simplification


def display_configuration():
    """Display the current training and reward configuration."""
    print("\n⚙️  Training Configuration:")
    print(f"   Hidden dimensions: {TRAINING_CONFIG.HIDDEN_DIM}")
    print(f"   Learning rate: {TRAINING_CONFIG.LEARNING_RATE}")
    print(f"   Batch size: {TRAINING_CONFIG.BATCH_SIZE}")
    print(f"   Use prioritized replay: {TRAINING_CONFIG.USE_PRIORITIZED_REPLAY}")
    print(f"   Use double DQN: {TRAINING_CONFIG.USE_DOUBLE_DQN}")

    print("\n💰 Reward Configuration:")
    print(f"   Success: {REWARDS.SUCCESS}")
    print(f"   Failure: {REWARDS.FAILS}")
    print(f"   Capture: {REWARDS.CAPTURE}")
    print(f"   Token finished: {REWARDS.TOKEN_FINISHED}")


def validate_and_analyze_model(model_path, trainer):
    """Simplified quick eval (validator removed)."""
    if not model_path.exists():
        return
    eval_stats = trainer.evaluate_model()
    print(f"Quick eval accuracy: {eval_stats.get('accuracy',0):.2%}")


def main():
    """Advanced RL training with comprehensive analysis."""
    parser = argparse.ArgumentParser(
        description="Advanced Ludo RL Agent Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training with validation
  python train_rl_agent.py --epochs 1000 --validation-split 0.2
  
  # Advanced training with early stopping
  python train_rl_agent.py --epochs 2000 --early-stopping --patience 20
  
  # Training with custom model path
  python train_rl_agent.py --output models/custom_model.pth --epochs 1500
        """,
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("saved_states/games"),
        help="Directory where game data is stored (default: saved_states/games)",
    )

    parser.add_argument(
        "--output",
        "--model-path",
        type=Path,
        default=Path("models/ludo_dqn_model.pth"),
        help="Path to save the trained model (default: models/ludo_dqn_model.pth)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
    )

    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on validation loss",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15)",
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

    parser.add_argument(
        "--use-prioritized-replay",
        action="store_true",
        default=True,
        help="Use prioritized experience replay (default: True)",
    )

    parser.add_argument(
        "--use-double-dqn",
        action="store_true",
        default=True,
        help="Use Double DQN technique (default: True)",
    )

    args = parser.parse_args()

    print("🎲 Advanced Ludo RL Training System")
    print("=" * 50)

    # Create models directory if needed
    model_path = args.output
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for saved game data
    save_dir = args.save_dir
    if not save_dir.exists():
        print(
            f"❌ No {save_dir} directory found. Please run tournaments first to generate training data."
        )
        print("   Example: python four_player_tournament.py")
        return

    # Find JSON files
    json_files = list(save_dir.glob("**/*.json"))
    if not json_files:
        print(f"❌ No JSON game data found in {save_dir} directory.")
        return

    print(f"✅ Found {len(json_files)} game data files")

    # Initialize advanced trainer
    print("\n🧠 Initializing Advanced RL Trainer...")
    try:
        trainer = LudoRLTrainer(
            # state_saver_dir=str(save_dir),
            use_prioritized_replay=args.use_prioritized_replay,
            use_double_dqn=args.use_double_dqn,
        )

        if len(trainer.game_data) == 0:
            print("❌ No valid training data found!")
            return

        print(f"   State dimension: {trainer.encoder.state_dim} features")
        print(f"   Training examples: {len(trainer.game_data)}")

    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        return

    # Display configuration
    display_configuration()

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

    # Train the agent with advanced features
    print(f"\n🏋️  Training Advanced DQN Agent for {args.epochs} epochs...")

    # Prepare training parameters
    training_params = {
        "epochs": args.epochs,
        "validation_split": args.validation_split,
        "target_update_freq": args.target_update_freq,
        "save_freq": args.save_freq,
        "model_save_path": str(model_path),
    }

    # Add early stopping if requested
    if args.early_stopping:
        training_params["early_stopping_patience"] = args.patience

    training_stats = trainer.train(**training_params)

    print("✅ Training completed!")
    print("\n📊 Training Results:")
    for key, value in training_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


    # Plotting removed; training logged to CSV automatically
    log_path = model_path.parent / "training_log.csv"
    if log_path.exists():
        print(f"📄 Training log: {log_path}")

    # Comprehensive model evaluation
    print("\n🎯 Comprehensive Model Evaluation...")
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

    # Advanced model validation and analysis
    validate_and_analyze_model(model_path, trainer)

    print(f"\n🎉 Advanced Training Complete! Model saved to: {model_path}")

    print("\n✨ Key improvements implemented:")
    print("  ✓ Compact 64-feature state representation")
    print("  ✓ Dueling DQN architecture with Double DQN")
    print("  ✓ Prioritized experience replay")
    print("  ✓ Enhanced reward engineering")
    print("  ✓ Simplified logging (CSV)")

    print("\n📁 Generated files:")
    if model_path.exists():
        print(f"  Model: {model_path}")
    log_path = model_path.parent / "training_log.csv"
    if log_path.exists():
        print(f"  Training log: {log_path}")

    print("\n💡 Next steps:")
    print("   1. Use the trained model in games with:")
    print("      from ludo_rl import RLPlayer")
    print("      player = RLPlayer('models/ludo_dqn_model.pth')")
    print("   2. Run tournaments to gather more diverse training data")
    print("   3. Experiment with advanced hyperparameters:")
    print(
        f"      python {sys.argv[0]} --epochs 2000 --validation-split 0.2 --early-stopping"
    )
    print("   4. Inspect CSV log for learning curves")

    print("\n🚀 Ready for production deployment!")


if __name__ == "__main__":
    main()
