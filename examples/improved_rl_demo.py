#!/usr/bin/env python3
"""
Example script demonstrating the improved Ludo RL system.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ludo_rl.config import REWARDS, TRAINING_CONFIG
from ludo_rl.rl_player import ImprovedRLPlayer
from ludo_rl.trainer import ImprovedLudoRLTrainer
from ludo_rl.validator import LudoRLValidator


def main():
    """Demonstrate the improved RL system."""
    print("🎯 Improved Ludo RL System Demo")
    print("=" * 50)

    # Check for saved game data
    saved_states_dir = project_root / "saved_states"
    if not saved_states_dir.exists():
        print(
            "❌ No saved_states directory found. Please run tournaments first to generate training data."
        )
        print("   Example: python four_player_tournament.py")
        return

    # Find JSON files
    json_files = list(saved_states_dir.glob("**/*.json"))
    if not json_files:
        print("❌ No JSON game data found in saved_states directory.")
        return

    print(f"✅ Found {len(json_files)} game data files")

    # 1. Create improved trainer
    print("\n🧠 Initializing Improved RL Trainer...")
    trainer = ImprovedLudoRLTrainer(
        state_saver_dir=str(saved_states_dir),
        use_prioritized_replay=True,
        use_double_dqn=True,
    )

    print(f"   State dimension: {trainer.encoder.state_dim} features")
    print(f"   Training examples: {len(trainer.game_data)}")

    # Display configuration
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

    # 2. Quick training demo (short epochs for demo)
    print("\n🏋️  Starting Training Demo (reduced epochs for demo)...")

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "improved_ludo_model_demo.pth"

    # Train with fewer epochs for demo
    training_results = trainer.train(
        epochs=50,  # Reduced for demo
        validation_split=0.1,
        early_stopping_patience=10,
        save_freq=20,
        model_save_path=str(model_path),
    )

    print("\n📊 Training Results:")
    for key, value in training_results.items():
        print(f"   {key}: {value}")

    # 3. Create training progress plot
    print("\n📈 Generating Training Progress Plot...")
    progress_plot_path = models_dir / "training_progress_demo.png"
    trainer.plot_training_progress(str(progress_plot_path))

    # 4. Test the trained model
    if model_path.exists():
        print("\n🤖 Testing Trained Model...")
        player = ImprovedRLPlayer(str(model_path), name="ImprovedRLAgent")

        # Create a sample game state for testing
        sample_game_state = create_sample_game_state()

        # Get move analysis
        analysis = player.choose_move_with_analysis(sample_game_state)

        print("   Sample decision analysis:")
        print(f"   Chosen move: {analysis['move_index']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Reasoning: {analysis.get('analysis', {}).get('reasoning', 'N/A')}")

        # 5. Model validation demo
        print("\n🔍 Model Validation Demo...")
        validator = LudoRLValidator(str(model_path))

        # Create some sample validation data
        validation_data = [sample_game_state] * 10  # Simple demo data

        try:
            analysis_results = validator.analyze_decision_patterns(validation_data)

            print("   Decision pattern analysis:")
            move_prefs = analysis_results.get("move_type_preferences", {})
            for move_type, count in move_prefs.items():
                print(f"     {move_type}: {count}")

            safety_prefs = analysis_results.get("safety_preferences", {})
            total_moves = sum(safety_prefs.values())
            if total_moves > 0:
                safe_rate = safety_prefs.get("safe", 0) / total_moves
                print(f"   Safety rate: {safe_rate:.2f}")

        except Exception as e:
            print(f"   Validation demo skipped due to: {e}")

    print("\n✨ Demo completed successfully!")
    print("\nKey improvements demonstrated:")
    print("  ✓ Compact 64-feature state representation")
    print("  ✓ Dueling DQN architecture with Double DQN")
    print("  ✓ Prioritized experience replay")
    print("  ✓ Enhanced reward engineering")
    print("  ✓ Validation and early stopping")
    print("  ✓ Comprehensive move analysis")
    print("  ✓ Model interpretation tools")

    print("\n📁 Generated files:")
    if model_path.exists():
        print(f"  Model: {model_path}")
    if progress_plot_path.exists():
        print(f"  Training plot: {progress_plot_path}")

    print("\n🚀 Ready for production use!")


def create_sample_game_state():
    """Create a sample game state for testing."""
    return {
        "game_context": {
            "player_state": {
                "tokens": [
                    {
                        "position": -1,
                        "is_active": False,
                        "is_finished": False,
                        "is_in_home_column": False,
                    },
                    {
                        "position": 5,
                        "is_active": True,
                        "is_finished": False,
                        "is_in_home_column": False,
                    },
                    {
                        "position": 15,
                        "is_active": True,
                        "is_finished": False,
                        "is_in_home_column": False,
                    },
                    {
                        "position": -1,
                        "is_active": False,
                        "is_finished": False,
                        "is_in_home_column": False,
                    },
                ],
                "tokens_in_home": 2,
                "active_tokens": 2,
                "finished_tokens": 0,
                "has_won": False,
            },
            "current_situation": {
                "dice_value": 4,
                "consecutive_sixes": 0,
                "turn_count": 15,
                "player_color": "red",
            },
            "opponents": [
                {"tokens_active": 2, "tokens_finished": 1, "threat_level": 0.3},
                {"tokens_active": 3, "tokens_finished": 0, "threat_level": 0.5},
                {"tokens_active": 1, "tokens_finished": 2, "threat_level": 0.7},
            ],
            "valid_moves": [
                {
                    "token_id": 1,
                    "move_type": "advance_main_board",
                    "strategic_value": 8,
                    "is_safe_move": True,
                    "captures_opponent": False,
                },
                {
                    "token_id": 2,
                    "move_type": "advance_main_board",
                    "strategic_value": 12,
                    "is_safe_move": False,
                    "captures_opponent": True,
                },
            ],
            "strategic_analysis": {
                "can_capture": True,
                "can_finish_token": False,
                "can_exit_home": False,
                "safe_moves": [0],
                "risky_moves": [1],
                "best_strategic_move": {"token_id": 2},
                "best_strategic_value": 12,
            },
        },
        "chosen_move": 1,
        "outcome": {
            "success": True,
            "old_position": 15,
            "new_position": 19,
            "captured_tokens": ["blue_token_1"],
            "token_finished": False,
            "extra_turn": False,
        },
        "timestamp": "2024-08-22T10:30:00Z",
        "strategy": "improved_rl",
    }


if __name__ == "__main__":
    main()
