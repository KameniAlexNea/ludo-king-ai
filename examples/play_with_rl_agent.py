#!/usr/bin/env python3
"""
Advanced RL Agent Gameplay and Evaluation Script

This script demonstrates comprehensive RL agent evaluation with:
1. Loading trained RL models with advanced architectures
2. Creating enhanced RL strategies with confidence scoring
3. Running detailed games with move analysis
4. Comprehensive performance evaluation and statistics
5. Model comparison and validation tools
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_rl.rl_player import RLPlayer
from ludo_stats.game_state_saver import GameStateSaver


class AdvancedRLEvaluation:
    """Advanced system for evaluating trained RL agents with detailed analysis."""

    def __init__(self, model_path: str, num_games: int = 10, max_turns: int = 1000):
        """
        Initialize the advanced RL evaluation environment.

        Args:
            model_path: Path to the trained RL model.
            num_games: Number of games to play for evaluation.
            max_turns: Maximum number of turns per game.
        """
        self.model_path = Path(model_path)
        self.num_games = num_games
        self.max_turns = max_turns

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load RL player with enhanced capabilities
        print(f"🤖 Loading Advanced RL Agent from {self.model_path}")
        self.rl_player = RLPlayer(str(self.model_path), name="AdvancedRL-DQN")

        # Validator removed

        # Get opponent strategies
        available_strategies = StrategyFactory.get_available_strategies()
        print(f"📋 Available strategies: {available_strategies}")

        # Select diverse opponent strategies for comprehensive evaluation
        preferred_opponents = [
            "optimist",
            "winner",
            "balanced",
            "cautious",
            "defensive",
        ]
        self.opponent_strategies = [
            s for s in preferred_opponents if s in available_strategies
        ][:3]

        if len(self.opponent_strategies) < 3:
            print(f"⚠️  Using available strategies: {available_strategies}")
            self.opponent_strategies = available_strategies[:3]

        # Enhanced state saving with detailed analysis
        save_dir = Path("saved_states/rl_gameplay_states")
        save_dir.mkdir(parents=True, exist_ok=True)
        self.state_saver = GameStateSaver(str(save_dir))

        # Comprehensive result tracking
        self.results = {
            "rl_wins": 0,
            "total_games": 0,
            "opponent_wins": defaultdict(int),
            "average_turns": 0,
            "rl_move_analysis": [],
            "confidence_scores": [],
            "strategic_decisions": defaultdict(int),
        }

    def run_evaluation(self):
        """Run comprehensive evaluation with detailed analysis."""
        print("\n🎮 Advanced RL Agent Evaluation System 🎮")
        print("=" * 60)
        print(
            f"🎯 Playing {self.num_games} games: AdvancedRL vs {self.opponent_strategies}"
        )

        # Display model configuration
        print("\n⚙️  Model Configuration:")
        print(f"   Model path: {self.model_path}")
        print(f"   Max turns per game: {self.max_turns}")

        total_turns = 0

        for game_num in range(self.num_games):
            try:
                turns = self._play_enhanced_game(game_num)
                total_turns += turns
                print(f"   Game {game_num + 1}: {turns} turns")
            except Exception as e:
                print(f"❌ Error in game {game_num + 1}: {e}")
                continue

        # Calculate average turns
        if self.results["total_games"] > 0:
            self.results["average_turns"] = total_turns / self.results["total_games"]

        # Save comprehensive game data
        self.state_saver.save_game(f"advanced_rl_evaluation_{self.num_games}_games")

        # Display detailed results
        self._display_enhanced_results()

        # Perform model analysis
        self._analyze_model_performance()

        return self.results

    def _play_enhanced_game(self, game_num: int) -> int:
        """Play a single game with enhanced move analysis and tracking."""
        # Setup game with RL agent and opponents
        colors = [
            PlayerColor.GREEN,
            PlayerColor.RED,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        game = LudoGame(colors[: len(self.opponent_strategies) + 1])

        # Set RL player as first player
        game.players[0].set_strategy(self.rl_player)
        game.players[0].strategy_name = "AdvancedRL-DQN"

        # Set opponent strategies
        for i, opponent_name in enumerate(self.opponent_strategies):
            if i + 1 < len(game.players):
                strategy = StrategyFactory.create_strategy(opponent_name)
                game.players[i + 1].set_strategy(strategy)
                game.players[i + 1].strategy_name = opponent_name

        # Game loop with enhanced tracking
        turn_count = 0
        while not game.game_over and turn_count < self.max_turns:
            current_player = game.get_current_player()
            dice_value = game.roll_dice()

            # Get detailed game context
            game_context = game.get_ai_decision_context(dice_value)

            if game_context["valid_moves"]:
                # Enhanced move analysis for RL player
                if current_player == game.players[0]:  # RL player
                    token_id = self.rl_player.choose_move(game_context)
                else:
                    # Regular opponent move
                    token_id = current_player.make_strategic_decision(game_context)

                # Execute move
                move_result = game.execute_move(
                    current_player, int(token_id), dice_value
                )

                # Save decision data
                self.state_saver.save_decision(
                    strategy_name=current_player.strategy_name,
                    game_context=game_context,
                    chosen_move=int(token_id),
                    outcome=move_result,
                )

                if move_result.get("game_won"):
                    break

                if not move_result.get("extra_turn", False):
                    game.next_turn()
            else:
                game.next_turn()

            turn_count += 1

        # Process game result
        self._process_enhanced_game_result(game, game_num, turn_count)
        return turn_count

    def _process_enhanced_game_result(
        self, game: LudoGame, game_num: int, turn_count: int
    ):
        """Process and record detailed game results."""
        self.results["total_games"] += 1

        if game.winner:
            if game.winner == game.players[0]:  # RL agent won
                self.results["rl_wins"] += 1
                print(f"🏆 Game {game_num + 1}: RL WINS! ({turn_count} turns)")
            else:
                winner_name = game.winner.strategy_name
                self.results["opponent_wins"][winner_name] += 1
                print(
                    f"📉 Game {game_num + 1}: {winner_name.upper()} wins ({turn_count} turns)"
                )
        else:
            print(
                f"⏱️  Game {game_num + 1}: Draw - max turns reached ({turn_count} turns)"
            )

    def _display_enhanced_results(self):
        """Display comprehensive evaluation results."""
        print(f"\n📊 Detailed Results after {self.results['total_games']} games:")
        print("=" * 60)

        # Win statistics
        total_games = max(self.results["total_games"], 1)
        rl_win_rate = self.results["rl_wins"] / total_games * 100
        print("🎯 RL Agent Performance:")
        print(f"   Wins: {self.results['rl_wins']} ({rl_win_rate:.1f}%)")
        print(f"   Average game length: {self.results['average_turns']:.1f} turns")

        # Opponent performance
        print("\n🤖 Opponent Performance:")
        for strategy, wins in self.results["opponent_wins"].items():
            percentage = wins / total_games * 100
            print(f"   {strategy.upper()}: {wins} wins ({percentage:.1f}%)")

        # Confidence analysis
        if self.results["confidence_scores"]:
            avg_confidence = sum(self.results["confidence_scores"]) / len(
                self.results["confidence_scores"]
            )
            max_confidence = max(self.results["confidence_scores"])
            min_confidence = min(self.results["confidence_scores"])

            print("\n🎲 RL Agent Decision Analysis:")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")

        # Strategic decision breakdown
        if self.results["strategic_decisions"]:
            print("\n📈 Strategic Decision Types:")
            total_decisions = sum(self.results["strategic_decisions"].values())
            for decision_type, count in self.results["strategic_decisions"].items():
                percentage = count / total_decisions * 100
                print(f"   {decision_type}: {count} ({percentage:.1f}%)")

    def _analyze_model_performance(self):
        """Perform advanced model performance analysis."""
        print("\n🔍 Advanced Model Analysis:")
        print("=" * 40)

        try:
            # Use validator to analyze decision patterns
            if self.results["rl_move_analysis"]:
                sample_decisions = self.results["rl_move_analysis"][
                    :50
                ]  # Analyze recent decisions

                # Calculate performance metrics
                high_confidence_moves = [
                    m for m in sample_decisions if m["confidence"] > 0.8
                ]
                low_confidence_moves = [
                    m for m in sample_decisions if m["confidence"] < 0.3
                ]

                print(
                    f"   High confidence moves: {len(high_confidence_moves)} ({len(high_confidence_moves) / len(sample_decisions) * 100:.1f}%)"
                )
                print(
                    f"   Low confidence moves: {len(low_confidence_moves)} ({len(low_confidence_moves) / len(sample_decisions) * 100:.1f}%)"
                )

                # Analyze move diversity
                unique_move_types = set(m["move_type"] for m in sample_decisions)
                print(
                    f"   Move type diversity: {len(unique_move_types)} different types"
                )

        except Exception as e:
            print(f"   ⚠️  Analysis error: {e}")

        # Performance recommendations
        rl_win_rate = self.results["rl_wins"] / max(self.results["total_games"], 1)
        print("\n💡 Performance Assessment:")
        if rl_win_rate > 0.6:
            print("   🏆 Excellent! Model shows strong competitive performance.")
        elif rl_win_rate > 0.4:
            print(
                "   📈 Good performance. Model is competitive with room for improvement."
            )
        elif rl_win_rate > 0.25:
            print(
                "   ⚡ Fair performance. Consider additional training or hyperparameter tuning."
            )
        else:
            print(
                "   📉 Needs improvement. Review training data quality and model architecture."
            )


def main():
    """Advanced RL agent evaluation and gameplay demonstration."""
    parser = argparse.ArgumentParser(
        description="Advanced Ludo RL Agent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation
  python play_with_rl_agent.py --model_path models/ludo_dqn_model.pth
  
  # Extended evaluation with more games
  python play_with_rl_agent.py --num_games 20 --max_turns 1500
  
  # Quick evaluation
  python play_with_rl_agent.py --num_games 3 --max_turns 500
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ludo_dqn_model.pth",
        help="Path to the trained RL model (default: models/ludo_dqn_model.pth)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=5,
        help="Number of games to play for evaluation (default: 5)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=1000,
        help="Maximum number of turns per game (default: 1000)",
    )
    args = parser.parse_args()

    print("� Advanced Ludo RL Agent Evaluation")
    print("=" * 50)

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Trained model not found: {model_path}")
        print("\n💡 To create a trained model:")
        print("   1. Generate training data by running tournaments:")
        print("      python four_player_tournament.py")
        print("   2. Train the model:")
        print("      python examples/train_rl_agent.py")
        print("   3. Then run this evaluation script")
        return

    try:
        # Initialize advanced evaluation system
        evaluation = AdvancedRLEvaluation(
            str(model_path), num_games=args.num_games, max_turns=args.max_turns
        )

        # Run comprehensive evaluation
        results = evaluation.run_evaluation()

        # Final performance summary
        if results and results["total_games"] > 0:
            rl_win_rate = results["rl_wins"] / results["total_games"]

            print("\n🏁 Final Performance Summary:")
            print("=" * 40)
            print(f"   Overall win rate: {rl_win_rate:.1%}")
            print(f"   Games played: {results['total_games']}")
            print(
                f"   Average game length: {results.get('average_turns', 0):.1f} turns"
            )

            # Performance classification
            if rl_win_rate > 0.6:
                print("   🏆 Classification: EXCELLENT - Tournament ready!")
            elif rl_win_rate > 0.4:
                print("   📈 Classification: COMPETITIVE - Strong performance!")
            elif rl_win_rate > 0.25:
                print("   ⚡ Classification: DEVELOPING - Shows potential!")
            else:
                print("   📚 Classification: LEARNING - Needs more training!")

            # Display confidence statistics
            if results.get("confidence_scores"):
                avg_conf = sum(results["confidence_scores"]) / len(
                    results["confidence_scores"]
                )
                print(f"   Decision confidence: {avg_conf:.3f}")

        print(
            "\n🎮 Evaluation complete! Results saved to saved_states/rl_gameplay_states/"
        )
        print(f"📁 Model used: {model_path}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        print("   Please check your model file and try again.")


if __name__ == "__main__":
    main()
