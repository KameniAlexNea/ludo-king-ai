#!/usr/bin/env python3
"""
Example script showing how to use a trained RL agent in Ludo games.

This demonstrates:
1. Loading a trained RL model
2. Creating an RL strategy
3. Running games with the RL agent against other strategies
4. Analyzing RL agent performance
"""

import os
import sys
from collections import defaultdict

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_rl import create_rl_strategy
from ludo_stats.game_state_saver import GameStateSaver


class RLEvaluation:
    """System for evaluating a trained RL agent against other strategies."""

    def __init__(self, model_path: str, num_games: int = 10):
        """
        Initialize the RL evaluation environment.

        Args:
            model_path: Path to the trained RL model.
            num_games: Number of games to play for evaluation.
        """
        self.model_path = model_path
        self.num_games = num_games
        self.max_turns = 1000  # Prevent infinite games

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Create RL strategy
        print(f"🤖 Loading RL agent from {self.model_path}")
        self.rl_strategy = create_rl_strategy(self.model_path, "RL-DQN")

        # Get other available strategies
        available_strategies = StrategyFactory.get_available_strategies()
        print(f"Available strategies: {available_strategies}")

        # Select opponent strategies
        opponent_strategies = ["optimist", "winner", "balanced"]
        self.opponent_strategies = [
            s for s in opponent_strategies if s in available_strategies
        ]

        if len(self.opponent_strategies) < 3:
            print(
                f"⚠️  Using all available strategies as opponents: {available_strategies}"
            )
            self.opponent_strategies = available_strategies[:3]

        # Initialize state saver
        self.state_saver = GameStateSaver("saved_states/rl_gameplay_states")

        # Track results
        self.results = {
            "rl_wins": 0,
            "total_games": 0,
            "opponent_wins": defaultdict(int),
        }

    def run_evaluation(self):
        """Run the full evaluation process."""
        print("\n🎮 RL Agent vs Other Strategies Evaluation 🎮")
        print("=" * 50)
        print(f"🎯 Playing {self.num_games} games: RL vs {self.opponent_strategies}")

        for game_num in range(self.num_games):
            try:
                self._play_game(game_num)
            except Exception as e:
                print(f"❌ Error in game {game_num + 1}: {e}")
                continue

        # Save RL gameplay data
        self.state_saver.save_game(f"rl_evaluation_{self.num_games}_games")

        self._display_results()

        return self.results

    def _play_game(self, game_num: int):
        """Plays a single game between the RL agent and opponents."""
        # Create game with RL agent as green and opponents as other colors
        colors = [
            PlayerColor.GREEN,
            PlayerColor.RED,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ]
        game = LudoGame(colors[: len(self.opponent_strategies) + 1])

        # Set strategies
        game.players[0].set_strategy(self.rl_strategy)
        game.players[0].strategy_name = self.rl_strategy.name

        for i, opponent_name in enumerate(self.opponent_strategies):
            if i + 1 < len(game.players):
                strategy = StrategyFactory.create_strategy(opponent_name)
                game.players[i + 1].set_strategy(strategy)
                game.players[i + 1].strategy_name = opponent_name

        # Play game
        turn_count = 0
        while not game.game_over and turn_count < self.max_turns:
            current_player = game.get_current_player()
            dice_value = game.roll_dice()

            # Get AI decision context
            game_context = game.get_ai_decision_context(dice_value)

            if game_context["valid_moves"]:
                # AI player makes decision
                token_id = current_player.make_strategic_decision(game_context)
                # Execute the move
                move_result = game.execute_move(
                    current_player, int(token_id), dice_value
                )

                self.state_saver.save_decision(
                    strategy_name=current_player.strategy.name,
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

        self._process_game_result(game, game_num)

    def _process_game_result(self, game: LudoGame, game_num: int):
        """Processes and records the result of a finished game."""
        self.results["total_games"] += 1

        if game.winner:
            if game.winner == game.players[0]:  # RL agent won
                self.results["rl_wins"] += 1
                print(f"Game {game_num + 1}: RL WINS! 🏆")
            else:
                winner_name = game.winner.strategy_name
                self.results["opponent_wins"][winner_name] += 1
                print(f"Game {game_num + 1}: {winner_name.upper()} wins")
        else:
            print(f"Game {game_num + 1}: Draw (max turns reached)")

    def _display_results(self):
        """Displays the final results of the evaluation."""
        print(f"\n📊 Results after {self.results['total_games']} games:")
        rl_win_rate = (
            self.results["rl_wins"] / max(self.results["total_games"], 1) * 100
        )
        print(f"   RL Agent wins: {self.results['rl_wins']} ({rl_win_rate:.1f}%)")

        for strategy, wins in self.results["opponent_wins"].items():
            percentage = wins / max(self.results["total_games"], 1) * 100
            print(f"   {strategy.upper()} wins: {wins} ({percentage:.1f}%)")


def main():
    """Main example function."""
    print("🎲 Ludo RL Agent Gameplay Example")
    print("=" * 50)

    model_path = "models/ludo_dqn_model.pth"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("\n💡 To create a trained model:")
        print("   1. Generate training data by running tournaments")
        print("   2. Run: python examples/train_rl_agent.py")
        print("   3. Then run this script again")
        return

    try:
        evaluation = RLEvaluation(model_path, num_games=5)
        results = evaluation.run_evaluation()

        if results and results["total_games"] > 0:
            rl_win_rate = results["rl_wins"] / results["total_games"]

            print("\n🎯 RL Agent Performance Summary:")
            print(f"   Win rate: {rl_win_rate:.1%}")

            if rl_win_rate > 0.4:
                print("   🏆 Strong performance! The RL agent is competitive.")
            elif rl_win_rate > 0.2:
                print("   📈 Decent performance. Consider more training or data.")
            else:
                print(
                    "   📉 Needs improvement. Try different hyperparameters or more data."
                )
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
