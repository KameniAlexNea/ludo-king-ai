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

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ludo import LudoGame, PlayerColor, StrategyFactory
from ludo_rl import create_rl_strategy
from ludo_stats.game_state_saver import GameStateSaver


def play_rl_vs_strategies(model_path: str, num_games: int = 10):
    """
    Play RL agent against existing strategies.

    Args:
        model_path: Path to trained RL model
        num_games: Number of games to play
    """
    print("🎮 RL Agent vs Other Strategies")
    print("=" * 50)

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Train a model first using examples/train_rl_agent.py")
        return

    # Create RL strategy
    print(f"🤖 Loading RL agent from {model_path}")
    rl_strategy = create_rl_strategy(model_path, "RL-DQN")

    # Get other available strategies
    available_strategies = StrategyFactory.get_available_strategies()
    print(f"Available strategies: {available_strategies}")

    # Select opponent strategies
    opponent_strategies = ["killer", "winner", "balanced"]
    available_opponents = [s for s in opponent_strategies if s in available_strategies]

    if len(available_opponents) < 3:
        print(f"⚠️  Using all available strategies: {available_strategies}")
        available_opponents = available_strategies[:3]

    # Initialize state saver
    state_saver = GameStateSaver("rl_gameplay_states")

    # Track results
    results = {
        "rl_wins": 0,
        "total_games": 0,
        "opponent_wins": {strategy: 0 for strategy in available_opponents},
    }

    print(f"\n🎯 Playing {num_games} games: RL vs {available_opponents}")

    for game_num in range(num_games):
        try:
            # Create game with RL agent as green and opponents as other colors
            colors = [
                PlayerColor.GREEN,
                PlayerColor.RED,
                PlayerColor.YELLOW,
                PlayerColor.BLUE,
            ]
            game = LudoGame(colors[: len(available_opponents) + 1])

            # Set strategies
            game.players[0].set_strategy(rl_strategy)  # RL agent
            for i, opponent_strategy in enumerate(available_opponents):
                if i + 1 < len(game.players):
                    strategy = StrategyFactory.create_strategy(opponent_strategy)
                    game.players[i + 1].set_strategy(strategy)

            # Play game
            turn_count = 0
            max_turns = 1000  # Prevent infinite games

            while not game.game_over and turn_count < max_turns:
                current_player = game.get_current_player()
                dice_value = game.roll_dice()

                # Get AI decision context
                game_context = game.get_ai_decision_context(dice_value)

                if game_context["valid_moves"]:
                    # AI player makes decision
                    token_id = current_player.make_strategic_decision(game_context)
                    # Execute the move
                    move_result = game.execute_move(
                        current_player, token_id, dice_value
                    )

                    # Save decision for RL agent
                    # if current_player == game.players[0]:  # RL agent
                    state_saver.save_decision(
                        strategy_name=current_player.strategy.name,
                        game_context=game_context,
                        chosen_move=int(token_id),
                        outcome=move_result,
                    )

                    # Execute turn
                    turn_result = game.play_turn(token_id)
                    turn_count += 1

                    if turn_result.get("game_over"):
                        break

            # Record results
            results["total_games"] += 1

            if game.winner:
                if game.winner == game.players[0]:  # RL agent won
                    results["rl_wins"] += 1
                    print(f"Game {game_num + 1}: RL WINS! 🏆")
                else:
                    # Find which opponent won
                    for i, opponent_strategy in enumerate(available_opponents):
                        if (
                            i + 1 < len(game.players)
                            and game.winner == game.players[i + 1]
                        ):
                            results["opponent_wins"][opponent_strategy] += 1
                            print(f"Game {game_num + 1}: {opponent_strategy} wins")
                            break
            else:
                print(f"Game {game_num + 1}: Draw (max turns reached)")

        except Exception as e:
            print(f"❌ Error in game {game_num + 1}: {e}")
            continue

    # Save RL gameplay data
    state_saver.save_game(f"rl_evaluation_{num_games}_games")

    # Print results
    print(f"\n📊 Results after {results['total_games']} games:")
    print(
        f"   RL Agent wins: {results['rl_wins']} ({results['rl_wins'] / max(results['total_games'], 1) * 100:.1f}%)"
    )

    for strategy, wins in results["opponent_wins"].items():
        percentage = wins / max(results["total_games"], 1) * 100
        print(f"   {strategy} wins: {wins} ({percentage:.1f}%)")

    return results


def analyze_rl_performance(model_path: str):
    """
    Analyze RL agent performance using a single game context.

    Args:
        model_path: Path to trained RL model
    """
    print("\n🔍 RL Agent Analysis")
    print("=" * 30)

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        # Create a test game scenario
        game = LudoGame([PlayerColor.GREEN, PlayerColor.RED])
        rl_strategy = create_rl_strategy(model_path, "RL-DQN")

        # Create a test scenario
        dice_value = 6
        game_context = game.get_ai_decision_context(dice_value)

        print("Test scenario:")
        print(f"   Dice value: {dice_value}")
        print(f"   Valid moves: {len(game_context['valid_moves'])}")

        if game_context["valid_moves"]:
            print("   Available moves:")
            for i, move in enumerate(game_context["valid_moves"]):
                print(
                    f"     {i}: Token {move['token_id']} - {move['move_type']} "
                    f"(strategic value: {move['strategic_value']:.1f})"
                )

        # Get RL agent decision
        chosen_token = rl_strategy.decide(game_context)
        print(f"\n🤖 RL Agent chose: Token {chosen_token}")

        # Compare with best strategic move
        strategic_analysis = game_context.get("strategic_analysis", {})
        best_move = strategic_analysis.get("best_strategic_move")

        if best_move:
            print(
                f"📋 Best strategic move: Token {best_move['token_id']} "
                f"(value: {best_move['strategic_value']:.1f})"
            )

            if chosen_token == best_move["token_id"]:
                print("✅ RL agent chose the optimal move!")
            else:
                print("⚠️  RL agent chose a different move")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")


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

    # Analyze RL performance
    analyze_rl_performance(model_path)

    # Play games
    print(f"\n{'=' * 50}")
    results = play_rl_vs_strategies(model_path, num_games=5)

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


if __name__ == "__main__":
    main()
