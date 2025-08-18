"""
Strategic AI Tournament Example
Demonstrates different AI personalities playing against each other.
"""

from ludo import LudoGame, PlayerColor, StrategyFactory
import random
from collections import defaultdict


def strategy_tournament():
    """Run a tournament between different AI strategies."""
    print("🏆 LUDO AI STRATEGY TOURNAMENT 🏆")
    print("=" * 60)

    # Available strategies
    strategies = ["killer", "winner", "optimist", "defensive", "balanced", "cautious"]

    # Show strategy descriptions
    print("\nParticipating Strategies:")
    print("-" * 40)
    descriptions = StrategyFactory.get_strategy_descriptions()
    for strategy in strategies:
        print(f"🤖 {strategy.upper()}: {descriptions[strategy]}")

    # Tournament settings
    games_per_matchup = 5
    results = defaultdict(lambda: defaultdict(int))
    total_games = 0

    print(f"\n🎮 Running tournament: {games_per_matchup} games per matchup")
    print("=" * 60)

    # Run all matchups
    for i, strategy1 in enumerate(strategies):
        for j, strategy2 in enumerate(strategies):
            if i < j:  # Avoid duplicate matchups
                print(f"\n{strategy1.upper()} vs {strategy2.upper()}")
                print("-" * 30)

                matchup_results = {"wins": {strategy1: 0, strategy2: 0}, "games": []}

                for game_num in range(games_per_matchup):
                    # Create game with strategies
                    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])

                    # Assign strategies
                    game.players[0].set_strategy(
                        StrategyFactory.create_strategy(strategy1)
                    )
                    game.players[1].set_strategy(
                        StrategyFactory.create_strategy(strategy2)
                    )

                    # Play the game
                    winner = play_strategic_game(game, max_turns=200)
                    total_games += 1

                    if winner:
                        winner_strategy = (
                            strategy1 if winner.color == PlayerColor.RED else strategy2
                        )
                        matchup_results["wins"][winner_strategy] += 1
                        results[winner_strategy]["wins"] += 1

                        loser_strategy = (
                            strategy2 if winner_strategy == strategy1 else strategy1
                        )
                        results[loser_strategy]["losses"] += 1

                        print(f"  Game {game_num + 1}: {winner_strategy.upper()} wins!")
                    else:
                        print(f"  Game {game_num + 1}: Draw (time limit)")
                        results[strategy1]["draws"] += 1
                        results[strategy2]["draws"] += 1

                # Show matchup summary
                s1_wins = matchup_results["wins"][strategy1]
                s2_wins = matchup_results["wins"][strategy2]
                print(
                    f"  Result: {strategy1.upper()} {s1_wins} - {s2_wins} {strategy2.upper()}"
                )

    # Tournament results
    print("\n🏆 FINAL TOURNAMENT RESULTS 🏆")
    print("=" * 60)

    # Calculate win rates
    strategy_stats = []
    for strategy in strategies:
        wins = results[strategy]["wins"]
        losses = results[strategy]["losses"]
        draws = results[strategy]["draws"]
        total = wins + losses + draws
        win_rate = (wins / total * 100) if total > 0 else 0

        strategy_stats.append(
            {
                "name": strategy,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total": total,
                "win_rate": win_rate,
            }
        )

    # Sort by win rate
    strategy_stats.sort(key=lambda x: x["win_rate"], reverse=True)

    print(
        f"{'Rank':<4} {'Strategy':<12} {'Wins':<6} {'Losses':<7} {'Draws':<6} {'Win Rate':<8}"
    )
    print("-" * 55)

    for rank, stats in enumerate(strategy_stats, 1):
        print(
            f"{rank:<4} {stats['name'].upper():<12} {stats['wins']:<6} {stats['losses']:<7} {stats['draws']:<6} {stats['win_rate']:.1f}%"
        )

    print(f"\nTotal games played: {total_games}")

    return strategy_stats


def play_strategic_game(game: LudoGame, max_turns=200):
    """Play a complete game with strategic AI players."""
    turn_count = 0

    while not game.game_over and turn_count < max_turns:
        current_player = game.get_current_player()
        dice_value = game.roll_dice()

        # Get AI decision context
        context = game.get_ai_decision_context(dice_value)

        if context["valid_moves"]:
            # AI makes strategic decision
            selected_token = current_player.make_strategic_decision(context)

            # Execute the move
            move_result = game.execute_move(current_player, selected_token, dice_value)

            if move_result["success"] and move_result.get("game_won"):
                return current_player

            if not move_result.get("extra_turn", False):
                game.next_turn()
        else:
            game.next_turn()

        turn_count += 1

    return None  # Draw/timeout


def demonstrate_strategic_thinking():
    """Demonstrate how different strategies think about the same situation."""
    print("\n🧠 STRATEGIC THINKING DEMONSTRATION 🧠")
    print("=" * 60)

    # Create a game situation
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])

    # Set up an interesting game state
    # Player has some tokens out, opponent has threats
    red_player = game.players[0]
    blue_player = game.players[1]

    # Move some tokens to create interesting situation
    # (This would normally be done through game play)
    from ludo.token import TokenState

    red_player.tokens[0].state = TokenState.ACTIVE
    red_player.tokens[0].position = 25
    red_player.tokens[1].state = TokenState.ACTIVE
    red_player.tokens[1].position = 45

    blue_player.tokens[0].state = TokenState.ACTIVE
    blue_player.tokens[0].position = 30

    # Roll a dice for decision making
    dice_value = 5
    game.last_dice_value = dice_value

    print(f"Game Situation: RED player to move, dice = {dice_value}")
    print("RED tokens: Token 0 at position 25, Token 1 at position 45")
    print("BLUE tokens: Token 0 at position 30")
    print()

    # Get game context
    context = game.get_ai_decision_context(dice_value)

    if context["valid_moves"]:
        print("Available moves:")
        for i, move in enumerate(context["valid_moves"]):
            print(
                f"  {i + 1}. Move Token {move['token_id']} from {move['current_position']} to {move['target_position']}"
            )
            print(
                f"     Type: {move['move_type']}, Safe: {move['is_safe_move']}, Value: {move['strategic_value']:.1f}"
            )
        print()

        # Test different strategies
        strategies_to_test = ["killer", "winner", "defensive", "optimist", "balanced"]

        print("Strategic Decisions:")
        print("-" * 30)

        for strategy_name in strategies_to_test:
            strategy = StrategyFactory.create_strategy(strategy_name)
            choice = strategy.decide(context)
            chosen_move = next(
                (m for m in context["valid_moves"] if m["token_id"] == choice), None
            )

            if chosen_move:
                print(
                    f"{strategy_name.upper():<10}: Token {choice} (to pos {chosen_move['target_position']}) - {strategy.description}"
                )
            else:
                print(f"{strategy_name.upper():<10}: No valid choice")
    else:
        print("No valid moves available in this situation.")


def compare_strategies_detailed():
    """Detailed comparison of strategy performance."""
    print("\n📊 DETAILED STRATEGY COMPARISON 📊")
    print("=" * 60)

    strategies = ["killer", "winner", "defensive", "optimist", "balanced"]
    num_games = 10

    detailed_stats = defaultdict(
        lambda: {
            "games_won": 0,
            "total_turns": 0,
            "tokens_captured": 0,
            "tokens_finished": 0,
            "risky_moves": 0,
            "safe_moves": 0,
        }
    )

    print(f"Running {num_games} games per strategy vs Random AI...")

    for strategy_name in strategies:
        print(f"\nTesting {strategy_name.upper()}...")

        for game_num in range(num_games):
            # Create game: strategy vs random
            game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
            game.players[0].set_strategy(StrategyFactory.create_strategy(strategy_name))
            game.players[1].set_strategy(StrategyFactory.create_strategy("random"))

            # Track game stats
            game_stats = {
                "turns": 0,
                "captures": 0,
                "finished": 0,
                "risky": 0,
                "safe": 0,
            }

            while not game.game_over and game_stats["turns"] < 200:
                current_player = game.get_current_player()
                dice_value = game.roll_dice()
                context = game.get_ai_decision_context(dice_value)

                if context["valid_moves"]:
                    selected_token = current_player.make_strategic_decision(context)
                    move_result = game.execute_move(
                        current_player, selected_token, dice_value
                    )

                    # Track stats for strategy player only
                    if current_player.color == PlayerColor.RED:
                        if move_result.get("captured_tokens"):
                            game_stats["captures"] += len(
                                move_result["captured_tokens"]
                            )
                        if move_result.get("token_finished"):
                            game_stats["finished"] += 1

                    if (
                        move_result.get("game_won")
                        and current_player.color == PlayerColor.RED
                    ):
                        detailed_stats[strategy_name]["games_won"] += 1
                        break

                    if not move_result.get("extra_turn", False):
                        game.next_turn()
                else:
                    game.next_turn()

                game_stats["turns"] += 1

            # Update detailed stats
            detailed_stats[strategy_name]["total_turns"] += game_stats["turns"]
            detailed_stats[strategy_name]["tokens_captured"] += game_stats["captures"]
            detailed_stats[strategy_name]["tokens_finished"] += game_stats["finished"]

    # Display detailed results
    print("\nDetailed Performance Analysis:")
    print("-" * 70)
    print(
        f"{'Strategy':<12} {'Win Rate':<10} {'Avg Turns':<10} {'Captures':<10} {'Finished':<10}"
    )
    print("-" * 70)

    for strategy_name in strategies:
        stats = detailed_stats[strategy_name]
        win_rate = (stats["games_won"] / num_games) * 100
        avg_turns = stats["total_turns"] / num_games if num_games > 0 else 0

        print(
            f"{strategy_name.upper():<12} {win_rate:<10.1f}% {avg_turns:<10.1f} {stats['tokens_captured']:<10} {stats['tokens_finished']:<10}"
        )


if __name__ == "__main__":
    # Run all demonstrations
    random.seed(42)  # For reproducible results

    # Strategy tournament
    tournament_results = strategy_tournament()

    # Strategic thinking demo
    demonstrate_strategic_thinking()

    # Detailed comparison
    compare_strategies_detailed()

    print("\n" + "=" * 60)
    print("🎯 STRATEGIC AI SYSTEM READY! 🎯")
    print("=" * 60)
    print("\nKey Features:")
    print("✅ 7 Different AI Personalities")
    print("✅ Adaptive Decision Making")
    print("✅ Strategic Analysis Integration")
    print("✅ Tournament System")
    print("✅ Performance Comparison")
    print("\nUse StrategyFactory.create_strategy('strategy_name') to create AIs!")
    print("Available strategies:", StrategyFactory.get_available_strategies())
