"""
Ludo King AI Environment - Main Example
Demonstrates how to use the Ludo environment for AI training and gameplay.
"""

from ludo import LudoGame, PlayerColor, StrategyFactory
import json


class SimpleAI:
    """
    A simple AI player that demonstrates how to interact with the Ludo environment.
    This can be enhanced with more sophisticated AI algorithms.
    """

    def __init__(self, player_color: str):
        self.player_color = player_color
        self.name = f"SimpleAI_{player_color}"

    def make_decision(self, game_context: dict) -> int:
        """
        Make a decision about which token to move based on game context.

        Args:
            game_context: Complete game state and available moves

        Returns:
            int: token_id to move (0-3)
        """
        valid_moves = game_context["valid_moves"]

        if not valid_moves:
            return 0  # No valid moves

        # Simple strategy priorities:
        # 1. Finish a token if possible
        # 2. Capture an opponent if possible
        # 3. Exit home if rolling a 6
        # 4. Move the token closest to finishing

        # Priority 1: Finish a token
        for move in valid_moves:
            if move["move_type"] == "finish":
                print(f"{self.name}: Finishing token {move['token_id']}!")
                return move["token_id"]

        # Priority 2: Capture opponent
        for move in valid_moves:
            if move["captures_opponent"]:
                print(f"{self.name}: Capturing opponent with token {move['token_id']}!")
                return move["token_id"]

        # Priority 3: Exit home with a 6
        for move in valid_moves:
            if move["move_type"] == "exit_home":
                print(f"{self.name}: Exiting home with token {move['token_id']}!")
                return move["token_id"]

        # Priority 4: Move token with highest strategic value
        best_move = max(valid_moves, key=lambda m: m["strategic_value"])
        print(
            f"{self.name}: Moving token {best_move['token_id']} (strategic value: {best_move['strategic_value']:.1f})"
        )
        return best_move["token_id"]


def play_game_demo():
    """Demonstrate a complete game with strategic AI players."""
    print("=" * 60)
    print("LUDO KING AI ENVIRONMENT DEMO WITH STRATEGIES")
    print("=" * 60)

    # Create a game with 4 players
    player_colors = [
        PlayerColor.RED,
        PlayerColor.BLUE,
        PlayerColor.GREEN,
        PlayerColor.YELLOW,
    ]
    game = LudoGame(player_colors)

    # Assign different strategies to players
    strategies = ["killer", "winner", "optimist", "defensive"]
    for i, player in enumerate(game.players):
        strategy = StrategyFactory.create_strategy(strategies[i])
        player.set_strategy(strategy)
        print(f"{player.color.value.upper()}: {strategy.name} - {strategy.description}")

    print("\nGame started with strategic AI players!")
    print(
        f"Turn order: {[f'{p.color.value}({p.get_strategy_name()})' for p in game.players]}"
    )

    turn_limit = 100  # Prevent infinite games
    turn_count = 0

    while not game.game_over and turn_count < turn_limit:
        current_player = game.get_current_player()

        print(
            f"\n--- Turn {turn_count + 1}: {current_player.color.value} ({current_player.get_strategy_name()}) ---"
        )

        # Roll dice
        dice_value = game.roll_dice()
        print(f"Dice roll: {dice_value}")

        # Get AI decision context
        context = game.get_ai_decision_context(dice_value)

        # Check for valid moves
        if not context["valid_moves"]:
            print("No valid moves available. Turn skipped.")
            game.next_turn()
            turn_count += 1
            continue

        # AI makes strategic decision
        selected_token = current_player.make_strategic_decision(context)

        # Execute the move
        move_result = game.execute_move(current_player, selected_token, dice_value)

        if move_result["success"]:
            chosen_move = next(
                (m for m in context["valid_moves"] if m["token_id"] == selected_token),
                None,
            )
            move_type = chosen_move["move_type"] if chosen_move else "unknown"
            print(
                f"{current_player.get_strategy_name()} AI: {move_type} with token {selected_token}"
            )
            print(
                f"Move executed: Token {selected_token} from {move_result['old_position']} to {move_result['new_position']}"
            )

            if move_result["captured_tokens"]:
                captured = move_result["captured_tokens"]
                print(
                    f"Captured: {[(t['player_color'], t['token_id']) for t in captured]}"
                )

            if move_result["token_finished"]:
                print(f"Token {selected_token} FINISHED!")

            if move_result.get("game_won"):
                print(f"\n🎉 {current_player.color.value} WINS THE GAME! 🎉")
                break

            if not move_result["extra_turn"]:
                game.next_turn()
        else:
            print(f"Move failed: {move_result['error']}")
            game.next_turn()

        turn_count += 1

        # Show current standings
        if turn_count % 10 == 0:
            print(f"\n--- Standings after {turn_count} turns ---")
            for player in game.players:
                finished = player.get_finished_tokens_count()
                print(f"{player.color.value}: {finished}/4 tokens finished")

    if turn_count >= turn_limit:
        print(f"\nGame ended after {turn_limit} turns (limit reached)")

    print("\nFinal Results:")
    for player in game.players:
        finished = player.get_finished_tokens_count()
        print(f"{player.color.value}: {finished}/4 tokens finished")


def demonstrate_ai_interface():
    """Demonstrate the AI interface and game state representation."""
    print("\n" + "=" * 60)
    print("AI INTERFACE DEMONSTRATION")
    print("=" * 60)

    # Create a simple 2-player game
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])

    print("\n1. Initial Game State:")
    game_state = game.get_game_state_for_ai()
    print(json.dumps(game_state, indent=2))

    print("\n2. Simulating a turn with dice roll...")
    dice_value = game.roll_dice()
    print(f"Dice rolled: {dice_value}")

    print("\n3. AI Decision Context:")
    context = game.get_ai_decision_context(dice_value)
    print(json.dumps(context, indent=2))

    # If there are valid moves, execute one
    if context["valid_moves"]:
        print("\n4. Executing first available move...")
        first_move = context["valid_moves"][0]
        current_player = game.get_current_player()

        move_result = game.execute_move(
            current_player, first_move["token_id"], dice_value
        )
        print("Move result:")
        print(json.dumps(move_result, indent=2))

    print("\n5. Updated Game State:")
    updated_state = game.get_game_state_for_ai()
    print(json.dumps(updated_state, indent=2))


def benchmark_game_performance():
    """Benchmark the game performance for AI training."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    import time

    num_games = 10
    total_time = 0
    total_turns = 0

    print(f"Running {num_games} games to measure performance...")

    for i in range(num_games):
        start_time = time.time()

        # Create and play a complete game
        game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
        ai_players = {
            PlayerColor.RED.value: SimpleAI(PlayerColor.RED.value),
            PlayerColor.BLUE.value: SimpleAI(PlayerColor.BLUE.value),
        }

        turn_count = 0
        max_turns = 200  # Limit for benchmark

        while not game.game_over and turn_count < max_turns:
            current_player = game.get_current_player()
            ai_player = ai_players[current_player.color.value]

            dice_value = game.roll_dice()
            context = game.get_ai_decision_context(dice_value)

            if context["valid_moves"]:
                selected_token = ai_player.make_decision(context)
                move_result = game.execute_move(
                    current_player, selected_token, dice_value
                )

                if not move_result["extra_turn"]:
                    game.next_turn()
            else:
                game.next_turn()

            turn_count += 1

        end_time = time.time()
        game_time = end_time - start_time
        total_time += game_time
        total_turns += turn_count

        print(f"Game {i + 1}: {turn_count} turns, {game_time:.3f}s")

    avg_time = total_time / num_games
    avg_turns = total_turns / num_games

    print("\nBenchmark Results:")
    print(f"Average game time: {avg_time:.3f} seconds")
    print(f"Average turns per game: {avg_turns:.1f}")
    print(f"Average time per turn: {(total_time / total_turns * 1000):.2f} ms")


def strategic_demo():
    """Quick demonstration of strategic AI system."""
    print("\n" + "=" * 60)
    print("STRATEGIC AI SYSTEM DEMONSTRATION")
    print("=" * 60)

    print("\nAvailable AI Strategies:")
    descriptions = StrategyFactory.get_strategy_descriptions()
    for name, desc in descriptions.items():
        print(f"🤖 {name.upper()}: {desc}")

    print("\nQuick 2-Player Strategic Match:")
    print("-" * 40)

    # Create a quick strategic match
    game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])
    game.players[0].set_strategy(StrategyFactory.create_strategy("killer"))
    game.players[1].set_strategy(StrategyFactory.create_strategy("defensive"))

    print(
        f"🔴 RED: {game.players[0].get_strategy_name()} vs 🔵 BLUE: {game.players[1].get_strategy_name()}"
    )

    # Play a few turns to show strategy differences
    for turn in range(10):
        if game.game_over:
            break

        current_player = game.get_current_player()
        dice_value = game.roll_dice()
        context = game.get_ai_decision_context(dice_value)

        if context["valid_moves"]:
            selected_token = current_player.make_strategic_decision(context)
            move_result = game.execute_move(current_player, selected_token, dice_value)

            if move_result["success"]:
                chosen_move = next(
                    (
                        m
                        for m in context["valid_moves"]
                        if m["token_id"] == selected_token
                    ),
                    None,
                )
                if chosen_move:
                    print(
                        f"Turn {turn + 1}: {current_player.color.value} ({current_player.get_strategy_name()}) - {chosen_move['move_type']}"
                    )

                if move_result.get("game_won"):
                    print(
                        f"🏆 {current_player.color.value} ({current_player.get_strategy_name()}) WINS!"
                    )
                    break

                if not move_result.get("extra_turn", False):
                    game.next_turn()
            else:
                print(f"Turn {turn + 1}: {current_player.color.value} - Move failed")
                game.next_turn()
        else:
            print(f"Turn {turn + 1}: {current_player.color.value} - No valid moves")
            game.next_turn()


if __name__ == "__main__":
    # Run the demonstrations
    play_game_demo()
    demonstrate_ai_interface()
    benchmark_game_performance()
    strategic_demo()

    print("\n" + "=" * 60)
    print("LUDO AI ENVIRONMENT WITH STRATEGIES READY!")
    print("=" * 60)
    print("\nTo use this environment in your AI projects:")
    print("1. Import: from ludo import LudoGame, PlayerColor, StrategyFactory")
    print("2. Create: game = LudoGame([PlayerColor.RED, PlayerColor.BLUE])")
    print("3. Strategy: player.set_strategy(StrategyFactory.create_strategy('killer'))")
    print("4. Decide: token_id = player.make_strategic_decision(context)")
    print("5. Execute: result = game.execute_move(player, token_id, dice_value)")
    print("\nAvailable strategies:", StrategyFactory.get_available_strategies())
    print("Run 'python strategic_tournament.py' for tournament mode!")
