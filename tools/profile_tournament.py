"""
Profiling script for Ludo tournament to identify bottlenecks.

This script runs a simplified tournament with profiling enabled to identify
performance bottlenecks in the game simulation and strategy decision making.
"""

from __future__ import annotations

import cProfile
import pstats
import random
import time
from io import StringIO
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Color
from ludo_rl.strategy.registry import STRATEGY_REGISTRY


def build_board_stack(game: Game, player_index: int) -> np.ndarray:
    """Build board state for strategy decision making."""
    player_color = int(game.players[player_index].color)
    return game.board.build_tensor(player_color)


def attach_strategy(player: Player, strategy_name: str, rng: random.Random) -> None:
    """Attach a strategy to a player."""
    cls = STRATEGY_REGISTRY[strategy_name]
    try:
        player.strategy = cls.create_instance(rng)
    except NotImplementedError:
        player.strategy = cls()
    player.strategy_name = strategy_name  # type: ignore[attr-defined]


def play_single_game(
    strategy_names: Sequence[str],
    rng: random.Random,
) -> dict:
    """Play a single game and return timing statistics."""
    # Create players with colors
    colors = [Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE]
    players = [Player(color=colors[i]) for i in range(config.NUM_PLAYERS)]
    game = Game(players=players)

    game_seed = rng.randint(0, 1_000_000)
    game.rng.seed(game_seed)
    random.seed(game_seed)

    # Initialize players
    for player, strategy_name in zip(game.players, strategy_names, strict=True):
        for piece in player.pieces:
            piece.position = 0
        player.has_finished = False
        player.strategy = None
        attach_strategy(player, strategy_name, rng)

    finish_order: List[int] = []
    turns_taken = 0
    current_index = 0
    # Timing stats
    time_get_board_state = 0.0
    time_decide = 0.0
    time_get_valid_moves = 0.0
    time_make_move = 0.0
    decision_count = 0

    while turns_taken < config.MAX_TURNS and len(finish_order) < config.NUM_PLAYERS:
        player = game.players[current_index]
        if player.check_won():
            if current_index not in finish_order:
                finish_order.append(current_index)
            current_index = (current_index + 1) % config.NUM_PLAYERS
            continue

        extra_turn = True
        while extra_turn:
            dice_roll = game.roll_dice()

            # Time legal_moves (new API)
            t0 = time.perf_counter()
            legal_moves = game.legal_moves(current_index, dice_roll)
            time_get_valid_moves += time.perf_counter() - t0

            if not legal_moves:
                extra_turn = False
                continue

            # Time get_board_state
            t0 = time.perf_counter()
            board_stack = build_board_stack(game, current_index)
            time_get_board_state += time.perf_counter() - t0

            # Time player.choose (new API)
            t0 = time.perf_counter()
            chosen_move = player.choose(board_stack, dice_roll, legal_moves)
            time_decide += time.perf_counter() - t0
            decision_count += 1

            move = chosen_move if chosen_move is not None else rng.choice(legal_moves)

            # Time apply_move (new API)
            t0 = time.perf_counter()
            result = game.apply_move(move)
            time_make_move += time.perf_counter() - t0

            if not result.events.move_resolved:
                extra_turn = False
            else:
                extra_turn = result.extra_turn

            if player.check_won() and current_index not in finish_order:
                finish_order.append(current_index)

        current_index = (current_index + 1) % config.NUM_PLAYERS
        turns_taken += 1

    return {
        "turns": turns_taken,
        "time_get_board_state": time_get_board_state,
        "time_decide": time_decide,
        "time_get_valid_moves": time_get_valid_moves,
        "time_make_move": time_make_move,
        "decision_count": decision_count,
    }


def run_profiled_games(num_games: int = 100, seed: int = 42):
    """Run multiple games with timing statistics."""
    rng = random.Random(seed)

    # Select 4 different strategies
    strategies = ["rusher", "killer", "cautious", "defensive"]

    print(f"Running {num_games} games with strategies: {', '.join(strategies)}")
    print("=" * 80)

    total_stats = {
        "turns": 0,
        "time_get_board_state": 0.0,
        "time_decide": 0.0,
        "time_get_valid_moves": 0.0,
        "time_make_move": 0.0,
        "decision_count": 0,
    }

    start_time = time.perf_counter()

    for i in range(num_games):
        stats = play_single_game(strategies, rng)
        for key in total_stats:
            total_stats[key] += stats[key]

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games...")

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("TIMING ANALYSIS")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Total games: {num_games}")
    print(f"Time per game: {total_time / num_games:.4f}s")
    print(f"Total turns: {total_stats['turns']}")
    print(f"Total decisions: {total_stats['decision_count']}")
    print()

    # Calculate percentages
    time_sum = (
        total_stats["time_get_board_state"]
        + total_stats["time_decide"]
        + total_stats["time_get_valid_moves"]
        + total_stats["time_make_move"]
    )

    print("Time breakdown:")
    print(
        f"  get_board_state:  {total_stats['time_get_board_state']:8.3f}s "
        f"({100 * total_stats['time_get_board_state'] / time_sum:5.1f}%) "
        f"- {total_stats['time_get_board_state'] / total_stats['decision_count'] * 1000:.3f}ms per call"
    )
    print(
        f"  player.decide:    {total_stats['time_decide']:8.3f}s "
        f"({100 * total_stats['time_decide'] / time_sum:5.1f}%) "
        f"- {total_stats['time_decide'] / total_stats['decision_count'] * 1000:.3f}ms per call"
    )
    print(
        f"  get_valid_moves:  {total_stats['time_get_valid_moves']:8.3f}s "
        f"({100 * total_stats['time_get_valid_moves'] / time_sum:5.1f}%) "
        f"- {total_stats['time_get_valid_moves'] / total_stats['decision_count'] * 1000:.3f}ms per call"
    )
    print(
        f"  make_move:        {total_stats['time_make_move']:8.3f}s "
        f"({100 * total_stats['time_make_move'] / time_sum:5.1f}%) "
        f"- {total_stats['time_make_move'] / total_stats['decision_count'] * 1000:.3f}ms per call"
    )
    print()
    print(
        f"Measured overhead: {time_sum:.3f}s ({100 * time_sum / total_time:.1f}% of total)"
    )
    print(
        f"Unmeasured overhead: {total_time - time_sum:.3f}s ({100 * (total_time - time_sum) / total_time:.1f}% of total)"
    )


def run_cprofile_single_game():
    """Run a single game with cProfile for detailed function-level profiling."""
    print("\n" + "=" * 80)
    print("DETAILED PROFILING (cProfile)")
    print("=" * 80)

    rng = random.Random(42)
    strategies = ["rusher", "killer", "cautious", "defensive"]

    profiler = cProfile.Profile()
    profiler.enable()

    # Run a single game
    play_single_game(strategies, rng)

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(50)  # Top 50 functions
    print(s.getvalue())

    # Save to file
    output_dir = Path(__file__).parent.parent / "training/profiling_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "tournament_profile.prof"
    profiler.dump_stats(str(output_file))
    print(f"\nProfile saved to: {output_file}")
    print(
        "View with: python -m pstats training/profiling_results/tournament_profile.prof"
    )


def main():
    """Main entry point."""
    print("Ludo Tournament Profiling")
    print("=" * 80)
    print()

    # First run timing analysis
    run_profiled_games(num_games=100, seed=42)

    # Then run detailed profiling
    run_cprofile_single_game()


if __name__ == "__main__":
    main()
