#!/usr/bin/env python3
"""Validate reward components by comparing different opponent strategies.

This tool runs tournament-style games to compare:
1. Random vs Random (baseline)
2. Weighted Random vs Random (intermediate)
3. Probabilistic vs Random (advanced strategy)
4. Probabilistic vs Probabilistic (expert vs expert)

This validates that reward shaping is working correctly by examining what
rewards each strategy receives in head-to-head matchups.

Key metrics:
- Reward breakdown per strategy
- Win rates and dominance
- What behaviors are being rewarded
"""

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ludo_engine import LudoGame
from ludo_engine.models import ALL_COLORS, PlayerColor, GameConstants, MoveType
from ludo_engine.strategies.strategy import StrategyFactory

sys.path.insert(0, str(Path(__file__).parent.parent))

from ludo_rl.config import EnvConfig
from ludo_rl.rewards.reward_adv_calculator import (
    AdvancedRewardCalculator,
    AdvRewardConfig,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate reward components by comparing opponent strategies"
    )
    p.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per matchup",
    )
    p.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players per game",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-game details",
    )
    return p.parse_args()


def format_header(text: str, width: int = 90) -> str:
    """Format a section header."""
    return f"\n{'=' * width}\n{text.center(width)}\n{'=' * width}\n"


def format_subsection(text: str, width: int = 90) -> str:
    """Format a subsection header."""
    return f"\n{'-' * width}\n{text}\n{'-' * width}"


def categorize_rewards(breakdown: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Categorize reward components into logical groups."""
    categories = {
        "Terminal": {},
        "Shaping - Opportunities": {},
        "Shaping - Risk Management": {},
        "Shaping - Strategic": {},
        "Shaping - Planning": {},
        "Basic Events": {},
        "Penalties": {},
    }

    categorization = {
        # Terminal
        "terminal": "Terminal",
        # Opportunities
        "capture_opportunity_taken": "Shaping - Opportunities",
        "capture_opportunity_missed": "Shaping - Opportunities",
        "exit_opportunity_taken": "Shaping - Opportunities",
        "exit_opportunity_missed": "Shaping - Opportunities",
        "finish_opportunity_taken": "Shaping - Opportunities",
        "finish_opportunity_missed": "Shaping - Opportunities",
        # Risk management
        "vulnerability_reduction": "Shaping - Risk Management",
        "vulnerability_increase": "Shaping - Risk Management",
        # Strategic
        "blocking_bonus": "Shaping - Strategic",
        "safe_zone": "Shaping - Strategic",
        # Planning
        "progress_efficiency": "Shaping - Planning",
        "opponent_pressure_relief": "Shaping - Planning",
        # Basic events
        "progress": "Basic Events",
        "capture": "Basic Events",
        "finish": "Basic Events",
        "exit_start": "Basic Events",
        "extra_turn": "Basic Events",
        "diversity_bonus": "Basic Events",
        # Penalties
        "illegal": "Penalties",
        "time_penalty": "Penalties",
        "got_captured": "Penalties",
        "all_captured": "Penalties",
    }

    for key, value in breakdown.items():
        category = categorization.get(key, "Other")
        if category not in categories:
            categories[category] = {}
        categories[category][key] = value

    return categories


def run_game(
    strategy_names: List[str],
    num_players: int,
    cfg: EnvConfig,
    reward_calc: AdvancedRewardCalculator,
    seed: Optional[int] = None,
) -> tuple:
    """Run a single game and track rewards for each strategy.

    Returns: (winner_color, rewards_by_color, breakdowns_by_color)
    """
    # Create player colors (2-player uses RED and YELLOW)
    if num_players == 2:
        colors = [ALL_COLORS[0], ALL_COLORS[2]]  # RED and YELLOW
    else:
        colors = ALL_COLORS[:num_players]

    if seed is not None:
        random.seed(seed)

    game = LudoGame(colors)

    # Initialize tracking (keyed by color)
    rewards_by_color: Dict[PlayerColor, float] = {color: 0.0 for color in colors}
    breakdowns_by_color: Dict[PlayerColor, Dict[str, float]] = {
        color: defaultdict(float) for color in colors
    }

    # Set up strategies for each player
    for strategy_name, color in zip(strategy_names, colors):
        player = game.get_player_from_color(color)
        strategy = StrategyFactory.create_strategy(strategy_name)
        player.set_strategy(strategy)
        player.strategy_name = strategy_name

    # Initialize reward calc
    reward_calc.reset_for_new_episode()
    
    # Track opportunities per color
    color_stats = {
        color: {
            "capture_ops_available": 0,
            "capture_ops_taken": 0,
            "finish_ops_available": 0,
            "finish_ops_taken": 0,
            "home_exit_ops_available": 0,
            "home_exit_ops_taken": 0,
        }
        for color in colors
    }

    # Play game
    max_turns = 500
    turn = 0
    while not game.game_over and turn < max_turns:
        player = game.get_current_player()
        current_color = player.color

        # Roll dice and get valid moves
        dice = game.roll_dice()
        ai_context = game.get_ai_decision_context(dice)
        valid_moves = ai_context.valid_moves if ai_context else []

        # Track available opportunities
        color_stats[current_color]["capture_ops_available"] += sum(1 for m in valid_moves if m.captures_opponent)
        color_stats[current_color]["finish_ops_available"] += sum(1 for m in valid_moves if m.target_position == GameConstants.FINISH_POSITION)
        color_stats[current_color]["home_exit_ops_available"] += sum(1 for m in valid_moves if m.move_type == MoveType.EXIT_HOME)

        # Use the player's built-in strategic decision making
        token_id = player.make_strategic_decision(ai_context)

        if token_id is None:
            # No valid moves, skip turn
            game.next_turn()
            turn += 1
            continue

        # Execute move
        move_result = game.execute_move(player, token_id, dice)
        
        # Track taken opportunities - find which move was executed
        chosen_move = next((m for m in valid_moves if m.token_id == token_id), None)
        if chosen_move:
            if chosen_move.captures_opponent:
                color_stats[current_color]["capture_ops_taken"] += 1
            if chosen_move.target_position == GameConstants.FINISH_POSITION:
                color_stats[current_color]["finish_ops_taken"] += 1
            if chosen_move.move_type == MoveType.EXIT_HOME:
                color_stats[current_color]["home_exit_ops_taken"] += 1

        # Build episode_info dict from accumulated stats
        episode_info = {
            "episode_capture_ops_available": color_stats[current_color]["capture_ops_available"],
            "episode_capture_ops_taken": color_stats[current_color]["capture_ops_taken"],
            "episode_home_exit_ops_available": color_stats[current_color]["home_exit_ops_available"],
            "episode_home_exit_ops_taken": color_stats[current_color]["home_exit_ops_taken"],
            "episode_finish_ops_available": color_stats[current_color]["finish_ops_available"],
            "episode_finish_ops_taken": color_stats[current_color]["finish_ops_taken"],
        }

        # Compute reward
        reward, breakdown = reward_calc.compute(
            game=game,
            agent_color=current_color,
            move=move_result,
            cfg=cfg,
            episode_info=episode_info,
            return_breakdown=True,
            is_illegal=False,
        )

        rewards_by_color[current_color] += reward
        for k, v in breakdown.items():
            breakdowns_by_color[current_color][k] += v

        # Move to next turn if no extra turn
        if not move_result.extra_turn:
            game.next_turn()
        turn += 1

    winner_color = game.winner.color if game.winner else None
    return (winner_color, rewards_by_color, breakdowns_by_color)


def run_matchup(
    strategy_names: List[str],
    num_games: int,
    num_players: int,
    cfg: EnvConfig,
    reward_calc: AdvancedRewardCalculator,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run multiple games and aggregate stats."""

    # Storage
    wins_by_strategy = defaultdict(int)
    total_rewards_by_strategy = defaultdict(list)
    total_breakdown_by_strategy = defaultdict(lambda: defaultdict(float))

    # Map strategy names to colors for tracking
    colors = (
        [
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ][:num_players]
        if num_players > 2
        else [ALL_COLORS[0], ALL_COLORS[2]]
    )

    print(f"Running {num_games} games: {' vs '.join(strategy_names)}...")
    print("Progress: ", end="", flush=True)

    for game_idx in range(num_games):
        if game_idx % max(1, num_games // 10) == 0:
            print(f"{game_idx}", end=" ", flush=True)

        winner, rewards_by_color, breakdowns_by_color = run_game(
            strategy_names=strategy_names,
            num_players=num_players,
            cfg=cfg,
            reward_calc=reward_calc,
            seed=seed + game_idx if seed else None,
        )

        # Track results (map colors to strategy names)
        for strategy_idx, strategy_name in enumerate(strategy_names):
            color = colors[strategy_idx]

            # Collect rewards
            total_rewards_by_strategy[strategy_name].append(
                rewards_by_color.get(color, 0.0)
            )

            # Collect breakdowns
            for k, v in breakdowns_by_color.get(color, {}).items():
                total_breakdown_by_strategy[strategy_name][k] += v

            # Check if this strategy won
            if winner == color:
                wins_by_strategy[strategy_name] += 1

        if verbose and game_idx < 3:
            winner_name = None
            for strategy_idx, strategy_name in enumerate(strategy_names):
                if colors[strategy_idx] == winner:
                    winner_name = strategy_name
                    break
            print(f"\n  Game {game_idx + 1}: {winner_name or 'Unknown'} won")

    print("\n")

    # Compute stats
    results = {}
    for strategy in strategy_names:
        rewards = total_rewards_by_strategy[strategy]
        if rewards:
            results[strategy] = {
                "wins": wins_by_strategy.get(strategy, 0),
                "games_played": len(rewards),
                "mean_reward": statistics.mean(rewards),
                "median_reward": statistics.median(rewards) if rewards else 0,
                "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0,
                "breakdown": dict(total_breakdown_by_strategy.get(strategy, {})),
            }

    return results


def print_results(matchup_name: str, results: Dict):
    """Print formatted results for a matchup."""
    print(format_subsection(f"MATCHUP: {matchup_name}"))

    for strategy, stats in results.items():
        wins = stats["wins"]
        games = stats["games_played"]
        mean_reward = stats["mean_reward"]
        median_reward = stats["median_reward"]
        std_reward = stats["std_reward"]
        breakdown = stats["breakdown"]

        win_rate = (wins / games * 100) if games > 0 else 0
        print(f"\n{strategy}:")
        print(f"  Win rate:      {wins}/{games} = {win_rate:.1f}%")
        print(f"  Mean reward:   {mean_reward:+.2f}")
        print(f"  Median reward: {median_reward:+.2f}")
        print(f"  Std dev:       {std_reward:.2f}")

        if breakdown:
            # Categorize and show top components
            categories = categorize_rewards(breakdown)
            print("  Reward breakdown:")

            for category_name, components in categories.items():
                if not components:
                    continue

                category_total = sum(components.values())
                print(f"    {category_name}: {category_total:+.2f}")

                # Show top 3 components
                for component, value in sorted(
                    components.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]:
                    print(f"      • {component}: {value:+.2f}")


def main():
    args = parse_args()
    print(format_header("REWARD VALIDATION - OPPONENT COMPARISON"))

    cfg = EnvConfig()
    cfg.fixed_num_players = args.num_players
    reward_calc = AdvancedRewardCalculator(AdvRewardConfig())

    # Run matchups
    matchups = [
        ("Random vs WRandom", ["random", "weighted_random"]),
        ("Random vs Prob", ["random", "probabilistic_v2"]),
        ("Probabilistic vs Random", ["probabilistic_v2", "random"]),
        ("Probabilistic vs Hybrid", ["probabilistic_v2", "hybrid_prob"]),
    ]

    all_results = {}
    for matchup_name, strategies in matchups:
        results = run_matchup(
            strategy_names=strategies,
            num_games=args.games,
            num_players=args.num_players,
            cfg=cfg,
            reward_calc=reward_calc,
            seed=args.seed,
            verbose=args.verbose,
        )
        all_results[matchup_name] = results
        print_results(matchup_name, results)

    # Summary
    print(format_header("VALIDATION SUMMARY"))
    print("\nKey insights:")
    print("- Compare rewards across different opponent strategies")
    print("- Strong strategies should receive consistent rewards")
    print("- Winning strategies should show positive terminal rewards")
    print()

    print("\nReward components by strategy:")
    for matchup_name, results in all_results.items():
        print(f"\n{matchup_name}:")
        for strategy, stats in results.items():
            mean = stats["mean_reward"]
            wins = stats["wins"]
            print(f"  {strategy:20s} -> mean reward: {mean:+7.2f}, wins: {wins:3d}")


if __name__ == "__main__":
    main()
