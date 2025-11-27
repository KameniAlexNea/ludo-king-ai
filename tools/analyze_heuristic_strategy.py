#!/usr/bin/env python3
"""Analyze heuristic strategies by tracking their move choices and rewards.

This script runs episodes with a fixed heuristic strategy (e.g., cautious, killer)
and collects detailed statistics about its decision-making patterns and rewards.

Usage:
    python tools/analyze_heuristic_strategy.py --strategy cautious --episodes 100
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

from ludo_rl.ludo_king import config as king_config
from ludo_rl.ludo_king.config import reward_config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import MoveEvents
from ludo_rl.strategy.registry import available as available_strategies
from ludo_rl.strategy.registry import create as get_strategy

load_dotenv()


@dataclass
class MoveStats:
    """Statistics for a single move."""

    episode: int
    step: int
    dice_roll: int
    piece_id: int
    old_pos: int
    new_pos: int
    # Move characteristics
    exits_yard: bool = False
    captures: bool = False
    finishes: bool = False
    enters_safe: bool = False
    forms_blockade: bool = False
    leaves_safe: bool = False
    hit_blockade: bool = False
    # Rewards received
    reward_total: float = 0.0
    reward_progress: float = 0.0
    reward_capture: float = 0.0
    reward_got_captured: float = 0.0
    reward_finish: float = 0.0
    reward_exit_home: float = 0.0
    reward_blockade: float = 0.0
    reward_hit_blockade: float = 0.0
    # Move category
    move_category: str = "other"


@dataclass
class StrategyAnalysis:
    """Aggregated statistics about a heuristic strategy."""

    strategy_name: str = ""
    total_moves: int = 0
    total_episodes: int = 0
    wins: int = 0
    total_pieces_finished: int = 0

    # Event counts
    exit_yard_count: int = 0
    capture_count: int = 0
    got_captured_count: int = 0
    finish_count: int = 0
    blockade_count: int = 0
    hit_blockade_count: int = 0
    enter_safe_count: int = 0
    leave_safe_count: int = 0

    # Reward tracking
    total_reward: float = 0.0
    reward_from_progress: float = 0.0
    reward_from_capture: float = 0.0
    reward_from_got_captured: float = 0.0
    reward_from_finish: float = 0.0
    reward_from_exit_home: float = 0.0
    reward_from_blockade: float = 0.0
    reward_from_hit_blockade: float = 0.0
    reward_from_win: float = 0.0
    reward_from_lose: float = 0.0

    # Move category counts
    category_counts: Counter = field(default_factory=Counter)
    category_rewards: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Per-episode tracking
    episode_rewards: List[float] = field(default_factory=list)
    episode_moves: List[int] = field(default_factory=list)

    # Detailed move tracking
    moves: List[MoveStats] = field(default_factory=list)


def compute_move_reward_breakdown(
    old_pos: int,
    new_pos: int,
    events: MoveEvents,
) -> Dict[str, float]:
    """Compute individual reward components for a move."""
    rewards = {
        "progress": 0.0,
        "capture": 0.0,
        "got_captured": 0.0,
        "finish": 0.0,
        "exit_home": 0.0,
        "blockade": 0.0,
        "hit_blockade": 0.0,
    }

    if events.move_resolved and old_pos != new_pos:
        rewards["progress"] = reward_config.progress

    if events.exited_home:
        rewards["exit_home"] = reward_config.exit_home

    if events.finished:
        rewards["finish"] = reward_config.finish

    if events.knockouts:
        rewards["capture"] = reward_config.capture * len(events.knockouts)

    if events.hit_blockade:
        rewards["hit_blockade"] = reward_config.hit_blockade

    if events.blockades:
        rewards["blockade"] = reward_config.blockade

    return rewards


def run_analysis_episode(
    strategy_name: str,
    opponents: List[str],
    episode_num: int,
    stats: StrategyAnalysis,
    collect_detailed: bool = True,
) -> bool:
    """Run one episode with heuristic strategy and collect statistics."""
    # Create game with strategy as player 0
    all_strategies = [strategy_name] + opponents[:3]
    while len(all_strategies) < king_config.NUM_PLAYERS:
        all_strategies.append(opponents[0] if opponents else "rusher")

    # Build players
    players = []
    for i, strat_name in enumerate(all_strategies):
        strategy = get_strategy(strat_name)
        player = Player(color=i, strategy=strategy, strategy_name=strat_name)
        players.append(player)

    game = Game(players)

    # Track our player (index 0)
    agent_idx = 0
    step = 0
    episode_reward = 0.0
    episode_moves = 0
    turn_count = 0

    while (
        not any(p.check_won() for p in game.players)
        and turn_count < king_config.MAX_TURNS
    ):
        current_idx = turn_count % len(game.players)

        # Roll dice
        dice = game.roll_dice()
        legal_moves = game.legal_moves(current_idx, dice)

        if not legal_moves:
            turn_count += 1
            continue

        # Get player's move choice
        player = game.players[current_idx]
        player_color = int(player.color)

        # Build board tensor for strategy
        board_tensor = game.board.build_tensor(player_color)

        # Player.choose returns Move object directly
        chosen_move = player.choose(board_tensor, dice, legal_moves)

        if chosen_move is None:
            chosen_move = legal_moves[0]

        # Get old position
        piece = player.pieces[chosen_move.piece_id]
        old_pos = piece.position

        # Apply move
        result = game.apply_move(chosen_move)
        events = result.events

        # Only track stats for our strategy (player 0)
        if current_idx == agent_idx:
            stats.total_moves += 1
            episode_moves += 1

            # Compute reward breakdown
            reward_breakdown = compute_move_reward_breakdown(
                old_pos, chosen_move.new_pos, events
            )
            move_reward = sum(reward_breakdown.values())
            episode_reward += move_reward

            # Update event counts
            if events.exited_home:
                stats.exit_yard_count += 1
            if events.knockouts:
                stats.capture_count += len(events.knockouts)
            if events.finished:
                stats.finish_count += 1
            if events.blockades:
                stats.blockade_count += 1
            if events.hit_blockade:
                stats.hit_blockade_count += 1

            # Update reward totals
            stats.reward_from_progress += reward_breakdown["progress"]
            stats.reward_from_capture += reward_breakdown["capture"]
            stats.reward_from_finish += reward_breakdown["finish"]
            stats.reward_from_exit_home += reward_breakdown["exit_home"]
            stats.reward_from_blockade += reward_breakdown["blockade"]
            stats.reward_from_hit_blockade += reward_breakdown["hit_blockade"]

            # Determine move category
            move_category = "other"
            if events.finished:
                move_category = "finish"
            elif events.knockouts:
                move_category = "capture"
            elif events.exited_home:
                move_category = "exit_yard"
            elif events.blockades:
                move_category = "blockade"
            else:
                # Check if entered safe zone
                new_pos = chosen_move.new_pos
                if new_pos in [1, 9, 14, 22, 27, 35, 40, 48] or (52 <= new_pos <= 56):
                    move_category = "enter_safe"
                    stats.enter_safe_count += 1
                else:
                    move_category = "progress"

            stats.category_counts[move_category] += 1
            stats.category_rewards[move_category].append(move_reward)

            # Store detailed move info
            if collect_detailed:
                move_stats = MoveStats(
                    episode=episode_num,
                    step=step,
                    dice_roll=dice,
                    piece_id=chosen_move.piece_id,
                    old_pos=old_pos,
                    new_pos=chosen_move.new_pos,
                    exits_yard=events.exited_home,
                    captures=bool(events.knockouts),
                    finishes=events.finished,
                    forms_blockade=events.blockades,
                    hit_blockade=events.hit_blockade,
                    reward_total=move_reward,
                    reward_progress=reward_breakdown["progress"],
                    reward_capture=reward_breakdown["capture"],
                    reward_finish=reward_breakdown["finish"],
                    reward_exit_home=reward_breakdown["exit_home"],
                    reward_blockade=reward_breakdown["blockade"],
                    reward_hit_blockade=reward_breakdown["hit_blockade"],
                    move_category=move_category,
                )
                stats.moves.append(move_stats)

            step += 1

        # Track when our pieces get captured by opponents
        if current_idx != agent_idx and events.knockouts:
            for ko in events.knockouts:
                if ko.player == agent_idx:
                    stats.got_captured_count += 1
                    stats.reward_from_got_captured += reward_config.got_capture
                    episode_reward += reward_config.got_capture

        turn_count += 1

    # Check win/lose
    won = game.players[agent_idx].check_won()
    if won:
        stats.wins += 1
        stats.reward_from_win += reward_config.win
        episode_reward += reward_config.win
    elif any(p.check_won() for p in game.players):
        stats.reward_from_lose += reward_config.lose
        episode_reward += reward_config.lose

    # Count finished pieces
    finished_pieces = sum(1 for p in game.players[agent_idx].pieces if p.position == 57)
    stats.total_pieces_finished += finished_pieces

    stats.total_reward += episode_reward
    stats.episode_rewards.append(episode_reward)
    stats.episode_moves.append(episode_moves)
    stats.total_episodes += 1

    return won


def print_stats(stats: StrategyAnalysis) -> None:
    """Print formatted strategy statistics."""
    print("\n" + "=" * 80)
    print(f"HEURISTIC STRATEGY ANALYSIS: {stats.strategy_name.upper()}")
    print("=" * 80)

    # Basic stats
    print(f"\nTotal Episodes: {stats.total_episodes}")
    print(f"Total Moves: {stats.total_moves}")
    print(
        f"Win Rate: {stats.wins / stats.total_episodes:.1%}"
        if stats.total_episodes > 0
        else "N/A"
    )
    print(
        f"Avg Moves per Episode: {stats.total_moves / stats.total_episodes:.1f}"
        if stats.total_episodes > 0
        else "N/A"
    )
    print(
        f"Avg Pieces Finished: {stats.total_pieces_finished / stats.total_episodes:.2f}"
        if stats.total_episodes > 0
        else "N/A"
    )

    # Event counts table
    print("\n" + "-" * 60)
    print("EVENT COUNTS")
    print("-" * 60)
    event_data = [
        [
            "Exit Yard",
            stats.exit_yard_count,
            (
                f"{stats.exit_yard_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
        [
            "Captures Made",
            stats.capture_count,
            (
                f"{stats.capture_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
        [
            "Got Captured",
            stats.got_captured_count,
            (
                f"{stats.got_captured_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
        [
            "Finished Piece",
            stats.finish_count,
            (
                f"{stats.finish_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
        [
            "Formed Blockade",
            stats.blockade_count,
            (
                f"{stats.blockade_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
        [
            "Hit Blockade",
            stats.hit_blockade_count,
            (
                f"{stats.hit_blockade_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
        [
            "Entered Safe",
            stats.enter_safe_count,
            (
                f"{stats.enter_safe_count / stats.total_moves:.1%}"
                if stats.total_moves > 0
                else "N/A"
            ),
        ],
    ]
    print(tabulate(event_data, headers=["Event", "Count", "Rate"], tablefmt="simple"))

    # Reward breakdown
    print("\n" + "-" * 60)
    print("REWARD BREAKDOWN")
    print("-" * 60)
    reward_data = [
        [
            "Progress",
            f"{stats.reward_from_progress:.2f}",
            f"{stats.reward_from_progress / stats.total_episodes:.3f}",
        ],
        [
            "Capture",
            f"{stats.reward_from_capture:.2f}",
            f"{stats.reward_from_capture / stats.total_episodes:.3f}",
        ],
        [
            "Got Captured",
            f"{stats.reward_from_got_captured:.2f}",
            f"{stats.reward_from_got_captured / stats.total_episodes:.3f}",
        ],
        [
            "Finish",
            f"{stats.reward_from_finish:.2f}",
            f"{stats.reward_from_finish / stats.total_episodes:.3f}",
        ],
        [
            "Exit Home",
            f"{stats.reward_from_exit_home:.2f}",
            f"{stats.reward_from_exit_home / stats.total_episodes:.3f}",
        ],
        [
            "Blockade",
            f"{stats.reward_from_blockade:.2f}",
            f"{stats.reward_from_blockade / stats.total_episodes:.3f}",
        ],
        [
            "Hit Blockade",
            f"{stats.reward_from_hit_blockade:.2f}",
            f"{stats.reward_from_hit_blockade / stats.total_episodes:.3f}",
        ],
        [
            "Win Bonus",
            f"{stats.reward_from_win:.2f}",
            f"{stats.reward_from_win / stats.total_episodes:.3f}",
        ],
        [
            "Lose Penalty",
            f"{stats.reward_from_lose:.2f}",
            f"{stats.reward_from_lose / stats.total_episodes:.3f}",
        ],
        ["---", "---", "---"],
        [
            "TOTAL",
            f"{stats.total_reward:.2f}",
            f"{stats.total_reward / stats.total_episodes:.3f}",
        ],
    ]
    print(
        tabulate(
            reward_data, headers=["Source", "Total", "Per Episode"], tablefmt="simple"
        )
    )

    # Move category breakdown
    print("\n" + "-" * 60)
    print("MOVE CATEGORY BREAKDOWN")
    print("-" * 60)
    category_data = []
    for cat, count in sorted(stats.category_counts.items(), key=lambda x: -x[1]):
        rewards = stats.category_rewards[cat]
        avg_reward = np.mean(rewards) if rewards else 0
        category_data.append(
            [
                cat.replace("_", " ").title(),
                count,
                f"{count / stats.total_moves:.1%}" if stats.total_moves > 0 else "N/A",
                f"{avg_reward:.4f}",
            ]
        )
    if category_data:
        print(
            tabulate(
                category_data,
                headers=["Category", "Count", "Frequency", "Avg Reward"],
                tablefmt="simple",
            )
        )

    # Episode statistics
    print("\n" + "-" * 60)
    print("EPISODE STATISTICS")
    print("-" * 60)
    if stats.episode_rewards:
        rewards = np.array(stats.episode_rewards)
        moves = np.array(stats.episode_moves)
        print(f"Avg Episode Reward: {rewards.mean():.2f} (std: {rewards.std():.2f})")
        print(f"Min/Max Episode Reward: {rewards.min():.2f} / {rewards.max():.2f}")
        print(f"Avg Episode Moves: {moves.mean():.1f} (std: {moves.std():.1f})")

    print("\n" + "=" * 80)


def print_reward_config_comparison(stats: StrategyAnalysis) -> None:
    """Print comparison of actual rewards vs config values."""
    print("\n" + "=" * 80)
    print("REWARD CONFIG VS ACTUAL")
    print("=" * 80)

    print("\nReward Config Values (COEF=5):")
    print(f"  finish: {reward_config.finish}")
    print(f"  capture: {reward_config.capture}")
    print(f"  got_capture: {reward_config.got_capture}")
    print(f"  blockade: {reward_config.blockade}")
    print(f"  hit_blockade: {reward_config.hit_blockade}")
    print(f"  exit_home: {reward_config.exit_home}")
    print(f"  progress: {reward_config.progress}")
    print(f"  win: {reward_config.win}")
    print(f"  lose: {reward_config.lose}")

    print("\nActual Contribution per Episode:")
    print(
        f"  finish: {stats.reward_from_finish / stats.total_episodes:.3f} ({stats.finish_count} events)"
    )
    print(
        f"  capture: {stats.reward_from_capture / stats.total_episodes:.3f} ({stats.capture_count} events)"
    )
    print(
        f"  got_capture: {stats.reward_from_got_captured / stats.total_episodes:.3f} ({stats.got_captured_count} events)"
    )
    print(
        f"  blockade: {stats.reward_from_blockade / stats.total_episodes:.3f} ({stats.blockade_count} events)"
    )
    print(
        f"  hit_blockade: {stats.reward_from_hit_blockade / stats.total_episodes:.3f} ({stats.hit_blockade_count} events)"
    )
    print(
        f"  exit_home: {stats.reward_from_exit_home / stats.total_episodes:.3f} ({stats.exit_yard_count} events)"
    )
    print(
        f"  progress: {stats.reward_from_progress / stats.total_episodes:.3f} (~{stats.total_moves / stats.total_episodes:.0f} moves)"
    )
    print(
        f"  win/lose: {(stats.reward_from_win + stats.reward_from_lose) / stats.total_episodes:.3f}"
    )

    # Highlight imbalances
    print("\n" + "-" * 60)
    print("REWARD SIGNAL ANALYSIS")
    print("-" * 60)

    total_positive = (
        stats.reward_from_progress
        + stats.reward_from_capture
        + stats.reward_from_finish
        + stats.reward_from_exit_home
        + stats.reward_from_blockade
        + stats.reward_from_win
    )
    total_negative = (
        abs(stats.reward_from_got_captured)
        + abs(stats.reward_from_hit_blockade)
        + abs(stats.reward_from_lose)
    )

    print(f"Total Positive Rewards: {total_positive:.2f}")
    print(f"Total Negative Rewards: {total_negative:.2f}")
    print(f"Net Reward: {stats.total_reward:.2f}")
    print(
        f"Positive/Negative Ratio: {total_positive / total_negative:.2f}"
        if total_negative > 0
        else "N/A"
    )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze heuristic strategy patterns and rewards"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=available_strategies(),
        help="Heuristic strategy to analyze",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to analyze"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="defensive,cautious,killer",
        help="Comma-separated opponent strategies",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--detailed", action="store_true", help="Collect detailed move-by-move data"
    )
    parser.add_argument(
        "--export-csv", type=str, default=None, help="Export detailed moves to CSV"
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    opponents = [s.strip() for s in args.opponents.split(",")]
    logger.info(f"Analyzing strategy: {args.strategy}")
    logger.info(f"Opponents: {opponents}")

    # Run analysis
    stats = StrategyAnalysis(strategy_name=args.strategy)
    logger.info(f"Running {args.episodes} episodes...")

    for ep in range(args.episodes):
        run_analysis_episode(
            args.strategy,
            opponents,
            ep,
            stats,
            collect_detailed=args.detailed,
        )
        if (ep + 1) % 20 == 0:
            logger.info(f"Completed {ep + 1}/{args.episodes} episodes")

    # Print results
    print_stats(stats)
    print_reward_config_comparison(stats)

    # Export to CSV if requested
    if args.export_csv and stats.moves:
        with open(args.export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "step",
                    "dice",
                    "piece_id",
                    "old_pos",
                    "new_pos",
                    "category",
                    "exits_yard",
                    "captures",
                    "finishes",
                    "blockade",
                    "hit_blockade",
                    "reward_total",
                    "reward_progress",
                    "reward_capture",
                    "reward_finish",
                    "reward_exit_home",
                    "reward_blockade",
                    "reward_hit_blockade",
                ]
            )
            for m in stats.moves:
                writer.writerow(
                    [
                        m.episode,
                        m.step,
                        m.dice_roll,
                        m.piece_id,
                        m.old_pos,
                        m.new_pos,
                        m.move_category,
                        int(m.exits_yard),
                        int(m.captures),
                        int(m.finishes),
                        int(m.forms_blockade),
                        int(m.hit_blockade),
                        f"{m.reward_total:.4f}",
                        f"{m.reward_progress:.4f}",
                        f"{m.reward_capture:.4f}",
                        f"{m.reward_finish:.4f}",
                        f"{m.reward_exit_home:.4f}",
                        f"{m.reward_blockade:.4f}",
                        f"{m.reward_hit_blockade:.4f}",
                    ]
                )
        logger.info(f"Exported {len(stats.moves)} moves to {args.export_csv}")


if __name__ == "__main__":
    main()
