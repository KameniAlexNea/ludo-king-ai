#!/usr/bin/env python3
"""Analyze RL agent strategy by tracking move choices and comparing to optimal plays.

This script runs episodes with a trained model and collects detailed statistics
about the agent's decision-making patterns:
- How often it prioritizes different move types (exit home, capture, finish, safe moves)
- Comparison of chosen action value vs alternative action values
- Move outcome analysis (what happened after the choice)

Usage:
    python tools/analyze_strategy.py --model-path training/models/best_model.zip --episodes 100
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from sb3_contrib import MaskablePPO
from tabulate import tabulate

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king import config as king_config
from ludo_rl.ludo_king.game import Game, Move
from ludo_rl.strategy.features import build_move_options
from ludo_rl.strategy.types import MoveOption

load_dotenv()


@dataclass
class MoveAnalysis:
    """Analysis of a single move decision."""

    episode: int
    step: int
    dice_roll: int
    chosen_action: int
    num_legal_moves: int
    # Move characteristics
    chosen_exits_yard: bool = False
    chosen_captures: bool = False
    chosen_finishes: bool = False
    chosen_enters_safe: bool = False
    chosen_forms_blockade: bool = False
    chosen_leaves_safe: bool = False
    chosen_progress: int = 0
    chosen_risk: float = 0.0
    chosen_distance_to_goal: int = 0
    # Alternatives available
    could_exit_yard: bool = False
    could_capture: bool = False
    could_finish: bool = False
    could_enter_safe: bool = False
    could_form_blockade: bool = False
    # Best alternative info
    best_capture_piece: int = -1
    best_finish_piece: int = -1
    best_safe_piece: int = -1
    # Value analysis
    action_probs: Optional[np.ndarray] = None
    chosen_prob: float = 0.0
    max_prob: float = 0.0
    value_estimate: float = 0.0
    # Outcome
    actual_reward: float = 0.0
    # Move category
    move_category: str = "other"


@dataclass
class StrategyStats:
    """Aggregated statistics about agent strategy."""

    total_moves: int = 0
    total_episodes: int = 0
    wins: int = 0

    # Choice frequencies (when opportunity exists)
    exit_yard_opportunities: int = 0
    exit_yard_chosen: int = 0
    capture_opportunities: int = 0
    capture_chosen: int = 0
    finish_opportunities: int = 0
    finish_chosen: int = 0
    safe_opportunities: int = 0
    safe_chosen: int = 0
    blockade_opportunities: int = 0
    blockade_chosen: int = 0
    left_safe_count: int = 0  # Times agent chose to leave a safe zone

    # Risk analysis
    total_risk_taken: float = 0.0
    high_risk_moves: int = 0  # risk > 0.3
    safe_alternative_ignored: int = 0  # Chose risky move when safe was available

    # Progress analysis
    total_progress: int = 0
    max_progress_chosen: int = 0  # Times agent chose the maximum progress move
    max_progress_opportunities: int = 0

    # Dice-specific stats
    six_rolls: int = 0
    six_exit_yard: int = 0  # Chose to exit yard on 6

    # Value estimation analysis
    chosen_prob_sum: float = 0.0
    suboptimal_choices: int = 0  # Chose action with < max probability
    value_estimates: List[float] = field(default_factory=list)
    actual_rewards: List[float] = field(default_factory=list)

    # Move category counts
    category_counts: Counter = field(default_factory=Counter)
    category_rewards: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Detailed move tracking
    moves: List[MoveAnalysis] = field(default_factory=list)


def classify_move(move: Move, game: Game, agent_idx: int) -> dict:
    """Classify a move by its characteristics."""
    player = game.players[agent_idx]
    piece = player.pieces[move.piece_id]
    old_pos = piece.position
    new_pos = move.new_pos

    return {
        "exits_yard": old_pos == 0 and new_pos > 0,
        "finishes": new_pos == king_config.PATH_LENGTH - 1,
        "enters_home_column": (
            old_pos < king_config.HOME_COLUMN_START
            and new_pos >= king_config.HOME_COLUMN_START
        ),
        "progress": new_pos - old_pos if old_pos > 0 else new_pos,
    }


def analyze_move_options(
    board_tensor: np.ndarray,
    dice: int,
    action_mask: np.ndarray,
    legal_moves: List[Move],
    game: Game,
) -> List[MoveOption]:
    """Build MoveOption objects for all legal moves."""
    # Build move_choices dict format expected by build_move_options
    move_choices = []
    for mv in legal_moves:
        player = game.players[mv.player_index]
        piece = player.pieces[mv.piece_id]
        move_choices.append({"piece": piece, "new_pos": mv.new_pos})

    # Pad to 4 moves if needed
    while len(move_choices) < 4:
        move_choices.append(None)

    ctx = build_move_options(board_tensor, dice, action_mask, move_choices)
    return ctx.moves


def run_analysis_episode(
    env: LudoEnv,
    model: MaskablePPO,
    episode_num: int,
    stats: StrategyStats,
    deterministic: bool = False,
    collect_detailed: bool = True,
) -> bool:
    """Run one episode and collect strategy statistics. Returns True if agent won."""
    obs, info = env.reset()
    terminated = False
    truncated = False
    step = 0
    won = False

    while not terminated and not truncated and env.current_turn < king_config.MAX_TURNS:
        mask = env.action_masks()
        if mask is None or not np.any(mask):
            # No legal moves - skip turn
            obs, reward, terminated, truncated, info = env.step(0)
            continue

        # Get model prediction with action probabilities
        action, _ = model.predict(
            obs, action_masks=mask[None, ...], deterministic=deterministic
        )
        action = int(np.asarray(action).item())

        # Get action probabilities and value estimate for analysis
        import torch

        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        model.policy.set_training_mode(False)
        with torch.no_grad():
            # Get distribution and value
            distribution = model.policy.get_distribution(obs_tensor)
            value = model.policy.predict_values(obs_tensor).cpu().numpy().flatten()[0]
            # Apply action mask
            probs = distribution.distribution.probs.cpu().numpy().flatten()
            # Zero out invalid actions for analysis
            probs = probs * mask.astype(float)
            if probs.sum() > 0:
                probs = probs / probs.sum()

        # Analyze available moves
        legal_moves = env.game.legal_moves(env.agent_index, env.current_dice_roll)
        dice = env.current_dice_roll
        stats.total_moves += 1

        # Build board tensor for move analysis
        board_tensor = env.game.board.build_tensor(
            int(env.game.players[env.agent_index].color)
        )
        move_options = analyze_move_options(
            board_tensor, dice, mask, legal_moves, env.game
        )

        # Track opportunities and choices
        chosen_option = None
        for opt in move_options:
            if opt.piece_id == action:
                chosen_option = opt
                break

        if chosen_option is None and len(move_options) > 0:
            # Fallback: use first legal option
            chosen_option = move_options[0]

        # Analyze opportunities
        can_exit = any(m.current_pos == 0 and m.new_pos > 0 for m in move_options)
        can_capture = any(m.can_capture for m in move_options)
        can_finish = any(m.enters_home for m in move_options)
        can_safe = any(m.enters_safe_zone for m in move_options)
        can_blockade = any(m.forms_blockade for m in move_options)
        max_progress = max((m.progress for m in move_options), default=0)

        # Update stats
        if can_exit:
            stats.exit_yard_opportunities += 1
            if chosen_option and chosen_option.current_pos == 0:
                stats.exit_yard_chosen += 1

        if can_capture:
            stats.capture_opportunities += 1
            if chosen_option and chosen_option.can_capture:
                stats.capture_chosen += 1

        if can_finish:
            stats.finish_opportunities += 1
            if chosen_option and chosen_option.enters_home:
                stats.finish_chosen += 1

        if can_safe:
            stats.safe_opportunities += 1
            if chosen_option and chosen_option.enters_safe_zone:
                stats.safe_chosen += 1

        if can_blockade:
            stats.blockade_opportunities += 1
            if chosen_option and chosen_option.forms_blockade:
                stats.blockade_chosen += 1

        if chosen_option:
            if chosen_option.leaving_safe_zone:
                stats.left_safe_count += 1
            stats.total_progress += chosen_option.progress
            stats.total_risk_taken += chosen_option.risk
            if chosen_option.risk > 0.3:
                stats.high_risk_moves += 1
            if (
                can_safe
                and not chosen_option.enters_safe_zone
                and chosen_option.risk > 0.1
            ):
                stats.safe_alternative_ignored += 1

        # Progress analysis
        if max_progress > 0:
            stats.max_progress_opportunities += 1
            if chosen_option and chosen_option.progress == max_progress:
                stats.max_progress_chosen += 1

        # Dice 6 analysis
        if dice == 6:
            stats.six_rolls += 1
            if chosen_option and chosen_option.current_pos == 0:
                stats.six_exit_yard += 1

        # Value analysis
        chosen_prob = probs[action] if action < len(probs) else 0.0
        max_prob = probs[mask].max() if mask.any() else 0.0
        stats.chosen_prob_sum += chosen_prob
        stats.value_estimates.append(value)
        if chosen_prob < max_prob - 0.01:  # Allow small tolerance
            stats.suboptimal_choices += 1

        # Determine move category for reward tracking
        move_category = "other"
        if chosen_option:
            if chosen_option.enters_home:
                move_category = "finish"
            elif chosen_option.can_capture:
                move_category = "capture"
            elif chosen_option.current_pos == 0:
                move_category = "exit_yard"
            elif chosen_option.enters_safe_zone:
                move_category = "enter_safe"
            elif chosen_option.forms_blockade:
                move_category = "blockade"
            elif chosen_option.leaving_safe_zone:
                move_category = "leave_safe"
            else:
                move_category = "progress"
        stats.category_counts[move_category] += 1

        # Find best alternatives
        best_capture = next((m.piece_id for m in move_options if m.can_capture), -1)
        best_finish = next((m.piece_id for m in move_options if m.enters_home), -1)
        best_safe = next((m.piece_id for m in move_options if m.enters_safe_zone), -1)

        # Store detailed move info if requested
        if collect_detailed:
            analysis = MoveAnalysis(
                episode=episode_num,
                step=step,
                dice_roll=dice,
                chosen_action=action,
                num_legal_moves=len(legal_moves),
                chosen_exits_yard=(
                    chosen_option.current_pos == 0 if chosen_option else False
                ),
                chosen_captures=chosen_option.can_capture if chosen_option else False,
                chosen_finishes=chosen_option.enters_home if chosen_option else False,
                chosen_enters_safe=(
                    chosen_option.enters_safe_zone if chosen_option else False
                ),
                chosen_forms_blockade=(
                    chosen_option.forms_blockade if chosen_option else False
                ),
                chosen_leaves_safe=(
                    chosen_option.leaving_safe_zone if chosen_option else False
                ),
                chosen_progress=chosen_option.progress if chosen_option else 0,
                chosen_risk=chosen_option.risk if chosen_option else 0.0,
                chosen_distance_to_goal=(
                    chosen_option.distance_to_goal if chosen_option else 0
                ),
                could_exit_yard=can_exit,
                could_capture=can_capture,
                could_finish=can_finish,
                could_enter_safe=can_safe,
                could_form_blockade=can_blockade,
                best_capture_piece=best_capture,
                best_finish_piece=best_finish,
                best_safe_piece=best_safe,
                action_probs=probs.copy(),
                chosen_prob=chosen_prob,
                max_prob=max_prob,
                value_estimate=value,
                move_category=move_category,
            )
            stats.moves.append(analysis)

        # Take the action
        obs, reward, terminated, truncated, info = env.step(action)
        stats.actual_rewards.append(reward)
        stats.category_rewards[move_category].append(reward)
        if collect_detailed and stats.moves:
            stats.moves[-1].actual_reward = reward
        step += 1

    # Check win
    won = info.get("win", False) if isinstance(info, dict) else False
    if won:
        stats.wins += 1
    stats.total_episodes += 1

    return won


def print_stats(stats: StrategyStats) -> None:
    """Print formatted strategy statistics."""
    print("\n" + "=" * 80)
    print("STRATEGY ANALYSIS REPORT")
    print("=" * 80)

    # Basic stats
    print(f"\nTotal Episodes: {stats.total_episodes}")
    print(f"Total Moves Analyzed: {stats.total_moves}")
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

    # Move priority table
    print("\n" + "-" * 60)
    print("MOVE PRIORITY ANALYSIS (when opportunity exists)")
    print("-" * 60)

    priority_data = []
    if stats.exit_yard_opportunities > 0:
        priority_data.append(
            [
                "Exit Yard",
                stats.exit_yard_opportunities,
                stats.exit_yard_chosen,
                f"{stats.exit_yard_chosen / stats.exit_yard_opportunities:.1%}",
            ]
        )
    if stats.capture_opportunities > 0:
        priority_data.append(
            [
                "Capture",
                stats.capture_opportunities,
                stats.capture_chosen,
                f"{stats.capture_chosen / stats.capture_opportunities:.1%}",
            ]
        )
    if stats.finish_opportunities > 0:
        priority_data.append(
            [
                "Finish Piece",
                stats.finish_opportunities,
                stats.finish_chosen,
                f"{stats.finish_chosen / stats.finish_opportunities:.1%}",
            ]
        )
    if stats.safe_opportunities > 0:
        priority_data.append(
            [
                "Enter Safe Zone",
                stats.safe_opportunities,
                stats.safe_chosen,
                f"{stats.safe_chosen / stats.safe_opportunities:.1%}",
            ]
        )
    if stats.blockade_opportunities > 0:
        priority_data.append(
            [
                "Form Blockade",
                stats.blockade_opportunities,
                stats.blockade_chosen,
                f"{stats.blockade_chosen / stats.blockade_opportunities:.1%}",
            ]
        )

    if priority_data:
        print(
            tabulate(
                priority_data,
                headers=["Move Type", "Opportunities", "Chosen", "Rate"],
                tablefmt="simple",
            )
        )

    # Risk analysis
    print("\n" + "-" * 60)
    print("RISK ANALYSIS")
    print("-" * 60)
    print(f"Total Risk Accumulated: {stats.total_risk_taken:.2f}")
    print(
        f"Avg Risk per Move: {stats.total_risk_taken / stats.total_moves:.3f}"
        if stats.total_moves > 0
        else "N/A"
    )
    print(
        f"High Risk Moves (>0.3): {stats.high_risk_moves} ({stats.high_risk_moves / stats.total_moves:.1%})"
        if stats.total_moves > 0
        else "N/A"
    )
    print(f"Times Left Safe Zone: {stats.left_safe_count}")
    print(f"Ignored Safe Alternative: {stats.safe_alternative_ignored}")

    # Progress analysis
    print("\n" + "-" * 60)
    print("PROGRESS ANALYSIS")
    print("-" * 60)
    print(f"Total Progress Made: {stats.total_progress}")
    print(
        f"Avg Progress per Move: {stats.total_progress / stats.total_moves:.2f}"
        if stats.total_moves > 0
        else "N/A"
    )
    if stats.max_progress_opportunities > 0:
        print(
            f"Chose Max Progress: {stats.max_progress_chosen}/{stats.max_progress_opportunities} ({stats.max_progress_chosen / stats.max_progress_opportunities:.1%})"
        )

    # Dice 6 analysis
    print("\n" + "-" * 60)
    print("DICE 6 ANALYSIS")
    print("-" * 60)
    if stats.six_rolls > 0:
        print(f"Total Dice 6 Rolls: {stats.six_rolls}")
        print(
            f"Chose Exit Yard on 6: {stats.six_exit_yard} ({stats.six_exit_yard / stats.six_rolls:.1%})"
        )

    # Value estimation
    print("\n" + "-" * 60)
    print("POLICY ANALYSIS")
    print("-" * 60)
    print(
        f"Avg Chosen Action Probability: {stats.chosen_prob_sum / stats.total_moves:.3f}"
        if stats.total_moves > 0
        else "N/A"
    )
    print(
        f"Suboptimal Choices (not max prob): {stats.suboptimal_choices} ({stats.suboptimal_choices / stats.total_moves:.1%})"
        if stats.total_moves > 0
        else "N/A"
    )

    # Value vs Reward correlation
    if stats.value_estimates and stats.actual_rewards:
        values = np.array(stats.value_estimates)
        rewards = np.array(stats.actual_rewards)
        corr = np.corrcoef(values, rewards)[0, 1] if len(values) > 1 else 0
        print(f"Avg Value Estimate: {values.mean():.3f} (std: {values.std():.3f})")
        print(f"Avg Actual Reward: {rewards.mean():.3f} (std: {rewards.std():.3f})")
        print(f"Value-Reward Correlation: {corr:.3f}")

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
                f"{count / stats.total_moves:.1%}",
                f"{avg_reward:.3f}",
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

    print("\n" + "=" * 80)


def analyze_missed_opportunities(stats: StrategyStats) -> None:
    """Analyze specific cases where agent missed good opportunities."""
    print("\n" + "=" * 80)
    print("MISSED OPPORTUNITY ANALYSIS")
    print("=" * 80)

    # Find moves where capture was available but not taken
    missed_captures = [
        m for m in stats.moves if m.could_capture and not m.chosen_captures
    ]
    print(f"\nMissed Captures: {len(missed_captures)}")
    if missed_captures[:5]:
        print("  Examples (first 5):")
        for m in missed_captures[:5]:
            print(
                f"    Episode {m.episode}, Step {m.step}: Dice={m.dice_roll}, Risk={m.chosen_risk:.2f}, "
                f"Chose prob={m.chosen_prob:.2f}, Best capture: piece {m.best_capture_piece}"
            )

    # Find moves where finish was available but not taken
    missed_finish = [m for m in stats.moves if m.could_finish and not m.chosen_finishes]
    print(f"\nMissed Finishes: {len(missed_finish)}")
    if missed_finish[:5]:
        print("  Examples (first 5):")
        for m in missed_finish[:5]:
            print(
                f"    Episode {m.episode}, Step {m.step}: Dice={m.dice_roll}, "
                f"Chose action={m.chosen_action}, Best finish: piece {m.best_finish_piece}"
            )

    # Analyze why finishes were missed
    if missed_finish:
        reasons = Counter()
        for m in missed_finish:
            if m.chosen_captures:
                reasons["chose_capture_instead"] += 1
            elif m.chosen_exits_yard:
                reasons["chose_exit_yard_instead"] += 1
            elif m.chosen_enters_safe:
                reasons["chose_safe_instead"] += 1
            else:
                reasons["other"] += 1
        print("  Reasons for missed finishes:")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    # High-confidence suboptimal moves
    bad_confidence = [
        m
        for m in stats.moves
        if m.chosen_prob > 0.5 and m.max_prob - m.chosen_prob > 0.2
    ]
    print(f"\nHigh-Confidence Wrong (chose >50% but not max): {len(bad_confidence)}")

    # Negative reward moves analysis
    negative_reward_moves = [m for m in stats.moves if m.actual_reward < -0.5]
    print(f"\nHigh Negative Reward Moves (<-0.5): {len(negative_reward_moves)}")
    if negative_reward_moves[:5]:
        print("  Examples (first 5):")
        for m in negative_reward_moves[:5]:
            print(
                f"    Episode {m.episode}, Step {m.step}: Category={m.move_category}, "
                f"Reward={m.actual_reward:.2f}, Value={m.value_estimate:.2f}"
            )

    # Distance to goal analysis
    print("\n" + "-" * 40)
    print("DISTANCE TO GOAL ANALYSIS")
    print("-" * 40)
    distances = [
        m.chosen_distance_to_goal for m in stats.moves if m.chosen_distance_to_goal > 0
    ]
    if distances:
        print(f"Avg Distance to Goal When Moving: {np.mean(distances):.1f}")
        print(
            f"Pieces at distance ≤10: {sum(1 for d in distances if d <= 10)} ({sum(1 for d in distances if d <= 10) / len(distances):.1%})"
        )
        print(
            f"Pieces at distance ≤5: {sum(1 for d in distances if d <= 5)} ({sum(1 for d in distances if d <= 5) / len(distances):.1%})"
        )

    print("=" * 80)


def analyze_value_by_category(stats: StrategyStats) -> None:
    """Analyze value estimates vs actual rewards by move category."""
    print("\n" + "=" * 80)
    print("VALUE ESTIMATION BY CATEGORY")
    print("=" * 80)

    category_value_data = []
    for cat in sorted(stats.category_counts.keys()):
        moves_in_cat = [m for m in stats.moves if m.move_category == cat]
        if not moves_in_cat:
            continue
        values = [m.value_estimate for m in moves_in_cat]
        rewards = [m.actual_reward for m in moves_in_cat]
        value_mean = np.mean(values)
        reward_mean = np.mean(rewards)
        error = value_mean - reward_mean
        category_value_data.append(
            [
                cat.replace("_", " ").title(),
                len(moves_in_cat),
                f"{value_mean:.3f}",
                f"{reward_mean:.3f}",
                f"{error:+.3f}",
            ]
        )

    if category_value_data:
        print(
            tabulate(
                category_value_data,
                headers=["Category", "Count", "Avg Value", "Avg Reward", "Error"],
                tablefmt="simple",
            )
        )

    print("=" * 80)


def print_recommendations(stats: StrategyStats) -> None:
    """Print actionable recommendations based on analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 80)

    recommendations = []

    # Finish rate
    finish_rate = (
        stats.finish_chosen / stats.finish_opportunities
        if stats.finish_opportunities > 0
        else 0
    )
    if finish_rate < 0.7:
        recommendations.append(
            f"⚠️  FINISH PRIORITY: Only {finish_rate:.0%} of finish opportunities taken. "
            f"Consider increasing finish reward or adding curriculum focusing on endgame."
        )

    # Capture rate
    capture_rate = (
        stats.capture_chosen / stats.capture_opportunities
        if stats.capture_opportunities > 0
        else 0
    )
    if capture_rate < 0.6:
        recommendations.append(
            f"⚠️  CAPTURE PRIORITY: Only {capture_rate:.0%} of capture opportunities taken. "
            f"Agent may be too risk-averse or not valuing captures enough."
        )

    # Exit yard on 6
    exit_on_6_rate = stats.six_exit_yard / stats.six_rolls if stats.six_rolls > 0 else 0
    if exit_on_6_rate < 0.6:
        recommendations.append(
            f"⚠️  EXIT STRATEGY: Only {exit_on_6_rate:.0%} of dice-6 rolls used to exit yard. "
            f"This is a critical strategic move that should be higher."
        )

    # Safe zone leaving
    leave_safe_rate = (
        stats.left_safe_count / stats.total_moves if stats.total_moves > 0 else 0
    )
    if leave_safe_rate > 0.25:
        recommendations.append(
            f"⚠️  SAFETY: Agent leaves safe zones {leave_safe_rate:.0%} of moves. "
            f"Consider if safe_position reward is sufficient."
        )

    # Value estimation
    if stats.value_estimates and stats.actual_rewards:
        values = np.array(stats.value_estimates)
        rewards = np.array(stats.actual_rewards)
        corr = np.corrcoef(values, rewards)[0, 1] if len(values) > 1 else 0
        if corr < 0.3:
            recommendations.append(
                f"⚠️  VALUE FUNCTION: Correlation between value estimates and rewards is only {corr:.2f}. "
                f"Consider training longer or adjusting vf_coef."
            )

    # Suboptimal choices
    subopt_rate = (
        stats.suboptimal_choices / stats.total_moves if stats.total_moves > 0 else 0
    )
    if subopt_rate > 0.3:
        recommendations.append(
            f"ℹ️  POLICY ENTROPY: {subopt_rate:.0%} of moves don't match max probability. "
            f"This suggests exploration or multi-modal policy. May be fine for diverse play."
        )

    # Win rate assessment
    win_rate = stats.wins / stats.total_episodes if stats.total_episodes > 0 else 0
    if win_rate < 0.25:
        recommendations.append(
            f"❌ WIN RATE: Only {win_rate:.0%} against opponents. "
            f"Baseline random is ~25% in 4-player. Agent may not be learning effectively."
        )
    elif win_rate > 0.35:
        recommendations.append(
            f"✅ WIN RATE: {win_rate:.0%} is above random baseline. Agent is learning."
        )

    if not recommendations:
        print("✅ No major issues detected. Agent shows reasonable strategy patterns.")
    else:
        for rec in recommendations:
            print(f"\n{rec}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze RL agent strategy patterns")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to analyze"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=None,
        help="Comma-separated opponent strategies (default: mixed)",
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for model inference"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--detailed", action="store_true", help="Collect detailed move-by-move analysis"
    )
    parser.add_argument(
        "--export-csv", type=str, default=None, help="Export detailed moves to CSV file"
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = MaskablePPO.load(args.model_path, device=args.device)
    model.policy.set_training_mode(False)

    # Create environment
    env = LudoEnv(use_fixed_opponents=args.opponents is not None)
    if args.opponents:
        opponents = [s.strip() for s in args.opponents.split(",")]
        env.opponents = opponents
        env._fixed_opponents_strategies = opponents

    # Run analysis
    stats = StrategyStats()
    logger.info(f"Running {args.episodes} episodes for analysis...")

    for ep in range(args.episodes):
        run_analysis_episode(
            env,
            model,
            ep,
            stats,
            deterministic=args.deterministic,
            collect_detailed=args.detailed,
        )
        if (ep + 1) % 20 == 0:
            logger.info(f"Completed {ep + 1}/{args.episodes} episodes")

    env.close()

    # Print results
    print_stats(stats)
    if args.detailed:
        analyze_missed_opportunities(stats)
        analyze_value_by_category(stats)
    print_recommendations(stats)

    # Export to CSV if requested
    if args.export_csv and stats.moves:
        with open(args.export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "step",
                    "dice",
                    "action",
                    "num_legal",
                    "category",
                    "exits_yard",
                    "captures",
                    "finishes",
                    "enters_safe",
                    "forms_blockade",
                    "leaves_safe",
                    "progress",
                    "risk",
                    "distance_to_goal",
                    "could_exit",
                    "could_capture",
                    "could_finish",
                    "could_safe",
                    "could_blockade",
                    "chosen_prob",
                    "max_prob",
                    "value_estimate",
                    "actual_reward",
                ]
            )
            for m in stats.moves:
                writer.writerow(
                    [
                        m.episode,
                        m.step,
                        m.dice_roll,
                        m.chosen_action,
                        m.num_legal_moves,
                        m.move_category,
                        int(m.chosen_exits_yard),
                        int(m.chosen_captures),
                        int(m.chosen_finishes),
                        int(m.chosen_enters_safe),
                        int(m.chosen_forms_blockade),
                        int(m.chosen_leaves_safe),
                        m.chosen_progress,
                        f"{m.chosen_risk:.3f}",
                        m.chosen_distance_to_goal,
                        int(m.could_exit_yard),
                        int(m.could_capture),
                        int(m.could_finish),
                        int(m.could_enter_safe),
                        int(m.could_form_blockade),
                        f"{m.chosen_prob:.4f}",
                        f"{m.max_prob:.4f}",
                        f"{m.value_estimate:.4f}",
                        f"{m.actual_reward:.4f}",
                    ]
                )
        logger.info(f"Exported {len(stats.moves)} moves to {args.export_csv}")


if __name__ == "__main__":
    main()
