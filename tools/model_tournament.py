"""
Tournament script for competing multiple RL model versions against each other.

Similar to tournament.py but uses trained PPO models instead of heuristic strategies.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.ludo.player import Player

POINTS_TABLE = (4, 3, 1, 0)


@dataclass
class ModelWrapper:
    """Wrapper for a loaded RL model."""

    name: str
    model: MaskablePPO
    device: str


@dataclass
class GameResult:
    index: int
    turns: int
    placements: List[str]
    points: Dict[str, int]


@dataclass
class CombinationSummary:
    participants: Tuple[str, ...]
    game_results: List[GameResult]
    totals: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tournament between multiple trained Ludo RL models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Base directory containing model checkpoints (e.g., training/ludo_models/ppo_ludo_1762453466)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="Comma-separated list of checkpoint step IDs (e.g., '1000000,2000000,5000000,10000000')",
    )
    parser.add_argument(
        "--games", type=int, default=10, help="Number of games per combination"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of heuristic strategies to include. "
            "If provided with fewer than 4 models, will mix models and strategies."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model inference (cpu/cuda)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions from models",
    )
    return parser.parse_args()


def _find_checkpoint_file(model_dir: str, step_str: str, suffix: str) -> str | None:
    """Locate a checkpoint file that contains the step string in its name."""
    target = f"_{step_str}_steps"
    for entry in os.listdir(model_dir):
        if not entry.endswith(suffix):
            continue
        if target in entry:
            return os.path.join(model_dir, entry)
    return None


def load_model_checkpoint(model_dir: str, step_id: str, device: str) -> ModelWrapper:
    """Load a model checkpoint and its VecNormalize stats.

    Args:
        model_dir: Base directory (e.g., training/ludo_models/ppo_ludo_1762453466)
        step_id: Checkpoint step ID (e.g., '1000000' or '1M')
        device: Device for inference (cpu/cuda)

    Returns:
        ModelWrapper with loaded model and normalization stats
    """
    # Normalize step_id format (handle both '1M' and '1000000')
    if step_id.upper().endswith("M"):
        # Convert '1M' to '1000000'
        multiplier = int(step_id[:-1])
        step_num = multiplier * 1_000_000
        step_str = str(step_num)
        display_name = step_id.upper()
    elif step_id.upper().endswith("K"):
        # Convert '500K' to '500000'
        multiplier = int(step_id[:-1])
        step_num = multiplier * 1_000
        step_str = str(step_num)
        display_name = step_id.upper()
    else:
        # Already in numeric format
        step_str = step_id
        step_num = int(step_id)
        # Create display name (e.g., '1000000' -> '1M')
        if step_num >= 1_000_000 and step_num % 1_000_000 == 0:
            display_name = f"{step_num // 1_000_000}M"
        elif step_num >= 1_000 and step_num % 1_000 == 0:
            display_name = f"{step_num // 1_000}K"
        else:
            display_name = step_str

    # Locate model checkpoint file (support multiple naming conventions)
    model_path = _find_checkpoint_file(model_dir, step_str, ".zip")
    if model_path is None:
        raise FileNotFoundError(
            f"Could not find model checkpoint matching pattern '*_{step_str}_steps.zip'"
        )

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = MaskablePPO.load(model_path, device=device)
    model.policy.set_training_mode(False)

    return ModelWrapper(name=display_name, model=model, device=device)


def parse_checkpoint_ids(checkpoint_str: str) -> List[str]:
    """Parse checkpoint IDs from comma-separated string.

    Args:
        checkpoint_str: String like '1M,2M,5M,10M' or '1000000,2000000,5000000'

    Returns:
        List of checkpoint ID strings
    """
    return [ckpt.strip() for ckpt in checkpoint_str.split(",") if ckpt.strip()]


def build_board_stack(game: LudoGame, player_index: int) -> np.ndarray:
    """Build board state tensor for RL model input."""
    return game.build_board_tensor(player_index)


def player_progress(player: Player) -> int:
    """Calculate total progress of all player pieces."""
    return sum(piece.position for piece in player.pieces)


def attach_model(player: Player, model_wrapper: ModelWrapper) -> None:
    """Attach an RL model to a player."""
    player.strategy_name = model_wrapper.name
    player._strategy = None  # Clear any strategy


def decide_with_model(
    model_wrapper: ModelWrapper,
    board_stack: np.ndarray,
    dice_roll: int,
    valid_moves: list[dict],
    deterministic: bool,
    rng: random.Random,
) -> dict | None:
    """Use RL model to decide on a move."""
    # Prepare observation
    obs = {
        "board": board_stack,
        "dice_roll": np.array([dice_roll - 1], dtype=np.int64),
    }

    # Create action mask
    action_mask = np.zeros(config.PIECES_PER_PLAYER, dtype=bool)
    piece_id_to_move = {}
    for move in valid_moves:
        piece_id = move["piece"].piece_id
        action_mask[piece_id] = True
        piece_id_to_move[piece_id] = move

    # Get action from model
    try:
        # Convert to format expected by model
        obs_tensor = {
            "board": obs["board"][None, ...],  # Add batch dimension
            "dice_roll": obs["dice_roll"][None, ...],
        }
        action, _state = model_wrapper.model.predict(
            obs_tensor, action_masks=action_mask[None, ...], deterministic=deterministic
        )
        piece_id = action.item()

        if piece_id in piece_id_to_move and action_mask[piece_id]:
            return piece_id_to_move[piece_id]
    except Exception as e:
        logger.warning(f"Model prediction failed: {e}")

    # Fallback to random
    return rng.choice(valid_moves) if valid_moves else None


def determine_rankings(game: LudoGame, finish_order: List[int]) -> List[int]:
    """Determine final rankings including unfinished players."""
    ordered = finish_order.copy()
    remaining = [idx for idx in range(config.NUM_PLAYERS) if idx not in ordered]
    remaining.sort(
        key=lambda idx: (
            sum(piece.position == 57 for piece in game.players[idx].pieces),
            player_progress(game.players[idx]),
        ),
        reverse=True,
    )
    ordered.extend(remaining)
    return ordered[: config.NUM_PLAYERS]


def play_game(
    participants: Sequence[str],
    models_dict: Dict[str, ModelWrapper],
    deterministic: bool,
    rng: random.Random,
    game_index: int,
) -> GameResult:
    """Play a single game with the given participants.

    participants are ModelWrapper instances.
    """
    game = LudoGame()
    game_seed = rng.randint(0, 1_000_000)
    game.rng.seed(game_seed)
    random.seed(game_seed)

    # Initialize players
    model_assignments: Dict[int, ModelWrapper] = {}

    for seat_index, (player, participant) in enumerate(
        zip(game.players, participants, strict=True)
    ):
        for piece in player.pieces:
            piece.position = 0
        player.has_finished = False
        player._strategy = None

        if participant not in models_dict:
            raise ValueError(f"Unknown participant '{participant}'")

        model_wrapper = models_dict[participant]
        attach_model(player, model_wrapper)
        model_assignments[seat_index] = model_wrapper

    finish_order: List[int] = []
    turns_taken = 0
    current_index = 0

    while turns_taken < config.MAX_TURNS and len(finish_order) < config.NUM_PLAYERS:
        player = game.players[current_index]
        if player.has_won():
            if current_index not in finish_order:
                finish_order.append(current_index)
            current_index = (current_index + 1) % config.NUM_PLAYERS
            continue

        extra_turn = True
        while extra_turn:
            dice_roll = game.roll_dice()
            valid_moves = game.get_valid_moves(current_index, dice_roll)
            if not valid_moves:
                extra_turn = False
                continue

            board_stack = build_board_stack(game, current_index)

            model_wrapper = model_assignments[current_index]
            decision = decide_with_model(
                model_wrapper,
                board_stack,
                dice_roll,
                valid_moves,
                deterministic,
                rng,
            )

            move = decision if decision is not None else rng.choice(valid_moves)

            result = game.make_move(
                current_index, move["piece"], move["new_pos"], dice_roll
            )

            if not result["events"]["move_resolved"]:
                extra_turn = False
            else:
                extra_turn = result["extra_turn"]

            if player.has_won() and current_index not in finish_order:
                finish_order.append(current_index)

        current_index = (current_index + 1) % config.NUM_PLAYERS
        turns_taken += 1

    rankings = determine_rankings(game, finish_order)
    placement_names = [game.players[idx].strategy_name for idx in rankings]

    points = {name: POINTS_TABLE[pos] for pos, name in enumerate(placement_names)}

    return GameResult(
        index=game_index,
        turns=turns_taken,
        placements=placement_names,
        points=points,
    )


def run_combination_tournament(
    participants: Sequence[str],
    models_dict: Dict[str, ModelWrapper],
    games: int,
    deterministic: bool,
    rng: random.Random,
) -> CombinationSummary:
    """Run a tournament for a specific combination of participants."""
    totals = {name: 0 for name in participants}
    results: List[GameResult] = []

    for game_index in range(1, games + 1):
        seats = list(participants)
        rng.shuffle(seats)
        result = play_game(seats, models_dict, deterministic, rng, game_index)
        results.append(result)

        for name, pts in result.points.items():
            totals[name] += pts

    return CombinationSummary(
        participants=tuple(participants),
        game_results=results,
        totals=totals,
    )


def run_league(
    participant_pool: Sequence[str],
    models_dict: Dict[str, ModelWrapper],
    games: int,
    deterministic: bool,
    rng: random.Random,
) -> tuple[Dict[str, int], Dict[str, int], List[CombinationSummary]]:
    """Run a full league tournament with all combinations."""
    league_totals = {name: 0 for name in participant_pool}
    games_played = {name: 0 for name in participant_pool}
    combination_summaries: List[CombinationSummary] = []

    combos = list(combinations(participant_pool, 4))
    logger.info(f"Running league with {len(combos)} combinations of 4 participants...")

    for combo_idx, combo in enumerate(combos, start=1):
        logger.info(f"[{combo_idx}/{len(combos)}] Running: {', '.join(combo)}")
        summary = run_combination_tournament(
            combo, models_dict, games, deterministic, rng
        )
        combination_summaries.append(summary)

        for name, pts in summary.totals.items():
            league_totals[name] += pts
            games_played[name] += len(summary.game_results)

        combo_table = sorted(
            summary.totals.items(), key=lambda item: (-item[1], item[0])
        )
        combo_results = ", ".join(
            f"{name}: {points} pts" for name, points in combo_table
        )
        logger.info(f"[{combo_idx}/{len(combos)}] Results: {combo_results}")

    return league_totals, games_played, combination_summaries


def print_summary(
    participant_pool: Sequence[str],
    league_totals: Dict[str, int],
    games_played: Dict[str, int],
    combination_summaries: Sequence[CombinationSummary],
) -> None:
    """Print tournament summary."""
    print("\n" + "=" * 80)
    print("TOURNAMENT SUMMARY")
    print("=" * 80)
    print("Participants:", ", ".join(participant_pool))
    print()

    for combo_index, summary in enumerate(combination_summaries, start=1):
        combo_label = ", ".join(summary.participants)
        print(f"Combination {combo_index:03d}: {combo_label}")
        combo_table = sorted(
            summary.totals.items(), key=lambda item: (-item[1], item[0])
        )
        print("  Points collected:")
        for rank, (name, points) in enumerate(combo_table, start=1):
            print(f"    {rank}. {name:20s} {points:3d} pts")
        print()

    print("=" * 80)
    print("LEAGUE STANDINGS")
    print("=" * 80)
    for rank, (name, points) in enumerate(
        sorted(league_totals.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        avg_points = points / games_played[name] if games_played[name] else 0
        print(
            f"  {rank:2d}. {name:20s} {points:5d} pts across {games_played[name]:4d} games "
            f"(avg: {avg_points:.2f} pts/game)"
        )

    winner = max(league_totals.items(), key=lambda item: (item[1], item[0]))[0]
    print()
    print(f"🏆 League winner: {winner}")
    print("=" * 80)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.strategies:
        raise SystemExit("Heuristic strategies are not supported in model tournaments.")

    # Parse checkpoint IDs
    checkpoint_ids = parse_checkpoint_ids(args.checkpoints)
    if not checkpoint_ids:
        raise SystemExit("No checkpoint IDs provided")

    # Load models
    print(f"Loading models from {args.model_dir}...")
    models_dict: Dict[str, ModelWrapper] = {}

    for checkpoint_id in checkpoint_ids:
        try:
            model_wrapper = load_model_checkpoint(
                args.model_dir, checkpoint_id, args.device
            )
            models_dict[model_wrapper.name] = model_wrapper
            print(f"  ✓ Loaded checkpoint {model_wrapper.name}")
        except Exception as e:
            print(f"  ✗ Error loading checkpoint {checkpoint_id}: {e}")
            raise SystemExit(f"Failed to load checkpoint {checkpoint_id}")

    print(f"\nSuccessfully loaded {len(models_dict)} model checkpoints.")
    print()

    # Build participant pool
    participant_pool: List[str] = list(models_dict.keys())

    if len(participant_pool) < 4:
        raise SystemExit(
            f"Need at least 4 participants for tournament. Got {len(participant_pool)}."
        )

    print(f"Tournament participants ({len(participant_pool)}):")
    for participant in participant_pool:
        participant_type = "Model" if participant in models_dict else "Strategy"
        print(f"  - {participant} ({participant_type})")
    print()

    # Run tournament
    league_totals, games_played, combo_summaries = run_league(
        participant_pool, models_dict, args.games, args.deterministic, rng
    )

    # Print results
    print_summary(participant_pool, league_totals, games_played, combo_summaries)


if __name__ == "__main__":
    main()
