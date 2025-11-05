from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from sb3_contrib import MaskablePPO

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.strategy.registry import STRATEGY_REGISTRY

load_dotenv()


@dataclass(slots=True)
class SampleBatch:
    board: np.ndarray
    dice: np.ndarray
    mask: np.ndarray
    action: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect behaviour-cloning data from scripted strategies and fine-tune "
            "a PPO checkpoint via supervised imitation."
        )
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Strategy name to imitate (must appear in opponents list).",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        required=True,
        help="Comma-separated list of four strategies to play each game with.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of games to simulate for data collection.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap on collected samples (useful for memory control).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the PPO checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Where to store the behaviour-cloned checkpoint.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=5,
        help="Number of behaviour-cloning epochs to run after data collection.",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=1024,
        help="Batch size for behaviour cloning updates.",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=3e-4,
        help="Learning rate for the behaviour cloning optimiser.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use for imitation updates (cpu / cuda / mps).",
    )
    parser.add_argument(
        "--dataset-out",
        type=str,
        default=None,
        help="Optional path to dump the collected dataset as NPZ for later use.",
    )
    return parser.parse_args()


def validate_strategies(opponents: Sequence[str]) -> None:
    unknown = [name for name in opponents if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown strategy names: {', '.join(unknown)}")


def build_board_stack(game: LudoGame, player_index: int) -> np.ndarray:
    board_state = game.get_board_state(player_index)
    zero_channel = np.zeros(config.PATH_LENGTH, dtype=np.float32)
    return np.stack(
        [
            np.asarray(board_state["my_pieces"], dtype=np.float32),
            np.asarray(board_state["opp1_pieces"], dtype=np.float32),
            np.asarray(board_state["opp2_pieces"], dtype=np.float32),
            np.asarray(board_state["opp3_pieces"], dtype=np.float32),
            np.asarray(board_state["safe_zones"], dtype=np.float32),
            zero_channel,
            zero_channel,
            zero_channel,
            zero_channel,
            zero_channel,
        ],
        dtype=np.float32,
    )


def prepare_players(game: LudoGame, strategies: Sequence[str]) -> None:
    for player, strategy_name in zip(game.players, strategies, strict=True):
        player.strategy_name = strategy_name
        player._strategy = None  # Reset cached strategy instance
        player.has_finished = False
        for piece in player.pieces:
            piece.position = 0


def action_mask_from_moves(valid_moves: Sequence[Dict]) -> np.ndarray:
    mask = np.zeros(config.PIECES_PER_PLAYER, dtype=bool)
    for move in valid_moves:
        mask[move["piece"].piece_id] = True
    return mask


def collect_teacher_samples(
    teacher: str,
    opponents: Sequence[str],
    games: int,
    rng: random.Random,
    max_samples: int | None,
) -> SampleBatch:
    records_board: List[np.ndarray] = []
    records_dice: List[np.ndarray] = []
    records_mask: List[np.ndarray] = []
    records_action: List[int] = []

    for game_idx in range(games):
        strategies = list(opponents)
        rng.shuffle(strategies)

        if teacher not in strategies:
            continue  # Should not happen but guard anyway

        game = LudoGame()
        prepare_players(game, strategies)

        finish_order: List[int] = []
        current_index = 0
        turns_taken = 0

        while turns_taken < config.MAX_TURNS and len(finish_order) < config.NUM_PLAYERS:
            player = game.players[current_index]
            if player.has_won():
                if current_index not in finish_order:
                    finish_order.append(current_index)
                current_index = (current_index + 1) % config.NUM_PLAYERS
                continue

            extra_turn = True
            while extra_turn and len(finish_order) < config.NUM_PLAYERS:
                dice_roll = game.roll_dice()
                valid_moves = game.get_valid_moves(current_index, dice_roll)

                if not valid_moves:
                    extra_turn = False
                    continue

                board_stack = build_board_stack(game, current_index)
                mask = action_mask_from_moves(valid_moves)
                decision = player.decide(board_stack, dice_roll, valid_moves)
                move = decision if decision is not None else rng.choice(valid_moves)

                if player.strategy_name == teacher and mask.any():
                    records_board.append(board_stack)
                    records_dice.append(np.array([dice_roll - 1], dtype=np.float32))
                    records_mask.append(mask.astype(bool))
                    records_action.append(move["piece"].piece_id)

                result = game.make_move(
                    current_index, move["piece"], move["new_pos"], dice_roll
                )

                extra_turn = result["extra_turn"] and result["events"].get(
                    "move_resolved", True
                )

                if player.has_won() and current_index not in finish_order:
                    finish_order.append(current_index)

                if max_samples is not None and len(records_action) >= max_samples:
                    break

            if max_samples is not None and len(records_action) >= max_samples:
                break

            current_index = (current_index + 1) % config.NUM_PLAYERS
            turns_taken += 1

        if max_samples is not None and len(records_action) >= max_samples:
            break

    if not records_action:
        raise RuntimeError(
            "No samples were collected. Check strategy names and teacher."
        )

    return SampleBatch(
        board=np.stack(records_board, axis=0),
        dice=np.stack(records_dice, axis=0),
        mask=np.stack(records_mask, axis=0),
        action=np.asarray(records_action, dtype=np.int64),
    )


def save_dataset(dataset: SampleBatch, path: str | None) -> None:
    if path is None:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        board=dataset.board,
        dice=dataset.dice,
        mask=dataset.mask,
        action=dataset.action,
    )
    logger.info(f"Saved dataset with {dataset.action.shape[0]} samples to {dest}")


def behaviour_clone(
    model_path: str,
    save_path: str,
    dataset: SampleBatch,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    model = MaskablePPO.load(model_path, device=device)
    policy = model.policy
    policy.train()

    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    num_samples = dataset.action.shape[0]

    board_tensor = torch.tensor(dataset.board, dtype=torch.float32, device=device)
    dice_tensor = torch.tensor(dataset.dice, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(dataset.mask, dtype=torch.bool, device=device)
    action_tensor = torch.tensor(dataset.action, dtype=torch.long, device=device)

    indices = np.arange(num_samples)

    for epoch in range(epochs):
        rng = np.random.permutation(indices)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = rng[start:end]

            obs = {
                "board": board_tensor[batch_idx],
                "dice_roll": dice_tensor[batch_idx],
            }
            mask_batch = mask_tensor[batch_idx]
            action_batch = action_tensor[batch_idx]

            distribution = policy.get_distribution(obs, action_masks=mask_batch)
            log_prob = distribution.log_prob(action_batch)
            loss = -log_prob.mean()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()
            batches += 1

        logger.info(
            "Epoch %d/%d - loss %.4f",
            epoch + 1,
            epochs,
            epoch_loss / max(1, batches),
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info("Saved behaviour-cloned checkpoint to %s", save_path)


def main() -> None:
    args = parse_args()

    opponents = [
        name.strip().lower() for name in args.opponents.split(",") if name.strip()
    ]
    validate_strategies(opponents)

    teacher = args.teacher.strip().lower()
    if teacher not in opponents:
        raise ValueError("Teacher strategy must be included in opponents list.")

    rng = random.Random(args.seed)

    logger.info("Collecting samples from teacher '%s' against %s", teacher, opponents)
    dataset = collect_teacher_samples(
        teacher=teacher,
        opponents=opponents,
        games=args.games,
        rng=rng,
        max_samples=args.max_samples,
    )
    logger.info("Collected %d samples", dataset.action.shape[0])

    save_dataset(dataset, args.dataset_out)

    if args.bc_epochs > 0:
        behaviour_clone(
            model_path=args.model_path,
            save_path=args.save_path,
            dataset=dataset,
            device=args.device,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
        )
    else:
        logger.info("Skipping behaviour cloning as --bc-epochs is 0")


if __name__ == "__main__":
    main()
