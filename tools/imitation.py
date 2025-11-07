from __future__ import annotations

import argparse
import os
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
        "--teacher",
        type=str,
        default=os.getenv("TEACHER", "homebody"),
        help=(
            "Strategy name to imitate when using fixed teacher mode. "
            "Ignored if --teacher-mode winner is selected."
        ),
    )
    parser.add_argument(
        "--teacher-mode",
        type=str,
        choices=("fixed", "winner"),
        default=os.getenv("TEACHER_MODE", "winner"),
        help=(
            "Select 'fixed' to clone a specific strategy provided via --teacher, "
            "or 'winner' to imitate whichever scripted player wins each game."
        ),
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=os.getenv(
            "OPPONENTS",
            "homebody,heatseeker,retaliator,hoarder,rusher,finish_line,probability",
        ),
        help="Comma-separated list of four strategies to play each game with.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=int(os.getenv("GAMES", 10000)),
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
        default=42,
        help="Optional RNG seed for reproducibility.",
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
        default=int(os.getenv("BC_BATCH_SIZE", 1024)),
        help="Batch size for behaviour cloning updates.",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=float(os.getenv("BC_LR", 1e-4)),
        help="Learning rate for the behaviour cloning optimiser.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
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
    return game.build_board_tensor(player_index)


def prepare_players(game: LudoGame, strategies: Sequence[str]) -> None:
    for player, strategy_name in zip(game.players, strategies):
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
    teacher: str | None,
    teacher_mode: str,
    opponents: Sequence[str],
    games: int,
    rng: random.Random,
    max_samples: int | None,
) -> SampleBatch:
    records_board: List[np.ndarray] = []
    records_dice: List[np.ndarray] = []
    records_mask: List[np.ndarray] = []
    records_action: List[int] = []

    dynamic_teacher = teacher_mode == "winner"

    for index_game in range(games):
        pool = list(opponents)
        rng.shuffle(pool)

        if dynamic_teacher:
            strategies = pool[: config.NUM_PLAYERS]
            if len(strategies) < config.NUM_PLAYERS:
                raise ValueError(
                    "Need at least four opponents to sample winner-based teachers."
                )
        else:
            assert teacher is not None
            others = [name for name in pool if name != teacher]
            if len(others) < config.NUM_PLAYERS - 1:
                raise ValueError(
                    "Opponents list must provide at least three non-teacher strategies."
                )
            strategies = [teacher] + others[: config.NUM_PLAYERS - 1]

        game = LudoGame()
        prepare_players(game, strategies)

        player_records: List[Dict[str, List[np.ndarray]]] = [
            {"board": [], "dice": [], "mask": [], "action": []}
            for _ in range(config.NUM_PLAYERS)
        ]

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

                if mask.any():
                    store = player_records[current_index]
                    store["board"].append(board_stack)
                    store["dice"].append(np.array([dice_roll - 1], dtype=np.float32))
                    store["mask"].append(mask.astype(bool))
                    store["action"].append(move["piece"].piece_id)

                result = game.make_move(
                    current_index, move["piece"], move["new_pos"], dice_roll
                )

                extra_turn = result["extra_turn"] and result["events"].get(
                    "move_resolved", True
                )

                if player.has_won() and current_index not in finish_order:
                    finish_order.append(current_index)

            current_index = (current_index + 1) % config.NUM_PLAYERS
            turns_taken += 1

        if dynamic_teacher:
            if finish_order:
                winner_index = finish_order[0]
                winner_store = player_records[winner_index]
                records_board.extend(winner_store["board"])
                records_dice.extend(winner_store["dice"])
                records_mask.extend(winner_store["mask"])
                records_action.extend(winner_store["action"])
        else:
            teacher_store = next(
                (
                    store
                    for store, player in zip(player_records, game.players)
                    if player.strategy_name == teacher
                ),
                None,
            )
            if teacher_store is not None:
                records_board.extend(teacher_store["board"])
                records_dice.extend(teacher_store["dice"])
                records_mask.extend(teacher_store["mask"])
                records_action.extend(teacher_store["action"])

        if max_samples is not None and len(records_action) >= max_samples:
            break

    if not records_action:
        raise RuntimeError(
            "No samples were collected. Check strategy names and teacher selection."
        )

    if max_samples is not None and len(records_action) > max_samples:
        trim = max_samples
        records_board = records_board[:trim]
        records_dice = records_dice[:trim]
        records_mask = records_mask[:trim]
        records_action = records_action[:trim]

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
            f"Epoch {epoch + 1}/{epochs} - loss {epoch_loss / max(1, batches):.4f}",
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"Saved behaviour-cloned checkpoint to {save_path}")


def main() -> None:
    args = parse_args()

    opponents = [
        name.strip().lower() for name in args.opponents.split(",") if name.strip()
    ]
    validate_strategies(opponents)

    if len(opponents) < config.NUM_PLAYERS:
        raise ValueError(
            "Need at least four opponent strategies to run imitation games."
        )

    teacher_mode = args.teacher_mode.strip().lower()
    teacher = args.teacher.strip().lower() if args.teacher else None

    if teacher_mode == "fixed":
        if not teacher:
            raise ValueError(
                "Teacher strategy name required when --teacher-mode is 'fixed'."
            )
        if teacher not in opponents:
            raise ValueError("Teacher strategy must be included in opponents list.")
    else:
        teacher = None

    rng = random.Random(args.seed)

    if teacher_mode == "winner":
        logger.info(
            f"Collecting samples from game winners against opponents {opponents}"
        )
    else:
        logger.info(f"Collecting samples from teacher '{teacher}' against {opponents}")
    dataset = collect_teacher_samples(
        teacher=teacher,
        teacher_mode=teacher_mode,
        opponents=opponents,
        games=args.games,
        rng=rng,
        max_samples=args.max_samples,
    )
    logger.info(f"Collected {dataset.action.shape[0]} samples")

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
