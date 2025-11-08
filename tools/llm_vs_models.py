"""Run quick matches between one LLM-driven player and three RL models."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sb3_contrib import MaskablePPO

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.strategy.llm_agent import DEFAULT_SYSTEM_PROMPT, LLMStrategy, init_chat_model
from ludo_rl.strategy.rl_agent import RLStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pit a single LLM strategy against three RL checkpoints across a number of games."
        )
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help="Identifier passed to LangChain's init_chat_model for the LLM player.",
    )
    parser.add_argument(
        "--llm-param",
        action="append",
        default=[],
        help="Optional key=value provider arguments forwarded to init_chat_model.",
    )
    parser.add_argument(
        "--llm-label",
        default="llm",
        help="Display label for the LLM participant (default: llm).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the default system prompt used by the LLM strategy.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts for the LLM strategy before falling back.",
    )
    parser.add_argument(
        "--rl-models",
        required=True,
        help="Comma-separated paths to three MaskablePPO checkpoints.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for RL model inference (cpu/cuda).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions when querying RL policies.",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=1,
        help="Number of matches to play (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible runs.",
    )
    return parser.parse_args()


def parse_provider_kwargs(pairs: Sequence[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Invalid provider parameter '{item}'. Use key=value format.")
        key, value = item.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def load_rl_strategy(path: str, *, device: str, deterministic: bool) -> RLStrategy:
    checkpoint = os.path.expanduser(path)
    model = MaskablePPO.load(checkpoint, device=device)
    model.policy.set_training_mode(False)
    return RLStrategy(model=model, deterministic=deterministic)


def attach_players(
    game: LudoGame,
    seats: Sequence[Tuple[str, RLStrategy | LLMStrategy]],
) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for idx, player in enumerate(game.players):
        player.reset()
        label, strategy = seats[idx]
        player.strategy_name = strategy.name
        player._strategy = strategy
        labels[idx] = label
    return labels


def determine_rankings(game: LudoGame, finish_order: List[int]) -> List[int]:
    ordered = finish_order.copy()
    remaining = [idx for idx in range(config.NUM_PLAYERS) if idx not in ordered]

    def player_progress(player_index: int) -> Tuple[int, int]:
        player = game.players[player_index]
        finished = sum(piece.position == 57 for piece in player.pieces)
        progress = sum(piece.position for piece in player.pieces)
        return finished, progress

    remaining.sort(key=lambda idx: player_progress(idx), reverse=True)
    ordered.extend(remaining)
    return ordered[: config.NUM_PLAYERS]


def run_single_game(
    base_rng: random.Random,
    llm_entry: Tuple[str, LLMStrategy],
    rl_entries: Sequence[Tuple[str, RLStrategy]],
) -> Dict[str, object]:
    game = LudoGame()

    round_seed = base_rng.randint(0, 1_000_000)
    random.seed(round_seed)
    game.rng.seed(round_seed)

    seats: List[Tuple[str, RLStrategy | LLMStrategy]] = [llm_entry, *rl_entries[:3]]
    seat_labels = attach_players(game, seats)

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

        outcome = game.take_turn(current_index, rng=base_rng)

        if not outcome.skipped and player.has_won() and current_index not in finish_order:
            finish_order.append(current_index)

        while (
            not outcome.skipped
            and outcome.extra_turn
            and len(finish_order) < config.NUM_PLAYERS
        ):
            outcome = game.take_turn(current_index, rng=base_rng)
            if not outcome.skipped and player.has_won() and current_index not in finish_order:
                finish_order.append(current_index)

        current_index = (current_index + 1) % config.NUM_PLAYERS
        turns_taken += 1

    rankings = determine_rankings(game, finish_order)
    ranking_labels = [seat_labels[idx] for idx in rankings]

    return {
        "turns": turns_taken,
        "rankings": ranking_labels,
    }


def main() -> None:
    args = parse_args()

    if args.n_games <= 0:
        raise SystemExit("n_games must be a positive integer.")

    if init_chat_model is None:
        raise SystemExit("LangChain chat models are unavailable. Install 'langchain'.")

    provider_kwargs = parse_provider_kwargs(args.llm_param)

    chat_model = init_chat_model(args.llm_model, **provider_kwargs)
    llm_strategy = LLMStrategy(
        model=chat_model,
        system_prompt=args.system_prompt,
        max_retries=max(0, args.max_retries),
    )
    llm_entry = (args.llm_label, llm_strategy)

    rl_paths = [path.strip() for path in args.rl_models.split(",") if path.strip()]
    if len(rl_paths) < 3:
        raise SystemExit("Provide at least three RL checkpoints via --rl-models.")

    rl_entries: List[Tuple[str, RLStrategy]] = []
    for idx, path in enumerate(rl_paths[:3]):
        label = Path(path).stem or f"rl_{idx}"
        strategy = load_rl_strategy(
            path,
            device=args.device,
            deterministic=args.deterministic,
        )
        rl_entries.append((label, strategy))

    rng = random.Random(args.seed)

    leaderboard: Dict[str, int] = {args.llm_label: 0}
    for label, _ in rl_entries:
        leaderboard.setdefault(label, 0)

    for game_idx in range(1, args.n_games + 1):
        result = run_single_game(rng, llm_entry, rl_entries)
        winner = result["rankings"][0]
        leaderboard[winner] += 1
        print(
            f"Game {game_idx:02d}: winner={winner} | rankings={result['rankings']} | turns={result['turns']}"
        )

    print("\nFinal standings (wins):")
    for label, wins in sorted(leaderboard.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label}: {wins}")


if __name__ == "__main__":
    main()
