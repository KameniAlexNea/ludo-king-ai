from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ludo_rl.ludo.config import config
from ludo_rl.ludo.game import LudoGame
from ludo_rl.ludo.player import Player
from ludo_rl.strategy.registry import STRATEGY_REGISTRY

POINTS_TABLE = (4, 3, 1, 0)


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
        description="Run a heuristic-only Ludo strategy tournament"
    )
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed to make the tournament reproducible",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated list of strategy names to include. "
            "League runs every combination of four distinct strategies."
        ),
    )
    return parser.parse_args()


def select_strategies(provided: str | None) -> List[str]:
    available = sorted(STRATEGY_REGISTRY.keys())
    if provided:
        chosen = [name.strip().lower() for name in provided.split(",") if name.strip()]
        unknown = [name for name in chosen if name not in STRATEGY_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown strategies requested: {', '.join(unknown)}")
        available = chosen

    if len(available) < 4:
        raise RuntimeError("Need at least four strategies to stage a league.")
    return available


def build_board_stack(game: LudoGame, player_index: int) -> np.ndarray:
    return game.build_board_tensor(player_index)


def player_progress(player: Player) -> int:
    return sum(piece.position for piece in player.pieces)


def attach_strategy(player: Player, strategy_name: str, rng: random.Random) -> None:
    cls = STRATEGY_REGISTRY[strategy_name]
    try:
        player._strategy = cls.create_instance(rng)
    except NotImplementedError:
        player._strategy = cls()
    player.strategy_name = strategy_name


def determine_rankings(game: LudoGame, finish_order: List[int]) -> List[int]:
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
    seats: Sequence[str],
    rng: random.Random,
    game_index: int,
) -> GameResult:
    game = LudoGame()
    game_seed = rng.randint(0, 1_000_000)
    game.rng.seed(game_seed)
    random.seed(game_seed)

    for player, strategy_name in zip(game.players, seats, strict=True):
        for piece in player.pieces:
            piece.position = 0
        player.has_finished = False
        player._strategy = None
        attach_strategy(player, strategy_name, rng)

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
            decision = player.decide(board_stack, dice_roll, valid_moves)
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
    participants: Sequence[str], games: int, rng: random.Random
) -> CombinationSummary:
    totals = {name: 0 for name in participants}
    results: List[GameResult] = []

    for game_index in range(1, games + 1):
        seats = list(participants)
        rng.shuffle(seats)
        result = play_game(seats, rng, game_index)
        results.append(result)

        for name, pts in result.points.items():
            totals[name] += pts

    return CombinationSummary(
        participants=tuple(participants),
        game_results=results,
        totals=totals,
    )


def run_league(
    strategy_pool: Sequence[str], games: int, rng: random.Random
) -> tuple[Dict[str, int], Dict[str, int], List[CombinationSummary]]:
    league_totals = {name: 0 for name in strategy_pool}
    games_played = {name: 0 for name in strategy_pool}
    combination_summaries: List[CombinationSummary] = []

    for combo in combinations(strategy_pool, 4):
        summary = run_combination_tournament(combo, games, rng)
        combination_summaries.append(summary)

        for name, pts in summary.totals.items():
            league_totals[name] += pts
            games_played[name] += len(summary.game_results)

    return league_totals, games_played, combination_summaries


def print_summary(
    strategy_pool: Sequence[str],
    league_totals: Dict[str, int],
    games_played: Dict[str, int],
    combination_summaries: Sequence[CombinationSummary],
) -> None:
    print("Strategy pool:", ", ".join(strategy_pool))
    print()

    for combo_index, summary in enumerate(combination_summaries, start=1):
        combo_label = ", ".join(summary.participants)
        print(f"Combination {combo_index:03d}: {combo_label}")
        combo_table = sorted(
            summary.totals.items(), key=lambda item: (-item[1], item[0])
        )
        print("  Points collected:")
        for rank, (name, points) in enumerate(combo_table, start=1):
            print(f"    {rank}. {name:12s} {points:3d} pts")
        print()

    print("League standings:")
    for rank, (name, points) in enumerate(
        sorted(league_totals.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        print(
            f"  {rank:2d}. {name:12s} {points:5d} pts across {games_played[name]:4d} games"
        )

    winner = max(league_totals.items(), key=lambda item: (item[1], item[0]))[0]
    print()
    print(f"League winner: {winner}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        strategy_pool = select_strategies(args.strategies)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))

    league_totals, games_played, combo_summaries = run_league(
        strategy_pool, args.games, rng
    )
    print_summary(strategy_pool, league_totals, games_played, combo_summaries)


if __name__ == "__main__":
    main()
