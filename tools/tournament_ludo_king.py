from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger

from ludo_rl.ludo_king import Game, Player, Simulator, Board, Color, config as king_config
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.registry import available as available_strategies

def _points_table(num_players: int) -> Tuple[int, ...]:
    # Highest rank gets most points; last gets 0
    return tuple(max(0, num_players - 1 - i) for i in range(num_players))


def seed_everything(seed: int | None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
        np.random.seed(seed)
    return rng


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
    parser = argparse.ArgumentParser(description="Run a Ludo King (new engine) strategy tournament")
    parser.add_argument(
        "--games",
        type=int,
        default=int(os.getenv("NGAMES", "10")),
        help="Number of games to play per combination",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=os.getenv("OPPONENTS", ",".join(available_strategies())),
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

    if len(available) < king_config.NUM_PLAYERS:
        raise RuntimeError(f"Need at least {king_config.NUM_PLAYERS} strategies to stage a league.")
    return available


def attach_strategy(player: Player, strategy_name: str, rng: random.Random) -> None:
    cls = STRATEGY_REGISTRY[strategy_name]
    try:
        player.strategy = cls.create_instance(rng)
    except NotImplementedError:
        player.strategy = cls()
    player.strategy_name = strategy_name  # type: ignore[attr-defined]


def build_board_stack(board: Board, player_color: int) -> np.ndarray:
    return board.build_tensor(player_color)


def player_progress(player: Player) -> int:
    return sum(p.position for p in player.pieces)


def determine_rankings(game: Game, finish_order: List[int]) -> List[int]:
    ordered = finish_order.copy()
    total_players = len(game.players)
    remaining = [idx for idx in range(total_players) if idx not in ordered]
    remaining.sort(
        key=lambda idx: (
            sum(1 for pc in game.players[idx].pieces if pc.position == king_config.HOME_FINISH),
            player_progress(game.players[idx]),
        ),
        reverse=True,
    )
    ordered.extend(remaining)
    return ordered[: total_players]


def play_game(seats: Sequence[str], rng: random.Random, game_index: int) -> GameResult:
    # Build seats based on configured number of players
    num = king_config.NUM_PLAYERS
    if num == 2:
        color_ids = [int(Color.RED), int(Color.YELLOW)]  # 0 vs 2 (opposite)
    else:
        color_ids = [int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.BLUE)][:num]
    players = [Player(color=c) for c in color_ids]
    game = Game(players=players)

    # Reset and attach strategies
    for player, strategy_name in zip(game.players, seats, strict=True):
        for piece in player.pieces:
            piece.position = 0
        player.has_finished = False
        attach_strategy(player, strategy_name, rng)

    finish_order: List[int] = []
    turns_taken = 0
    current_index = 0
    total_players = len(game.players)

    # Main loop
    while turns_taken < king_config.MAX_TURNS and len(finish_order) < total_players:
        player = game.players[current_index]
        if player.check_won():
            if current_index not in finish_order:
                finish_order.append(current_index)
            current_index = (current_index + 1) % total_players
            continue

        extra_turn = True
        while extra_turn:
            dice_roll = game.roll_dice()
            legal = game.legal_moves(current_index, dice_roll)
            if not legal:
                extra_turn = False
                continue

            board_stack = build_board_stack(game.board, int(player.color))
            decision = player.choose(board_stack, dice_roll, legal)
            move = decision if decision is not None else rng.choice(legal)

            result = game.apply_move(move)
            extra_turn = result.extra_turn and result.events.move_resolved

            if player.check_won() and current_index not in finish_order:
                finish_order.append(current_index)

        current_index = (current_index + 1) % total_players
        turns_taken += 1

    rankings = determine_rankings(game, finish_order)
    placement_names = [getattr(game.players[idx], "strategy_name", "?") for idx in rankings]
    pt = _points_table(len(game.players))
    points = {name: pt[pos] for pos, name in enumerate(placement_names)}

    return GameResult(
        index=game_index,
        turns=turns_taken,
        placements=placement_names,
        points=points,
    )


def run_combination_tournament(participants: Sequence[str], games: int, rng: random.Random) -> CombinationSummary:
    totals = {name: 0 for name in participants}
    results: List[GameResult] = []

    for game_index in range(1, games + 1):
        seats = list(participants)
        rng.shuffle(seats)
        result = play_game(seats, rng, game_index)
        results.append(result)
        for name, pts in result.points.items():
            totals[name] += pts

    return CombinationSummary(participants=tuple(participants), game_results=results, totals=totals)


def log_combination_summary(summary: CombinationSummary, index: int | None = None) -> None:
    combo_label = ", ".join(summary.participants)
    prefix = f"Combination {index:03d}: " if index is not None else "Combination: "
    logger.info(f"{prefix}{combo_label}")
    combo_table = sorted(summary.totals.items(), key=lambda item: (-item[1], item[0]))
    logger.info("  Points collected:")
    for rank, (name, points) in enumerate(combo_table, start=1):
        logger.info(f"    {rank}. {name:12s} {points:3d} pts")


def run_league(strategy_pool: Sequence[str], games: int, rng: random.Random) -> tuple[Dict[str, int], Dict[str, int], List[CombinationSummary]]:
    league_totals = {name: 0 for name in strategy_pool}
    games_played = {name: 0 for name in strategy_pool}
    combination_summaries: List[CombinationSummary] = []

    all_combos = list(combinations(strategy_pool, king_config.NUM_PLAYERS))
    seeds = [rng.randint(0, 2**32 - 1) for _ in all_combos]

    def _run_combo(combo: Tuple[str, ...], seed: int) -> CombinationSummary:
        local_rng = random.Random(seed)
        return run_combination_tournament(combo, games, local_rng)

    max_workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_combo, combo, seed): (combo, idx)
            for idx, (combo, seed) in enumerate(zip(all_combos, seeds, strict=True))
        }
        combo_counter = 0
        for future in as_completed(future_map):
            summary = future.result()
            combination_summaries.append(summary)
            for name, pts in summary.totals.items():
                league_totals[name] += pts
                games_played[name] += len(summary.game_results)
            combo_counter += 1
            log_combination_summary(summary, combo_counter)

    return league_totals, games_played, combination_summaries


def print_summary(strategy_pool: Sequence[str], league_totals: Dict[str, int], games_played: Dict[str, int], combination_summaries: Sequence[CombinationSummary]) -> None:
    logger.info(f"Strategy pool: {', '.join(strategy_pool)}")

    print("League standings:")
    for rank, (name, points) in enumerate(sorted(league_totals.items(), key=lambda item: (-item[1], item[0])), start=1):
        print(f"  {rank:2d}. {name:12s} {points:5d} pts across {games_played[name]:4d} games")

    winner = max(league_totals.items(), key=lambda item: (item[1], item[0]))[0]
    print()
    print(f"League winner: {winner}")


def main() -> None:
    args = parse_args()
    rng = seed_everything(args.seed)

    try:
        strategy_pool = select_strategies(args.strategies)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))

    league_totals, games_played, combo_summaries = run_league(strategy_pool, args.games, rng)
    print_summary(strategy_pool, league_totals, games_played, combo_summaries)


if __name__ == "__main__":
    main()
