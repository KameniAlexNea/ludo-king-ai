import argparse
import itertools
import os
import random
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from sb3_contrib import MaskablePPO

from ludo_rl.ludo_king import Board, Color, Game, Player, Simulator
from ludo_rl.ludo_king import config as king_config
from ludo_rl.strategy.registry import STRATEGY_REGISTRY
from ludo_rl.strategy.registry import available as available_strategies

load_dotenv()


def seed_everything(seed: int | None) -> random.Random:
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
        np.random.seed(seed)
    return rng


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Ludo agent (ludo_king)"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--episodes-per-combo",
        type=int,
        default=int(os.getenv("NGAMES", "20")),
        help="Number of evaluation games per opponent lineup",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=os.getenv("OPPONENTS", ",".join(available_strategies())),
        help="Comma-separated list of opponent strategies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for NumPy and Python random generators",
    )
    parser.add_argument(
        "--limit-combos",
        type=int,
        default=None,
        help="Evaluate only the first N opponent triplets (random order if seed provided)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to run inference with the loaded policy",
    )
    return parser.parse_args()


def attach_strategy(player: Player, strategy_name: str, rng: random.Random) -> None:
    cls = STRATEGY_REGISTRY[strategy_name]
    try:
        player.strategy = cls.create_instance(rng)
    except NotImplementedError:
        player.strategy = cls()
    player.strategy_name = strategy_name  # type: ignore[attr-defined]


def build_board_stack(board: Board, player_color: int) -> np.ndarray:
    return board.build_tensor(player_color)


def _extract_piece_index(move: object) -> int | None:
    for attr in ("piece_id", "token", "piece_index"):
        if hasattr(move, attr):
            try:
                return int(getattr(move, attr))
            except Exception:
                pass
    piece = getattr(move, "piece", None)
    if piece is not None:
        for attr in ("piece_id", "index", "id"):
            if hasattr(piece, attr):
                try:
                    return int(getattr(piece, attr))
                except Exception:
                    pass
    return None


def decide_with_model(
    model: MaskablePPO,
    simulator: Simulator,
    dice_roll: int,
    legal_moves: list[object],
    deterministic: bool,
    rng: random.Random,
) -> object | None:
    # Get token-sequence observation from simulator
    obs = simulator.get_token_sequence_observation(dice_roll)

    action_mask = np.zeros(king_config.PIECES_PER_PLAYER, dtype=bool)
    moves_by_piece: dict[int, list[object]] = {}
    for mv in legal_moves:
        pid = _extract_piece_index(mv)
        if pid is None:
            continue
        if 0 <= pid < king_config.PIECES_PER_PLAYER:
            action_mask[pid] = True
            moves_by_piece.setdefault(pid, []).append(mv)

    try:
        action, _ = model.predict(
            obs, action_masks=action_mask[None, ...], deterministic=deterministic
        )
        pid = int(action.item())
        candidates = moves_by_piece.get(pid)
        if candidates:
            return rng.choice(candidates)
    except Exception:
        pass
    return rng.choice(legal_moves) if legal_moves else None


def determine_rankings(game: Game, finish_order: list[int]) -> list[int]:
    ordered = finish_order.copy()
    total = len(game.players)
    remaining = [i for i in range(total) if i not in ordered]
    remaining.sort(
        key=lambda idx: (
            sum(
                1
                for pc in game.players[idx].pieces
                if pc.position == king_config.HOME_FINISH
            ),
            sum(pc.position for pc in game.players[idx].pieces),
        ),
        reverse=True,
    )
    ordered.extend(remaining)
    return ordered[:total]


def evaluate_triplet(
    model: MaskablePPO,
    triplet: Sequence[str],
    episodes: int,
    deterministic: bool,
    rng: random.Random,
) -> dict:
    rank_counter: Counter[int] = Counter()
    wins = 0

    for _ in range(episodes):
        # Build game and seat opponents
        num = king_config.NUM_PLAYERS
        # Support 2 players (opposite seats) or 4 players
        if num == 2:
            color_ids = [int(Color.RED), int(Color.YELLOW)]
        else:
            color_ids = [
                int(Color.RED),
                int(Color.GREEN),
                int(Color.YELLOW),
                int(Color.BLUE),
            ][:num]
        players = [Player(color=c) for c in color_ids]
        game = Game(players=players)
        simulator = Simulator.for_game(game, agent_index=0)

        # Agent at seat 0 (RED); opponents occupy seats 1..(num-1)
        for seat_idx, player in enumerate(game.players):
            for piece in player.pieces:
                piece.position = 0
            player.has_finished = False
            if seat_idx > 0:
                attach_strategy(player, triplet[seat_idx - 1], rng)
            else:
                player.strategy_name = "agent"  # type: ignore[attr-defined]

        finish_order: list[int] = []
        turns = 0
        cur = 0

        while turns < king_config.MAX_TURNS and len(finish_order) < len(game.players):
            pl = game.players[cur]
            if pl.check_won():
                if cur not in finish_order:
                    finish_order.append(cur)
                cur = (cur + 1) % len(game.players)
                continue

            extra = True
            while extra:
                dice = game.roll_dice()
                legal = game.legal_moves(cur, dice)
                if not legal:
                    extra = False
                    continue

                # Append move to history for both agent and opponents
                simulator._append_history(dice, cur)

                if cur == 0:
                    mv = decide_with_model(
                        model, simulator, dice, legal, deterministic, rng
                    )
                else:
                    board_stack = build_board_stack(game.board, int(pl.color))
                    decision = pl.choose(board_stack, dice, legal)
                    mv = decision if decision is not None else rng.choice(legal)

                result = game.apply_move(mv)
                extra = result.extra_turn and result.events.move_resolved

                if pl.check_won() and cur not in finish_order:
                    finish_order.append(cur)

            cur = (cur + 1) % len(game.players)
            turns += 1

        rankings = determine_rankings(game, finish_order)
        agent_rank = rankings.index(0) + 1  # 1-based
        if agent_rank == 1:
            wins += 1
        rank_counter[agent_rank] += 1

    return {
        "triplet": triplet,
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes if episodes else 0.0,
        "avg_rank": (
            sum(rank * count for rank, count in rank_counter.items()) / episodes
            if episodes
            else 0.0
        ),
        "rank_counts": dict(rank_counter),
    }


def iter_triplets(
    limit: int | None, strategies: Sequence[str]
) -> Iterable[Sequence[str]]:
    # Build opponent lineups of size NUM_PLAYERS-1
    opp_count = max(1, king_config.NUM_PLAYERS - 1)
    combos = list(itertools.combinations(strategies, opp_count))
    if limit is not None and limit < len(combos):
        random.shuffle(combos)
        combos = combos[:limit]
    return combos


def main() -> None:
    args = parse_args()
    rng = seed_everything(args.seed)

    model = MaskablePPO.load(args.model_path, device=args.device)
    model.policy.set_training_mode(False)

    results = []
    opponents = [s.strip() for s in args.opponents.split(",") if s.strip()]

    print(f"Evaluating against opponent strategies: {', '.join(opponents)}")
    for triplet in iter_triplets(args.limit_combos, opponents):
        stats = evaluate_triplet(
            model=model,
            triplet=triplet,
            episodes=args.episodes_per_combo,
            deterministic=args.deterministic,
            rng=rng,
        )
        results.append(stats)

        triplet_label = ",".join(triplet)
        logger.info(
            f"RL vs Opponents {triplet_label:<40} | Win-rate: {stats['win_rate']:.2%} | Avg Rank: {stats['avg_rank']:.2f}"
        )

    if not results:
        logger.warning("No opponent triplets evaluated.")
        return

    best = max(results, key=lambda item: item["win_rate"])
    print("\nBest performing triplet:")
    print(
        f"{','.join(best['triplet'])} -> Win-rate {best['win_rate']:.2%}, Average rank {best['avg_rank']:.2f}"
    )


if __name__ == "__main__":
    main()
