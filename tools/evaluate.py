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

from ludo_rl.ludo_env import LudoEnv
from ludo_rl.ludo_king import config as king_config
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


def _run_episode_with_env(
    env: LudoEnv,
    model: MaskablePPO,
    deterministic: bool,
) -> int:
    """Run a single episode in LudoEnv and return final rank (1=win, 0=draw/unknown)."""
    obs, info = env.reset()
    terminated = False
    truncated = False
    final_rank: int = 0

    while not terminated and not truncated and env.current_turn < king_config.MAX_TURNS:
        mask = env.action_masks()
        if mask is None or not np.any(mask):
            action = 0
        else:
            action, _ = model.predict(
                obs, action_masks=mask[None, ...], deterministic=deterministic
            )
            action = int(np.asarray(action).item())

        obs, reward, terminated, truncated, info = env.step(action)

    fr = info.get("final_rank") if isinstance(info, dict) else None
    if isinstance(fr, (int, np.integer)):
        final_rank = int(fr)
    return final_rank


def evaluate_triplet(
    model: MaskablePPO,
    triplet: Sequence[str],
    episodes: int,
    deterministic: bool,
    rng: random.Random,
) -> dict:
    """Evaluate a fixed opponent lineup using LudoEnv episodes."""
    rank_counter: Counter[int] = Counter()
    wins = 0

    # Build one env and fix its opponents to this triplet
    env = LudoEnv(use_fixed_opponents=True)
    # Override opponent pool and selection to ensure exact lineup
    triplet = list(triplet)
    env.opponents = triplet
    env._fixed_opponents_strategies = triplet
    env.strategy_selection = 1  # sequential; with fixed_opponents this locks the lineup
    env._reset_count = 0  # start from first lineup ordering

    for _ in range(episodes):
        rng.shuffle(triplet)
        final_rank = _run_episode_with_env(env, model, deterministic)
        if final_rank == 1:
            wins += 1
        if final_rank <= 0:
            # Treat draw/unknown as worst rank (optional: keep 0)
            rank_counter[king_config.NUM_PLAYERS] += 1
        else:
            rank_counter[final_rank] += 1

    # Clean up
    try:
        env.close()
    except Exception:
        pass

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
