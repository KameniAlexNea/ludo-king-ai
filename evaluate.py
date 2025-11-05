import argparse
import itertools
import os
import random
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ludo_rl.ludo_env import LudoEnv
from loguru import logger

ALL_STRATEGIES: Sequence[str] = (
    "probability",
    "cautious",
    "killer",
    "defensive",
    "finish_line",
    "heatseeker",
    "hoarder",
    "homebody",
    "rusher",
    "support",
    "retaliator",
)


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Ludo agent")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--episodes-per-combo",
        type=int,
        default=10,
        help="Number of evaluation games per opponent triplet",
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


def build_env(model_path: str | None = None) -> DummyVecEnv:
    """Create eval env and, if available, load VecNormalize stats.

    If `model_path` is provided, this will look for a `vecnormalize.pkl` file
    in the same directory and load it to ensure observations are normalized
    exactly like during training. Rewards are not normalized during eval.
    """
    base_env = DummyVecEnv([lambda: LudoEnv(render_mode=None)])

    if model_path is None:
        return base_env

    stats_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        try:
            env = VecNormalize.load(stats_path, base_env)
            env.training = False
            env.norm_reward = False
            return env
        except Exception:
            # Fallback to unnormalized env if loading fails
            return base_env
    return base_env


def evaluate_triplet(
    model: MaskablePPO,
    triplet: Sequence[str],
    episodes: int,
    deterministic: bool,
) -> dict:
    os.environ["OPPONENTS"] = ",".join(triplet)
    os.environ["STRATEGY_SELECTION"] = "1"

    env = build_env(getattr(model, "path", None) or getattr(model, "_loaded_params", {}).get("model_path", None))

    rank_counter: Counter[int] = Counter()
    wins = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        truncated = False
        final_info: dict = {}

        while not done and not truncated:
            action_masks = env.env_method("action_masks")[0]
            action, _state = model.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )
            obs, rewards, dones, infos = env.step([action.item()])

            done = bool(dones[0])
            truncated = bool(infos[0].get("truncated", False))
            final_info = infos[0]

        rank = int(final_info.get("final_rank", 0))
        if rank == 1:
            wins += 1
        rank_counter[rank] += 1

    env.close()

    return {
        "triplet": triplet,
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes if episodes else 0.0,
        "avg_rank": sum(rank * count for rank, count in rank_counter.items())
        / episodes
        if episodes
        else 0.0,
        "rank_counts": dict(rank_counter),
    }


def iter_triplets(limit: int | None, strategies: Sequence[str]) -> Iterable[Sequence[str]]:
    combos = list(itertools.combinations(strategies, 3))
    if limit is not None and limit < len(combos):
        random.shuffle(combos)
        combos = combos[:limit]
    return combos


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model = MaskablePPO.load(args.model_path, device=args.device)
    # Attach model_path so build_env can find VecNormalize stats alongside it
    setattr(model, "path", args.model_path)
    model.policy.set_training_mode(False)

    results = []
    for triplet in iter_triplets(args.limit_combos, ALL_STRATEGIES):
        stats = evaluate_triplet(
            model=model,
            triplet=triplet,
            episodes=args.episodes_per_combo,
            deterministic=args.deterministic,
        )
        results.append(stats)

        triplet_label = ",".join(triplet)
        logger.info(
            f"Triplet {triplet_label:<40} | Win-rate: {stats['win_rate']:.2%} | "
            f"Avg Rank: {stats['avg_rank']:.2f}"
        )

    if not results:
        logger.warning("No opponent triplets evaluated.")
        return

    best = max(results, key=lambda item: item["win_rate"])
    print("\nBest performing triplet:")
    print(
        f"{','.join(best['triplet'])} -> Win-rate {best['win_rate']:.2%}, "
        f"Average rank {best['avg_rank']:.2f}"
    )


if __name__ == "__main__":
    main()
