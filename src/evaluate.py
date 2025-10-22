"""Minimal evaluation script for running a trained policy against scripted opponents."""

from __future__ import annotations

import argparse
from typing import Sequence

from sb3_contrib import MaskablePPO
import pandas as pd
from models.config import EnvConfig
from models.eval_utils import EvalStats, evaluate_against


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Ludo model.")
    parser.add_argument(
        "model", type=str, help="Path to the saved MaskablePPO model .zip file."
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="probabilistic_v3,killer,cautious",
        help="Comma-separated list of opponent strategies to evaluate against.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of evaluation games per opponent.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=300,
        help="Maximum turns per episode before declaring a draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on (cpu or cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    opponents = [op.strip() for op in args.opponents.split(",") if op.strip()]
    if not opponents:
        raise ValueError("At least one opponent strategy must be provided.")

    model: MaskablePPO = MaskablePPO.load(args.model, device=args.device)
    print(model.policy)
    env_cfg = EnvConfig(max_turns=args.max_turns, seed=args.seed)

    summaries: Sequence[EvalStats] = [
        evaluate_against(model, opponent, args.games, env_cfg, args.deterministic)
        for opponent in opponents
    ]

    print("Opponent,Games,WinRate,LossRate,DrawRate,AvgReward,AvgLength")
    for summary in summaries:
        data = summary.as_dict()
        print(
            f"{data['opponent']},{int(data['episodes'])},"
            f"{data['win_rate']:.3f},{data['loss_rate']:.3f},"
            f"{data['draw_rate']:.3f},{data['avg_reward']:.2f},"
            f"{data['avg_length']:.1f}"
        )


if __name__ == "__main__":
    main()
