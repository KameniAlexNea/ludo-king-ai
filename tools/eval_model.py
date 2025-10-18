#!/usr/bin/env python3
"""Simple evaluation runner for a saved MaskablePPO model.

Runs N stochastic episodes in a single-process env and reports
mean/median/percentiles for total reward, per-term breakdown and win-rate.
"""

import argparse
import os
import statistics
from collections import defaultdict
from typing import Dict

from dotenv import load_dotenv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.utils.move_utils import MoveUtils
from ludo_rl.utils.opponents import build_opponent_combinations

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default="training/models/best_model.zip",
        help="Path to saved MaskablePPO model",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=os.getenv("EVAL_EPISODES", 100),
        help="Number of evaluation episodes",
    )
    p.add_argument("--fixed-num-players", type=int, default=2)
    p.add_argument(
        "--opponent",
        type=str,
        nargs="+",
        default=None,
        help="Name of opponent strategy to use for all opponents (optional)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for env")
    return p.parse_args()


def safe_add_dict(dst: Dict[str, float], src: Dict[str, float]):
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v)


def run_eval(
    model_path: str, n_episodes: int, fixed_num_players: int, opponent: str | None
):
    cfg = EnvConfig()
    cfg.fixed_num_players = fixed_num_players

    # single-process env for easy introspection
    env = LudoRLEnv(cfg)
    env = ActionMasker(env, MoveUtils.get_action_mask_for_env)
    opponents_counter: Dict[str, int] = defaultdict(int)

    # set opponents if requested (will be applied on reset)
    opponents = None
    if opponent is None:
        opponent = cfg.opponents.evaluation_candidates
    print("Using opponents:", opponent)
    opponents = build_opponent_combinations(
        opponent, n_episodes, fixed_num_players - 1, True
    )

    print(f"Loading model from {model_path}")
    model = MaskablePPO.load(model_path, device="cpu")

    rewards = []
    wins = 0
    captures = []
    finishes = []
    breakdown_sums: Dict[str, float] = defaultdict(float)
    breakdown_counts: Dict[str, int] = defaultdict(int)
    seed = args.seed
    for ep in range(n_episodes):
        curr_opponents = opponents[ep]

        reset_ret = env.reset(seed=seed, options={"opponents": curr_opponents})
        seed += 1
        # Gym returns (obs, info) while SB3 VecEnv expects obs only. We're using raw env.
        if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
            obs, _ = reset_ret
        else:
            obs = reset_ret
        done = False
        total_reward = 0.0
        ep_breakdown: Dict[str, float] = defaultdict(float)
        ep_captures = 0
        ep_finishes = 0

        while not done:
            # model.predict expects a raw observation, not (obs, info)
            if isinstance(obs, tuple) and len(obs) == 2:
                obs_in, _ = obs
            else:
                obs_in = obs
            action, _ = model.predict(obs_in, deterministic=False)
            step_ret = env.step(int(action))
            if isinstance(step_ret, tuple) and len(step_ret) == 5:
                obs, r, terminated, truncated, info = step_ret
            else:
                # Fallback for other env APIs
                obs, r, done, info = step_ret
                terminated = done
                truncated = False
            total_reward += float(r)
            # accumulate breakdown if present
            rb = info.get("reward_breakdown")
            if rb:
                for k, v in rb.items():
                    ep_breakdown[k] += float(v)
            # step-level stats
            ep_captures = info.get("episode_captured_opponents", ep_captures)
            ep_finishes = info.get("finished_tokens", ep_finishes)

            done = bool(terminated or truncated)

        # Determine win from underlying env state
        try:
            winner = env.unwrapped.game.winner
            is_win = winner is not None and winner.color == env.unwrapped.agent_color
        except Exception:
            is_win = False

        if is_win:
            wins += 1

        print(
            f"Episode {ep + 1}/{n_episodes}, opponents: {curr_opponents} - Win: {is_win} - Reward: {total_reward:.2f}"
        )
        opponents_counter[str(curr_opponents)] += is_win

        rewards.append(total_reward)
        captures.append(ep_captures)
        finishes.append(ep_finishes)

        # aggregate breakdown
        for k, v in ep_breakdown.items():
            breakdown_sums[k] += v
            breakdown_counts[k] += 1

    # Summarize
    mean_r = statistics.mean(rewards)
    median_r = statistics.median(rewards)
    std_r = statistics.pstdev(rewards)
    pct10 = sorted(rewards)[max(0, int(0.1 * len(rewards)) - 1)]
    pct90 = sorted(rewards)[min(len(rewards) - 1, int(0.9 * len(rewards)) - 1)]

    print("\nEvaluation summary")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {mean_r:.2f}  Median: {median_r:.2f}  Std: {std_r:.2f}")
    print(f"10th / 90th percentiles: {pct10:.2f} / {pct90:.2f}")
    print(f"Win rate: {wins}/{n_episodes} = {wins / n_episodes:.3f}")
    print(
        f"Avg captures: {statistics.mean(captures):.2f}  Avg finishes: {statistics.mean(finishes):.2f}"
    )

    print("\nAverage reward breakdown (per-episode where present):")
    for k, total in sorted(breakdown_sums.items(), key=lambda x: -x[1]):
        cnt = breakdown_counts.get(k, 1)
        print(f"  {k}: {total / cnt:.4f} (mean over {cnt} episodes)")

    print("Opponent winning counts:")
    for k, v in sorted(opponents_counter.items(), key=lambda x: -x[1]):
        print(
            f"  {k}: {v} wins - {v / n_episodes // len(curr_opponents):.3f} win rate - contrib {v / max(1, wins):.3f} of total wins"
        )


if __name__ == "__main__":
    args = parse_args()
    run_eval(args.model, args.n_episodes, args.fixed_num_players, args.opponent)
