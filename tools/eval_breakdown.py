#!/usr/bin/env python3
"""Run deterministic evaluation episodes and collect reward breakdowns.

Usage:
  python tools/eval_breakdown.py --model ./training/models/mymodel.zip --vec ./training/models/vecnormalize.pkl --n 240 --baseline probabilistic_v2 --out results.csv

The script creates a single-process eval env, loads VecNormalize if provided
so normalization matches training, runs N episodes and saves per-episode
aggregated reward breakdowns to CSV.
"""
from __future__ import annotations
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.utils.move_utils import MoveUtils


def make_eval_venv(env_cfg: EnvConfig):
    def _init():
        env = LudoRLEnv(env_cfg)
        return ActionMasker(env, MoveUtils.get_action_mask_for_env)

    return DummyVecEnv([_init])


def aggregate_breakdowns(breakdowns_list):
    totals = defaultdict(float)
    for bd in breakdowns_list:
        if not bd:
            continue
        for k, v in bd.items():
            try:
                totals[k] += float(v)
            except Exception:
                # ignore non-numeric
                pass
    return totals


def run_eval(model_path: str, vec_path: str | None, n_games: int, baseline: str | None, fixed_num_players: int | None, out_csv: str):
    env_cfg = EnvConfig()
    if fixed_num_players is not None:
        env_cfg.fixed_num_players = int(fixed_num_players)

    venv = make_eval_venv(env_cfg)
    venv = VecMonitor(venv)

    if vec_path is not None:
        if os.path.exists(vec_path):
            print(f"Loading VecNormalize from {vec_path}")
            venv = VecNormalize.load(vec_path, venv)
        else:
            raise FileNotFoundError(f"VecNormalize file not found: {vec_path}")

    # Load model (policy), attach to env so it uses wrappers
    model = MaskablePPO.load(model_path, env=venv)

    # If baseline provided, set opponents attribute on the env
    if baseline is not None:
        # Use VecEnv.set_attr to assign attribute on each wrapped env
        venv.set_attr("opponents", baseline)
    if fixed_num_players is not None:
        venv.set_attr("fixed_num_players", int(fixed_num_players))

    rows = []
    keyset = set()

    for epi in range(n_games):
        obs = venv.reset()
        done = False
        ep_breakdowns = []
        total_reward = 0.0
        length = 0
        while True:
            # Model expects the observation shape as provided by the vec env
            action, _ = model.predict(obs, deterministic=True)
            step_result = venv.step(action)
            # Support both gym (obs, reward, done, info) and gymnasium (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                # normalize to gymnasium-like values
                terminated = done
                truncated = np.array([False]) if isinstance(done, (list, tuple, np.ndarray)) else False
            # vec env returns arrays; take first element
            info0 = info[0] if isinstance(info, (list, tuple, np.ndarray)) else info
            bd = info0.get("reward_breakdown") if info0 is not None else None
            ep_breakdowns.append(bd or {})
            total_reward += float(reward[0]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
            length += 1
            done_flag = bool(terminated[0]) if isinstance(terminated, (list, tuple, np.ndarray)) else bool(terminated)
            if done_flag or (isinstance(truncated, (list, tuple, np.ndarray)) and bool(truncated[0])):
                break

        totals = aggregate_breakdowns(ep_breakdowns)
        keyset.update(totals.keys())
        row = {"episode": epi, "total_reward": total_reward, "length": length}
        row.update(totals)
        rows.append(row)

    # Write CSV with union of keys
    keys = ["episode", "total_reward", "length"] + sorted(k for k in keyset)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, 0.0) for k in keys})

    # Print summary averages
    agg = defaultdict(float)
    for r in rows:
        for k in keys:
            agg[k] += float(r.get(k, 0.0))
    n = len(rows)
    print("Evaluation summary (averages):")
    for k in keys:
        print(f"  {k}: {agg[k]/n:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the saved MaskablePPO model (.zip)")
    parser.add_argument("--vec", required=False, help="Path to VecNormalize file (optional)")
    parser.add_argument("--n", type=int, default=240, help="Number of eval episodes")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline opponents string (optional)")
    parser.add_argument("--players", type=int, default=None, help="Fixed number of players for eval (optional)")
    parser.add_argument("--out", type=str, default="eval_breakdown.csv", help="Output CSV path")
    args = parser.parse_args()

    run_eval(args.model, args.vec, args.n, args.baseline, args.players, args.out)


if __name__ == "__main__":
    main()
