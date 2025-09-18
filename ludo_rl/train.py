from __future__ import annotations

import argparse
import copy
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.callbacks.curriculum import ProgressCallback
from ludo_rl.callbacks.eval_baselines import SimpleBaselineEvalCallback
from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv
from ludo_rl.ludo_env.ludo_env_selfplay import LudoRLEnvSelfPlay
from ludo_rl.callbacks.selfplay_sync import SelfPlaySyncCallback
from ludo_rl.utils.move_utils import MoveUtils
# build per-rank env


def make_env(rank: int, seed: int, base_cfg: EnvConfig, env_type: str = "classic"):
    def _init():
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed + rank
        if env_type == "selfplay":
            env = LudoRLEnvSelfPlay(cfg)
        else:
            env = LudoRLEnv(cfg)
        return ActionMasker(env, MoveUtils.get_action_mask_for_env)

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--logdir", type=str, default=TrainConfig.logdir)
    p.add_argument("--model-dir", type=str, default=TrainConfig.model_dir)
    p.add_argument("--eval-freq", type=int, default=TrainConfig.eval_freq)
    p.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    p.add_argument(
        "--eval-baselines",
        type=str,
        default=TrainConfig.eval_baselines,
        help="Comma-separated list of opponent strategy names",
    )
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--n-steps", type=int, default=TrainConfig.n_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    p.add_argument("--max-turns", type=int, default=TrainConfig.max_turns)
    p.add_argument("--env-type", type=str, default="classic", choices=["classic", "selfplay"], help="Environment type")
    p.add_argument("--selfplay-sync-freq", type=int, default=100_000, help="Timesteps between frozen model snapshots for self-play")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env_cfg = EnvConfig(max_turns=args.max_turns)

    if args.n_envs == 1:
        venv = DummyVecEnv([make_env(0, 42, env_cfg, args.env_type)])
    else:
        venv = SubprocVecEnv(
            [make_env(i, 42 + i * 100, env_cfg, args.env_type) for i in range(args.n_envs)]
        )

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_reward=False)

    # Separate eval env with same wrappers (always classic for evaluation vs baselines)
    eval_env = DummyVecEnv([make_env(999, 1337, env_cfg, "classic")])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    try:
        eval_env.obs_rms = venv.obs_rms
    except Exception:
        pass

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        tensorboard_log=args.logdir,
        verbose=1,
        device="auto",
    )

    progress_cb = ProgressCallback(total_timesteps=args.total_steps, update_freq=10_000)
    eval_cb = SimpleBaselineEvalCallback(
        baselines=[s.strip() for s in args.eval_baselines.split(",") if s.strip()],
        n_games=args.eval_games,
        eval_freq=args.eval_freq,
        env_cfg=env_cfg,
        verbose=1,
    )
    callbacks = [progress_cb, eval_cb]
    if args.env_type == "selfplay":
        callbacks.append(SelfPlaySyncCallback(save_dir=args.model_dir, save_freq=args.selfplay_sync_freq, verbose=1))

    model.learn(total_timesteps=args.total_steps, callback=callbacks)
    model.save(os.path.join(args.model_dir, "maskable_ppo_ludo_rl_final"))


if __name__ == "__main__":
    main()
