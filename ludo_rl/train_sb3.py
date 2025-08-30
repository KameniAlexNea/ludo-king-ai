"""Stable-Baselines3 training script for Ludo.

Usage (after installing requirements):
    python -m ludo_rl.train_sb3 --total-steps 2000000 --n-envs 8

Produces models/, logs/ and tensorboard metrics.
"""

from __future__ import annotations

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from .envs.ludo_env import EnvConfig, LudoGymEnv


def make_env(rank: int, seed: int, base_cfg: EnvConfig):
    def _init():
        cfg = EnvConfig(**{**base_cfg.__dict__})
        cfg.seed = seed + rank
        env = LudoGymEnv(cfg)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="Max turns per episode (reduce to increase variance)",
    )
    parser.add_argument(
        "--no-reward-norm",
        action="store_true",
        help="Disable VecNormalize reward normalization for debugging",
    )
    parser.add_argument(
        "--no-probabilistic-rewards",
        action="store_true",
        help="Disable probabilistic reward shaping for debugging",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient for exploration (higher = more exploration)",
    )
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    base_cfg = EnvConfig(max_turns=args.max_turns)
    if args.no_probabilistic_rewards:
        base_cfg.reward_cfg.use_probabilistic_rewards = False

    if args.n_envs == 1:
        env_fns = [make_env(0, 42, base_cfg)]
        vec_env = DummyVecEnv(env_fns)
    else:
        # Use different seeds for each environment to add stochasticity
        import random
        env_fns = [make_env(i, 42 + i * 100, base_cfg) for i in range(args.n_envs)]
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env, filename=os.path.join(args.logdir, "monitor.csv"))
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=not args.no_reward_norm,
        clip_obs=10.0,
    )

    # Build evaluation env with identical wrapper stack (Monitor + VecNormalize) so
    # normalization stats can be synchronized without assertion errors.
    eval_env_raw = DummyVecEnv([make_env(999, 42, base_cfg)])
    eval_env_raw = VecMonitor(eval_env_raw)
    # For evaluation we typically want raw rewards; keep norm_reward False always
    eval_env = VecNormalize(
        eval_env_raw,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,  # Use CLI argument
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.logdir,
        device="auto",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq // len(env_fns),
        save_path=args.model_dir,
        name_prefix="ppo_ludo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.logdir,
        eval_freq=args.eval_freq // len(env_fns),
        deterministic=False,  # Changed to False for stochastic evaluation
        n_eval_episodes=5,  # Evaluate more episodes
    )

    model.learn(total_timesteps=args.total_steps, callback=[checkpoint_cb, eval_cb])
    model.save(os.path.join(args.model_dir, "ppo_ludo_final"))


if __name__ == "__main__":
    main()
