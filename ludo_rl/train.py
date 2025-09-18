from __future__ import annotations

import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from ludo_rl.callbacks.curriculum import ProgressCallback
from ludo_rl.config import EnvConfig, TrainConfig
from ludo_rl.ludo_env.ludo_env import LudoRLEnv


# build per-rank env

def make_env(rank: int, seed: int, cfg: EnvConfig):
    def _init():
        env = LudoRLEnv(cfg)

        def mask_fn(e):
            return e.step(0)[-1]["action_mask"]  # Not used; we override via ActionMasker below

        # Use ActionMasker with env's mask via info on step; but better: attach callable
        def mask_fn2(env_inst):
            return env_inst._pending_valid and np.array([vm.token_id in [m.token_id for m in env_inst._pending_valid] for vm in env_inst._pending_valid])

        # Proper mask from env utility
        def mask_fn3(env_inst):
            return env_inst.action_space.contains(0) or env_inst._pending_valid

        # Minimal mask callback bridging env state to masker
        def mask_fn_final(env_inst):
            try:
                from ludo_rl.utils.move_utils import MoveUtils

                return MoveUtils.action_mask(getattr(env_inst, "_pending_valid", None))
            except Exception:
                return np.ones(4, dtype=bool)

        return ActionMasker(env, mask_fn_final)

    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--logdir", type=str, default=TrainConfig.logdir)
    p.add_argument("--model-dir", type=str, default=TrainConfig.model_dir)
    p.add_argument("--eval-freq", type=int, default=TrainConfig.eval_freq)
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--n-steps", type=int, default=TrainConfig.n_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    p.add_argument("--max-turns", type=int, default=TrainConfig.max_turns)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env_cfg = EnvConfig(max_turns=args.max_turns)

    if args.n_envs == 1:
        venv = DummyVecEnv([make_env(0, 42, env_cfg)])
    else:
        venv = SubprocVecEnv([make_env(i, 42 + i * 100, env_cfg) for i in range(args.n_envs)])

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_reward=False)

    # Separate eval env with same wrappers
    eval_env = DummyVecEnv([make_env(999, 1337, env_cfg)])
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

    model.learn(total_timesteps=args.total_steps, callback=[progress_cb])
    model.save(os.path.join(args.model_dir, "maskable_ppo_ludo_rl_final"))


if __name__ == "__main__":
    main()
