"""Simple training entry-point that wires the minimal environment with MaskablePPO."""

from __future__ import annotations

import copy
import math
import os

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from models.ludo_env import LudoRLEnv
from models.callbacks import PeriodicEvalCallback
from models.arguments import parse_args


def lr_schedule(lr_min, lr_max, lr_warmup) -> float:
    def function(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < lr_warmup:
            return lr_min + (lr_max - lr_min) * (progress / lr_warmup)
        adjusted_progress = (progress - lr_warmup) / (1 - lr_warmup)
        return lr_max + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * adjusted_progress)
        )

    return function



def _mask_fn(env: LudoRLEnv):
    return env.valid_action_mask()


def main() -> None:
    train_cfg, env_cfg = parse_args()

    os.makedirs(train_cfg.logdir, exist_ok=True)
    os.makedirs(train_cfg.model_dir, exist_ok=True)

    def make_env_fn(rank: int):
        def _init():
            cfg_copy = copy.deepcopy(env_cfg)
            env = LudoRLEnv(cfg_copy)
            if cfg_copy.seed is not None:
                env.reset(seed=cfg_copy.seed + rank)
            return ActionMasker(env, _mask_fn)

        return _init

    if train_cfg.n_envs > 1:
        env_fns = [make_env_fn(i) for i in range(train_cfg.n_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([make_env_fn(0)])
    vec_env = VecMonitor(vec_env, train_cfg.logdir)

    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {
            "pi": list(train_cfg.pi_net_arch),
            "vf": list(train_cfg.vf_net_arch),
        },
    }

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=lr_schedule(
            lr_min=5e-6, lr_max=train_cfg.learning_rate, lr_warmup=0.05
        ),
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        ent_coef=train_cfg.ent_coef,
        vf_coef=train_cfg.vf_coef,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        tensorboard_log=train_cfg.logdir,
        verbose=1,
        seed=train_cfg.seed,
        device=train_cfg.device,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, train_cfg.save_steps // train_cfg.n_envs),
        save_path=train_cfg.model_dir,
        name_prefix="ludo_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = PeriodicEvalCallback(
        env_cfg=env_cfg,
        opponents=train_cfg.eval_opponents,
        episodes=train_cfg.eval_episodes,
        eval_freq=train_cfg.eval_freq,
        deterministic=train_cfg.eval_deterministic,
    )

    save_path = os.path.join(train_cfg.model_dir, "ppo_ludo_minimal.zip")
    model.save(save_path)

    model.learn(
        total_timesteps=train_cfg.total_steps,
        callback=[eval_callback, checkpoint_callback],
    )

    save_path = os.path.join(train_cfg.model_dir, "ppo_ludo_minimal.zip")
    model.save(save_path)


if __name__ == "__main__":
    main()
