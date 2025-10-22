"""Simple training entry-point that wires the minimal environment with MaskablePPO."""

from __future__ import annotations

import argparse
import copy
import math
import os
from collections.abc import Mapping

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from models.config import EnvConfig, TrainConfig
from models.ludo_env import LudoRLEnv


def lr_schedule(lr_min, lr_max, lr_warmup) -> float:
    def function(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < lr_warmup:
            return lr_min + (lr_max - lr_min) * (progress / lr_warmup)
        else:
            adjusted_progress = (progress - lr_warmup) / (1 - lr_warmup)
            return lr_max + 0.5 * (lr_max - lr_min) * (
                1 + math.cos(math.pi * adjusted_progress)
            )

    return function


def _parse_args() -> tuple[TrainConfig, EnvConfig]:
    defaults = TrainConfig()
    env_defaults = EnvConfig()

    parser = argparse.ArgumentParser(description="Train a PPO agent on classic Ludo.")
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--ent-coef", type=float, default=defaults.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=defaults.vf_coef)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--logdir", type=str, default=defaults.logdir)
    parser.add_argument("--model-dir", type=str, default=defaults.model_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed, nargs="?")
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=None,
        help="Optional custom policy net architecture shared between policy and value, e.g. --net-arch 512 256.",
    )
    parser.add_argument(
        "--pi-net-arch",
        type=int,
        nargs="+",
        default=None,
        help="Optional policy branch architecture, e.g. --pi-net-arch 256 128.",
    )
    parser.add_argument(
        "--vf-net-arch",
        type=int,
        nargs="+",
        default=None,
        help="Optional value branch architecture, e.g. --vf-net-arch 256 128.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=defaults.n_envs,
        help="Number of parallel environments (>=1). Eight CPU workers are a good default.",
    )
    parser.add_argument("--max-turns", type=int, default=env_defaults.max_turns)
    parser.add_argument(
        "--fixed-agent-color",
        type=str,
        default=None,
        help="Optional fixed color for the learning agent (e.g. BLUE).",
    )
    parser.add_argument(
        "--opponent-strategy",
        type=str,
        default=env_defaults.opponent_strategy,
        help="Strategy name understood by ludo_engine.StrategyFactory.",
    )

    args = parser.parse_args()

    if args.net_arch and (args.pi_net_arch or args.vf_net_arch):
        parser.error(
            "--net-arch cannot be used together with --pi-net-arch/--vf-net-arch"
        )

    net_arch = None
    if args.pi_net_arch or args.vf_net_arch:
        net_arch = {}
        if args.pi_net_arch:
            net_arch["pi"] = tuple(args.pi_net_arch)
        if args.vf_net_arch:
            net_arch["vf"] = tuple(args.vf_net_arch)
    elif args.net_arch:
        net_arch = tuple(args.net_arch)

    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        logdir=args.logdir,
        model_dir=args.model_dir,
        seed=args.seed,
        device=args.device,
        save_steps=args.save_steps,
        net_arch=net_arch,
        n_envs=max(1, args.n_envs),
    )

    env_cfg = EnvConfig(
        max_turns=args.max_turns,
        seed=args.seed,
        randomize_agent=args.fixed_agent_color is None,
        opponent_strategy=args.opponent_strategy,
    )
    if args.fixed_agent_color:
        env_cfg.randomize_agent = False
        env_cfg.fixed_agent_color = args.fixed_agent_color
        env_cfg.opponent_strategy = args.opponent_strategy

    return train_cfg, env_cfg


def _mask_fn(env: LudoRLEnv):
    return env.valid_action_mask()


def main() -> None:
    train_cfg, env_cfg = _parse_args()

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

    # Optionally use a custom feature extractor when discrete observations are enabled
    policy_kwargs = {"activation_fn": torch.nn.LeakyReLU}
    if train_cfg.net_arch:
        if isinstance(train_cfg.net_arch, Mapping):
            net_arch_cfg = {}
            if "pi" in train_cfg.net_arch:
                net_arch_cfg["pi"] = list(train_cfg.net_arch["pi"])
            if "vf" in train_cfg.net_arch:
                net_arch_cfg["vf"] = list(train_cfg.net_arch["vf"])
            if not net_arch_cfg:
                raise ValueError(
                    "net_arch mapping must contain at least one of 'pi' or 'vf'."
                )
            policy_kwargs["net_arch"] = net_arch_cfg
        else:
            shared_layers = list(train_cfg.net_arch)
            policy_kwargs["net_arch"] = {
                "pi": shared_layers.copy(),
                "vf": shared_layers.copy(),
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

    checkpoint_callback = None
    if train_cfg.save_steps and train_cfg.save_steps > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=train_cfg.save_steps // train_cfg.n_envs,
            save_path=train_cfg.model_dir,
            name_prefix="ppo_checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=False,
        )
    save_path = os.path.join(train_cfg.model_dir, "ppo_ludo_minimal.zip")
    model.save(save_path)

    model.learn(
        total_timesteps=train_cfg.total_steps,
        callback=checkpoint_callback,
    )

    save_path = os.path.join(train_cfg.model_dir, "ppo_ludo_minimal.zip")
    model.save(save_path)


if __name__ == "__main__":
    main()
