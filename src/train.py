"""Simple training entry-point that wires the minimal environment with MaskablePPO."""
from __future__ import annotations

import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from models.config import EnvConfig, TrainConfig
from models.ludo_env import LudoRLEnv


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

    def make_env():
        return ActionMasker(LudoRLEnv(env_cfg), _mask_fn)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env, train_cfg.logdir)

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=train_cfg.learning_rate,
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
    )

    model.learn(total_timesteps=train_cfg.total_steps)

    save_path = os.path.join(train_cfg.model_dir, "ppo_ludo_minimal")
    model.save(save_path)


if __name__ == "__main__":
    main()
