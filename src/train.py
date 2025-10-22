"""Simple training entry-point that wires the minimal environment with MaskablePPO."""

from __future__ import annotations

import argparse
import copy
import math
import os

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from models.config import EnvConfig, TrainConfig
from models.eval_utils import evaluate_against_many
from models.ludo_env import LudoRLEnv


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
        "--pi-net-arch",
        type=int,
        nargs="+",
        default=list(defaults.pi_net_arch),
        help="Policy branch architecture, e.g. --pi-net-arch 256 128.",
    )
    parser.add_argument(
        "--vf-net-arch",
        type=int,
        nargs="+",
        default=list(defaults.vf_net_arch),
        help="Value branch architecture, e.g. --vf-net-arch 256 128.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=defaults.n_envs,
        help=(
            "Number of parallel environments (>=1). Eight CPU workers are a good default."
        ),
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
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=defaults.eval_freq,
        help="Run evaluations every N timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=defaults.eval_episodes,
        help="Number of evaluation games per opponent when evaluations run.",
    )
    parser.add_argument(
        "--eval-opponents",
        type=str,
        default=",".join(defaults.eval_opponents),
        help="Comma separated list of opponent strategies for evaluation runs.",
    )
    parser.add_argument(
        "--eval-stochastic",
        dest="eval_deterministic",
        action="store_false",
        default=defaults.eval_deterministic,
        help="Use stochastic actions during evaluation instead of deterministic default.",
    )

    args = parser.parse_args()

    eval_opponents = tuple(
        opponent.strip()
        for opponent in args.eval_opponents.split(",")
        if opponent.strip()
    )
    if not eval_opponents:
        eval_opponents = defaults.eval_opponents
    if args.eval_freq <= 0:
        parser.error("--eval-freq must be positive")

    if args.save_steps <= 0:
        parser.error("--save-steps must be positive")

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
        n_envs=max(1, args.n_envs),
        eval_freq=args.eval_freq,
        eval_episodes=max(1, args.eval_episodes),
        eval_opponents=eval_opponents,
        eval_deterministic=args.eval_deterministic,
        pi_net_arch=tuple(args.pi_net_arch),
        vf_net_arch=tuple(args.vf_net_arch),
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


class PeriodicEvalCallback(BaseCallback):
    """Run lightweight policy evaluations at a fixed timestep cadence."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        opponents: tuple[str, ...],
        episodes: int,
        eval_freq: int,
        deterministic: bool,
    ) -> None:
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        super().__init__(verbose=1)
        self.base_cfg = copy.deepcopy(env_cfg)
        self.opponents = opponents
        self.episodes = episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self._next_eval = eval_freq
        self._last_eval_step = 0

    def _run_eval(self) -> None:
        summaries = evaluate_against_many(
            self.model,
            self.opponents,
            self.episodes,
            self.base_cfg,
            self.deterministic,
        )
        step = int(self.num_timesteps)
        if self.verbose > 0:
            print(f"\n[Eval] timesteps={step}")
            for summary in summaries:
                print(
                    f"  vs {summary.opponent}: win_rate={summary.win_rate:.3f} "
                    f"avg_reward={summary.avg_reward:.2f} avg_length={summary.avg_length:.1f}"
                )
        for summary in summaries:
            prefix = f"eval/{summary.opponent}"
            self.logger.record(f"{prefix}/win_rate", summary.win_rate)
            self.logger.record(f"{prefix}/avg_reward", summary.avg_reward)
            self.logger.record(f"{prefix}/avg_length", summary.avg_length)
        self.logger.dump(step)
        self._last_eval_step = step

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if int(self.num_timesteps) != self._last_eval_step:
            self._run_eval()


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
