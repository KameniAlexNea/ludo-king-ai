"""RLlib PPO training script configured to mirror the Stable-Baselines3 setup.

Usage:
        python -m ludo_rls.train_rllib --stop-timesteps 2000000 \
                --num-workers 8 --rollout-steps 512 --sgd-minibatch-size 256 \
                --num-sgd-epochs 10 --lr 3e-4 --ent-coef 0.01 --clip-range 0.2

Matches (as closely as RLlib allows) the SB3 hyperparameters used in `train_sb3.py`:
    * rollout_fragment_length ~= n_steps
    * train_batch_size = rollout_fragment_length * num_workers (one collection pass before update)
    * sgd_minibatch_size == SB3 batch_size
    * num_sgd_iter == SB3 n_epochs
    * gamma, gae_lambda, clip_param, entropy_coeff, vf_loss_coeff, grad_clip

Differences / Notes:
    * RLlib internally normalizes advantages and handles batching; we approximate SB3 update cadence.
    * Evaluation & tournament logic not included here; rely on Tune metrics.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
from ray import init, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

from .envs.ludo_env import EnvConfig, LudoGymEnv

ENV_NAME = "LudoGymEnv-v0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=2_000_000,
        help="Total environment timesteps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of rollout workers (parallel env actors).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=512,
        help="Per-worker rollout fragment length (SB3 n_steps).",
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=256,
        help="Minibatch size for PPO SGD (SB3 batch_size).",
    )
    parser.add_argument(
        "--num-sgd-epochs",
        type=int,
        default=10,
        help="Number of SGD passes per batch (SB3 n_epochs).",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate (SB3 learning_rate)."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient (SB3 ent_coef).",
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function loss coefficient."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.5, help="Global gradient clipping norm."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./rllib_results",
        help="Output directory for Tune logs.",
    )
    args = parser.parse_args()

    # Register env (gymnasium style) so RLlib knows it
    gym.register(id=ENV_NAME, entry_point=lambda **cfg: LudoGymEnv(EnvConfig()))

    init(ignore_reinit_error=True, include_dashboard=False)

    # Derive train_batch_size = rollout_steps * num_workers (single collection pass)
    train_batch_size = args.rollout_steps * max(1, args.num_workers)

    config: AlgorithmConfig = (
        PPOConfig()
        .environment(ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .training(
            model={"fcnet_hiddens": [256, 256]},
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_range,
            entropy_coeff=args.ent_coef,
            vf_loss_coeff=args.vf_coef,
            grad_clip=args.grad_clip,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_epochs,
        )
    )

    algo_config = config.resources(num_gpus=0)
    algo_config.rollouts(
        num_rollout_workers=args.num_workers,
        rollout_fragment_length=args.rollout_steps,
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=tune.RunConfig(
            stop={"timesteps_total": args.stop_timesteps},
            local_dir=args.logdir,
            verbose=1,
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
