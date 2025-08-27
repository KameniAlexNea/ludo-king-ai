"""RLlib training entry point.
Register env via gymnasium interface.
Run:
    python -m ludo_rl.train_rllib --stop-timesteps 2000000
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
from ray import tune, init
from ray.rllib.algorithms.ppo import PPOConfig

from .envs.ludo_env import LudoGymEnv, EnvConfig

ENV_NAME = "LudoGymEnv-v0"

def _env_creator(config):
    return LudoGymEnv(EnvConfig())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="./rllib_results")
    args = parser.parse_args()

    # Register env (gymnasium style) so RLlib knows it
    gym.register(id=ENV_NAME, entry_point=lambda **cfg: LudoGymEnv(EnvConfig()))

    init(ignore_reinit_error=True, include_dashboard=False)

    algo_config = (
        PPOConfig()
        .environment(ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=args.num_workers)
        .training(model={"fcnet_hiddens": [256, 256]}, train_batch_size=4000, sgd_minibatch_size=256)
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
        run_config=tune.RunConfig(stop={"timesteps_total": args.stop_timesteps}, local_dir=args.logdir),
    )
    tuner.fit()

if __name__ == "__main__":
    main()
