"""Multi-agent training script using PettingZoo + SuperSuit + Stable-Baselines3."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Optional

import supersuit as ss
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from models.analysis.eval_utils import evaluate_against_many
from models.arguments import parse_args
from models.callbacks.callbacks import PeriodicEvalCallback
from models.callbacks.self_play import SelfPlayCallback
from models.configs.config import EnvConfig, MultiAgentConfig
from models.envs.ludo_env_aec import OpponentPoolManager, TurnBasedSelfPlayEnv
from models.envs.ludo_env_aec import env as make_aec_env


def lr_schedule(lr_min, lr_max, lr_warmup) -> float:
    """Cosine annealing with warmup learning rate schedule."""

    def function(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < lr_warmup:
            return lr_min + (lr_max - lr_min) * (progress / lr_warmup)
        adjusted_progress = (progress - lr_warmup) / (1 - lr_warmup)
        return lr_max + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * adjusted_progress)
        )

    return function


def create_multiagent_env(
    env_cfg: EnvConfig,
    n_envs: int = 4,
    max_cycles: int = 300,
    opponent_pool: Optional[OpponentPoolManager] = None,
    ma_cfg: Optional[MultiAgentConfig] = None,
):
    """Create a vectorized multi-agent environment using SuperSuit.

    Args:
        env_cfg: Environment configuration
        n_envs: Number of parallel environments
        max_cycles: Maximum cycles per episode (maps to max_turns)
        opponent_pool: Pool of opponent models for self-play
        ma_cfg: Multi-agent configuration

    Returns:
        Vectorized environment compatible with SB3
    """

    # Create base PettingZoo environment
    def make_env_fn():
        def _init():
            cfg_copy = copy.deepcopy(env_cfg)
            if max_cycles is not None:
                cfg_copy.max_turns = max_cycles
            base_env = make_aec_env(cfg_copy)
            base_env = ss.pad_action_space_v0(base_env)
            return TurnBasedSelfPlayEnv(
                base_env, opponent_pool=opponent_pool, ma_cfg=ma_cfg
            )

        return _init

    env_fns = [make_env_fn() for _ in range(max(1, n_envs))]
    vec_env = DummyVecEnv(env_fns)
    return vec_env


def main() -> None:
    train_cfg, env_cfg = parse_args()
    env_cfg.multi_agent = True

    # Multi-agent specific config
    ma_cfg = MultiAgentConfig()

    os.makedirs(train_cfg.logdir, exist_ok=True)
    os.makedirs(train_cfg.model_dir, exist_ok=True)

    # Initialize opponent pool for self-play
    opponent_pool = None
    if ma_cfg.enable_self_play:
        pool_dir = os.path.join(train_cfg.model_dir, "opponent_pool")
        opponent_pool = OpponentPoolManager(pool_dir, ma_cfg.opponent_pool_size)
        print(f"Self-play enabled with pool size {ma_cfg.opponent_pool_size}")
        print(f"Existing opponents: {len(opponent_pool.get_all_opponents())}")

    # Update env config max_turns to match what SuperSuit expects
    env_cfg_copy = copy.deepcopy(env_cfg)

    # Create vectorized multi-agent environment
    print("Creating multi-agent environment...")
    vec_env = create_multiagent_env(
        env_cfg_copy,
        n_envs=train_cfg.n_envs,
        max_cycles=env_cfg_copy.max_turns,
        opponent_pool=opponent_pool,
        ma_cfg=ma_cfg,
    )

    # Wrap with monitor for logging
    vec_env = VecMonitor(vec_env, train_cfg.logdir)

    print(f"Environment created with {train_cfg.n_envs} parallel environments")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Policy configuration
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {
            "pi": list(train_cfg.pi_net_arch),
            "vf": list(train_cfg.vf_net_arch),
        },
    }

    # Create MaskablePPO model
    # Note: With SuperSuit's conversion, all 4 agents share the same policy
    # This is ideal for symmetric games like Ludo
    print("Initializing MaskablePPO model...")
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

    print("Model initialized successfully")
    print(f"Policy network: {train_cfg.pi_net_arch}")
    print(f"Value network: {train_cfg.vf_net_arch}")

    # Set up callbacks
    callbacks = []

    # Checkpoint callback - also manages self-play opponent pool
    if opponent_pool:
        checkpoint_callback = SelfPlayCallback(
            save_freq=max(1, ma_cfg.save_opponent_freq // train_cfg.n_envs),
            save_path=train_cfg.model_dir,
            opponent_pool=opponent_pool,
            name_prefix="ludo_ppo_ma",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, train_cfg.save_steps // train_cfg.n_envs),
            save_path=train_cfg.model_dir,
            name_prefix="ludo_ppo_ma",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    callbacks.append(checkpoint_callback)

    # Evaluation callback (evaluates against fixed opponents)
    eval_callback = PeriodicEvalCallback(
        env_cfg=env_cfg,
        opponents=train_cfg.eval_opponents,
        episodes=train_cfg.eval_episodes,
        eval_freq=train_cfg.eval_freq,
        deterministic=train_cfg.eval_deterministic,
    )
    callbacks.append(eval_callback)

    # Save initial model
    initial_path = os.path.join(train_cfg.model_dir, "ppo_ludo_multiagent_initial.zip")
    model.save(initial_path)
    print(f"Initial model saved to {initial_path}")

    # Initial evaluation before training begins
    print("\nRunning initial evaluation...")
    eval_env_cfg = copy.deepcopy(env_cfg)
    initial_eval = evaluate_against_many(
        model,
        train_cfg.eval_opponents,
        train_cfg.eval_episodes,
        eval_env_cfg,
        train_cfg.eval_deterministic,
    )
    for summary in initial_eval:
        print(
            f"  vs {summary.opponent}: win_rate={summary.win_rate:.3f} "
            f"avg_reward={summary.avg_reward:.2f} avg_length={summary.avg_length:.1f}"
        )

    # Train the model
    print(f"\nStarting training for {train_cfg.total_steps:,} timesteps...")
    print("=" * 80)

    model.learn(
        total_timesteps=train_cfg.total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    print("=" * 80)
    print("Training completed!")

    # Save final model
    final_path = os.path.join(train_cfg.model_dir, "ppo_ludo_multiagent_final.zip")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    # Print self-play statistics
    if opponent_pool:
        opponents = opponent_pool.get_all_opponents()
        print(f"\nFinal opponent pool size: {len(opponents)}")
        for i, opp in enumerate(opponents, 1):
            print(f"  {i}. {Path(opp).name}")


if __name__ == "__main__":
    main()
