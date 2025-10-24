"""Multi-agent training script using PettingZoo + SuperSuit + Stable-Baselines3."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import supersuit as ss
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from models.arguments import parse_args
from models.callbacks import PeriodicEvalCallback
from models.config import EnvConfig, MultiAgentConfig
from models.ludo_env_aec import TurnBasedSelfPlayEnv
from models.ludo_env_aec import env as make_aec_env


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


class OpponentPoolManager:
    """Manages a pool of opponent models for self-play training."""

    def __init__(self, pool_dir: str, pool_size: int = 5):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.pool_size = pool_size
        self.opponents: list[str] = []
        self._load_existing_opponents()

    def _load_existing_opponents(self):
        """Load existing opponent models from disk."""
        if not self.pool_dir.exists():
            return

        opponent_files = sorted(
            self.pool_dir.glob("opponent_*.zip"), key=lambda p: p.stat().st_mtime
        )
        self.opponents = [str(f) for f in opponent_files[-self.pool_size :]]

    def add_opponent(self, model_path: str, timestep: int):
        """Add a new opponent to the pool."""
        opponent_path = self.pool_dir / f"opponent_{timestep}.zip"

        # Copy model to opponent pool
        import shutil

        shutil.copy(model_path, opponent_path)

        self.opponents.append(str(opponent_path))

        # Maintain pool size by removing oldest
        if len(self.opponents) > self.pool_size:
            old_opponent = Path(self.opponents.pop(0))
            if old_opponent.exists():
                old_opponent.unlink()

    def sample_opponent(self) -> Optional[str]:
        """Sample a random opponent from the pool."""
        if not self.opponents:
            return None
        return np.random.choice(self.opponents)

    def get_all_opponents(self) -> list[str]:
        """Return all opponents in the pool."""
        return self.opponents.copy()


class SelfPlayCallback(CheckpointCallback):
    """Callback that saves models to opponent pool periodically."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        opponent_pool: OpponentPoolManager,
        name_prefix: str = "ludo_ppo",
        **kwargs,
    ):
        super().__init__(
            save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, **kwargs
        )
        self.opponent_pool = opponent_pool

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Every time we save a checkpoint, also add to opponent pool
        if self.n_calls % self.save_freq == 0:
            # Find the most recent model
            latest_model = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            if os.path.exists(latest_model):
                self.opponent_pool.add_opponent(latest_model, self.num_timesteps)
                print(f"Added opponent to pool at timestep {self.num_timesteps}")

        return result


def create_multiagent_env(env_cfg: EnvConfig, n_envs: int = 4, max_cycles: int = 300):
    """Create a vectorized multi-agent environment using SuperSuit.

    Args:
        env_cfg: Environment configuration
        n_envs: Number of parallel environments
        max_cycles: Maximum cycles per episode (maps to max_turns)

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
            return TurnBasedSelfPlayEnv(base_env)

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
        env_cfg_copy, n_envs=train_cfg.n_envs, max_cycles=env_cfg_copy.max_turns
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
