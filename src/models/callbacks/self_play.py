"""Callback utilities for self-play training."""

from __future__ import annotations

import os

from stable_baselines3.common.callbacks import CheckpointCallback

from ..envs.ludo_env_aec import OpponentPoolManager


class SelfPlayCallback(CheckpointCallback):
    """Callback that saves models to the opponent pool at checkpoints."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        opponent_pool: OpponentPoolManager,
        name_prefix: str = "ludo_ppo",
        **kwargs,
    ) -> None:
        super().__init__(
            save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, **kwargs
        )
        self.opponent_pool = opponent_pool

    def _on_step(self) -> bool:
        result = super()._on_step()

        # When a checkpoint is written, also add it to the opponent pool.
        if self.n_calls % self.save_freq == 0:
            latest_model = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            if os.path.exists(latest_model):
                self.opponent_pool.add_opponent(latest_model, self.num_timesteps)
                print(f"Added opponent to pool at timestep {self.num_timesteps}")

        return result


__all__ = ["SelfPlayCallback"]
