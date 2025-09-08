from __future__ import annotations

from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class ProgressCurriculumCallback(BaseCallback):
    """Updates envs with normalized training progress for curriculum sampling.

    Args:
        total_timesteps: Total timesteps planned for training (used to normalize progress 0..1).
        update_freq: How often to push progress to envs.
        attr_name: The attribute on the env to set (default: '_training_progress').
    """

    def __init__(
        self,
        total_timesteps: int,
        update_freq: int = 10_000,
        attr_name: str = "_training_progress",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.update_freq = max(1, int(update_freq))
        self.attr_name = attr_name

    def _on_step(self) -> bool:
        # Push progress periodically
        if self.num_timesteps % self.update_freq != 0:
            return True
        p = min(1.0, max(0.0, self.num_timesteps / float(self.total_timesteps)))
        try:
            # Works for DummyVecEnv and SubprocVecEnv
            self.training_env.set_attr(self.attr_name, p)
        except Exception:
            pass
        return True
