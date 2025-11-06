import math
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


def lr_schedule(
    lr_min: float = 1e-5, lr_max: float = 3e-4, warmup_steps: float = 0.03
) -> Callable[[float], float]:
    lr_min, lr_max = min(lr_min, lr_max), max(lr_min, lr_max)

    def schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < warmup_steps:
            return lr_min + (lr_max - lr_min) * (progress / warmup_steps)
        else:
            adjusted_progress = (progress - warmup_steps) / (1 - warmup_steps)
            return lr_min + 0.5 * (lr_max - lr_min) * (
                1 + math.cos(math.pi * adjusted_progress)
            )

    return schedule


class CoefScheduler(BaseCallback):
    """Dynamically adjust the entropy coefficient using a cosine schedule."""

    def __init__(
        self, total_timesteps: int, att: str, schedule: Callable[[float], float]
    ):
        super().__init__()
        self.total_timesteps = max(1, total_timesteps)
        self.schedule = schedule
        self.att = att

    def _on_step(self) -> bool:
        progress_remaining = 1.0 - (self.num_timesteps / self.total_timesteps)
        progress_remaining = float(max(0.0, min(1.0, progress_remaining)))
        setattr(self.model, self.att, float(self.schedule(progress_remaining)))
        return True
