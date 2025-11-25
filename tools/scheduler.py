import math
from typing import Callable

from loguru import logger
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


def target_kl_schedule(
    kl_start: float = 0.02,
    kl_peak: float = 0.06,
    kl_end: float = 0.025,
    warmup_fraction: float = 0.15,
    cooldown_fraction: float = 0.15,
) -> Callable[[float], float]:
    """
    Schedule for target_kl that:
    - Starts conservative (kl_start) to avoid early convergence
    - Increases to kl_peak during warmup
    - Maintains kl_peak during mid-training
    - Decreases to kl_end during cooldown for stability

    Args:
        kl_start: Initial target_kl value (conservative)
        kl_peak: Maximum target_kl value (mid-training)
        kl_end: Final target_kl value (for stability)
        warmup_fraction: Fraction of training for warmup phase
        cooldown_fraction: Fraction of training for cooldown phase
    """

    def schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining

        if progress < warmup_fraction:
            # Warmup: smoothly increase from kl_start to kl_peak
            warmup_progress = progress / warmup_fraction
            # Use smooth interpolation (cosine)
            factor = 0.5 * (1 - math.cos(math.pi * warmup_progress))
            return kl_start + (kl_peak - kl_start) * factor

        elif progress > (1 - cooldown_fraction):
            # Cooldown: smoothly decrease from kl_peak to kl_end
            cooldown_progress = (progress - (1 - cooldown_fraction)) / cooldown_fraction
            # Use smooth interpolation (cosine)
            factor = 0.5 * (1 - math.cos(math.pi * cooldown_progress))
            return kl_peak - (kl_peak - kl_end) * factor

        else:
            # Mid-training: maintain peak value
            return kl_peak

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

        # Log every 10% progress
        if self.num_timesteps % (self.total_timesteps // 10) < self.model.n_envs:
            new_value = getattr(self.model, self.att)
            logger.debug(f"train/{self.att}: {new_value}")
        return True
