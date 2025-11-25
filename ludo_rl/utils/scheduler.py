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


def warmup_plateau_decay_schedule(
    start: float,
    peak: float,
    end: float,
    warmup_fraction: float,
    plateau_fraction: float,
) -> Callable[[float], float]:
    """
    Generic schedule with warmup, plateau, and decay phases.

    Phases:
    1. Warmup (0 -> warmup_fraction): cosine ramp from start to peak
    2. Plateau (warmup -> warmup+plateau): maintain peak value
    3. Decay (remaining): cosine decay from peak to end

    Args:
        start: Initial value
        peak: Peak value (maintained during plateau)
        end: Final value after decay
        warmup_fraction: Fraction of training for warmup ramp
        plateau_fraction: Fraction of training to maintain peak
    """

    def schedule(progress_remaining: float) -> float:
        progress = 1 - progress_remaining

        if progress < warmup_fraction:
            # Warmup: cosine ramp up
            t = progress / warmup_fraction
            factor = 0.5 * (1 - math.cos(math.pi * t))
            return start + (peak - start) * factor

        elif progress < warmup_fraction + plateau_fraction:
            # Plateau: maintain peak
            return peak

        else:
            # Decay: cosine ramp down
            t = (progress - warmup_fraction - plateau_fraction) / (
                1 - warmup_fraction - plateau_fraction
            )
            factor = 0.5 * (1 + math.cos(math.pi * t))
            return end + (peak - end) * factor

    return schedule


def entropy_schedule(
    ent_start: float = 0.02,
    ent_peak: float = 0.03,
    ent_end: float = 0.005,
    warmup_fraction: float = 0.1,
    plateau_fraction: float = 0.3,
) -> Callable[[float], float]:
    """
    Entropy coefficient schedule that encourages exploration early, then decays.

    Args:
        ent_start: Initial entropy coefficient (moderate exploration)
        ent_peak: Peak entropy coefficient (maximum exploration)
        ent_end: Final entropy coefficient (exploitation focus)
        warmup_fraction: Fraction of training for warmup ramp
        plateau_fraction: Fraction of training to maintain peak entropy
    """
    return warmup_plateau_decay_schedule(
        start=ent_start,
        peak=ent_peak,
        end=ent_end,
        warmup_fraction=warmup_fraction,
        plateau_fraction=plateau_fraction,
    )


def target_kl_schedule(
    kl_start: float = 0.02,
    kl_peak: float = 0.06,
    kl_end: float = 0.025,
    warmup_fraction: float = 0.15,
    cooldown_fraction: float = 0.15,
) -> Callable[[float], float]:
    """
    Schedule for target_kl with warmup, plateau, and cooldown.

    Args:
        kl_start: Initial target_kl value (conservative)
        kl_peak: Maximum target_kl value (mid-training)
        kl_end: Final target_kl value (for stability)
        warmup_fraction: Fraction of training for warmup phase
        cooldown_fraction: Fraction of training for cooldown phase
    """
    # Convert cooldown to plateau: plateau = 1 - warmup - cooldown
    plateau_fraction = 1.0 - warmup_fraction - cooldown_fraction
    return warmup_plateau_decay_schedule(
        start=kl_start,
        peak=kl_peak,
        end=kl_end,
        warmup_fraction=warmup_fraction,
        plateau_fraction=plateau_fraction,
    )


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


class CurriculumSyncCallback(BaseCallback):
    """
    Syncs global training timesteps to all environments' curriculum samplers.
    
    In multi-env (vectorized) training, each env has its own OpponentLineupSampler.
    Without syncing, each env would track its own reset count, leading to slow/incorrect
    curriculum progression. This callback ensures all envs use the global timestep count
    for curriculum progress calculation.
    """

    def __init__(self, sync_interval: int = 1000, verbose: int = 0):
        """
        Args:
            sync_interval: How often (in timesteps) to sync. Lower = more overhead,
                          higher = more lag in curriculum updates. 1000 is a good default.
            verbose: Verbosity level (0 = silent, 1 = log syncs)
        """
        super().__init__(verbose)
        self.sync_interval = max(1, sync_interval)
        self._last_sync = 0

    def _on_step(self) -> bool:
        # Only sync every sync_interval timesteps to reduce overhead
        if self.num_timesteps - self._last_sync >= self.sync_interval:
            self._sync_timesteps()
            self._last_sync = self.num_timesteps
        return True

    def _sync_timesteps(self) -> None:
        """Sync global timesteps to all environments."""
        try:
            # Access the underlying envs through the VecEnv wrapper chain
            vec_env = self.model.get_env()
            if vec_env is None:
                return

            # Call env_method to set timesteps on all envs
            # This works with both DummyVecEnv and SubprocVecEnv
            vec_env.env_method("set_curriculum_timesteps", self.num_timesteps)

            if self.verbose > 0 and self.num_timesteps % (self.sync_interval * 100) == 0:
                logger.info(
                    f"Curriculum sync: {self.num_timesteps:,} timesteps -> all envs"
                )
        except AttributeError:
            # Environment doesn't have the method - skip silently
            pass
        except Exception as e:
            # Log but don't crash training
            logger.warning(f"Curriculum sync failed: {e}")
