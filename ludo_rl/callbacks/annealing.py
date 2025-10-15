from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from ludo_rl.config import TrainConfig


class AnnealingCallback(BaseCallback):
    """Anneal entropy coefficient and capture reward scale over training.

    This callback adjusts:
      - model.ent_coef linearly from entropy_coef_initial -> entropy_coef_final

    It expects that the training script passes in the TrainConfig and that the
    VecEnv exposes each underlying env with an attribute `cfg` referencing EnvConfig.
    Gracefully no-ops if attributes not present.
    """

    def __init__(
        self, train_cfg: TrainConfig, verbose: int = 0, scale_reward: bool = False
    ):
        super().__init__(verbose)
        self.train_cfg = train_cfg
        self.scale_reward = scale_reward

    def _on_step(self) -> bool:
        # Current step
        t = self.num_timesteps

        # Entropy annealing
        if self.train_cfg.entropy_anneal_steps > 0:
            frac = min(1.0, t / float(self.train_cfg.entropy_anneal_steps))
            new_ent = self.train_cfg.entropy_coef_initial + frac * (
                self.train_cfg.entropy_coef_final - self.train_cfg.entropy_coef_initial
            )
            try:
                # Directly set ent_coef - SB3 uses this in loss computation
                self.model.ent_coef = float(new_ent)
                if t % 1_000_000 == 0:
                    logger.info(f"[Annealing] Set ent_coef to {new_ent}")
            except Exception as e:
                if self.verbose > 0:
                    logger.error(
                        f"[Annealing] Failed to set ent_coef to {new_ent}: {e}"
                    )
        return True
