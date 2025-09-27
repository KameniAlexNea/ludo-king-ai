from typing import Optional

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from ludo_rl.config import EnvConfig, TrainConfig


class AnnealingCallback(BaseCallback):
    """Anneal entropy coefficient and capture reward scale over training.

    This callback adjusts:
      - model.ent_coef linearly from entropy_coef_initial -> entropy_coef_final
      - env.cfg.reward.capture_reward_scale from capture_scale_initial -> capture_scale_final

    It expects that the training script passes in the TrainConfig and that the
    VecEnv exposes each underlying env with an attribute `cfg` referencing EnvConfig.
    Gracefully no-ops if attributes not present.
    """

    def __init__(self, train_cfg: TrainConfig, verbose: int = 0, scale_reward: bool = False):
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
            except Exception as e:
                if self.verbose > 0:
                    logger.error(
                        f"[Annealing] Failed to set ent_coef to {new_ent}: {e}"
                    )

        # Log current values periodically
        if (
            self.train_cfg.anneal_log_freq > 0
            and t % self.train_cfg.anneal_log_freq == 0
        ):
            try:
                current_lr = self.model.policy.optimizer.param_groups[0]["lr"]
                ent = getattr(self.model, "ent_coef", None)
                logger.info(
                    f"[Anneal] step={t} lr={current_lr:.6g} ent={ent} capture_scale={self.train_cfg.capture_scale_initial}->{self.train_cfg.capture_scale_final}"
                )
            except Exception as e:
                if self.verbose > 0:
                    logger.error(f"[Annealing] Failed to log annealing values: {e}")

        # Capture reward scale annealing (env side)
        if self.scale_reward and self.train_cfg.capture_scale_anneal_steps > 0:
            frac_c = min(1.0, t / float(self.train_cfg.capture_scale_anneal_steps))
            new_scale = self.train_cfg.capture_scale_initial + frac_c * (
                self.train_cfg.capture_scale_final
                - self.train_cfg.capture_scale_initial
            )
            try:
                # Access underlying envs if vectorized
                vec_env = getattr(self.model, "env", None)
                if vec_env is not None and hasattr(vec_env, "envs"):
                    for e in vec_env.envs:
                        cfg: Optional[EnvConfig] = getattr(e, "cfg", None)
                        if cfg is not None:
                            cfg.reward.capture_reward_scale = float(new_scale)
            except Exception:
                pass

        return True
