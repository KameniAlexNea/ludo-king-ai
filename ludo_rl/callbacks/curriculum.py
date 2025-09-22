from __future__ import annotations

from sb3_contrib.common.maskable.callbacks import *
from stable_baselines3.common.callbacks import EveryNTimesteps, EvalCallback


class ProgressCallback(EvalCallback):
    def __init__(
        self, total_timesteps: int, update_freq: int = 10_000, verbose: int = 0
    ):
        super().__init__(verbose)
        self.total = max(1, int(total_timesteps))
        self.freq = max(1, int(update_freq))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq != 0:
            return True
        p = min(1.0, self.num_timesteps / float(self.total))
        try:
            self.training_env.set_attr("_progress", p)
        except Exception:
            pass
        return True
