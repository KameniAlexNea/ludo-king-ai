from __future__ import annotations

import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class SelfPlaySyncCallback(BaseCallback):
    """Periodically save a snapshot of the current model and update envs.

    The envs are expected to implement a `load_frozen_model(path, obs_rms)` method.
    """

    def __init__(self, save_dir: str, save_freq: int = 100_000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.save_freq = int(save_freq)
        os.makedirs(self.save_dir, exist_ok=True)
        self._last_path: Optional[str] = None

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if self.num_timesteps == 0 or (self.num_timesteps % self.save_freq) != 0:
            return True
        # Save snapshot
        path = os.path.join(self.save_dir, f"selfplay_snapshot_{self.num_timesteps}.zip")
        try:
            self.model.save(path)
            self._last_path = path
        except Exception as e:
            if self.verbose:
                print(f"[SelfPlaySync] Failed to save model: {e}")
            return True

        # Broadcast to all envs
        try:
            obs_rms = getattr(self.model.env, "obs_rms", None)
        except Exception:
            obs_rms = None
        try:
            # Vector envs support set_attr
            self.model.get_env().set_attr("_", None)  # no-op to ensure call works
            self.model.get_env().env_method("load_frozen_model", path, obs_rms)
        except Exception:
            # Fallback: try single env attribute
            try:
                env = self.model.get_env()
                if hasattr(env, "load_frozen_model"):
                    env.load_frozen_model(path, obs_rms)
            except Exception:
                pass
        if self.verbose:
            print(f"[SelfPlaySync] Updated frozen opponents from {path}")
        return True
