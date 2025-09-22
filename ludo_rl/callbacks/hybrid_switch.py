from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class HybridSwitchCallback(BaseCallback):
    """Callback to switch hybrid environment from self-play to classic mode after a certain number of steps."""

    def __init__(self, switch_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.switch_step = int(switch_step)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.switch_step:
            try:
                # Switch all envs to classic mode
                self.training_env.set_attr("switch_to_classic", True)
                if self.verbose:
                    print(f"[HybridSwitch] Switched to classic mode at step {self.num_timesteps}")
            except Exception as e:
                if self.verbose:
                    print(f"[HybridSwitch] Error switching modes: {e}")
        return True