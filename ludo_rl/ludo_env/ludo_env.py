from typing import Any, Dict, List, Optional

from ludo_engine.models import ALL_COLORS
from ludo_engine.strategies.strategy import StrategyFactory

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env_base import LudoRLEnvBase
from ludo_rl.utils.opponents import sample_opponents


class LudoRLEnv(LudoRLEnvBase):
    metadata = {"render_modes": ["human"], "name": "LudoRLEnv-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
        self._progress: Optional[float] = None

    # ---- opponent sampling ----
    def _sample_opponents(self) -> List[str]:
        return sample_opponents(
            self.cfg.opponents.candidates,
            self._progress,
            self.cfg.curriculum.boundaries,
            self.rng,
        )

    def attach_opponents(self, options: Optional[Dict[str, Any]] = None) -> None:
        if options and isinstance(options, dict) and "opponents" in options:
            strategies = options["opponents"]
        else:
            strategies = self._sample_opponents()
        # Derive opponent colors from the current game player order (skip agent)
        colors = [p.color for p in self.game.players if p.color != self.agent_color]
        for name, color in zip(strategies, colors):
            player = self.game.get_player_from_color(color)
            try:
                player.set_strategy(StrategyFactory.create_strategy(name))
            except Exception:
                pass

    # ---- gym api hooks ----
    def extra_reset_info(self) -> Dict[str, Any]:
        return {"progress": self._progress}

    # called by callback to set progress 0..1
    def set_training_progress(self, p: float):
        self._progress = max(0.0, min(1.0, float(p)))
