from typing import Dict, List, Optional

from ludo_engine.models import ALL_COLORS
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env_base import LudoRLEnvBase
from ludo_rl.ludo_env.observation import ObservationBuilder
from ludo_rl.strategies.frozen_policy_strategy import FrozenPolicyStrategy
from ludo_rl.utils.opponents import sample_opponents


class LudoRLEnvHybrid(LudoRLEnvBase):
    """Hybrid environment that starts with self-play and switches to classic opponents after a certain number of steps.

    - Initially uses self-play: agent vs frozen copies of itself.
    - After switching, uses sampled scripted opponents with curriculum.
    """

    metadata = {"render_modes": ["human"], "name": "LudoRLEnvHybrid-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
        self.mode = "selfplay"  # Start in self-play mode
        self._progress: Optional[float] = None

        # Self-play components
        self.model: MaskablePPO = None
        self._frozen_policy: MaskableActorCriticPolicy = None
        self._opponent_builders: Dict[str, ObservationBuilder] = {}

    def set_model(self, model: MaskablePPO) -> None:
        """Inject the live model for self-play opponents."""
        self.model = model

    def switch_to_classic(self) -> None:
        """Switch from self-play to classic opponent sampling mode."""
        self.mode = "classic"

    def _snapshot_policy(self) -> None:
        try:
            self._frozen_policy = getattr(self.model, "policy", None)
        except Exception:
            self._frozen_policy = None

    def _sample_opponents(self) -> List[str]:
        """Sample opponents for classic mode."""
        return sample_opponents(
            self.cfg.opponents.candidates,
            self._progress,
            self.cfg.curriculum.boundaries,
            self.rng,
        )

    def on_reset_before_attach(self, options: Optional[Dict] = None) -> None:
        # Check if callback has signaled to switch to classic mode
        if getattr(self, "switch_to_classic", False):
            self.mode = "classic"

        if self.mode == "selfplay":
            # Build per-opponent observation builders for strategy perspectives
            self._opponent_builders = {}
            for c in ALL_COLORS:
                if c != self.agent_color:
                    self._opponent_builders[c] = ObservationBuilder(
                        self.cfg, self.game, c
                    )
            # Snapshot current policy for this episode (used by opponents)
            self._snapshot_policy()
        # For classic mode, no special setup needed here

    def attach_opponents(self, options: Optional[Dict] = None) -> None:
        if options and isinstance(options, dict) and "opponents" in options:
            strategies = options["opponents"]
            self._attach_strategies_mixed(strategies)
            return

        if self.mode == "selfplay":
            # Attach FrozenPolicyStrategy for each opponent
            strategies = [
                FrozenPolicyStrategy(
                    policy=self._frozen_policy,
                    obs_builder=self._opponent_builders[color],
                    deterministic=True,
                )
                for color in ALL_COLORS
                if color != self.agent_color
            ]
            self._attach_strategies_mixed(strategies)
        else:  # classic mode
            strategies = self._sample_opponents()
            colors = [c for c in ALL_COLORS if c != self.agent_color]
            for name, color in zip(strategies, colors):
                player = self.game.get_player_from_color(color)
                try:
                    from ludo_engine.strategies.strategy import StrategyFactory

                    player.set_strategy(StrategyFactory.create_strategy(name))
                except Exception:
                    pass

    def extra_reset_info(self) -> Dict:
        return {"progress": self._progress, "mode": self.mode}

    def set_training_progress(self, p: float):
        self._progress = max(0.0, min(1.0, float(p)))
