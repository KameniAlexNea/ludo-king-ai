from typing import Dict, List, Optional

from loguru import logger
from ludo_engine import StrategyFactory
from ludo_engine.models import ALL_COLORS
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ludo_rl.config import EnvConfig
from ludo_rl.ludo_env.ludo_env_base import LudoRLEnvBase
from ludo_rl.ludo_env.observation import (
    ContinuousObservationBuilder,
    DiscreteObservationBuilder,
    ObservationBuilderBase,
)
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
        self.env_mode = "selfplay"  # "selfplay" or "classic"
        self._progress: Optional[float] = None

        # Self-play components
        self.model: MaskablePPO = None
        self._frozen_policy: MaskableActorCriticPolicy = None
        self._opponent_builders: Dict[str, ObservationBuilderBase] = {}
        self.obs_normalizer = None
        self.obs_builder_cls = (
            DiscreteObservationBuilder
            if self.cfg.obs.discrete
            else ContinuousObservationBuilder
        )

    def set_model(self, model: MaskablePPO) -> None:
        """Inject the live model for self-play opponents."""
        self.model = model

    def set_obs_normalizer(self, obs_normalizer) -> None:
        """Inject the observation normalizer for frozen policy strategies."""
        self.obs_normalizer = obs_normalizer

    def _sample_opponents(self, num_opponents: int) -> List[str]:
        return sample_opponents(
            self.cfg.opponents.candidates,
            self._progress,
            self.cfg.curriculum.boundaries,
            self.rng,
            num_opponents,
        )

    def _snapshot_policy(self) -> None:
        try:
            self._frozen_policy = getattr(self.model, "policy", None)
        except Exception as e:
            logger.error(f"Failed to snapshot policy: {e}")
            self._frozen_policy = None

    def on_reset_before_attach(self, options: Optional[Dict] = None) -> None:
        # Check if callback has signaled to switch to classic mode
        if self.env_mode == "selfplay":
            # Build per-opponent observation builders for strategy perspectives
            self._opponent_builders = {}
            for c in ALL_COLORS:
                if c != self.agent_color:
                    self._opponent_builders[c] = self.obs_builder_cls(
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

        if self.env_mode == "selfplay":
            # Attach FrozenPolicyStrategy for each opponent
            strategies = [
                FrozenPolicyStrategy(
                    policy=self._frozen_policy,
                    obs_builder=self._opponent_builders[color],
                    deterministic=False,
                    obs_normalizer=self.obs_normalizer,
                )
                for color in ALL_COLORS
                if color != self.agent_color
            ]
            self._attach_strategies_mixed(strategies)
        else:  # classic mode
            # Derive opponent colors from the current game player order (skip agent)
            colors = [p.color for p in self.game.players if p.color != self.agent_color]
            strategies = self._sample_opponents(len(colors))
            if len(strategies) != len(colors):
                raise ValueError(
                    f"Number of strategies ({len(strategies)}) must match number of opponent colors ({len(colors)})"
                )

            for name, color in zip(strategies, colors):
                player = self.game.get_player_from_color(color)
                player.set_strategy(StrategyFactory.create_strategy(name))

    def extra_reset_info(self) -> Dict:
        return {"progress": self._progress, "env_mode": self.env_mode}

    def set_training_progress(self, p: float):
        self._progress = max(0.0, min(1.0, float(p)))
