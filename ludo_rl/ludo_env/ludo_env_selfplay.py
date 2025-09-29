from typing import Dict, Optional

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


class LudoRLEnvSelfPlay(LudoRLEnvBase):
    """Self-play environment using a frozen copy of the policy as opponents.

    - At reset, the agent color is selected (optionally randomized).
    - The learning agent controls only that color; other 3 colors use a frozen model.
    - A callback can periodically provide a new frozen model path and obs_rms stats.
    """

    metadata = {"render_modes": ["human"], "name": "LudoRLEnvSelfPlay-v0"}

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)

        # Live training model and frozen snapshot used by opponents
        self.model: MaskablePPO = None
        self._frozen_policy: MaskableActorCriticPolicy = None
        self._opponent_builders: Dict[str, ObservationBuilderBase] = {}
        self.obs_normalizer = None

    # ---- Model snapshot management (in-memory) ----
    def set_model(self, model: MaskablePPO) -> None:
        """Inject the live model; env will snapshot its policy on reset for opponents."""
        self.model = model

    def _snapshot_policy(self) -> None:
        try:
            self._frozen_policy = getattr(self.model, "policy", None)
        except Exception:
            self._frozen_policy = None

    def set_obs_normalizer(self, obs_normalizer) -> None:
        """Inject the observation normalizer for frozen policy strategies."""
        self.obs_normalizer = obs_normalizer

    # ---- gym api ----
    def on_reset_before_attach(self, options: Optional[Dict] = None) -> None:
        # Build per-opponent observation builders for strategy perspectives
        self._opponent_builders = {}
        for c in ALL_COLORS:
            if c != self.agent_color:
                if getattr(self.cfg, "obs", None) and getattr(
                    self.cfg.obs, "discrete", False
                ):
                    self._opponent_builders[c] = DiscreteObservationBuilder(
                        self.cfg, self.game, c
                    )
                else:
                    self._opponent_builders[c] = ContinuousObservationBuilder(
                        self.cfg, self.game, c
                    )
        # Snapshot current policy for this episode (used by opponents)
        self._snapshot_policy()

    def attach_opponents(self, options: Optional[Dict] = None) -> None:
        # Allow passing explicit strategies via options
        if options and isinstance(options, dict) and "opponents" in options:
            strategies = options["opponents"]
            self._attach_strategies_mixed(strategies)
            return
        # Default: attach FrozenPolicyStrategy for each opponent
        strategies = [
            FrozenPolicyStrategy(
                policy=self._frozen_policy,
                obs_builder=self._opponent_builders[color],
                deterministic=True,
                obs_normalizer=self.obs_normalizer,
            )
            for color in ALL_COLORS
            if color != self.agent_color
        ]
        self._attach_strategies_mixed(strategies)
