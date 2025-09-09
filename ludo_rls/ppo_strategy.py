from rl_base.strategies.base_ppo_strategy import BasePPOStrategy
from .envs.ludo_env import LudoGymEnv
from .envs.model import EnvConfig


class PPOStrategy(BasePPOStrategy):
    """Single-seat PPO strategy wrapper using shared ObservationBuilder."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        env_config: EnvConfig | None = None,
        deterministic: bool = True,
    ):
        # Convert to base config for initialization
        if env_config is None:
            base_config = EnvConfig()
        else:
            base_config = env_config
            
        super().__init__(model_path, model_name, base_config, deterministic)

    def _create_dummy_env(self):
        """Create the dummy environment for this implementation."""
        return LudoGymEnv(self.env_cfg)

    def _inject_context_into_dummy_game(self, ctx: dict) -> tuple[int, int]:
        """Use the shorter method name variant for ludo_rls compatibility."""
        return self._inject_context(ctx)
    
    def _inject_context(self, ctx: dict) -> tuple[int, int]:
        """Inject context into dummy game - ludo_rls variant method name."""
        return super()._inject_context_into_dummy_game(ctx)
