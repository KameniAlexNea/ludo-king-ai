from .base_ppo_strategy import BasePPOStrategy

from ..envs.ludo_env.ludo_env_multi_seat import LudoGymEnv
from ..envs.models.model_multi_seat import EnvConfig


class PPOStrategy(BasePPOStrategy):
    """Classic multi-seat PPO strategy wrapper."""

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
