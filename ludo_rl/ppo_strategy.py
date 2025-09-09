from ludo.constants import Colors

from rl_base.strategies.base_ppo_strategy import BasePPOStrategy
from .envs.ludo_env import LudoGymEnv
from .envs.model import EnvConfig


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
            
        # Enforce fixed training seat (important for meaning of observation ordering)
        if base_config.agent_color != Colors.RED:
            raise ValueError(
                f"PPOStrategy expects agent_color RED (training seat); got {base_config.agent_color}"
            )
            
        super().__init__(model_path, model_name, base_config, deterministic)

    def _create_dummy_env(self):
        """Create the dummy environment for this implementation."""
        return LudoGymEnv(self.env_cfg)
