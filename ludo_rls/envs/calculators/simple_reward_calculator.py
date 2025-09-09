"""Simple deterministic reward calculation utilities for LudoGymEnv (ludo_rls version).

This inherits from the base calculator and customizes for the ludo_rls environment.
"""

from rl_base.envs.calculators.reward_calculator import RewardCalculator
from ..model import EnvConfig


class SimpleRewardCalculator(RewardCalculator):
    """Self-play reward calculator."""
    
    def __init__(self, cfg: EnvConfig, game, agent_color: str):
        super().__init__(cfg, game, agent_color)
