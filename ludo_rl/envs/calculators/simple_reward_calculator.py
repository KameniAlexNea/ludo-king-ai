"""Simple deterministic reward calculation utilities for LudoGymEnv (ludo_rl version).

This inherits from the base calculator and customizes for the ludo_rl environment.
"""

from typing import Dict, List

from ludo.constants import GameConstants
from rl_base.envs.calculators.reward_calculator import RewardCalculator

from ..model import EnvConfig


class SimpleRewardCalculator(RewardCalculator):
    """Classic multi-seat reward calculator."""

    def __init__(self, cfg: EnvConfig, game, agent_color: str):
        super().__init__(cfg, game, agent_color)

    def compute_comprehensive_reward(
        self,
        move_res: Dict,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
        *args, **kwargs
    ) -> Dict[str, float]:
        """Compute comprehensive reward components using parent class logic.

        Returns:
            Dict of reward components (not total reward like base class)
        """
        # Call parent method to get base components
        components = super().compute_comprehensive_reward(
            move_res=move_res,
            progress_delta=progress_delta,
            extra_turn=extra_turn,
            diversity_bonus=diversity_bonus,
            illegal_action=illegal_action,
            reward_components=reward_components,
            *args, **kwargs
        )

        # SimpleRewardCalculator specific customizations
        rcfg = self.cfg.reward_cfg

        # Handle got_captured (not in base class)
        if move_res.get("got_captured"):
            components["got_captured"] = rcfg.got_captured

        # Ensure component names match expected format
        if "finish" in components:
            components["token_finished"] = components.pop("finish")

        return components
