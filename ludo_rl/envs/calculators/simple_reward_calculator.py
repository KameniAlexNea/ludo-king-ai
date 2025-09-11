"""Simple deterministic reward calculation utilities for LudoGymEnv (ludo_rl version).

This inherits from the base calculator and customizes for the ludo_rl environment.
"""

from typing import Dict, List

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
    ) -> Dict[str, float]:
        """Compute comprehensive reward components.

        Returns:
            Dict of reward components (not total reward like base class)
        """
        components = {}

        rcfg = self.cfg.reward_cfg

        # Handle illegal actions
        if illegal_action:
            components["illegal"] = rcfg.illegal_action

        # Handle other rewards
        if move_res.get("captured_tokens"):
            capture_count = len(move_res["captured_tokens"])
            components["capture"] = rcfg.capture * capture_count

        if move_res.get("got_captured"):
            components["got_captured"] = rcfg.got_captured

        if extra_turn:
            components["extra_turn"] = rcfg.extra_turn

        if move_res.get("token_finished"):
            components["token_finished"] = rcfg.finish_token

        # Progress reward
        if progress_delta > 0:
            components["progress"] = progress_delta * rcfg.progress_scale

        # Time penalty (small per-step cost)
        components["time"] = rcfg.time_penalty

        return components
