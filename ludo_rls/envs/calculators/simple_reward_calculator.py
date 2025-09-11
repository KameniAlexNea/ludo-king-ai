"""Simple deterministic reward calculation utilities for LudoGymEnv (ludo_rls version).

This inherits from the base calculator and customizes for the ludo_rls environment.
"""

from typing import Dict, List

from ludo.constants import GameConstants
from rl_base.envs.calculators.reward_calculator import RewardCalculator

from ..model import EnvConfig


class SimpleRewardCalculator(RewardCalculator):
    """Self-play reward calculator with ludo_rls specific features."""

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
        masked_autocorrect: bool = True,
        *args, **kwargs
    ) -> Dict[str, float]:
        """Extended version that supports ludo_rls specific parameters.

        Args:
            token_positions_before: Token positions before the move (for ludo_rls compatibility)
            masked_autocorrect: Whether action was auto-corrected due to masking

        Returns:
            Dict of reward components (not total reward like base class)
        """
        # Call parent method to get base components (ignore extra parameters)
        components = super().compute_comprehensive_reward(
            move_res=move_res,
            progress_delta=progress_delta,
            extra_turn=extra_turn,
            diversity_bonus=diversity_bonus,
            illegal_action=illegal_action,
            reward_components=reward_components,
            *args, **kwargs
        )

        # ludo_rls specific customizations
        rcfg = self.cfg.reward_cfg

        # Handle masked_autocorrect for illegal actions
        if illegal_action and masked_autocorrect:
            penalty = rcfg.illegal_action
            penalty *= getattr(rcfg, "illegal_masked_scale", 0.25)
            components["illegal"] = penalty

        # Handle got_captured (not in base class)
        if move_res.get("got_captured"):
            components["got_captured"] = rcfg.got_captured

        # Ensure component names match expected format
        if "finish" in components:
            components["token_finished"] = components.pop("finish")

        return components

    def get_terminal_reward(
        self, agent_player, opponents: list, truncated: bool = False
    ) -> float:
        """Extended version that supports truncated parameter for ludo_rls.

        Args:
            agent_player: The agent player
            opponents: List of opponent players
            truncated: Whether the episode was truncated due to timeout

        Returns:
            Terminal reward value
        """
        rcfg = self.cfg.reward_cfg

        if self.game.game_over:
            if agent_player.has_won():
                return rcfg.win
            return rcfg.lose
        if truncated:
            # Only treat as draw if nobody actually won
            if not any(p.has_won() for p in self.game.players):
                return getattr(rcfg, "draw_penalty", -2.0)
        return 0.0
