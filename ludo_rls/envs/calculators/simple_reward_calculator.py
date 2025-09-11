"""Simple deterministic reward calculation utilities for LudoGymEnv (ludo_rls version).

This inherits from the base calculator and customizes for the ludo_rls environment.
"""

from typing import Dict, List

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
        token_positions_before: List[int] = None,
        masked_autocorrect: bool = False,
    ) -> Dict[str, float]:
        """Extended version that supports ludo_rls specific parameters.

        Args:
            token_positions_before: Token positions before the move (for ludo_rls compatibility)
            masked_autocorrect: Whether action was auto-corrected due to masking

        Returns:
            Dict of reward components (not total reward like base class)
        """
        # Use the base method but ignore the extra parameters for now
        # and return a dict instead of total reward
        components = {}

        rcfg = self.cfg.reward_cfg

        # Handle illegal actions
        if illegal_action:
            penalty = rcfg.illegal_action
            if masked_autocorrect:
                penalty *= getattr(rcfg, "illegal_masked_scale", 0.25)
            components["illegal"] = penalty

        # Handle other rewards similar to base class
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
