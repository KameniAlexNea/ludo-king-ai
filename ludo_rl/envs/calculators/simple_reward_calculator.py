"""Simple deterministic reward calculation utilities for LudoGymEnv (ludo_rl version).

This inherits from the base calculator and customizes for the ludo_rl environment.
"""

from typing import Dict, List

from ludo_engine import MoveResult
from ludo_engine.models import BoardConstants, GameConstants

from ludo_rl.envs.model import EnvConfig
from rl_base.envs.calculators.reward_calculator import RewardCalculator


class SimpleRewardCalculator(RewardCalculator):
    """Classic multi-seat reward calculator."""

    def __init__(self, cfg: EnvConfig, game, agent_color: str):
        super().__init__(cfg, game, agent_color)

    def compute_comprehensive_reward(
        self,
        move_res: MoveResult,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
        start_position: int = -1,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """Compute comprehensive reward components with strategic depth.

        Returns:
            Dict of reward components (not total reward like base class)
        """
        if move_res is None:
            return {
                "illegal": (
                    self.cfg.reward_cfg.illegal_action if illegal_action else 0.0
                ),
                "time": self.cfg.reward_cfg.time_penalty,
            }
        components = {}

        rcfg = self.cfg.reward_cfg

        # Handle illegal actions
        if illegal_action:
            components["illegal"] = rcfg.illegal_action

        # Handle other rewards
        if move_res.captured_tokens:
            capture_count = len(move_res.captured_tokens)
            components["capture"] = rcfg.capture * capture_count

        if (
            move_res.new_position == GameConstants.HOME_POSITION
            and move_res.old_position != GameConstants.HOME_POSITION
        ):
            components["got_captured"] = rcfg.got_captured

        if extra_turn:
            components["extra_turn"] = rcfg.extra_turn

        if move_res.finished_token:
            components["token_finished"] = rcfg.finish_token

        # Enhanced progress reward with strategic milestones
        if progress_delta > 0:
            components["progress"] = progress_delta * rcfg.progress_scale

            # Strategic milestone bonuses - only for the token that actually moved
            agent_player = self.game.get_player_from_color(self.agent_color)
            home_entry_pos = BoardConstants.HOME_COLUMN_ENTRIES[self.agent_color]

            # Find the token that moved
            moved_token = agent_player.tokens[move_res.token_id]
            moved_token_start = start_position

            if moved_token:
                # Home column entry bonus (when entering home column approach)
                if (
                    moved_token.position >= home_entry_pos
                    and moved_token_start < home_entry_pos
                ):
                    components["home_column_entry"] = rcfg.home_column_entry_bonus

                # Critical position bonuses (safe squares, near home)
                if moved_token.position in BoardConstants.STAR_SQUARES:  # Safe squares
                    components["safe_square_bonus"] = rcfg.safe_square_bonus
                elif moved_token.position >= home_entry_pos + 1:  # Very close to home
                    components["near_home_bonus"] = rcfg.near_home_bonus

        if diversity_bonus:
            # Inactivity penalty (encourage activating tokens) - configurable scaling
            agent_player = self.game.get_player_from_color(self.agent_color)
            tokens_at_home = sum(1 for t in agent_player.tokens if t.position < 0)
            inactivity_penalty = (
                tokens_at_home * rcfg.inactivity_penalty * rcfg.diversity_penalty_scale
            )
            components["inactivity"] = inactivity_penalty

            # Active token bonus (reward having tokens on board) - configurable scaling
            active_tokens = GameConstants.TOKENS_PER_PLAYER - tokens_at_home
            active_bonus = (
                active_tokens * rcfg.active_token_bonus * rcfg.diversity_bonus_scale
            )
            components["active_bonus"] = active_bonus

        # Time penalty (small per-step cost)
        components["time"] = rcfg.time_penalty

        return components
