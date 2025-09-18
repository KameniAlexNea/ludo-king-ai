"""Unified simple deterministic reward calculation utilities for LudoGymEnv.

This calculator supports both single-seat and multi-seat environments with
conditional features based on the environment configuration.
"""

from typing import Dict, List, Union

from ludo_engine import MoveResult, Player
from ludo_engine.models import BoardConstants, GameConstants

from ..models.model_multi_seat import EnvConfig as MultiSeatEnvConfig
from ..models.model_single_seat import EnvConfig as SingleSeatEnvConfig
from .reward_calculator_base import RewardCalculator

# Type alias for both config types
EnvConfig = Union[MultiSeatEnvConfig, SingleSeatEnvConfig]


class SimpleRewardCalculator(RewardCalculator):
    """Unified reward calculator supporting both single-seat and multi-seat environments."""

    def __init__(self, cfg: EnvConfig, game, agent_color: str):
        super().__init__(cfg, game, agent_color)
        # Detect environment type
        self.is_single_seat = isinstance(cfg, SingleSeatEnvConfig)
        self.is_multi_seat = isinstance(cfg, MultiSeatEnvConfig)

    def compute_comprehensive_reward(
        self,
        move_res: MoveResult,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
        start_position: int = -1,
        token_positions_before: List[int] = None,
        masked_autocorrect: bool = False,
    ) -> Dict[str, float]:
        """Compute comprehensive reward components with conditional features.

        Supports both single-seat and multi-seat specific parameters.

        Args:
            start_position: Starting position of the moved token (multi-seat feature)
            token_positions_before: Token positions before move (single-seat compatibility)
            masked_autocorrect: Whether action was auto-corrected (single-seat feature)

        Returns:
            Dict of reward components
        """
        if move_res is None:
            illegal_penalty = self.cfg.reward_cfg.illegal_action if illegal_action else 0.0
            if self.is_single_seat and masked_autocorrect and illegal_action:
                illegal_penalty *= getattr(self.cfg.reward_cfg, "illegal_masked_scale", 0.25)

            return {
                "illegal": illegal_penalty,
                "time": self.cfg.reward_cfg.time_penalty,
            }

        components = {}
        rcfg = self.cfg.reward_cfg

        # Handle illegal actions with single-seat specific scaling
        if illegal_action:
            penalty = rcfg.illegal_action
            if self.is_single_seat and masked_autocorrect:
                penalty *= getattr(rcfg, "illegal_masked_scale", 0.25)
            components["illegal"] = penalty

        # Basic rewards (common to both environments)
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

        if move_res and move_res.finished_token:
            components["token_finished"] = rcfg.finish_token

        # Progress reward with multi-seat strategic enhancements
        if progress_delta > 0:
            components["progress"] = progress_delta * rcfg.progress_scale

            # Multi-seat specific strategic bonuses
            if self.is_multi_seat and hasattr(rcfg, 'home_column_entry_bonus'):
                agent_player = self.game.get_player_from_color(self.agent_color)
                home_entry_pos = BoardConstants.HOME_COLUMN_ENTRIES[self.agent_color]

                # Find the token that moved
                moved_token = agent_player.tokens[move_res.token_id]

                if moved_token and start_position >= 0:
                    # Home column entry bonus
                    if (
                        moved_token.position >= home_entry_pos
                        and start_position < home_entry_pos
                    ):
                        components["home_column_entry"] = rcfg.home_column_entry_bonus

                    # Critical position bonuses
                    if moved_token.position in BoardConstants.STAR_SQUARES:  # Safe squares
                        components["safe_square_bonus"] = getattr(rcfg, 'safe_square_bonus', 0.0)
                    elif moved_token.position >= home_entry_pos + 1:  # Very close to home
                        components["near_home_bonus"] = getattr(rcfg, 'near_home_bonus', 0.0)

        # Diversity bonuses with multi-seat scaling
        if diversity_bonus:
            agent_player = self.game.get_player_from_color(self.agent_color)
            tokens_at_home = sum(1 for t in agent_player.tokens if t.position < 0)

            # Inactivity penalty
            penalty_scale = getattr(rcfg, 'diversity_penalty_scale', 1.0) if self.is_multi_seat else 1.0
            inactivity_penalty = tokens_at_home * rcfg.inactivity_penalty * penalty_scale
            components["inactivity"] = inactivity_penalty

            # Active token bonus
            active_tokens = GameConstants.TOKENS_PER_PLAYER - tokens_at_home
            bonus_scale = getattr(rcfg, 'diversity_bonus_scale', 1.0) if self.is_multi_seat else 1.0
            active_bonus = active_tokens * rcfg.active_token_bonus * bonus_scale
            components["active_bonus"] = active_bonus

        # Time penalty (common to both)
        components["time"] = rcfg.time_penalty

        return components

    def get_terminal_reward(
        self, agent_player: Player, opponents: list, truncated: bool = False
    ) -> float:
        """Terminal reward calculation supporting single-seat truncated episodes.

        Args:
            agent_player: The agent player
            opponents: List of opponent players
            truncated: Whether episode was truncated (single-seat feature)

        Returns:
            Terminal reward value
        """
        rcfg = self.cfg.reward_cfg

        if self.game.game_over:
            if agent_player.has_won():
                return rcfg.win
            return rcfg.lose

        # Single-seat specific: handle truncated episodes
        if self.is_single_seat and truncated:
            # Only treat as draw if nobody actually won
            if not any(p.has_won() for p in self.game.players):
                return getattr(rcfg, 'draw_penalty', -2.0)

        return 0.0
