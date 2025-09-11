"""Reward calculation utilities for LudoGymEnv."""

from typing import Dict, List, Optional

from ludo.constants import BoardConstants, GameConstants
from ludo.game import LudoGame
from ludo.player import Player

from ..model import BaseEnvConfig
from .probabilistic_calculator import ProbabilisticCalculator


class RewardCalculator:
    """Handles reward computation, including probabilistic modifiers."""

    def __init__(self, cfg: BaseEnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self.prob_calc = ProbabilisticCalculator(cfg, game, agent_color)

    # NOTE: Use color-aware forward distance to model capture feasibility.
    def _compute_positional_reward(self, move_res: Dict) -> float:
        """Reward smart positioning (approaching finish, staying together, etc.)."""
        target_pos = move_res.get("target_position")
        if target_pos is None or target_pos < 0:
            return 0.0

        reward = 0.0

        # Reward approaching home entry
        home_entry = BoardConstants.HOME_COLUMN_ENTRIES.get(self.agent_color, 0)
        if target_pos < GameConstants.HOME_COLUMN_START:
            distance_to_home = (home_entry - target_pos) % GameConstants.MAIN_BOARD_SIZE
            if distance_to_home <= 12:  # Within 12 spaces of home
                reward += (
                    self.cfg.reward_cfg.home_approach_bonus
                    * (12 - distance_to_home)
                    / 12
                )

        return reward

    def _compute_safety_reward(self, move_res: Dict) -> float:
        """Reward moving to safe positions when threatened."""
        target_pos = move_res.get("target_position")
        source_pos = move_res.get("source_position")

        if target_pos is None:
            return 0.0

        # Only reward if moving from dangerous to safe position
        if (
            source_pos is not None
            and not BoardConstants.is_safe_position(source_pos)
            and BoardConstants.is_safe_position(target_pos)
        ):
            source_risk = (
                self.prob_calc._compute_position_risk(source_pos)
                if source_pos >= 0
                else 0.0
            )
            if source_risk > 0.3:  # Only if there was significant risk
                return self.cfg.reward_cfg.safety_bonus * source_risk

        return 0.0

    def compute_progress_reward(
        self, progress_before: float, progress_after: float
    ) -> float:
        """Compute progress-based reward."""
        delta = progress_after - progress_before
        if abs(delta) > 1e-9:
            return delta * self.cfg.reward_cfg.progress_scale
        return 0.0

    def _compute_probabilistic_multiplier(self, move: Optional[Dict]) -> float:
        """Compute a simple, bounded risk-based multiplier for positive rewards."""
        if not self.cfg.reward_cfg.use_probabilistic_rewards or not move:
            return 1.0

        target_pos = move.get("target_position")
        if target_pos is None or not isinstance(target_pos, int) or target_pos < 0:
            return 1.0

        risk = self.prob_calc._compute_position_risk(target_pos)
        risk_reduction = self.cfg.reward_cfg.risk_weight * risk
        multiplier = 1.0 - risk_reduction

        # Reasonable bounds
        return max(0.5, min(1.0, multiplier))

    def compute_comprehensive_reward(
        self,
        move_res: Dict,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
    ) -> float:
        """Comprehensive reward system with strategic depth."""
        rcfg = self.cfg.reward_cfg
        total_reward = 0.0

        multiplier = self._compute_probabilistic_multiplier(move_res)

        # Major event rewards (highest priority)
        if move_res.get("captured_tokens"):
            capture_reward = rcfg.capture * len(move_res["captured_tokens"])
            if capture_reward > 0:
                capture_reward *= multiplier
            total_reward += capture_reward
            reward_components.append(capture_reward)  # capture

        if move_res.get("token_finished"):
            finish_reward = rcfg.finish_token
            total_reward += finish_reward  # Don't modulate - finishing is always good
            reward_components.append(finish_reward)  # finish

        # Moderate event rewards
        if extra_turn:
            extra_turn_reward = rcfg.extra_turn
            if extra_turn_reward > 0:
                extra_turn_reward *= multiplier
            total_reward += extra_turn_reward
            reward_components.append(extra_turn_reward)  # extra_turn

        # Strategic rewards (modulated by risk)
        positional_reward = self._compute_positional_reward(move_res)
        if positional_reward > 0:
            positional_reward *= multiplier
        total_reward += positional_reward
        reward_components.append(positional_reward)  # positional

        # Add safety reward after positional reward
        safety_reward = self._compute_safety_reward(move_res)
        if safety_reward > 0:
            safety_reward *= multiplier
        total_reward += safety_reward
        reward_components.append(safety_reward)  # safety

        # Small bonuses and penalties
        if diversity_bonus:
            diversity_reward = rcfg.diversity_bonus * multiplier
            total_reward += diversity_reward
            reward_components.append(diversity_reward)  # diversity

        if illegal_action:
            total_reward += rcfg.illegal_action  # Don't modulate penalties
            reward_components.append(rcfg.illegal_action)  # illegal

        # Progress reward (minimal, just for learning continuity)
        if abs(progress_delta) > 1e-9:
            progress_reward = progress_delta * rcfg.progress_scale
            if progress_reward > 0:
                progress_reward *= multiplier
            total_reward += progress_reward
            reward_components.append(progress_reward)  # progress

        # Time penalty (very small, just to encourage efficiency)
        total_reward += rcfg.time_penalty
        reward_components.append(rcfg.time_penalty)  # time

        # Inactivity penalty (encourage activating tokens)
        agent_player = next(p for p in self.game.players if p.color.value == self.agent_color)
        tokens_at_home = sum(1 for t in agent_player.tokens if t.position < 0)
        inactivity_penalty = tokens_at_home * rcfg.inactivity_penalty
        total_reward += inactivity_penalty
        reward_components.append(inactivity_penalty)  # inactivity

        # Active token bonus (reward having tokens on board)
        active_tokens = GameConstants.TOKENS_PER_PLAYER - tokens_at_home
        active_bonus = active_tokens * rcfg.active_token_bonus
        total_reward += active_bonus
        reward_components.append(active_bonus)  # active_bonus

        return total_reward

    def get_terminal_reward(
        self, agent_player: Player, opponents: list[Player]
    ) -> float:
        """Compute terminal rewards (win/lose)."""
        if agent_player.has_won():
            return self.cfg.reward_cfg.win
        elif any(p.has_won() for p in opponents):
            return self.cfg.reward_cfg.lose
        return 0.0
