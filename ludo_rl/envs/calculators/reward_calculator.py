"""Reward calculation utilities for LudoGymEnv."""

from typing import Dict, List, Optional

from ludo.constants import BoardConstants, GameConstants
from ludo.game import LudoGame
from ludo.player import Player
from ludo.token import Token

from ..model import EnvConfig


class RewardCalculator:
    """Handles reward computation, including probabilistic modifiers."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self._risk_cache = {}  # Add cache
        self._cache_turn = -1  # Track when to invalidate

    def _backward_distance(self, from_pos: int, opp_pos: int) -> Optional[int]:
        """Compute backward distance from opp_pos to from_pos on the board."""
        if opp_pos < 0 or from_pos < 0:
            return None
        # Assuming circular board, but simplified
        if opp_pos >= from_pos:
            return opp_pos - from_pos
        else:
            return GameConstants.MAIN_BOARD_SIZE - from_pos + opp_pos

    # NOTE: Use color-aware forward distance to model capture feasibility.
    def _forward_distance_color_path(
        self, opp_pos: int, target_pos: int, opp_color: str
    ) -> Optional[int]:
        """Distance in steps an opponent must travel to reach target before entering its home column.

        If target is not reachable before the opponent would exit to its home column,
        returns None. Only main-board positions (< HOME_COLUMN_START) considered.
        """
        if opp_pos < 0 or BoardConstants.is_home_column_position(opp_pos):
            return None
        if target_pos < 0 or BoardConstants.is_home_column_position(target_pos):
            # Opponents in main board cannot capture inside home columns
            return None
        entry = BoardConstants.HOME_COLUMN_ENTRIES.get(opp_color, 0)
        # Steps until opponent reaches its home entry
        steps_to_entry = (entry - opp_pos) % GameConstants.MAIN_BOARD_SIZE
        # Steps to target along circular path
        steps_to_target = (target_pos - opp_pos) % GameConstants.MAIN_BOARD_SIZE
        if steps_to_target == 0:
            return None  # same square now, not a forward capture distance
        # If target requires moving beyond home entry, not capturable on this lap
        if 0 < steps_to_entry < steps_to_target:
            return None
        return steps_to_target

    def _single_turn_capture_probability(self, distance: Optional[int]) -> float:
        """Probability a specific backward distance can be rolled in one turn."""
        if (
            distance is None
            or distance < GameConstants.DICE_MIN
            or distance > GameConstants.DICE_MAX
        ):
            return 0.0
        # Uniform dice assumption
        return 1.0 / GameConstants.DICE_MAX

    def _compute_capture_probability(self, target_pos: int, opponent: Player) -> float:
        """Compute probability opponent can capture at target_pos next turn."""
        if target_pos < 0 or BoardConstants.is_safe_position(target_pos):
            return 0.0

        # Count unique dice rolls that enable capture
        capture_rolls = set()
        for token in opponent.tokens:
            if token.position < 0 or BoardConstants.is_home_column_position(
                token.position
            ):
                continue

            # Calculate what roll would move this token to target_pos
            required_roll = self._calculate_required_roll(
                token.position, target_pos, opponent.color.value
            )
            if (
                required_roll
                and GameConstants.DICE_MIN <= required_roll <= GameConstants.DICE_MAX
            ):
                capture_rolls.add(required_roll)

        return len(capture_rolls) / GameConstants.DICE_MAX

    def _compute_blocking_reward(self, move_res: Dict) -> float:
        """Enhanced blocking reward that considers strategic value."""
        target_pos = move_res.get("target_position")
        source_pos = move_res.get("source_position")

        if target_pos is None or target_pos < 0:
            return 0.0

        reward = 0.0
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )

        # Count tokens at target position after move
        tokens_at_target = sum(
            1 for t in agent_player.tokens if t.position == target_pos
        )

        # Reward creating blocks
        if tokens_at_target >= 2:
            block_strength = tokens_at_target - 1
            # Strategic locations get higher rewards
            location_multiplier = 1.0

            # Start positions are extra valuable to block
            if target_pos in BoardConstants.START_POSITIONS.values():
                location_multiplier = 2.0
            # Near home entries are valuable
            elif any(
                abs(target_pos - entry) <= 6
                for entry in BoardConstants.HOME_COLUMN_ENTRIES.values()
            ):
                location_multiplier = 1.5

            reward += (
                self.cfg.reward_cfg.blocking_bonus
                * block_strength
                * location_multiplier
            )

        # Penalize breaking valuable blocks
        if source_pos is not None and source_pos >= 0:
            tokens_remaining = sum(
                1 for t in agent_player.tokens if t.position == source_pos
            )
            if tokens_remaining == 1:  # Just broke a 2-token block
                # Less penalty if we're doing something valuable (capturing, finishing)
                if move_res.get("captured_tokens") or move_res.get("token_finished"):
                    penalty_multiplier = 0.3  # Reduced penalty for good moves
                else:
                    penalty_multiplier = 0.7
                reward -= self.cfg.reward_cfg.blocking_bonus * penalty_multiplier

        return reward

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

        # Reward being in home column (closer to finish)
        elif target_pos >= GameConstants.HOME_COLUMN_START:
            progress_in_home = (target_pos - GameConstants.HOME_COLUMN_START) / (
                GameConstants.HOME_COLUMN_SIZE - 1
            )
            reward += self.cfg.reward_cfg.home_progress_bonus * progress_in_home

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
                self._compute_position_risk(source_pos) if source_pos >= 0 else 0.0
            )
            if source_risk > 0.3:  # Only if there was significant risk
                return self.cfg.reward_cfg.safety_bonus * source_risk

        return 0.0

    def _can_token_reach_position(
        self, token: Token, target_pos: int, dice_roll: int
    ) -> bool:
        """Check if token can legally move to target_pos with dice_roll."""
        if token.position < 0:
            # Token in home - can only exit with 6
            player_color = token.player_color  # Fix attribute access
            start_pos = BoardConstants.START_POSITIONS.get(player_color, 0)
            return dice_roll == 6 and target_pos == start_pos

        if BoardConstants.is_home_column_position(token.position):
            # Token in home column - simple addition if valid
            expected_pos = token.position + dice_roll
            return (
                expected_pos == target_pos
                and expected_pos <= GameConstants.FINISH_POSITION
            )

        # Token on main board - calculate required roll
        required_roll = self._calculate_required_roll(
            token.position, target_pos, token.player_color
        )
        return required_roll == dice_roll

    def _calculate_required_roll(
        self, from_pos: int, to_pos: int, player_color: str
    ) -> Optional[int]:
        """Calculate dice roll needed to move from from_pos to to_pos."""
        if from_pos < 0 or to_pos < 0:
            return None

        # Handle home column entry
        home_entry = BoardConstants.HOME_COLUMN_ENTRIES.get(player_color, 0)

        if (
            from_pos < GameConstants.HOME_COLUMN_START
            and to_pos >= GameConstants.HOME_COLUMN_START
        ):
            # Moving from main board to home column
            steps_to_entry = (home_entry - from_pos) % GameConstants.MAIN_BOARD_SIZE
            if steps_to_entry == 0:
                steps_to_entry = GameConstants.MAIN_BOARD_SIZE  # Full lap needed

            # Steps within home column
            steps_in_home = to_pos - GameConstants.HOME_COLUMN_START
            total_steps = steps_to_entry + steps_in_home
            return total_steps if 1 <= total_steps <= GameConstants.DICE_MAX else None

        elif (
            from_pos < GameConstants.HOME_COLUMN_START
            and to_pos < GameConstants.HOME_COLUMN_START
        ):
            # Both on main board
            distance = (to_pos - from_pos) % GameConstants.MAIN_BOARD_SIZE
            return distance if distance > 0 else None

        elif (
            from_pos >= GameConstants.HOME_COLUMN_START
            and to_pos >= GameConstants.HOME_COLUMN_START
        ):
            # Both in home column
            return to_pos - from_pos if to_pos > from_pos else None

        return None

    def _compute_position_risk(self, position: int) -> float:
        """Cached risk calculation."""
        current_turn = getattr(self.game, "turn_count", 0)

        # Invalidate cache if game state changed
        if current_turn != self._cache_turn:
            self._risk_cache.clear()
            self._cache_turn = current_turn

        if position in self._risk_cache:
            return self._risk_cache[position]

        # Calculate risk (existing logic)
        if position < 0 or BoardConstants.is_safe_position(position):
            risk = 0.0
        else:
            risk = 0.0
            for opp_player in self.game.players:
                if opp_player.color.value == self.agent_color:
                    continue
                opp_risk = self._compute_capture_probability(position, opp_player)
                risk = risk + opp_risk - (risk * opp_risk)
            risk = min(1.0, risk)

        self._risk_cache[position] = risk
        return risk

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

        risk = self._compute_position_risk(target_pos)
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
        blocking_reward = self._compute_blocking_reward(move_res)
        if blocking_reward != 0:
            if blocking_reward > 0:
                blocking_reward *= multiplier
            total_reward += blocking_reward
            reward_components.append(blocking_reward)  # blocking

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
