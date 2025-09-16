"""Probabilistic calculations for risk and rewards."""

from typing import Dict, Optional

from ludo.constants import BoardConstants, GameConstants
from ludo.game import LudoGame
from ludo.player import Player
from ludo.token import Token
from rl_base.envs.model import BaseEnvConfig


class ProbabilisticCalculator:
    """Handles probabilistic calculations for risk assessment and rewards."""

    def __init__(self, cfg: BaseEnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color
        self._risk_cache = {}  # Cache for risk calculations
        self._cache_turn = -1  # Track when to invalidate cache

    def _backward_distance(self, from_pos: int, opp_pos: int) -> Optional[int]:
        """Compute backward distance from opp_pos to from_pos on the board."""
        if opp_pos < 0 or from_pos < 0:
            return None
        # Assuming circular board, but simplified
        if opp_pos >= from_pos:
            return opp_pos - from_pos
        else:
            return GameConstants.MAIN_BOARD_SIZE - from_pos + opp_pos

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
        steps_to_entry = self._backward_distance(opp_pos, entry)
        # Steps to target along circular path
        steps_to_target = self._backward_distance(opp_pos, target_pos)
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

    def _can_token_reach_position(
        self, token: Token, target_pos: int, dice_roll: int
    ) -> bool:
        """Check if token can legally move to target_pos with dice_roll."""
        if token.position < 0:
            # Token in home - can only exit with 6
            player_color = token.player_color
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
            steps_to_entry = self._backward_distance(from_pos, home_entry)

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

    def compute_probabilistic_multiplier(self, move: Optional[Dict]) -> float:
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
