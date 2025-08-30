"""Reward calculation utilities for LudoGymEnv."""

from typing import Dict, List, Optional

from ludo.constants import BoardConstants, Colors, GameConstants, StrategyConstants
from ludo.game import LudoGame
from ludo.player import Player

from ..model import EnvConfig


class RewardCalculator:
    """Handles reward computation, including probabilistic modifiers."""

    def __init__(self, cfg: EnvConfig, game: LudoGame, agent_color: str):
        self.cfg = cfg
        self.game = game
        self.agent_color = agent_color

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
        """Compute probability this opponent can capture token at target_pos next turn."""
        if not self._can_reach_position(target_pos, opponent):
            return 0.0
        
        capture_ways = 0
        for dice_roll in range(GameConstants.DICE_MIN, GameConstants.DICE_MAX + 1):
            # Check if ANY of opponent's tokens can reach target_pos with this roll
            for token in opponent.tokens:
                if self._can_token_reach_position(token, target_pos, dice_roll):
                    capture_ways += 1
                    break  # Only need one token per dice roll
        
        return capture_ways / GameConstants.DICE_MAX

    def _can_reach_position(self, target_pos: int, opponent: Player) -> bool:
        """Check if opponent has any token that can potentially reach target_pos."""
        for token in opponent.tokens:
            if token.position >= 0 and not BoardConstants.is_home_column_position(token.position):
                # Simplified: assume if on board, could potentially move
                return True
        return False

    def _can_token_reach_position(self, token, target_pos: int, dice_roll: int) -> bool:
        """Check if a specific token can reach target_pos with given dice_roll."""
        if token.position < 0 or BoardConstants.is_home_column_position(token.position):
            return False
        # Simplified distance check (you may need to implement proper movement logic)
        distance = (target_pos - token.position) % GameConstants.MAIN_BOARD_SIZE
        return distance == dice_roll

    def _compute_position_risk(self, position: int) -> float:
        """Total risk of being captured at this position."""
        if BoardConstants.is_safe_position(position):
            return 0.0
        
        total_risk = 0.0
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                continue
            risk = self._compute_capture_probability(position, opp_player)
            # Risk compounds but with diminishing returns
            total_risk = total_risk + risk - (total_risk * risk)
        
        return total_risk

    def _compute_probabilistic_reward_modifier(
        self, move: Dict, reward_components: List[float]
    ) -> float:
        """Compute modifier for rewards based on probabilities."""
        if not self.cfg.reward_cfg.use_probabilistic_rewards:
            return 1.0

        opponent_positions = []
        for color in Colors.ALL_COLORS:
            if color.lower() == self.agent_color.lower():
                continue
            opp = next(
                p for p in self.game.players if p.color.value.lower() == color.lower()
            )
            opponent_positions.extend(t.position for t in opp.tokens if t.position >= 0)

        risk = self._compute_position_risk(move.get("target_position", -1))
        # Reduce reward if high risk
        modifier = 1.0 - self.cfg.reward_cfg.risk_weight * risk
        return max(0.1, modifier)  # floor to prevent negative

    def compute_capture_reward(
        self, move_res: Dict, reward_components: List[float]
    ) -> float:
        """Compute capture reward with probabilistic modifier."""
        if not move_res.get("captured_tokens"):
            return 0.0
        base_capture_reward = self.cfg.reward_cfg.capture * len(
            move_res["captured_tokens"]
        )
        modifier = self._compute_probabilistic_reward_modifier(
            move_res, reward_components
        )
        return base_capture_reward * modifier

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
        if not move or not isinstance(move, dict):
            return 1.0

        target_pos = move.get("target_position")
        if target_pos is None or not isinstance(target_pos, int) or target_pos < 0:
            return 1.0

        risk = self._compute_position_risk(target_pos)
        
        # Simple risk modulation: higher risk = lower multiplier
        risk_multiplier = 1.0 - (0.3 * risk)  # Max 30% reduction
        return max(0.7, risk_multiplier)  # Floor at 0.7 to keep signals meaningful

    def compute_comprehensive_reward(
        self,
        move_res: Dict,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
    ) -> float:
        """Simple, effective reward system with risk-modulated positive rewards.

        Clear reward signals with meaningful magnitudes:
        - Event rewards dominate step rewards
        - Risk modulates positive actions consistently
        - No systematic bias against actions
        """
        rcfg = self.cfg.reward_cfg
        total_reward = 0.0

        multiplier = self._compute_probabilistic_multiplier(move_res)

        # Event-based rewards (clear, meaningful signals)
        if move_res.get("captured_tokens"):
            capture_reward = rcfg.capture * len(move_res["captured_tokens"])
            if capture_reward > 0:
                capture_reward *= multiplier
            total_reward += capture_reward
            reward_components.append(capture_reward)

        if move_res.get("token_finished"):
            finish_reward = rcfg.finish_token
            if finish_reward > 0:
                finish_reward *= multiplier
            total_reward += finish_reward
            reward_components.append(finish_reward)

        if extra_turn:
            extra_turn_reward = rcfg.extra_turn
            if extra_turn_reward > 0:
                extra_turn_reward *= multiplier
            total_reward += extra_turn_reward
            reward_components.append(extra_turn_reward)

        if diversity_bonus:
            diversity_reward = rcfg.diversity_bonus
            if diversity_reward > 0:
                diversity_reward *= multiplier
            total_reward += diversity_reward
            reward_components.append(diversity_reward)

        if illegal_action:
            illegal_reward = rcfg.illegal_action
            total_reward += illegal_reward  # Don't modulate penalties
            reward_components.append(illegal_reward)

        # Simple progress reward (scaled up to be meaningful)
        if abs(progress_delta) > 1e-9:
            progress_reward = progress_delta * rcfg.progress_scale
            if progress_reward > 0:
                progress_reward *= multiplier
            total_reward += progress_reward
            reward_components.append(progress_reward)

        # Small time penalty (not modulated)
        total_reward += rcfg.time_penalty

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
