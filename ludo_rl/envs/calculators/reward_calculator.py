"""Reward calculation utilities for LudoGymEnv."""

from typing import Dict, List, Optional

from ludo.constants import Colors, GameConstants
from ludo.game import LudoGame

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

    def _single_turn_capture_probability(self, distance: Optional[int]) -> float:
        """Probability of capturing in one turn given backward distance."""
        if distance is None or distance < 1 or distance > 6:
            return 0.0
        # Simplified: assume dice uniform, but account for exact rolls
        if distance <= 6:
            return 1.0 / 6.0  # can roll exactly
        return 0.0

    def _compute_move_risk(self, move: Dict, opponent_positions: List[int]) -> float:
        """Compute probabilistic risk for a move."""
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or tgt < 0:
            return 0.0
        if move.get("move_type") == "finish" or move.get("is_safe_move") or tgt >= 100:
            return 0.0

        # Single step risk
        threats = sum(
            1
            for opp in opponent_positions
            if 1 <= self._backward_distance(tgt, opp) <= 6
        )
        immediate_risk = 1 - (5 / 6) ** threats if threats > 0 else 0.0

        # Horizon risk (simplified)
        horizon_risk = 0.0
        for opp in opponent_positions:
            d = self._backward_distance(tgt, opp)
            if d is None:
                continue
            p_turn = self._single_turn_capture_probability(d)
            # Discounted over horizon
            p_capture = 1 - (1 - p_turn) ** self.cfg.reward_cfg.horizon_turns
            horizon_risk = max(horizon_risk, p_capture)

        return (immediate_risk + horizon_risk) / 2.0  # blend

    def _compute_probabilistic_reward_modifier(
        self, move: Dict, reward_components: List[float]
    ) -> float:
        """Compute modifier for rewards based on probabilities."""
        if not self.cfg.reward_cfg.use_probabilistic_rewards:
            return 1.0

        opponent_positions = []
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = next(p for p in self.game.players if p.color.value == color)
            opponent_positions.extend(t.position for t in opp.tokens if t.position >= 0)

        risk = self._compute_move_risk(move, opponent_positions)
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

    def get_terminal_reward(self, agent_player, opponents) -> float:
        """Compute terminal rewards (win/lose)."""
        if agent_player.has_won():
            return self.cfg.reward_cfg.win
        elif any(p.has_won() for p in opponents):
            return self.cfg.reward_cfg.lose
        return 0.0
