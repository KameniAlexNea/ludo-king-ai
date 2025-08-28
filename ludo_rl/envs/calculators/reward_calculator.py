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
            if color.lower() == self.agent_color.lower():
                continue
            opp = next(
                p for p in self.game.players if p.color.value.lower() == color.lower()
            )
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

    def _compute_opportunity_score(
        self, move: Dict, opponent_positions: List[int]
    ) -> float:
        """Compute opportunity score for capturing opponents or strategic advantage."""
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or tgt < 0:
            return 0.0

        opportunity_score = 0.0

        # Capture opportunity: probability of capturing opponent tokens from this position
        for opp_pos in opponent_positions:
            if opp_pos < 0:
                continue
            # Distance from target position to opponent
            distance_to_opp = self._backward_distance(opp_pos, tgt)
            if distance_to_opp is not None and 1 <= distance_to_opp <= 6:
                # Probability of capturing this opponent in next turn
                capture_prob = self._single_turn_capture_probability(distance_to_opp)
                # Discounted probability over horizon
                horizon_capture_prob = (
                    1 - (1 - capture_prob) ** self.cfg.reward_cfg.horizon_turns
                )
                opportunity_score += horizon_capture_prob

        # Finishing opportunity: bonus for moves that set up finishing
        if move.get("move_type") == "finish":
            opportunity_score += 1.0
        elif tgt >= 50:  # Close to finishing
            finishing_distance = 56 - tgt  # Assuming finish at 56
            if finishing_distance <= 6:
                opportunity_score += (
                    self.cfg.reward_cfg.finishing_probability_weight
                    * (1.0 - finishing_distance / 6.0)
                )

        return opportunity_score

    def _compute_agent_vulnerability(
        self, agent_positions: List[int], opponent_positions: List[int]
    ) -> float:
        """Compute overall vulnerability of agent's tokens to capture."""
        if not agent_positions:
            return 0.0

        total_risk = 0.0
        for agent_pos in agent_positions:
            if agent_pos < 0:
                continue

            # Immediate risk from current position
            immediate_threats = sum(
                1
                for opp_pos in opponent_positions
                if opp_pos >= 0
                and 1 <= self._backward_distance(agent_pos, opp_pos) <= 6
            )
            immediate_risk = (
                1 - (5 / 6) ** immediate_threats if immediate_threats > 0 else 0.0
            )

            # Horizon risk
            horizon_risk = 0.0
            for opp_pos in opponent_positions:
                if opp_pos < 0:
                    continue
                distance = self._backward_distance(agent_pos, opp_pos)
                if distance is not None:
                    p_turn = self._single_turn_capture_probability(distance)
                    p_capture = 1 - (1 - p_turn) ** self.cfg.reward_cfg.horizon_turns
                    horizon_risk = max(horizon_risk, p_capture)

            position_risk = (immediate_risk + horizon_risk) / 2.0
            total_risk += position_risk

        return (
            total_risk / len([p for p in agent_positions if p >= 0])
            if agent_positions
            else 0.0
        )

    def _compute_probabilistic_move_reward(
        self, move: Dict, reward_components: List[float]
    ) -> float:
        """Compute comprehensive probabilistic reward modifier for any move."""
        if not self.cfg.reward_cfg.use_probabilistic_rewards:
            return 0.0

        # Get current positions
        agent_positions = []
        opponent_positions = []

        agent_player = next(
            p
            for p in self.game.players
            if p.color.value.lower() == self.agent_color.lower()
        )
        agent_positions.extend(
            t.position for t in agent_player.tokens if t.position >= 0
        )

        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = next(p for p in self.game.players if p.color.value == color)
            opponent_positions.extend(t.position for t in opp.tokens if t.position >= 0)

        # Calculate risk and opportunity
        move_risk = self._compute_move_risk(move, opponent_positions)
        opportunity_score = self._compute_opportunity_score(move, opponent_positions)

        # Calculate overall agent vulnerability (for risk penalty)
        agent_vulnerability = self._compute_agent_vulnerability(
            agent_positions, opponent_positions
        )

        # Combine into probabilistic reward
        risk_penalty = (
            self.cfg.reward_cfg.risk_weight * (move_risk + agent_vulnerability) / 2.0
        )
        opportunity_bonus = self.cfg.reward_cfg.opportunity_weight * opportunity_score

        probabilistic_reward = self.cfg.reward_cfg.opportunity_bonus_scale * (
            opportunity_bonus - risk_penalty
        )

        return probabilistic_reward

    def apply_probabilistic_modifier(
        self, base_reward: float, move: Optional[Dict] = None
    ) -> float:
        """Apply probabilistic modifier to any base reward."""
        if not self.cfg.reward_cfg.use_probabilistic_rewards or move is None:
            return base_reward

        probabilistic_adjustment = self._compute_probabilistic_move_reward(move, [])
        return base_reward + probabilistic_adjustment

    def compute_comprehensive_reward(
        self,
        move_res: Dict,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
    ) -> float:
        """Compute comprehensive reward including all components with probabilistic modifiers."""
        rcfg = self.cfg.reward_cfg
        total_reward = 0.0

        # Base rewards with probabilistic modifiers
        if move_res.get("captured_tokens"):
            capture_reward = self.apply_probabilistic_modifier(
                rcfg.capture * len(move_res["captured_tokens"]), move_res
            )
            total_reward += capture_reward
            reward_components.append(capture_reward)

        if move_res.get("token_finished"):
            finish_reward = self.apply_probabilistic_modifier(
                rcfg.finish_token, move_res
            )
            total_reward += finish_reward
            reward_components.append(finish_reward)

        if extra_turn:
            extra_turn_reward = self.apply_probabilistic_modifier(
                rcfg.extra_turn, move_res
            )
            total_reward += extra_turn_reward
            reward_components.append(extra_turn_reward)

        if diversity_bonus:
            diversity_reward = self.apply_probabilistic_modifier(
                rcfg.diversity_bonus, move_res
            )
            total_reward += diversity_reward
            reward_components.append(diversity_reward)

        if illegal_action:
            illegal_reward = self.apply_probabilistic_modifier(
                rcfg.illegal_action, move_res
            )
            total_reward += illegal_reward
            reward_components.append(illegal_reward)

        # Progress reward with probabilistic consideration
        if abs(progress_delta) > 1e-9:
            progress_reward = self.apply_probabilistic_modifier(
                progress_delta * rcfg.progress_scale, move_res
            )
            total_reward += progress_reward
            reward_components.append(progress_reward)

        # Time penalty (usually not modified probabilistically)
        total_reward += rcfg.time_penalty
        reward_components.append(rcfg.time_penalty)

        return total_reward
