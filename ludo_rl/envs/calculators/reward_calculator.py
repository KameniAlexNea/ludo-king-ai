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

    def _compute_move_risk(
        self, move: Dict, opponent_positions: List[int]
    ) -> float:  # opponent_positions unused (legacy param)
        """Compute capture risk for the landing position using color-aware path & safety squares."""
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or tgt < 0:
            return 0.0
        if move.get("move_type") == "finish" or BoardConstants.is_safe_position(tgt):
            return 0.0

        threats = 0
        horizon_risk = 0.0
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                continue
            for tok in opp_player.tokens:
                pos = tok.position
                if pos < 0 or BoardConstants.is_home_column_position(pos):
                    continue
                dist = self._forward_distance_color_path(
                    pos, tgt, opp_player.color.value
                )
                if dist is None:
                    continue
                if GameConstants.DICE_MIN <= dist <= GameConstants.DICE_MAX:
                    threats += 1
                    p_turn = self._single_turn_capture_probability(dist)
                    p_capture = 1 - (1 - p_turn) ** self.cfg.reward_cfg.horizon_turns
                    horizon_risk = max(horizon_risk, p_capture)

        immediate_risk = 1 - (5 / 6) ** threats if threats > 0 else 0.0
        return (immediate_risk + horizon_risk) / 2.0

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
    ) -> float:  # opponent_positions legacy
        """Compute opportunity score (capture chances, finishing proximity, safety)."""
        tgt = move.get("target_position")
        if not isinstance(tgt, int) or tgt < 0:
            return 0.0
        score = 0.0

        # Capture opportunities (opponents we could reach next turn from landing square)
        for opp_player in self.game.players:
            if opp_player.color.value == self.agent_color:
                continue
            for tok in opp_player.tokens:
                opp_pos = tok.position
                if opp_pos < 0 or BoardConstants.is_home_column_position(opp_pos):
                    continue
                # If opponent could move to our target next turn (i.e., we threaten them on following turn?)
                dist = self._forward_distance_color_path(tgt, opp_pos, self.agent_color)
                if (
                    dist is not None
                    and GameConstants.DICE_MIN <= dist <= GameConstants.DICE_MAX
                ):
                    p = self._single_turn_capture_probability(dist)
                    horizon = 1 - (1 - p) ** self.cfg.reward_cfg.horizon_turns
                    score += horizon

        # Finishing / home-column proximity
        if move.get("move_type") == "finish":
            score += 1.0
        elif BoardConstants.is_home_column_position(tgt):
            remaining = GameConstants.FINISH_POSITION - tgt
            if remaining >= 0:
                score += self.cfg.reward_cfg.finishing_probability_weight * (
                    1.0 - remaining / (GameConstants.HOME_COLUMN_SIZE - 1)
                )
        else:
            entry = BoardConstants.HOME_COLUMN_ENTRIES.get(self.agent_color, 0)
            distance_to_entry = (entry - tgt) % GameConstants.MAIN_BOARD_SIZE
            if 0 < distance_to_entry <= GameConstants.DICE_MAX:
                score += self.cfg.reward_cfg.finishing_probability_weight * (
                    1.0 - distance_to_entry / GameConstants.DICE_MAX
                )

        # Safety bonus (use strategy constant scaled down)
        if BoardConstants.is_safe_position(tgt):
            score += StrategyConstants.SAFE_MOVE_BONUS / 100.0

        return score

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
                and GameConstants.DICE_MIN
                <= (self._backward_distance(agent_pos, opp_pos) or 0)
                <= GameConstants.DICE_MAX
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

    def _compute_probabilistic_multiplier(self, move: Optional[Dict]) -> float:
        """Compute a single multiplicative scaling factor for this move.

        Applied only to meaningful positive rewards (capture / finish / extra_turn /
        diversity / progress). Negative base penalties (illegal/time, terminal) stay
        absolute so they define clear boundaries.
        """
        if (
            not self.cfg.reward_cfg.use_probabilistic_rewards
            or not move
            or not isinstance(move, dict)
        ):
            return 1.0

        # Collect positions once
        agent_player = next(
            p for p in self.game.players if p.color.value == self.agent_color
        )
        agent_positions = [t.position for t in agent_player.tokens if t.position >= 0]

        opponent_positions: List[int] = []
        for color in Colors.ALL_COLORS:
            if color == self.agent_color:
                continue
            opp = next(p for p in self.game.players if p.color.value == color)
            opponent_positions.extend(t.position for t in opp.tokens if t.position >= 0)

        move_risk = self._compute_move_risk(move, opponent_positions)
        opportunity_score = self._compute_opportunity_score(move, opponent_positions)
        agent_vulnerability = self._compute_agent_vulnerability(
            agent_positions, opponent_positions
        )

        # Raw signal (positive = opportunity dominated, negative = risk dominated)
        raw = (
            self.cfg.reward_cfg.opportunity_weight * opportunity_score
            - self.cfg.reward_cfg.risk_weight * (move_risk + agent_vulnerability) / 2.0
        )
        scale = self.cfg.reward_cfg.opportunity_bonus_scale
        multiplier = 1.0 + scale * raw
        # Clamp to keep training stable
        if multiplier < 0.5:
            multiplier = 0.5
        elif multiplier > 1.5:
            multiplier = 1.5
        return multiplier

    def compute_comprehensive_reward(
        self,
        move_res: Dict,
        progress_delta: float,
        extra_turn: bool,
        diversity_bonus: bool,
        illegal_action: bool,
        reward_components: List[float],
    ) -> float:
        """Compute per-move reward.

        Probabilistic signal is a single multiplier applied to positive strategic
        components (capture / finish / extra / diversity / progress). Penalties and
        terminal rewards are unaffected to preserve their absolute semantics.
        """
        rcfg = self.cfg.reward_cfg
        total_reward = 0.0

        multiplier = self._compute_probabilistic_multiplier(move_res)

        # Base rewards (scaled)
        if move_res.get("captured_tokens"):
            capture_reward = (
                rcfg.capture * len(move_res["captured_tokens"]) * multiplier
            )
            total_reward += capture_reward
            reward_components.append(capture_reward)

        if move_res.get("token_finished"):
            finish_reward = rcfg.finish_token * multiplier
            total_reward += finish_reward
            reward_components.append(finish_reward)

        if extra_turn:
            extra_turn_reward = rcfg.extra_turn * multiplier
            total_reward += extra_turn_reward
            reward_components.append(extra_turn_reward)

        if diversity_bonus:
            diversity_reward = rcfg.diversity_bonus * multiplier
            total_reward += diversity_reward
            reward_components.append(diversity_reward)

        if illegal_action:
            total_reward += rcfg.illegal_action
            reward_components.append(rcfg.illegal_action)

        if abs(progress_delta) > 1e-9:
            progress_reward = progress_delta * rcfg.progress_scale * multiplier
            total_reward += progress_reward
            reward_components.append(progress_reward)

        # Time penalty (usually not modified probabilistically)
        total_reward += rcfg.time_penalty
        reward_components.append(rcfg.time_penalty)

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
