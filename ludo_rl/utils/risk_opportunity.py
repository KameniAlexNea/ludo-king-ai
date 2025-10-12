from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from ludo_engine.core import LudoGame
from ludo_engine.models import BoardConstants, GameConstants, MoveResult, PlayerColor

from ludo_rl.config import RewardConfig
from ludo_rl.utils.reward_calculator import RewardCalculator as BaseRewardCalculator


@dataclass
class SimpleROWeights(RewardConfig):
    """Weights for the simple Risk+Opportunity score.

    Keep these modest. The intent is to provide a light heuristic signal,
    not to replace the main reward shaping.
    """

    # Additional knobs beyond RewardConfig
    progress_per_step: float = 0.5 / 6.0  # up to ~0.5 for a 6-step move
    leave_home: float = 1.0
    safe_square: float = 0.5
    risk_per_threat: float = 1.5  # subtract per opponent that can capture next turn


class RiskOpportunityCalculator:
    """Simple risk + opportunity heuristic as a class (like RewardCalculator).

    Usage:
        calc = RiskOpportunityCalculator()
        score = calc.compute(game, agent_color, move)
    """

    def __init__(self, weights: SimpleROWeights | None = None) -> None:
        self.weights = weights or SimpleROWeights()

    @staticmethod
    def _forward_distance(from_pos: int, opp_pos: int) -> int:
        if opp_pos < 0 or from_pos < 0:
            return None  # type: ignore
        if opp_pos >= from_pos:
            return opp_pos - from_pos
        else:
            return GameConstants.MAIN_BOARD_SIZE - from_pos + opp_pos

    @staticmethod
    def _iter_opponent_positions(
        game: LudoGame, agent_color: PlayerColor
    ) -> Iterable[int]:
        for p in game.players:
            if p.color == agent_color:
                continue
            for t in p.tokens:
                yield t.position

    def _risk_score(
        self,
        game: LudoGame,
        agent_color: PlayerColor,
        target_pos: int,
        w: SimpleROWeights,
    ) -> float:
        if target_pos is None or target_pos < 0:
            return 0.0
        if BoardConstants.is_safe_position(target_pos):
            return 0.0

        threats = 0
        for opp_pos in self._iter_opponent_positions(game, agent_color):
            if opp_pos is None or opp_pos < 0:
                continue
            d = self._forward_distance(opp_pos, target_pos)
            if 1 <= d <= 6:  # type: ignore
                threats += 1
        return float(threats) * w.risk_per_threat

    def _opportunity_score(
        self, move: MoveResult, old_pos: int, new_pos: int, w: SimpleROWeights
    ) -> float:
        score = 0.0
        if move.captured_tokens:
            score += w.capture * len(move.captured_tokens)
        if move.finished_token:
            score += w.finish_token
        if move.extra_turn:
            score += w.extra_turn
        if (old_pos is None or old_pos < 0) and (new_pos is not None and new_pos >= 0):
            score += w.leave_home
        if (
            new_pos is not None
            and new_pos >= 0
            and BoardConstants.is_safe_position(new_pos)
        ):
            score += w.safe_square
        if (old_pos is not None and old_pos >= 0) and (
            new_pos is not None and new_pos >= 0
        ):
            steps = min(6, self._forward_distance(old_pos, new_pos))  # type: ignore
            score += steps * w.progress_per_step
        return score

    def compute(
        self,
        game: LudoGame,
        agent_color: PlayerColor,
        move: MoveResult,
        return_breakdown: bool = False,
        is_illegal: bool = False,
    ) -> float | Tuple[float, Dict[str, float]]:
        """Return opportunity - risk (optionally with a breakdown)."""
        w = self.weights
        old_pos = move.old_position
        new_pos = move.new_position
        opp = self._opportunity_score(move, old_pos, new_pos, w)
        risk = self._risk_score(game, agent_color, new_pos, w)
        score = opp - risk
        if not return_breakdown:
            return score
        return score, {"opportunity": opp, "risk": -risk}


class MergedRewardCalculator:
    """Compute shaped rewards based on game events and state changes."""

    def __init__(self):
        self.reward_calculator = BaseRewardCalculator()
        self.ro_calculator = RiskOpportunityCalculator()

    def compute(
        self,
        game: LudoGame,
        agent_color: PlayerColor,
        move: MoveResult,
        return_breakdown: bool = False,
        is_illegal: bool = False,
    ) -> float:
        # Get RO score (includes move-specific rewards and risk penalties)
        ro_reward = self.ro_calculator.compute(
            game, agent_color, move, return_breakdown=True, is_illegal=is_illegal
        )
        # Get base reward breakdown for non-overlapping components
        _, base_breakdown = self.reward_calculator.compute(
            game,
            agent_color,
            move,
            return_breakdown=True,
            is_illegal=is_illegal,
        )

        # Components already handled by RO: progress, safe_zone, capture, finish, extra_turn, exit_start
        overlapping_keys = {
            "progress",
            "safe_zone",
            "capture",
            "finish",
            "extra_turn",
            "exit_start",
        }

        # Add non-overlapping components from base reward
        additional_reward = 0.0
        additional_breakdown = {}
        for key, value in base_breakdown.items():
            if key not in overlapping_keys:
                additional_reward += value
                additional_breakdown[key] = value

        if return_breakdown:
            ro_r, ro_b = ro_reward
            total_r = ro_r + additional_reward
            total_b = {**ro_b, **additional_breakdown}
            return total_r, total_b
        ro_r, _ = ro_reward
        return ro_r + additional_reward
