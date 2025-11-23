from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ludo_rl.strategy.base import BaseStrategy, BaseStrategyConfig
from ludo_rl.strategy.features import build_move_options, model_arena_results

from .cautious import CautiousStrategy, CautiousStrategyConfig
from .defensive import DefensiveStrategy, DefensiveStrategyConfig
from .hoarder import HoarderStrategy, HoarderStrategyConfig
from .homebody import HomebodyStrategy, HomebodyStrategyConfig
from .killer import KillerStrategy, KillerStrategyConfig


@dataclass(slots=True)
class HybridStrategyConfig(BaseStrategyConfig):
    cautious: CautiousStrategyConfig = field(default_factory=CautiousStrategyConfig)
    defensive: DefensiveStrategyConfig = field(default_factory=DefensiveStrategyConfig)
    hoarder: HoarderStrategyConfig = field(default_factory=HoarderStrategyConfig)
    homebody: HomebodyStrategyConfig = field(default_factory=HomebodyStrategyConfig)
    killer: KillerStrategyConfig = field(default_factory=KillerStrategyConfig)

    def sample(self, rng=None):
        return {
            "cautious": self.cautious.sample(rng),
            "defensive": self.defensive.sample(rng),
            "hoarder": self.hoarder.sample(rng),
            "homebody": self.homebody.sample(rng),
            "killer": self.killer.sample(rng),
        }


class HybridStrategy(BaseStrategy):
    """Aggregates softmax decisions from top 5 scripted strategies, weighted by tournament points."""

    name: ClassVar[str] = "hybrid"

    config: ClassVar[HybridStrategyConfig] = HybridStrategyConfig()
    weights: np.ndarray
    strategy_names = ["cautious", "defensive", "hoarder", "homebody", "killer"]

    def __init__(
        self,
        cautious: dict,
        defensive: dict,
        hoarder: dict,
        homebody: dict,
        killer: dict,
    ):
        self.sub_strategies: dict[str, BaseStrategy] = {
            "cautious": CautiousStrategy(**cautious),
            "defensive": DefensiveStrategy(**defensive),
            "hoarder": HoarderStrategy(**hoarder),
            "homebody": HomebodyStrategy(**homebody),
            "killer": KillerStrategy(**killer),
        }

        results = dict(model_arena_results())
        points = np.array([results[name] for name in self.strategy_names])
        self.weights = points / points.sum()

    def decide(
        self,
        board_stack: np.ndarray,
        dice_roll: int,
        action_mask: np.ndarray,
        move_choices: list[dict | None],
    ) -> int | None:
        ctx = build_move_options(board_stack, dice_roll, action_mask, move_choices)
        legal_moves = list(ctx.iter_legal())
        if not legal_moves:
            return None

        n_moves = len(legal_moves)
        n_strats = len(self.strategy_names)
        probs = np.zeros((n_strats, n_moves))

        scores = np.array(
            [
                [self.sub_strategies[strat]._score_move(ctx, m) for m in legal_moves]
                for strat in self.strategy_names
            ]
        )
        scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = scores / scores.sum(axis=1, keepdims=True)

        # Weighted average probs
        avg_probs = np.average(probs, axis=0, weights=self.weights)
        return random.choices(legal_moves, weights=avg_probs, k=1)[0]

    @classmethod
    def create_instance(cls, rng: random.Random | None = None) -> "HybridStrategy":
        params = cls.config.sample(rng)
        return cls(**params)
