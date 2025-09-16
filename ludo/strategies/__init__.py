"""
Strategies module - Collection of all available Ludo AI strategies.
"""

from ludo_engine.strategies.balanced import BalancedStrategy
from ludo_engine.strategies.base import Strategy
from ludo_engine.strategies.cautious import CautiousStrategy
from ludo_engine.strategies.defensive import DefensiveStrategy
from ludo_engine.strategies.hybrid_prob import HybridConfig, HybridProbStrategy
from ludo_engine.strategies.killer import KillerStrategy
from ludo_engine.strategies.llm import LLMStrategy
from ludo_engine.strategies.optimist import OptimistStrategy
from ludo_engine.strategies.probabilistic import ProbabilisticStrategy
from ludo_engine.strategies.probabilistic_v2 import ProbabilisticV2Strategy
from ludo_engine.strategies.probabilistic_v3 import ProbabilisticV3Strategy, V3Config
from ludo_engine.strategies.random_strategy import RandomStrategy
from ludo_engine.strategies.weighted_random import WeightedRandomStrategy
from ludo_engine.strategies.winner import WinnerStrategy

# Strategy Mapping - Centralized mapping of strategy names to classes
STRATEGIES: dict[str, Strategy] = {
    "killer": KillerStrategy,
    "winner": WinnerStrategy,
    "optimist": OptimistStrategy,
    "defensive": DefensiveStrategy,
    "balanced": BalancedStrategy,
    "probabilistic": ProbabilisticStrategy,
    "probabilistic_v3": ProbabilisticV3Strategy,
    "probabilistic_v2": ProbabilisticV2Strategy,
    "hybrid_prob": HybridProbStrategy,
    "random": RandomStrategy,
    "weighted_random": WeightedRandomStrategy,
    "cautious": CautiousStrategy,
    "llm": LLMStrategy,
}

__all__ = [
    "Strategy",
    "KillerStrategy",
    "WinnerStrategy",
    "OptimistStrategy",
    "DefensiveStrategy",
    "BalancedStrategy",
    "ProbabilisticStrategy",
    "ProbabilisticV2Strategy",
    "HybridProbStrategy",
    "RandomStrategy",
    "WeightedRandomStrategy",
    "CautiousStrategy",
    "LLMStrategy",
    "STRATEGIES",
    "V3Config",
    "HybridConfig",
]
