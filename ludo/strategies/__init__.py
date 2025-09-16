"""
Strategies module - Collection of all available Ludo AI strategies.
"""

from ludo.strategies.balanced import BalancedStrategy
from ludo.strategies.base import Strategy
from ludo.strategies.cautious import CautiousStrategy
from ludo.strategies.defensive import DefensiveStrategy
from ludo.strategies.hybrid_prob import HybridConfig, HybridProbStrategy
from ludo.strategies.killer import KillerStrategy
from ludo.strategies.llm import LLMStrategy
from ludo.strategies.optimist import OptimistStrategy
from ludo.strategies.probabilistic import ProbabilisticStrategy
from ludo.strategies.probabilistic_v2 import ProbabilisticV2Strategy
from ludo.strategies.probabilistic_v3 import ProbabilisticV3Strategy, V3Config
from ludo.strategies.random_strategy import RandomStrategy
from ludo.strategies.weighted_random import WeightedRandomStrategy
from ludo.strategies.winner import WinnerStrategy

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
