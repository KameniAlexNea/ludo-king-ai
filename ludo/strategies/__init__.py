"""
Strategies module - Collection of all available Ludo AI strategies.
"""

from .balanced import BalancedStrategy
from .base import Strategy
from .cautious import CautiousStrategy
from .defensive import DefensiveStrategy
from .hybrid_prob import HybridConfig, HybridProbStrategy
from .killer import KillerStrategy
from .llm import LLMStrategy
from .optimist import OptimistStrategy
from .probabilistic import ProbabilisticStrategy
from .probabilistic_v2 import ProbabilisticV2Strategy
from .probabilistic_v3 import ProbabilisticV3Strategy, V3Config
from .random_strategy import RandomStrategy
from .weighted_random import WeightedRandomStrategy
from .winner import WinnerStrategy

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
