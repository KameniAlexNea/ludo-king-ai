"""
Strategies module - Collection of all available Ludo AI strategies.
"""

from .base import Strategy
from .killer import KillerStrategy
from .winner import WinnerStrategy
from .optimist import OptimistStrategy
from .defensive import DefensiveStrategy
from .balanced import BalancedStrategy
from .random_strategy import RandomStrategy
from .cautious import CautiousStrategy
from .llm_strategy import LLMStrategy

# Strategy Mapping - Centralized mapping of strategy names to classes
STRATEGIES: dict[str, Strategy] = {
    "killer": KillerStrategy,
    "winner": WinnerStrategy,
    "optimist": OptimistStrategy,
    "defensive": DefensiveStrategy,
    "balanced": BalancedStrategy,
    "random": RandomStrategy,
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
    "RandomStrategy",
    "CautiousStrategy",
    "LLMStrategy",
    "STRATEGIES",
]
