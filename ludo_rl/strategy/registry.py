from __future__ import annotations

from typing import Dict, Type

from .base import BaseStrategy
from .human import HumanStrategy
from .ml_strategies.llm_agent import LLMStrategy
from .ml_strategies.rl_agent import RLStrategy
from .scripted_strategies import (
    CautiousStrategy,
    DefensiveStrategy,
    HoarderStrategy,
    HomebodyStrategy,
    HybridStrategy,
    KillerStrategy,
)

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    CautiousStrategy.name: CautiousStrategy,
    DefensiveStrategy.name: DefensiveStrategy,
    HoarderStrategy.name: HoarderStrategy,
    HomebodyStrategy.name: HomebodyStrategy,
    KillerStrategy.name: KillerStrategy,
    HybridStrategy.name: HybridStrategy,
}


def create(strategy_name: str, use_create=True, **kwargs):
    cls = STRATEGY_REGISTRY.get(strategy_name.lower())
    if cls is None:
        raise KeyError(f"'{strategy_name}' is not a registered strategy.")
    if use_create:
        return cls.create_instance()
    return cls(**kwargs)


def available(ignore_human: bool = True) -> Dict[str, Type]:
    if ignore_human:
        return {
            name: cls
            for name, cls in STRATEGY_REGISTRY.items()
            if name not in (HumanStrategy.name, LLMStrategy.name, RLStrategy.name)
        }
    return dict(STRATEGY_REGISTRY)
