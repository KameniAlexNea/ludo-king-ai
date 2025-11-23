from __future__ import annotations

from typing import Dict, Type

from .base import BaseStrategy
from .cautious import CautiousStrategy
from .defensive import DefensiveStrategy
from .hoarder import HoarderStrategy
from .homebody import HomebodyStrategy
from .human import HumanStrategy
from .killer import KillerStrategy
from .llm_agent import LLMStrategy
from .rl_agent import RLStrategy

STRATEGY_REGISTRY: Dict[str, BaseStrategy] = {
    CautiousStrategy.name: CautiousStrategy,
    KillerStrategy.name: KillerStrategy,
    DefensiveStrategy.name: DefensiveStrategy,
    HoarderStrategy.name: HoarderStrategy,
    HomebodyStrategy.name: HomebodyStrategy,
    RLStrategy.name: RLStrategy,
    LLMStrategy.name: LLMStrategy,
    HumanStrategy.name: HumanStrategy,
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
