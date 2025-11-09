from __future__ import annotations

from typing import Dict, Type

from .base import BaseStrategy
from .cautious import CautiousStrategy
from .defensive import DefensiveStrategy
from .finish_line import FinishLineStrategy
from .heatseeker import HeatSeekerStrategy
from .hoarder import HoarderStrategy
from .homebody import HomebodyStrategy
from .human import HumanStrategy
from .killer import KillerStrategy
from .llm_agent import LLMStrategy
from .probability import ProbabilityStrategy
from .retaliator import RetaliatorStrategy
from .rl_agent import RLStrategy
from .rusher import RusherStrategy
from .support import SupportStrategy

STRATEGY_REGISTRY: Dict[str, BaseStrategy] = {
    ProbabilityStrategy.name: ProbabilityStrategy,
    CautiousStrategy.name: CautiousStrategy,
    KillerStrategy.name: KillerStrategy,
    DefensiveStrategy.name: DefensiveStrategy,
    FinishLineStrategy.name: FinishLineStrategy,
    HeatSeekerStrategy.name: HeatSeekerStrategy,
    HoarderStrategy.name: HoarderStrategy,
    HomebodyStrategy.name: HomebodyStrategy,
    RusherStrategy.name: RusherStrategy,
    SupportStrategy.name: SupportStrategy,
    RetaliatorStrategy.name: RetaliatorStrategy,
    RLStrategy.name: RLStrategy,
    LLMStrategy.name: LLMStrategy,
    HumanStrategy.name: HumanStrategy,
}


def create(strategy_name: str, use_create=True, **kwargs):
    cls = STRATEGY_REGISTRY.get(strategy_name.lower())
    if cls is None:
        raise KeyError(f"Unknown strategy '{strategy_name}'.")
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
