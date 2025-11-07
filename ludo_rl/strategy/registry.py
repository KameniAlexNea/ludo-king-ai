from __future__ import annotations

from typing import Dict, Type

from .cautious import CautiousStrategy
from .defensive import DefensiveStrategy
from .finish_line import FinishLineStrategy
from .heatseeker import HeatSeekerStrategy
from .hoarder import HoarderStrategy
from .homebody import HomebodyStrategy
from .killer import KillerStrategy
from .probability import ProbabilityStrategy
from .retaliator import RetaliatorStrategy
from .rusher import RusherStrategy
from .support import SupportStrategy

STRATEGY_REGISTRY: Dict[str, Type] = {
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
}


def create(strategy_name: str, use_create=True, **kwargs):
    cls = STRATEGY_REGISTRY.get(strategy_name.lower())
    if cls is None:
        raise KeyError(f"Unknown strategy '{strategy_name}'.")
    if use_create:
        return cls.create_instance()
    return cls(**kwargs)


def available() -> Dict[str, Type]:
    return dict(STRATEGY_REGISTRY)
