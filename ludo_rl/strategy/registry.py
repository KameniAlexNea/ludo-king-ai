from __future__ import annotations

from typing import Dict, Type

from .cautious import CautiousStrategy
from .killer import KillerStrategy
from .probability import ProbabilityStrategy

STRATEGY_REGISTRY: Dict[str, Type] = {
    ProbabilityStrategy.name: ProbabilityStrategy,
    CautiousStrategy.name: CautiousStrategy,
    KillerStrategy.name: KillerStrategy,
}


def create(strategy_name: str, **kwargs):
    cls = STRATEGY_REGISTRY.get(strategy_name.lower())
    if cls is None:
        raise KeyError(f"Unknown strategy '{strategy_name}'.")
    return cls(**kwargs)


def available() -> Dict[str, Type]:
    return dict(STRATEGY_REGISTRY)
