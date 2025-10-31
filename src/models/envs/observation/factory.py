"""Factory helpers for constructing observation builders."""

from __future__ import annotations

from ludo_engine.core import LudoGame
from ludo_engine.models import PlayerColor

from ...configs.config import EnvConfig
from .base import ObservationBuilderBase
from .continuous import ContinuousObservationBuilder
from .flattened import FlattenedObservationBuilder


def make_observation_builder(
    cfg: EnvConfig, game: LudoGame, agent_color: PlayerColor
) -> ObservationBuilderBase:
    if cfg.multi_agent:
        return FlattenedObservationBuilder(cfg, game, agent_color)
    return ContinuousObservationBuilder(cfg, game, agent_color)


__all__ = [
    "make_observation_builder",
]
