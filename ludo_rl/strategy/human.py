import random
from typing import ClassVar

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


class HumanStrategyConfig(BaseStrategyConfig):
    def sample(self, rng: random.Random | None = None) -> dict[str, object]:
        return {}


class HumanStrategy(BaseStrategy):
    name = "human"
    config: ClassVar[HumanStrategyConfig] = HumanStrategyConfig()

    def _softmax(self, ctx: StrategyContext, move: MoveOption):
        return 1.0
