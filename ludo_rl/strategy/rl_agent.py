from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import numpy as np
from sb3_contrib import MaskablePPO

from .base import BaseStrategy, BaseStrategyConfig
from .types import MoveOption, StrategyContext


@dataclass(slots=True)
class RLStrategyConfig(BaseStrategyConfig):
    model: MaskablePPO
    deterministic: bool = True

    def sample(self, rng=None) -> Dict[str, object]:  # noqa: D401 - interface impl
        return {"model": self.model, "deterministic": self.deterministic}


class RLStrategy(BaseStrategy):
    """Delegate move selection to a pre-trained MaskablePPO policy."""

    name = "rl"
    config: ClassVar[Optional[BaseStrategyConfig]] = None

    def __init__(self, model: MaskablePPO, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic
        self.model.policy.set_training_mode(False)

    @classmethod
    def configure(cls, model: MaskablePPO, *, deterministic: bool = True) -> None:
        return cls(model=model, deterministic=deterministic)

    @classmethod
    def configure_from_path(
        cls, model_path: str, *, device: str = "cpu", deterministic: bool = True
    ) -> None:
        model = MaskablePPO.load(model_path, device=device)
        model.policy.set_training_mode(False)
        return cls.configure(model=model, deterministic=deterministic)

    def select_move(self, ctx: StrategyContext) -> Optional[MoveOption]:  # type: ignore[override]
        if not np.any(ctx.action_mask):
            return None

        observation = {
            "board": np.asarray(ctx.board, dtype=np.float32),
            "dice_roll": np.asarray([ctx.dice_roll - 1], dtype=np.int64),
        }

        action, _ = self.model.predict(
            observation,
            action_masks=ctx.action_mask.astype(bool),
            deterministic=self.deterministic,
        )

        action_id = int(np.asarray(action).item())
        if action_id < 0 or action_id >= len(ctx.action_mask):
            return next(ctx.iter_legal(), None)

        if not ctx.action_mask[action_id]:
            return next(ctx.iter_legal(), None)

        for move in ctx.moves:
            if move.piece_id == action_id:
                return move

        return next(ctx.iter_legal(), None)
