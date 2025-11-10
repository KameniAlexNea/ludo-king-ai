from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .types import Move


class Strategy(Protocol):
    def select_move(self, legal_moves: Sequence[Move]) -> Move | None:
        ...


@dataclass(slots=True)
class RandomStrategy:
    rng_seed: int | None = None

    def select_move(self, legal_moves: Sequence[Move]) -> Move | None:
        import random

        if not legal_moves:
            return None
        rng = random.Random(self.rng_seed)
        return rng.choice(list(legal_moves))
