import random

from .base import BaseStrategy


class HumanStrategy(BaseStrategy):
    name = "human"

    def select_move(self, ctx):
        moves = list(ctx.iter_legal())
        return random.choice(moves) if moves else None
