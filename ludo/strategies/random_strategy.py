"""
Random Strategy - Makes completely random valid moves.
"""

import random
from typing import Dict

from ludo_engine.strategies.base import Strategy


class RandomStrategy(Strategy):
    """
    Random strategy for baseline comparison.
    Makes completely random valid moves.
    """

    def __init__(self):
        super().__init__("Random", "Baseline strategy that makes random valid moves")

    def decide(self, game_context: Dict) -> int:
        valid_moves = self._get_valid_moves(game_context)

        if not valid_moves:
            return 0

        # Completely random choice
        random_move = random.choice(valid_moves)
        return random_move["token_id"]
