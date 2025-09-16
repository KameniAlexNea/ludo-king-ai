import unittest

from ludo_engine.game import LudoGame
from ludo_engine.player import PlayerColor
from ludo_engine.strategies.cautious import CautiousStrategy
from ludo_engine.strategies.killer import KillerStrategy
from ludo_engine.strategies.random_strategy import RandomStrategy


class TestHeuristicStrategies(unittest.TestCase):
    def setUp(self):
        # Create game with 4 canonical colors
        self.game = LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
                PlayerColor.YELLOW,
                PlayerColor.BLUE,
            ]
        )
        self.strategies = [RandomStrategy(), CautiousStrategy(), KillerStrategy()]

    def test_choose_move_returns_valid(self):
        # Roll dice and build AI decision context the way strategies expect
        dice = self.game.roll_dice()
        ctx = self.game.get_ai_decision_context(dice)
        valid_moves = ctx["valid_moves"]
        valid_ids = [m["token_id"] for m in valid_moves]
        for strat in self.strategies:
            token_id = strat.decide(ctx)
            if valid_moves:
                self.assertIn(token_id, valid_ids)
            else:
                # When no moves strategies default to 0; accept 0 only if there truly are no moves
                self.assertEqual(token_id, 0)


if __name__ == "__main__":
    unittest.main()
