import unittest

from ludo_engine.game import LudoGame
from ludo_engine.player import PlayerColor


class TestGameCore(unittest.TestCase):
    def setUp(self):
        self.game = LudoGame(
            [PlayerColor.RED, PlayerColor.BLUE, PlayerColor.GREEN, PlayerColor.YELLOW]
        )

    def test_initial_state(self):
        self.assertEqual(len(self.game.players), 4)
        self.assertFalse(self.game.game_over)
        for p in self.game.players:
            self.assertEqual(p.get_finished_tokens_count(), 0)

    def test_roll_dice_range(self):
        vals = [self.game.roll_dice() for _ in range(50)]
        for v in vals:
            self.assertTrue(1 <= v <= 6)

    def test_three_consecutive_sixes_blocks_moves(self):
        # Force three sixes
        self.game.consecutive_sixes = 2
        self.game.last_dice_value = 6
        self.game.consecutive_sixes = 3
        player = self.game.get_current_player()
        moves = self.game.get_valid_moves(player, 6)
        self.assertEqual(moves, [])

    def test_execute_invalid_token(self):
        player = self.game.get_current_player()
        res = self.game.execute_move(player, 99, 3)
        self.assertFalse(res["success"])


if __name__ == "__main__":
    unittest.main()
