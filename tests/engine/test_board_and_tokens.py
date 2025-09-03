import unittest
from ludo.game import LudoGame
from ludo.player import PlayerColor

class TestBoardAndTokens(unittest.TestCase):
    def setUp(self):
        self.game = LudoGame([
            PlayerColor.RED,
            PlayerColor.GREEN,
            PlayerColor.YELLOW,
            PlayerColor.BLUE,
        ])

    def test_initial_tokens_home(self):
        for player in self.game.players:
            for token in player.tokens:
                self.assertTrue(token.is_in_home())

    def test_enter_board_on_six(self):
        player = self.game.get_current_player()
        dice = 6
        moves = self.game.get_valid_moves(player, dice)
        # Any move that exits home will have move_type 'exit_home'
        exit_moves = [m for m in moves if m['move_type'] == 'exit_home']
        self.assertTrue(len(exit_moves) > 0)

    def test_no_enter_without_six(self):
        player = self.game.get_current_player()
        dice = 5
        moves = self.game.get_valid_moves(player, dice)
        exit_moves = [m for m in moves if m['move_type'] == 'exit_home']
        self.assertEqual(len(exit_moves), 0)

if __name__ == '__main__':
    unittest.main()
