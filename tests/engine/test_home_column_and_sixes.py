import unittest

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.token import TokenState


class TestHomeColumnAndSixes(unittest.TestCase):
    def setUp(self):
        self.game = LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
            ]
        )

    def test_exact_finish_required(self):
        player = self.game.players[0]
        token = player.tokens[0]
        # Put token in home column near finish (position 103) needs +2 to 105
        token.state = TokenState.HOME_COLUMN
        token.position = 103
        dice = 3  # overshoots (would reach 106)
        # can_move = token.can_move(dice, None)
        # In our simplified logic, can_move just checks <=105 via get_target_position logic in move(); emulate by get_target_position
        target = token.get_target_position(dice, player.start_position)
        self.assertGreater(target, 105)  # overshoot raw target
        # Move should fail as move() will reject beyond 105
        moved = token.move(dice, player.start_position)
        self.assertFalse(moved)
        # Now exact dice
        dice = 2
        moved2 = token.move(dice, player.start_position)
        self.assertTrue(moved2)
        self.assertTrue(token.is_finished())

    def test_three_consecutive_sixes_forfeit(self):
        player = self.game.get_current_player()
        # Simulate rolling three sixes
        self.game.consecutive_sixes = 3
        # Now any get_valid_moves should return [] due to rule
        dice = 6
        moves = self.game.get_valid_moves(player, dice)
        self.assertEqual(len(moves), 0)


if __name__ == "__main__":
    unittest.main()
