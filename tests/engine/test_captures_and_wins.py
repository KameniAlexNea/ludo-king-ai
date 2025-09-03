import unittest

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.token import TokenState


class TestCapturesAndWins(unittest.TestCase):
    def setUp(self):
        self.game = LudoGame(
            [
                PlayerColor.RED,
                PlayerColor.GREEN,
            ]
        )

    def force_position(self, player_idx, token_idx, board_index):
        token = self.game.players[player_idx].tokens[token_idx]
        token.state = TokenState.ACTIVE
        token.position = board_index
        # Register on board so capture detection works
        if board_index >= 0:
            self.game.board.add_token(token, board_index)
        return token

    def test_capture(self):
        # Position opponent at 5, our token at 2; roll 3 so we land on 5 and capture
        opponent_token = self.force_position(1, 0, 5)
        our_token = self.force_position(0, 0, 2)
        dice = 3
        player = self.game.players[0]
        # Validate move exists
        moves = self.game.get_valid_moves(player, dice)
        target_moves = [m for m in moves if m["token_id"] == our_token.token_id]
        self.assertTrue(target_moves)
        res = self.game.execute_move(player, our_token.token_id, dice)
        self.assertTrue(res["success"])
        # Captured token should be sent home
        self.assertTrue(opponent_token.is_in_home())

    def test_win_progress(self):
        player = self.game.players[0]
        for t in player.tokens:
            t.state = TokenState.FINISHED
        self.assertTrue(player.has_won())


if __name__ == "__main__":
    unittest.main()
