import unittest

from ludo.game import LudoGame
from ludo.player import PlayerColor
from ludo.token import TokenState


class TestCaptureExtraTurnFlag(unittest.TestCase):
    def test_capture_sets_extra_turn(self):
        game = LudoGame([PlayerColor.RED, PlayerColor.GREEN])
        red = game.players[0]
        green = game.players[1]
        r_tok = red.tokens[0]
        g_tok = green.tokens[0]
        r_tok.state = TokenState.ACTIVE
        r_tok.position = 2
        game.board.add_token(r_tok, 2)
        g_tok.state = TokenState.ACTIVE
        g_tok.position = 5
        game.board.add_token(g_tok, 5)
        dice = 3
        moves = game.get_valid_moves(red, dice)
        self.assertTrue(any(m["token_id"] == 0 for m in moves))
        res = game.execute_move(red, 0, dice)
        self.assertTrue(res["success"])
        self.assertTrue(res["captured_tokens"])
        self.assertTrue(res["extra_turn"])


if __name__ == "__main__":
    unittest.main()
