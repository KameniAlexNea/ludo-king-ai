"""
Tests for ludo_rl/ludo_king/player.py
"""

import unittest

import numpy as np

from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Color, Move


class TestPlayerInit(unittest.TestCase):
    """Tests for Player initialization."""

    def test_creates_four_pieces(self):
        player = Player(Color.RED)
        self.assertEqual(len(player.pieces), 4)

    def test_pieces_start_in_yard(self):
        player = Player(Color.RED)
        for piece in player.pieces:
            self.assertEqual(piece.position, 0)

    def test_default_strategy_is_unknown(self):
        player = Player(Color.RED)
        self.assertEqual(player.strategy_name, "unknown")


class TestActivePositions(unittest.TestCase):
    """Tests for Player.active_positions()."""

    def test_returns_all_positions(self):
        player = Player(Color.RED)
        player.pieces[0].position = 5
        player.pieces[1].position = 10
        positions = player.active_positions()
        self.assertEqual(positions, [5, 10, 0, 0])


class TestCheckWon(unittest.TestCase):
    """Tests for Player.check_won()."""

    def test_not_won_initially(self):
        player = Player(Color.RED)
        self.assertFalse(player.check_won())

    def test_won_when_all_finished(self):
        player = Player(Color.RED)
        for piece in player.pieces:
            piece.position = config.HOME_FINISH
        self.assertTrue(player.check_won())

    def test_caches_won_state(self):
        player = Player(Color.RED)
        for piece in player.pieces:
            piece.position = config.HOME_FINISH
        player.check_won()
        self.assertTrue(player.has_finished)
        # Even if we move pieces back, cached flag stays true
        player.pieces[0].position = 0
        self.assertTrue(player.check_won())


class TestChoose(unittest.TestCase):
    """Tests for Player.choose()."""

    def test_returns_none_with_empty_moves(self):
        player = Player(Color.RED)
        board = np.zeros((10, config.PATH_LENGTH), dtype=np.float32)
        result = player.choose(board, 3, [])
        self.assertIsNone(result)

    def test_random_strategy_picks_from_moves(self):
        player = Player(Color.RED, strategy_name="random")
        board = np.zeros((10, config.PATH_LENGTH), dtype=np.float32)
        moves = [
            Move(player_index=0, piece_id=0, new_pos=5, dice_roll=5),
            Move(player_index=0, piece_id=1, new_pos=6, dice_roll=6),
        ]
        result = player.choose(board, 5, moves)
        self.assertIn(result, moves)


if __name__ == "__main__":
    unittest.main()
