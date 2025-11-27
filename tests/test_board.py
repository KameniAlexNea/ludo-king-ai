"""
Tests for ludo_rl/ludo_king/board.py
"""

import unittest

import numpy as np

from ludo_rl.ludo_king.board import Board
from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.piece import Piece


def _make_board(num_players: int = 4) -> Board:
    """Create a board with pieces for testing."""
    colors = list(range(num_players))
    players = [[Piece(c, i) for i in range(4)] for c in colors]
    return Board(players=players, colors=colors)


class TestAbsolutePosition(unittest.TestCase):
    """Tests for Board.absolute_position()."""

    def test_player_0_relative_1_is_start(self):
        board = _make_board()
        # Player 0 starts at abs 1
        self.assertEqual(board.absolute_position(0, 1), 1)

    def test_player_1_relative_1_is_start(self):
        board = _make_board()
        # Player 1 starts at abs 14
        self.assertEqual(board.absolute_position(1, 1), 14)

    def test_wraps_around_track(self):
        board = _make_board()
        # Player 0 at rel 51: (1 + 51 - 1) % 52 = 51
        # Player 1 at rel 51: (14 + 51 - 1) % 52 = 12
        self.assertEqual(board.absolute_position(0, 51), 51)
        self.assertEqual(board.absolute_position(1, 51), 12)

    def test_invalid_position_returns_minus_one(self):
        board = _make_board()
        self.assertEqual(board.absolute_position(0, 0), -1)
        self.assertEqual(board.absolute_position(0, 100), -1)


class TestRelativePosition(unittest.TestCase):
    """Tests for Board.relative_position()."""

    def test_inverse_of_absolute(self):
        board = _make_board()
        for color in range(4):
            for rel in range(1, 52):
                abs_pos = board.absolute_position(color, rel)
                if abs_pos > 0:
                    self.assertEqual(board.relative_position(color, abs_pos), rel)

    def test_invalid_abs_returns_minus_one(self):
        board = _make_board()
        self.assertEqual(board.relative_position(0, 0), -1)
        self.assertEqual(board.relative_position(0, -5), -1)


class TestTranslateRelative(unittest.TestCase):
    """Tests for Board.translate_relative()."""

    def test_same_color_no_change(self):
        board = _make_board()
        self.assertEqual(board.translate_relative(0, 0, 10), 10)

    def test_different_colors_translate(self):
        board = _make_board()
        # Player 1 rel 1 -> abs 14 -> Player 0 rel = (14 - 1 + 52) % 52 + 1 = 14
        self.assertEqual(board.translate_relative(1, 0, 1), 14)


class TestPiecesAtAbsolute(unittest.TestCase):
    """Tests for Board.pieces_at_absolute()."""

    def test_finds_piece_at_position(self):
        board = _make_board()
        board.players[0][0].position = 1  # rel 1 -> abs 1
        pieces = board.pieces_at_absolute(1)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0][0], 0)  # color 0

    def test_exclude_color_filters(self):
        board = _make_board()
        board.players[0][0].position = 1
        pieces = board.pieces_at_absolute(1, exclude_color=0)
        self.assertEqual(len(pieces), 0)

    def test_empty_when_no_pieces(self):
        board = _make_board()
        pieces = board.pieces_at_absolute(25)
        self.assertEqual(len(pieces), 0)


class TestCountAtRelative(unittest.TestCase):
    """Tests for Board.count_at_relative()."""

    def test_counts_pieces(self):
        board = _make_board()
        board.players[0][0].position = 5
        board.players[0][1].position = 5
        self.assertEqual(board.count_at_relative(0, 5), 2)

    def test_zero_when_empty(self):
        board = _make_board()
        self.assertEqual(board.count_at_relative(0, 10), 0)


class TestGetBlockadePositions(unittest.TestCase):
    """Tests for Board.get_blockade_positions()."""

    def test_detects_blockade(self):
        board = _make_board()
        board.players[0][0].position = 10
        board.players[0][1].position = 10
        blockades = board.get_blockade_positions()
        abs_pos = board.absolute_position(0, 10)
        self.assertIn(abs_pos, blockades)
        self.assertEqual(blockades[abs_pos], 0)

    def test_no_blockade_with_one_piece(self):
        board = _make_board()
        board.players[0][0].position = 10
        blockades = board.get_blockade_positions()
        self.assertEqual(len(blockades), 0)

    def test_ignores_yard(self):
        board = _make_board()
        # All pieces start at 0 (yard) - not a blockade
        blockades = board.get_blockade_positions()
        self.assertEqual(len(blockades), 0)


class TestResetTransitionSummaries(unittest.TestCase):
    """Tests for Board.reset_transition_summaries()."""

    def test_clears_all_arrays(self):
        board = _make_board()
        board.movement_heatmap[5] = 1.0
        board.my_knockouts[10] = 1.0
        board.opp_knockouts[15] = 1.0
        board.new_blockades[20] = 1.0
        board.reward_heatmap[25] = 1.0

        board.reset_transition_summaries()

        self.assertEqual(board.movement_heatmap.sum(), 0.0)
        self.assertEqual(board.my_knockouts.sum(), 0.0)
        self.assertEqual(board.opp_knockouts.sum(), 0.0)
        self.assertEqual(board.new_blockades.sum(), 0.0)
        self.assertEqual(board.reward_heatmap.sum(), 0.0)


class TestTokenOrderForAgent(unittest.TestCase):
    """Tests for Board.token_order_for_agent()."""

    def test_agent_first(self):
        board = _make_board()
        order = board.token_order_for_agent(0)
        self.assertEqual(order[0], 0)

    def test_cyclic_order(self):
        board = _make_board()
        order = board.token_order_for_agent(2)
        self.assertEqual(order, [2, 3, 0, 1])


class TestAllTokenPositions(unittest.TestCase):
    """Tests for Board.all_token_positions()."""

    def test_returns_16_positions(self):
        board = _make_board()
        positions = board.all_token_positions(0)
        self.assertEqual(len(positions), 16)

    def test_initial_positions_are_zero(self):
        board = _make_board()
        positions = board.all_token_positions(0)
        self.assertTrue(np.all(positions == 0))

    def test_reflects_piece_positions(self):
        board = _make_board()
        board.players[0][0].position = 10
        positions = board.all_token_positions(0)
        self.assertEqual(positions[0], 10)


class TestBuildTensor(unittest.TestCase):
    """Tests for Board.build_tensor()."""

    def test_shape_is_10_by_path_length(self):
        board = _make_board()
        tensor = board.build_tensor(0)
        self.assertEqual(tensor.shape, (10, config.PATH_LENGTH))

    def test_initial_pieces_in_yard(self):
        board = _make_board()
        tensor = board.build_tensor(0)
        # 4 pieces in yard for each player
        for ch in range(4):
            self.assertEqual(tensor[ch][0], 4.0)

    def test_piece_on_track_shows_in_tensor(self):
        board = _make_board()
        board.players[0][0].position = 5
        tensor = board.build_tensor(0)
        self.assertEqual(tensor[0][5], 1.0)
        self.assertEqual(tensor[0][0], 3.0)  # 3 remaining in yard

    def test_safe_zones_channel(self):
        board = _make_board()
        tensor = board.build_tensor(0)
        # Home column is safe
        for pos in range(config.HOME_COLUMN_START, config.HOME_FINISH):
            self.assertEqual(tensor[4][pos], 1.0)

    def test_transition_summaries_in_tensor(self):
        board = _make_board()
        board.movement_heatmap[10] = 2.0
        board.reward_heatmap[20] = 0.5
        tensor = board.build_tensor(0)
        self.assertEqual(tensor[5][10], 2.0)
        self.assertEqual(tensor[9][20], 0.5)

    def test_reuses_buffer(self):
        board = _make_board()
        t1 = board.build_tensor(0)
        t2 = board.build_tensor(0)
        self.assertIs(t1, t2)


if __name__ == "__main__":
    unittest.main()
