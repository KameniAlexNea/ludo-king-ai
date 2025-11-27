"""
Tests for ludo_rl/ludo_king/game.py
"""

import unittest

from ludo_rl.ludo_king.config import config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.types import Color, Move


def _make_game() -> Game:
    players = [Player(c) for c in [Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE]]
    return Game(players=players)


class TestRollDice(unittest.TestCase):
    """Tests for Game.roll_dice()."""

    def test_returns_1_to_6(self):
        game = _make_game()
        for _ in range(100):
            roll = game.roll_dice()
            self.assertIn(roll, range(1, 7))


class TestDestinationForRoll(unittest.TestCase):
    """Tests for Game._destination_for_roll()."""

    def test_yard_needs_six(self):
        self.assertIsNone(Game._destination_for_roll(0, 5))
        self.assertEqual(Game._destination_for_roll(0, 6), config.START_POSITION)

    def test_normal_move_on_track(self):
        self.assertEqual(Game._destination_for_roll(5, 3), 8)

    def test_enters_home_column(self):
        # At pos 50, roll 3 -> 53 which is home column
        result = Game._destination_for_roll(50, 3)
        # 50 + 3 = 53, overflow = 53 - 51 = 2, home = 52 + 2 - 1 = 53
        self.assertEqual(result, 53)

    def test_overshoot_home_returns_none(self):
        # At home column pos 55, roll 6 would overshoot
        self.assertIsNone(Game._destination_for_roll(55, 6))

    def test_exact_finish(self):
        # At pos 55, roll 2 -> 57 (finish)
        self.assertEqual(Game._destination_for_roll(55, 2), config.HOME_FINISH)

    def test_already_finished_returns_none(self):
        self.assertIsNone(Game._destination_for_roll(config.HOME_FINISH, 1))


class TestLegalMoves(unittest.TestCase):
    """Tests for Game.legal_moves()."""

    def test_from_yard_with_six(self):
        game = _make_game()
        moves = game.legal_moves(0, 6)
        self.assertEqual(len(moves), 4)  # All 4 pieces can exit

    def test_from_yard_without_six(self):
        game = _make_game()
        moves = game.legal_moves(0, 3)
        self.assertEqual(len(moves), 0)

    def test_piece_on_track(self):
        game = _make_game()
        game.players[0].pieces[0].position = 10
        moves = game.legal_moves(0, 3)
        self.assertTrue(any(m.piece_id == 0 and m.new_pos == 13 for m in moves))

    def test_blocked_by_blockade(self):
        game = _make_game()
        game.players[0].pieces[0].position = 5
        # Create opponent blockade at position 7
        abs_7 = game.board.absolute_position(0, 7)
        opp_rel = game.board.relative_position(1, abs_7)
        game.players[1].pieces[0].position = opp_rel
        game.players[1].pieces[1].position = opp_rel
        # Move through blockade should not be legal
        moves = game.legal_moves(0, 3)  # 5 -> 8 crosses 7
        self.assertFalse(any(m.piece_id == 0 and m.new_pos == 8 for m in moves))


class TestApplyMove(unittest.TestCase):
    """Tests for Game.apply_move()."""

    def test_moves_piece(self):
        game = _make_game()
        game.players[0].pieces[0].position = 5
        move = Move(player_index=0, piece_id=0, new_pos=8, dice_roll=3)
        result = game.apply_move(move)
        self.assertEqual(game.players[0].pieces[0].position, 8)
        self.assertTrue(result.events.move_resolved)

    def test_exit_yard_sets_event(self):
        game = _make_game()
        move = Move(player_index=0, piece_id=0, new_pos=1, dice_roll=6)
        result = game.apply_move(move)
        self.assertTrue(result.events.exited_home)

    def test_finish_sets_event(self):
        game = _make_game()
        game.players[0].pieces[0].position = 55
        move = Move(player_index=0, piece_id=0, new_pos=57, dice_roll=2)
        result = game.apply_move(move)
        self.assertTrue(result.events.finished)
        self.assertTrue(result.extra_turn)

    def test_capture_opponent(self):
        game = _make_game()
        # Place agent at pos 10, opponent at same absolute position
        game.players[0].pieces[0].position = 10
        abs_11 = game.board.absolute_position(0, 11)
        opp_rel = game.board.relative_position(1, abs_11)
        game.players[1].pieces[0].position = opp_rel

        move = Move(player_index=0, piece_id=0, new_pos=11, dice_roll=1)
        result = game.apply_move(move)

        self.assertTrue(result.events.knockouts)
        self.assertEqual(game.players[1].pieces[0].position, 0)  # Sent home
        self.assertTrue(result.extra_turn)

    def test_no_capture_on_safe_square(self):
        game = _make_game()
        safe_abs = config.SAFE_SQUARES_ABS[0]
        agent_rel = game.board.relative_position(0, safe_abs)
        opp_rel = game.board.relative_position(1, safe_abs)

        game.players[0].pieces[0].position = agent_rel - 1
        game.players[1].pieces[0].position = opp_rel

        move = Move(player_index=0, piece_id=0, new_pos=agent_rel, dice_roll=1)
        result = game.apply_move(move)

        self.assertFalse(result.events.knockouts)
        self.assertNotEqual(game.players[1].pieces[0].position, 0)

    def test_hit_blockade(self):
        game = _make_game()
        game.players[0].pieces[0].position = 5
        # Create blockade at pos 6
        abs_6 = game.board.absolute_position(0, 6)
        opp_rel = game.board.relative_position(1, abs_6)
        game.players[1].pieces[0].position = opp_rel
        game.players[1].pieces[1].position = opp_rel

        move = Move(player_index=0, piece_id=0, new_pos=6, dice_roll=1)
        result = game.apply_move(move)

        self.assertTrue(result.events.hit_blockade)
        self.assertFalse(result.events.move_resolved)
        self.assertEqual(game.players[0].pieces[0].position, 5)  # Didn't move

    def test_form_blockade_sets_event(self):
        game = _make_game()
        game.players[0].pieces[0].position = 10
        game.players[0].pieces[1].position = 9

        move = Move(player_index=0, piece_id=1, new_pos=10, dice_roll=1)
        result = game.apply_move(move)

        self.assertTrue(result.events.blockades)

    def test_extra_turn_on_six(self):
        game = _make_game()
        game.players[0].pieces[0].position = 5
        move = Move(player_index=0, piece_id=0, new_pos=11, dice_roll=6)
        result = game.apply_move(move)
        self.assertTrue(result.extra_turn)

    def test_rewards_is_none(self):
        game = _make_game()
        game.players[0].pieces[0].position = 5
        move = Move(player_index=0, piece_id=0, new_pos=8, dice_roll=3)
        result = game.apply_move(move)
        self.assertIsNone(result.rewards)


if __name__ == "__main__":
    unittest.main()
