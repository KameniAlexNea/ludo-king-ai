from __future__ import annotations

import random
import unittest

import numpy as np

from ludo_rl.ludo_king.board import Board
from ludo_rl.ludo_king.config import config, reward_config
from ludo_rl.ludo_king.game import Game
from ludo_rl.ludo_king.player import Player
from ludo_rl.ludo_king.reward import compute_move_rewards
from ludo_rl.ludo_king.types import Color, Move


def make_players() -> list[Player]:
    return [
        Player(Color.RED),
        Player(Color.GREEN),
        Player(Color.YELLOW),
        Player(Color.BLUE),
    ]


class BoardAndGameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.players = make_players()
        self.game = Game(players=self.players)
        self.board: Board = self.game.board

    def test_absolute_and_relative_position_conversion(self) -> None:
        abs_pos = self.board.absolute_position(int(Color.RED), 10)
        self.assertEqual(abs_pos, 10)
        self.assertEqual(self.board.relative_position(int(Color.RED), abs_pos), 10)
        self.assertEqual(self.board.absolute_position(int(Color.RED), 0), -1)
        self.assertEqual(self.board.relative_position(int(Color.RED), 0), -1)

    def test_legal_moves_from_yard_and_blockade_rule(self) -> None:
        moves = self.game.legal_moves(0, 6)
        # All four pieces can enter
        self.assertEqual(len(moves), 4)
        # Create opponent blockade at absolute position corresponding to our relative 5
        abs_block = self.board.absolute_position(int(Color.RED), 5)
        opp_rel = self.board.relative_position(int(Color.GREEN), abs_block)
        self.players[1].pieces[0].position = opp_rel
        self.players[1].pieces[1].position = opp_rel
        # Place our piece at 4; with dice 1 move to 5 would hit blockade and be illegal
        self.players[0].pieces[0].position = 4
        blocked = [mv for mv in self.game.legal_moves(0, 1) if mv.new_pos == 5]
        if blocked:
            res = self.game.apply_move(blocked[0])
            self.assertTrue(res.events.hit_blockade)
            self.assertFalse(res.events.move_resolved)
        else:
            self.assertFalse(blocked)

    def test_home_column_limits(self) -> None:
        self.players[0].pieces[0].position = config.HOME_COLUMN_START
        moves = self.game.legal_moves(0, 3)
        self.assertTrue(
            any(
                mv.piece_id == 0 and mv.new_pos == config.HOME_COLUMN_START + 3
                for mv in moves
            )
        )
        moves = self.game.legal_moves(0, 6)
        self.assertFalse(any(mv.piece_id == 0 for mv in moves))

    def test_apply_move_capture_and_extra_turn(self) -> None:
        mover = self.players[0].pieces[0]
        mover.position = 10
        target_rel = 11
        target_abs = self.board.absolute_position(int(Color.RED), target_rel)
        opponent_piece = self.players[1].pieces[0]
        opponent_rel = self.board.relative_position(int(Color.GREEN), target_abs)
        opponent_piece.position = opponent_rel

        mv = Move(
            player_index=0, piece_id=mover.piece_id, new_pos=target_rel, dice_roll=1
        )
        resolution = self.game.apply_move(mv)
        rewards = compute_move_rewards(
            len(self.players),
            0,
            resolution.old_position,
            resolution.new_position,
            resolution.events,
        )
        self.assertTrue(resolution.events.knockouts)
        self.assertEqual(opponent_piece.position, 0)
        self.assertGreater(rewards[0], 0)
        self.assertTrue(resolution.extra_turn)

    def test_apply_move_hits_blockade(self) -> None:
        mover = self.players[0].pieces[0]
        new_position = 12
        abs_target = self.board.absolute_position(int(Color.RED), new_position)
        while abs_target in config.SAFE_SQUARES_ABS or new_position <= 1:
            new_position += 1
            abs_target = self.board.absolute_position(int(Color.RED), new_position)
        opponent_rel = self.board.relative_position(int(Color.GREEN), abs_target)
        self.players[1].pieces[0].position = opponent_rel
        self.players[1].pieces[1].position = opponent_rel

        mover.position = new_position - 1
        mv = Move(
            player_index=0, piece_id=mover.piece_id, new_pos=new_position, dice_roll=1
        )
        resolution = self.game.apply_move(mv)
        rewards = compute_move_rewards(
            len(self.players),
            0,
            resolution.old_position,
            resolution.new_position,
            resolution.events,
        )
        self.assertTrue(resolution.events.hit_blockade)
        self.assertFalse(resolution.events.move_resolved)
        self.assertEqual(mover.position, new_position - 1)
        self.assertEqual(rewards[0], reward_config.hit_blockade)

    def test_roll_dice_range(self) -> None:
        rolls = {self.game.roll_dice() for _ in range(50)}
        self.assertTrue(all(1 <= roll <= 6 for roll in rolls))
        self.assertEqual(len(rolls), 6)

    def test_safe_square_no_capture(self) -> None:
        # Place opponent single piece on a global safe square and land on it
        safe_abs = config.SAFE_SQUARES_ABS[1]
        opp_rel = self.board.relative_position(int(Color.GREEN), safe_abs)
        self.players[1].pieces[0].position = opp_rel
        # Move our piece to the corresponding relative ring pos
        our_rel = self.board.relative_position(int(Color.RED), safe_abs)
        self.players[0].pieces[0].position = (
            our_rel - 1 if our_rel > 1 else config.MAIN_TRACK_END
        )
        mv = Move(player_index=0, piece_id=0, new_pos=our_rel, dice_roll=1)
        res = self.game.apply_move(mv)
        # No knockout should happen on safe square
        self.assertFalse(res.events.knockouts)
        self.assertTrue(res.events.move_resolved)
        self.assertNotEqual(self.players[1].pieces[0].position, 0)

    def test_cannot_cross_blockade(self) -> None:
        # Create opponent blockade ahead on the ring
        start_rel = 6
        self.players[0].pieces[0].position = start_rel
        abs_block = self.board.absolute_position(int(Color.RED), 8)
        opp_rel = self.board.relative_position(int(Color.GREEN), abs_block)
        self.players[1].pieces[0].position = opp_rel
        self.players[1].pieces[1].position = opp_rel
        # Attempt to move 3 steps (crosses relative squares 7,8,9)
        mv = Move(player_index=0, piece_id=0, new_pos=9, dice_roll=3)
        res = self.game.apply_move(mv)
        self.assertTrue(res.events.hit_blockade)
        self.assertFalse(res.events.move_resolved)
        self.assertEqual(self.players[0].pieces[0].position, start_rel)

    def test_forming_own_blockade_sets_event_and_reward(self) -> None:
        # Choose a ring square that is not a global safe square
        target_rel = 10
        target_abs = self.board.absolute_position(int(Color.RED), target_rel)
        while target_abs in config.SAFE_SQUARES_ABS or target_rel <= 2:
            target_rel += 1
            target_abs = self.board.absolute_position(int(Color.RED), target_rel)

        # Place one of our pieces on target_rel and move another onto it to form a blockade
        p0a = self.players[0].pieces[0]
        p0b = self.players[0].pieces[1]
        p0a.position = target_rel
        # Place the second one just behind (wrap if needed)
        p0b.position = target_rel - 1 if target_rel > 1 else config.MAIN_TRACK_END

        mv = Move(
            player_index=0, piece_id=p0b.piece_id, new_pos=target_rel, dice_roll=1
        )
        res = self.game.apply_move(mv)

        # Move should resolve and create a blockade event
        self.assertTrue(res.events.move_resolved)
        self.assertTrue(res.events.blockades)
        self.assertGreaterEqual(
            self.game.board.count_at_relative(int(Color.RED), target_rel), 2
        )
        # Rewards should at least include blockade bonus
        self.assertIsNotNone(res.rewards)
        self.assertGreaterEqual(res.rewards[0], reward_config.blockade)


class PlayerBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1)
        self.player = Player(Color.RED)

    def test_check_won_states(self) -> None:
        self.assertFalse(self.player.check_won())
        for piece in self.player.pieces:
            piece.position = config.HOME_FINISH
        self.assertTrue(self.player.check_won())
        self.assertTrue(self.player.check_won())  # cached flag path

    def test_choose_no_valid_moves_returns_none(self) -> None:
        board = np.zeros((10, config.PATH_LENGTH), dtype=float)
        self.assertIsNone(self.player.choose(board, 3, []))


if __name__ == "__main__":
    unittest.main()
